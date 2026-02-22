"""
DRaFT-style test-time training (TTT) for DPS.

For each test measurement y, fine-tunes the score model's attention layers
via LoRA so that DPS produces reconstructions with better measurement
consistency.  The reward signal is R(x0) = -||y - A(x0)||^2, backpropagated
through the final Tweedie step (ReFL, K=1) or multiple trailing steps
(DRaFT, K>1).
"""

import yaml
import torch
import torch.nn as nn
import numpy as np
import tqdm
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchvision.utils import save_image

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from eval import get_eval_fn, Evaluator
from sampler import get_sampler
from cores.scheduler import get_diffusion_scheduler
from lora import apply_lora, remove_lora, get_lora_params, frozen_tweedie


# ---------------------------------------------------------------------------
# DPS sampling prefix (no-grad) — replicates sampler.py DPS.sample logic
# ---------------------------------------------------------------------------

def dps_sample_prefix(model, scheduler, guidance_scale, x_start, operator,
                      measurement, num_steps):
    """Run the first ``num_steps`` of DPS (Euler PF-ODE with guidance).

    This mirrors the loop in :class:`sampler.DPS.sample` but stops early and
    returns the state so the caller can attach a differentiable Tweedie step.

    Args:
        model: Diffusion model (DDPM wrapper).
        scheduler: Diffusion scheduler with sigma_steps etc.
        guidance_scale: DPS guidance weight.
        x_start: Initial noisy sample (B, C, H, W).
        operator: Forward measurement operator.
        measurement: Observed y (B, …).
        num_steps: How many Euler steps to run (the *prefix*).

    Returns:
        tuple: (xt, sigma, st) — current sample, noise level, and scaling
        at the step right after the prefix ends.
    """
    sigma_steps = scheduler.sigma_steps
    total_steps = len(sigma_steps) - 1  # last entry is 0
    assert num_steps <= total_steps, (
        f"num_steps ({num_steps}) must be <= total DPS steps ({total_steps})"
    )

    xt = x_start
    for step in range(num_steps):
        sigma = sigma_steps[step]
        sigma_next = sigma_steps[step + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        # --- guidance gradient (same as DPS.sample) ---
        model.requires_grad_(True)
        xt_in = xt.detach().requires_grad_(True)
        x0hat = model.tweedie(xt_in / st, sigma)
        loss_per_sample = operator.loss(x0hat, measurement)
        grad_xt = torch.autograd.grad(loss_per_sample.sum(), xt_in)[0]
        model.requires_grad_(False)

        with torch.no_grad():
            norm_factor = loss_per_sample.sqrt().view(-1, *([1] * (grad_xt.ndim - 1)))
            norm_factor = norm_factor.clamp(min=1e-8)
            normalized_grad = grad_xt / norm_factor

        # --- PF-ODE Euler step ---
        with torch.no_grad():
            score = (x0hat.detach() - xt / st) / sigma ** 2
            deriv = dst / st * xt - st * dsigma * sigma * score
            xt_next = xt + dt * deriv
            xt = xt_next - guidance_scale * normalized_grad

            if torch.isnan(xt).any():
                break

    # Return the state at the boundary: sigma/st for the *next* step
    sigma_boundary = sigma_steps[num_steps]
    t_boundary = scheduler.get_sigma_inv(sigma_boundary)
    st_boundary = scheduler.get_scaling(t_boundary)
    return xt.detach(), sigma_boundary, st_boundary


# ---------------------------------------------------------------------------
# Per-image test-time training
# ---------------------------------------------------------------------------

@torch.no_grad()
def dps_final_sample(model, sampler, x_start, operator, measurement):
    """Full DPS sample using the (possibly LoRA-adapted) model."""
    return sampler.sample(model, x_start, operator, measurement, verbose=False)


def run_ttt(model, sampler, operator, measurement, gt, evaluator,
            lora_rank=4, lora_alpha=1.0, lr=1e-4, num_ttt_steps=50,
            lambda_kl=0.01, grad_clip=1.0, K=1, verbose=True):
    """Test-time train LoRA on the score model for a single measurement.

    Args:
        model: DDPM diffusion model.
        sampler: DPS sampler instance (carries scheduler + guidance_scale).
        operator: Forward operator A.
        measurement: Observed y, shape (B, …).
        gt: Ground-truth image (for evaluation only).
        evaluator: Evaluator instance.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lr: Learning rate for AdamW.
        num_ttt_steps: Number of TTT optimisation iterations.
        lambda_kl: Weight on KL (frozen-model) regularisation.
        grad_clip: Max gradient norm for clipping.
        K: Number of trailing Tweedie steps with gradient (1 = ReFL).
        verbose: Print progress.

    Returns:
        dict with keys 'sample', 'metrics_before', 'metrics_after', 'losses'.
    """
    device = next(model.parameters()).device
    scheduler = sampler.scheduler
    guidance_scale = sampler.guidance_scale
    sigma_steps = scheduler.sigma_steps
    total_dps_steps = len(sigma_steps) - 1
    prefix_steps = max(total_dps_steps - K, 0)

    # --- baseline (before TTT) ---
    with torch.no_grad():
        x_start_eval = sampler.get_start(measurement.shape[0], model)
        sample_before = dps_final_sample(model, sampler, x_start_eval, operator, measurement)
        metrics_before = evaluator(gt, measurement, sample_before)

    # --- inject LoRA ---
    lora_modules = apply_lora(model, rank=lora_rank, alpha=lora_alpha)
    lora_params = get_lora_params(lora_modules)
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)

    losses = []
    pbar = tqdm.trange(num_ttt_steps, desc="TTT") if verbose else range(num_ttt_steps)

    for ttt_step in pbar:
        optimizer.zero_grad()

        # fresh noise
        x_T = sampler.get_start(measurement.shape[0], model)

        # --- prefix: run DPS steps without LoRA gradient ---
        if prefix_steps > 0:
            xt, sigma_boundary, st_boundary = dps_sample_prefix(
                model, scheduler, guidance_scale, x_T, operator,
                measurement, prefix_steps,
            )
        else:
            xt = x_T.detach()
            sigma_boundary = sigma_steps[0]
            t_boundary = scheduler.get_sigma_inv(sigma_boundary)
            st_boundary = scheduler.get_scaling(t_boundary)

        # --- gradient-tracked trailing step(s) ---
        # For K=1 (ReFL): single Tweedie at the boundary
        # For K>1 (DRaFT): run K Euler+Tweedie steps with grad
        if K == 1:
            # Single Tweedie prediction at the boundary noise level
            model.requires_grad_(True)
            x0_hat = model.tweedie(xt / st_boundary, sigma_boundary)
            model.requires_grad_(False)
        else:
            # DRaFT: K trailing PF-ODE steps with gradient
            xt_grad = xt.requires_grad_(False).clone()
            for k_step in range(K):
                step_idx = prefix_steps + k_step
                if step_idx >= total_dps_steps:
                    break
                sigma = sigma_steps[step_idx]
                sigma_next = sigma_steps[step_idx + 1]
                t = scheduler.get_sigma_inv(sigma)
                t_next = scheduler.get_sigma_inv(sigma_next)
                dt = t_next - t
                st = scheduler.get_scaling(t)
                dst = scheduler.get_scaling_derivative(t)
                dsigma = scheduler.get_sigma_derivative(t)

                model.requires_grad_(True)
                x0_hat = model.tweedie(xt_grad / st, sigma)

                # Euler step (keep in graph for DRaFT backprop)
                score = (x0_hat - xt_grad / st) / sigma ** 2
                deriv = dst / st * xt_grad - st * dsigma * sigma * score
                xt_grad = xt_grad + dt * deriv
                model.requires_grad_(False)

            # Final Tweedie at the last reached noise level
            final_step_idx = min(prefix_steps + K, total_dps_steps)
            sigma_final = sigma_steps[final_step_idx] if final_step_idx < len(sigma_steps) else sigma_steps[-1]
            t_final = scheduler.get_sigma_inv(sigma_final)
            st_final = scheduler.get_scaling(t_final)

            model.requires_grad_(True)
            x0_hat = model.tweedie(xt_grad / st_final, sigma_final)
            model.requires_grad_(False)

        # --- reward loss: measurement consistency ---
        reward_loss = operator.loss(x0_hat, measurement).mean()

        # --- KL regularisation (optional) ---
        kl_loss = torch.tensor(0.0, device=device)
        if lambda_kl > 0:
            with torch.no_grad():
                x0_frozen = frozen_tweedie(model, lora_modules, xt / st_boundary, sigma_boundary)
            kl_loss = ((x0_hat - x0_frozen) ** 2).mean()

        total_loss = reward_loss + lambda_kl * kl_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
        optimizer.step()

        loss_val = total_loss.item()
        losses.append(loss_val)
        if verbose:
            pbar.set_postfix(loss=f"{loss_val:.4f}", reward=f"{reward_loss.item():.4f}")

    # --- final evaluation with finetuned model ---
    with torch.no_grad():
        x_start_eval = sampler.get_start(measurement.shape[0], model)
        sample_after = dps_final_sample(model, sampler, x_start_eval, operator, measurement)
        metrics_after = evaluator(gt, measurement, sample_after)

    # --- cleanup ---
    remove_lora(model)

    return {
        "sample_before": sample_before,
        "sample_after": sample_after,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="default.yaml")
def main(args: DictConfig):
    # --- reproducibility ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # --- TTT hyperparameters (with defaults) ---
    ttt_cfg = OmegaConf.to_container(args.get("ttt", {}), resolve=True)
    lora_rank = ttt_cfg.get("lora_rank", 4)
    lora_alpha = ttt_cfg.get("lora_alpha", 1.0)
    lr = ttt_cfg.get("lr", 1e-4)
    num_ttt_steps = ttt_cfg.get("num_ttt_steps", 50)
    lambda_kl = ttt_cfg.get("lambda_kl", 0.01)
    grad_clip = ttt_cfg.get("grad_clip", 1.0)
    K = ttt_cfg.get("K", 1)

    # --- data ---
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # --- operator & measurement ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # --- sampler (must be DPS) ---
    sampler = get_sampler(**args.sampler, mcmc_sampler_config=task_group.get("mcmc_sampler_config", None))

    # --- model ---
    model = get_model(**args.model)

    # --- evaluator ---
    eval_fn_list = [get_eval_fn(name) for name in args.eval_fn_list]
    evaluator = Evaluator(eval_fn_list)

    # --- output directory ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    root = save_dir / args.name
    root.mkdir(exist_ok=True)
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    # --- per-image TTT loop ---
    all_before, all_after = [], []
    for img_idx in range(total_number):
        print(f"\n=== Image {img_idx + 1}/{total_number} ===")
        gt_i = images[img_idx: img_idx + 1]
        y_i = y[img_idx: img_idx + 1]

        result = run_ttt(
            model, sampler, operator, y_i, gt_i, evaluator,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lr=lr,
            num_ttt_steps=num_ttt_steps, lambda_kl=lambda_kl,
            grad_clip=grad_clip, K=K, verbose=True,
        )

        # log per-image metrics
        for name in result["metrics_before"]:
            before_val = result["metrics_before"][name].item()
            after_val = result["metrics_after"][name].item()
            print(f"  {name}: {before_val:.4f} -> {after_val:.4f}")

        all_before.append(result["sample_before"])
        all_after.append(result["sample_after"])

        # save samples
        img_dir = root / "samples"
        img_dir.mkdir(exist_ok=True)
        save_image(
            result["sample_after"] * 0.5 + 0.5,
            str(img_dir / f"{img_idx:05d}_ttt.png"),
        )
        save_image(
            result["sample_before"] * 0.5 + 0.5,
            str(img_dir / f"{img_idx:05d}_baseline.png"),
        )

    # --- aggregate metrics ---
    all_before = torch.cat(all_before, dim=0)
    all_after = torch.cat(all_after, dim=0)
    print("\n=== Aggregate Results ===")
    with torch.no_grad():
        results_before = evaluator(images, y, all_before)
        results_after = evaluator(images, y, all_after)
    for name in results_before:
        print(f"  {name}: baseline={results_before[name].item():.4f}  "
              f"TTT={results_after[name].item():.4f}")

    print(f"\nFinished TTT for {args.name}!")


if __name__ == "__main__":
    main()
