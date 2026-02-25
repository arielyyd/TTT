"""
Single-image diagnostic for online TTT pipeline.

Runs 1 image through the full pipeline and prints detailed per-round stats:
  - DRaFT loss, buffer loss, total loss
  - LoRA gradient norm (before & after clipping)
  - LoRA weight norm & delta per round
  - DPS measurement loss during sampling (prefix vs suffix)
  - Per-step DPS guidance norm inside dps_draft_k_sample
  - Final metrics (PSNR, SSIM, LPIPS) + comparison with DPS baseline
  - GPU memory usage

Usage:
  python3 diagnose_single.py \
      +data=test-imagenet data.end_id=1 \
      +model=imagenet256ddpm +sampler=edm_dps \
      +task=gaussian_blur task_group=pixel \
      +ttt.draft_k=10 +ttt.num_draft_rounds=5 +ttt.lora_rank=4 \
      gpu=0
"""

import json
import random
import yaml
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchvision.utils import save_image

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from eval import get_eval_fn, Evaluator
from sampler import get_sampler
from lora import (apply_lora, remove_lora, get_lora_params, save_lora,
                  frozen_tweedie)


# ---------------------------------------------------------------------------
# Instrumented dps_draft_k_sample — collects per-step diagnostics
# ---------------------------------------------------------------------------

def dps_draft_k_sample_diag(model, scheduler, forward_op, y, device,
                            draft_k=1, guidance_scale=1.0, lora_params=None):
    """Same as dps_draft_k_sample but returns (x_0, diag_dict)."""
    in_shape = model.get_in_shape()
    x = torch.randn(1, *in_shape, device=device) * scheduler.get_prior_sigma()
    sigma_steps = scheduler.sigma_steps
    num_steps = len(sigma_steps) - 1
    grad_start = max(num_steps - draft_k, 0)

    diag = {"dps_loss": [], "dps_grad_norm": [], "phase": []}

    for i in range(num_steps):
        sigma = sigma_steps[i]
        sigma_next = sigma_steps[i + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        if i < grad_start:
            model.requires_grad_(True)
            xt_in = x.detach().requires_grad_(True)
            x0hat = model.tweedie(xt_in / st, sigma)
            loss_dps = forward_op.loss(x0hat, y)
            grad_xt = torch.autograd.grad(loss_dps.sum(), xt_in)[0]
            model.requires_grad_(False)

            diag["dps_loss"].append(loss_dps.item())
            diag["dps_grad_norm"].append(grad_xt.norm().item())
            diag["phase"].append("prefix")

            with torch.no_grad():
                norm_factor = loss_dps.sqrt().view(
                    -1, *([1] * (grad_xt.ndim - 1))).clamp(min=1e-8)
                normalized_grad = grad_xt / norm_factor
                score = (x0hat.detach() - x / st) / sigma ** 2
                deriv = dst / st * x - st * dsigma * sigma * score
                x = x + dt * deriv - guidance_scale * normalized_grad
        else:
            if i == grad_start:
                x = x.detach().requires_grad_(True)
                if lora_params is not None:
                    for p in lora_params:
                        p.requires_grad_(True)

            x0hat = model.tweedie(x / st, sigma)
            loss_dps = forward_op.loss(x0hat, y)
            grad_xt = torch.autograd.grad(
                loss_dps.sum(), x, retain_graph=True)[0]
            norm_factor = loss_dps.detach().sqrt().view(
                -1, *([1] * (grad_xt.ndim - 1))).clamp(min=1e-8)
            normalized_grad = (grad_xt / norm_factor).detach()

            diag["dps_loss"].append(loss_dps.item())
            diag["dps_grad_norm"].append(grad_xt.norm().item())
            diag["phase"].append("suffix")

            score = (x0hat - x / st) / sigma ** 2
            deriv = dst / st * x - st * dsigma * sigma * score
            x = x + dt * deriv - guidance_scale * normalized_grad

    return x, diag


# ---------------------------------------------------------------------------
# Buffer helpers (copied from run_online_ttt.py)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.data = []

    def add(self, x0, y):
        self.data.append((x0.detach().cpu(), y.detach().cpu()))
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.data))
        indices = random.sample(range(len(self.data)), batch_size)
        xs = torch.cat([self.data[i][0] for i in indices], dim=0)
        ys = torch.cat([self.data[i][1] for i in indices], dim=0)
        return xs, ys

    def __len__(self):
        return len(self.data)


def buffer_update_loss(model, scheduler, forward_op, buffer, batch_size,
                       device):
    sigma_steps = scheduler.sigma_steps
    valid_sigmas = sigma_steps[sigma_steps > 0]
    if len(valid_sigmas) == 0:
        return torch.tensor(0.0, device=device)

    x0_batch, y_batch = buffer.sample(batch_size)
    x0_batch = x0_batch.to(device)
    y_batch = y_batch.to(device)

    idx = torch.randint(0, len(valid_sigmas), (1,)).item()
    sigma = valid_sigmas[idx]

    noise = torch.randn_like(x0_batch)
    x_noisy = x0_batch + sigma * noise
    x_hat = model.tweedie(x_noisy, sigma)
    loss = forward_op.loss(x_hat, y_batch).mean()
    return loss


def dps_sample_clean(model, scheduler, forward_op, y, device,
                     guidance_scale=1.0):
    """No-grad DPS sample for final output."""
    in_shape = model.get_in_shape()
    x = torch.randn(1, *in_shape, device=device) * scheduler.get_prior_sigma()
    sigma_steps = scheduler.sigma_steps
    num_steps = len(sigma_steps) - 1
    for i in range(num_steps):
        sigma = sigma_steps[i]
        sigma_next = sigma_steps[i + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        model.requires_grad_(True)
        xt_in = x.detach().requires_grad_(True)
        x0hat = model.tweedie(xt_in / st, sigma)
        loss_dps = forward_op.loss(x0hat, y)
        grad_xt = torch.autograd.grad(loss_dps.sum(), xt_in)[0]
        model.requires_grad_(False)

        with torch.no_grad():
            norm_factor = loss_dps.sqrt().view(
                -1, *([1] * (grad_xt.ndim - 1))).clamp(min=1e-8)
            normalized_grad = grad_xt / norm_factor
            score = (x0hat.detach() - x / st) / sigma ** 2
            deriv = dst / st * x - st * dsigma * sigma * score
            x = x + dt * deriv - guidance_scale * normalized_grad
    return x


def norm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs",
            config_name="default.yaml")
def main(args: DictConfig):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    ttt = OmegaConf.to_container(args.get("ttt", {}), resolve=True)
    lora_rank = ttt.get("lora_rank", 4)
    lora_alpha = ttt.get("lora_alpha", 1.0)
    target_modules = ttt.get("target_modules", "all")
    lr = ttt.get("lr", 1e-3)
    grad_clip = ttt.get("grad_clip", 1.0)
    draft_k = ttt.get("draft_k", 1)
    num_draft_rounds = ttt.get("num_draft_rounds", 1)
    buffer_batch_size = ttt.get("buffer_batch_size", 4)
    lambda_buffer = ttt.get("lambda_buffer", 1.0)
    guidance_scale = ttt.get("guidance_scale", 1.0)

    # --- Setup ---
    dataset = get_dataset(**args.data)
    gt = dataset.get_data(1, 0)  # first image only
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(gt)

    sampler = get_sampler(
        **args.sampler,
        mcmc_sampler_config=task_group.get("mcmc_sampler_config", None))
    scheduler = sampler.scheduler

    model = get_model(**args.model)
    device = next(model.parameters()).device

    eval_fn_list = [get_eval_fn(name) for name in args.eval_fn_list]
    evaluator = Evaluator(eval_fn_list)

    save_dir = Path("results/diagnose")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Single-Image Diagnostic")
    print(f"  rank={lora_rank}  alpha={lora_alpha}  lr={lr}")
    print(f"  draft_k={draft_k}  rounds={num_draft_rounds}")
    print(f"  guidance_scale={guidance_scale}")
    print(f"  num_steps={len(scheduler.sigma_steps) - 1}")
    print("=" * 60)

    # =================================================================
    # Step 1: DPS baseline (no LoRA)
    # =================================================================
    print("\n--- DPS Baseline ---")
    torch.manual_seed(args.seed)
    x_baseline = sampler.sample(model, sampler.get_start(1, model),
                                operator, y, verbose=False)
    with torch.no_grad():
        metrics_base = evaluator(gt, y, x_baseline)
        meas_base = operator.loss(x_baseline, y).mean()
    print(f"  PSNR:    {metrics_base['psnr'].item():.4f}")
    print(f"  SSIM:    {metrics_base['ssim'].item():.4f}")
    print(f"  LPIPS:   {metrics_base['lpips'].item():.4f}")
    print(f"  meas_l2: {meas_base.item():.4f}")

    if torch.cuda.is_available():
        print(f"  GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # =================================================================
    # Step 2: Online TTT rounds with diagnostics
    # =================================================================
    print("\n--- Online TTT ---")
    lora_modules = apply_lora(model, rank=lora_rank, alpha=lora_alpha,
                              target_modules=target_modules)
    lora_params = get_lora_params(lora_modules)
    optimizer = torch.optim.Adam(lora_params, lr=lr)

    num_lora_params = sum(p.numel() for p in lora_params)
    print(f"  LoRA params: {num_lora_params:,}")

    # Snapshot initial LoRA weights
    w0 = torch.cat([p.detach().flatten() for p in lora_params]).clone()

    buffer_D = ReplayBuffer()
    round_logs = []

    for rnd in range(num_draft_rounds):
        optimizer.zero_grad()

        # --- DRaFT sample with diagnostics ---
        torch.manual_seed(args.seed + rnd)
        x_0, sampling_diag = dps_draft_k_sample_diag(
            model, scheduler, operator, y, device,
            draft_k=draft_k, guidance_scale=guidance_scale,
            lora_params=lora_params)

        # --- Losses ---
        loss_current = operator.loss(x_0, y).mean()

        loss_buffer = torch.tensor(0.0, device=device)
        if len(buffer_D) > 0:
            loss_buffer = buffer_update_loss(
                model, scheduler, operator, buffer_D,
                buffer_batch_size, device)

        total_loss = loss_current + lambda_buffer * loss_buffer
        total_loss.backward()

        # --- Gradient stats (before clipping) ---
        grad_vec = torch.cat([p.grad.flatten() for p in lora_params
                              if p.grad is not None])
        grad_norm_pre = grad_vec.norm().item()
        grad_max = grad_vec.abs().max().item()
        grad_mean = grad_vec.abs().mean().item()

        # --- Clip & step ---
        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
        grad_norm_post = torch.cat([p.grad.flatten() for p in lora_params
                                    if p.grad is not None]).norm().item()
        optimizer.step()

        # --- LoRA weight stats ---
        w_now = torch.cat([p.detach().flatten() for p in lora_params])
        w_norm = w_now.norm().item()
        w_delta = (w_now - w0).norm().item()

        # --- DPS sampling summary ---
        prefix_losses = [l for l, p in zip(sampling_diag["dps_loss"],
                                           sampling_diag["phase"])
                         if p == "prefix"]
        suffix_losses = [l for l, p in zip(sampling_diag["dps_loss"],
                                           sampling_diag["phase"])
                         if p == "suffix"]
        prefix_gnorms = [g for g, p in zip(sampling_diag["dps_grad_norm"],
                                           sampling_diag["phase"])
                         if p == "prefix"]
        suffix_gnorms = [g for g, p in zip(sampling_diag["dps_grad_norm"],
                                           sampling_diag["phase"])
                         if p == "suffix"]

        rnd_log = {
            "round": rnd,
            "loss_draft": loss_current.item(),
            "loss_buffer": loss_buffer.item(),
            "loss_total": total_loss.item(),
            "grad_norm_pre_clip": grad_norm_pre,
            "grad_norm_post_clip": grad_norm_post,
            "grad_max": grad_max,
            "grad_mean": grad_mean,
            "lora_w_norm": w_norm,
            "lora_w_delta_from_init": w_delta,
            "dps_prefix_loss_start": prefix_losses[0] if prefix_losses else None,
            "dps_prefix_loss_end": prefix_losses[-1] if prefix_losses else None,
            "dps_suffix_loss_start": suffix_losses[0] if suffix_losses else None,
            "dps_suffix_loss_end": suffix_losses[-1] if suffix_losses else None,
            "dps_prefix_gnorm_mean": np.mean(prefix_gnorms) if prefix_gnorms else None,
            "dps_suffix_gnorm_mean": np.mean(suffix_gnorms) if suffix_gnorms else None,
        }

        if torch.cuda.is_available():
            rnd_log["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

        round_logs.append(rnd_log)

        clipped = " (clipped)" if grad_norm_pre > grad_clip else ""
        print(f"\n  Round {rnd}:")
        print(f"    loss:  draft={loss_current.item():.4f}  "
              f"buffer={loss_buffer.item():.4f}  "
              f"total={total_loss.item():.4f}")
        print(f"    grad:  norm={grad_norm_pre:.4f}{clipped}  "
              f"max={grad_max:.6f}  mean={grad_mean:.6f}")
        print(f"    lora:  ||w||={w_norm:.6f}  "
              f"||w-w0||={w_delta:.6f}")
        print(f"    DPS prefix:  loss {prefix_losses[0]:.2f} → "
              f"{prefix_losses[-1]:.4f}  "
              f"grad_norm_avg={np.mean(prefix_gnorms):.4f}"
              if prefix_losses else "    DPS prefix:  (none)")
        print(f"    DPS suffix:  loss {suffix_losses[0]:.4f} → "
              f"{suffix_losses[-1]:.4f}  "
              f"grad_norm_avg={np.mean(suffix_gnorms):.4f}"
              if suffix_losses else "    DPS suffix:  (none)")

    # =================================================================
    # Step 3: Final clean sample + metrics
    # =================================================================
    print("\n--- Final DPS Sample (with trained LoRA) ---")
    torch.manual_seed(args.seed + 999)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    x_final = dps_sample_clean(model, scheduler, operator, y, device,
                               guidance_scale=guidance_scale)

    with torch.no_grad():
        metrics_ttt = evaluator(gt, y, x_final)
        meas_ttt = operator.loss(x_final, y).mean()

    print(f"  PSNR:    {metrics_ttt['psnr'].item():.4f}  "
          f"(baseline: {metrics_base['psnr'].item():.4f}  "
          f"delta: {metrics_ttt['psnr'].item() - metrics_base['psnr'].item():+.4f})")
    print(f"  SSIM:    {metrics_ttt['ssim'].item():.4f}  "
          f"(baseline: {metrics_base['ssim'].item():.4f}  "
          f"delta: {metrics_ttt['ssim'].item() - metrics_base['ssim'].item():+.4f})")
    print(f"  LPIPS:   {metrics_ttt['lpips'].item():.4f}  "
          f"(baseline: {metrics_base['lpips'].item():.4f}  "
          f"delta: {metrics_ttt['lpips'].item() - metrics_base['lpips'].item():+.4f})")
    print(f"  meas_l2: {meas_ttt.item():.4f}  "
          f"(baseline: {meas_base.item():.4f})")

    if torch.cuda.is_available():
        print(f"  GPU peak mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # =================================================================
    # Step 4: Save outputs
    # =================================================================
    # Visual comparison: GT | blurred | baseline | TTT
    y_vis = y
    if y.shape != gt.shape:
        y_vis = torch.nn.functional.interpolate(
            y, size=gt.shape[-2:], mode='bilinear', align_corners=False)

    grid = torch.cat([norm(gt), norm(y_vis), norm(x_baseline), norm(x_final)],
                     dim=0)
    save_image(grid, str(save_dir / "comparison.png"), nrow=4, padding=2)

    # Save diagnostics JSON
    output = {
        "config": {
            "lora_rank": lora_rank, "lora_alpha": lora_alpha, "lr": lr,
            "draft_k": draft_k, "num_draft_rounds": num_draft_rounds,
            "guidance_scale": guidance_scale, "grad_clip": grad_clip,
            "num_lora_params": num_lora_params,
        },
        "baseline": {
            "psnr": metrics_base["psnr"].item(),
            "ssim": metrics_base["ssim"].item(),
            "lpips": metrics_base["lpips"].item(),
            "meas_l2": meas_base.item(),
        },
        "ttt": {
            "psnr": metrics_ttt["psnr"].item(),
            "ssim": metrics_ttt["ssim"].item(),
            "lpips": metrics_ttt["lpips"].item(),
            "meas_l2": meas_ttt.item(),
        },
        "rounds": round_logs,
    }
    with open(str(save_dir / "diagnostics.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {save_dir}/")
    print(f"  - comparison.png")
    print(f"  - diagnostics.json")

    remove_lora(model)


if __name__ == "__main__":
    main()
