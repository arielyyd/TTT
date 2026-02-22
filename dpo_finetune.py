"""
Diffusion-DPO test-time training for DPS.

Adapts DPO (Direct Preference Optimization) to DDPM-based diffusion models
for per-image test-time training.  Following Salesforce DiffusionDPO
(Wallace et al., CVPR 2024), the DPO loss operates on the DDPM training
objective: at a random timestep t, noise both the winner and loser clean
images with the SAME Gaussian noise, predict the noise with the model, and
use the MSE as an ELBO proxy for log-likelihood.

Key insight: our model is a DDPM noise predictor wrapped in VPPrecond.
  VPPrecond.forward(x, sigma):
      F_x = UNet(c_in * x, c_noise)        # F_x = noise prediction (ε̂)
      D_x = x - sigma * F_x                # D_x = Tweedie estimate  (x̂_0)
So we can recover ε̂ = (x - D_x) / sigma from any Tweedie call, and
MSE(ε̂, ε) is exactly the DDPM training loss at that noise level.

The DPO loss is:
  L = -log σ( -β/2 * [ (MSE_w^θ - MSE_l^θ) - (MSE_w^ref - MSE_l^ref) ] )
where w/l are winner/loser, θ is LoRA-adapted, ref is frozen.

Reference: https://github.com/SalesforceAIResearch/DiffusionDPO
"""

import yaml
import torch
import torch.nn.functional as F
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
from lora import apply_lora, remove_lora, get_lora_params, frozen_tweedie


# ---------------------------------------------------------------------------
# DDPM forward process helpers
# ---------------------------------------------------------------------------

def sample_sigma(model, batch_size, device):
    """Sample noise levels from the VP training distribution.

    The model (VPPrecond) was trained with t ~ Uniform(ε, 1) and
    σ(t) = sqrt(exp(∫β(t)) - 1).  We sample t uniformly and convert
    to σ so the DPO loss matches the training objective.
    """
    precond = model.model  # VPPrecond
    # t ~ Uniform(epsilon, 1)  — same range the DDPM was trained on
    eps = 1e-5
    t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps
    sigma = precond.sigma(t)  # VP: sigma(t) = sqrt(exp(beta_int(t)) - 1)
    return sigma


def ddpm_noise_and_predict(model, x_clean, sigma):
    """Add DDPM noise at level σ and return (noise_pred, true_noise).

    Forward process (in VPPrecond / EDM convention, s(t) factored out):
        x_noisy = x_clean + σ * ε

    The model's tweedie output D_x satisfies:
        D_x = x_noisy - σ * ε̂
    so:
        ε̂ = (x_noisy - D_x) / σ

    Returns:
        noise_pred: predicted noise ε̂, shape [B, C, H, W]
        noise:      actual noise ε,    shape [B, C, H, W]
    """
    noise = torch.randn_like(x_clean)
    sigma_bc = sigma.view(-1, *([1] * (x_clean.ndim - 1)))  # [B, 1, 1, 1]
    x_noisy = x_clean + sigma_bc * noise

    # Tweedie prediction: D_x = x̂_0
    D_x = model.tweedie(x_noisy, sigma)

    # Recover noise prediction from Tweedie
    noise_pred = (x_noisy - D_x) / sigma_bc

    return noise_pred, noise


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_candidates(model, sampler, operator, measurement, num_candidates):
    """Run DPS multiple times with different seeds → candidate reconstructions.

    Returns:
        candidates: tensor [N, C, H, W] of reconstructions
        losses:     tensor [N] of measurement consistency ||y - A(x̂)||²
    """
    candidates = []
    losses = []
    for _ in range(num_candidates):
        x_start = sampler.get_start(measurement.shape[0], model)
        x_hat = sampler.sample(model, x_start, operator, measurement, verbose=False)
        loss = operator.loss(x_hat, measurement)  # [B]
        candidates.append(x_hat)
        losses.append(loss)

    # stack along a new dim: [N, B, C, H, W] then take B=0 since single image
    candidates = torch.cat(candidates, dim=0)  # [N, C, H, W]
    losses = torch.cat(losses, dim=0)           # [N]
    return candidates, losses


def make_pairs(candidates, losses):
    """Rank candidates by loss, pair best with worst.

    Returns:
        winners: tensor [P, C, H, W]
        losers:  tensor [P, C, H, W]
    """
    order = losses.argsort()  # ascending: best (lowest loss) first
    n = len(order)
    num_pairs = n // 2

    winner_idx = order[:num_pairs]
    loser_idx = order[n - num_pairs:].flip(0)  # worst paired with best

    return candidates[winner_idx], candidates[loser_idx]


# ---------------------------------------------------------------------------
# DPO loss (Salesforce DiffusionDPO style)
# ---------------------------------------------------------------------------

def dpo_loss(model, lora_modules, winners, losers, beta_dpo=5000.0):
    """Compute the Diffusion-DPO loss at a random DDPM timestep.

    Following DiffusionDPO (Wallace et al.):
      1. Sample random σ (noise level)
      2. Sample shared noise ε
      3. Noise both winner and loser with same (σ, ε)
      4. MSE of noise predictions = ELBO proxy for log π
      5. DPO loss on the log-likelihood differences

    Args:
        model:        DDPM diffusion model (with LoRA injected).
        lora_modules: list of LoRAConv1d modules (for frozen_tweedie).
        winners:      clean winner images [P, C, H, W].
        losers:       clean loser  images [P, C, H, W].
        beta_dpo:     DPO temperature. Higher = more conservative updates.

    Returns:
        loss:         scalar DPO loss.
        implicit_acc: fraction where model already prefers winner (for logging).
    """
    device = winners.device
    P = winners.shape[0]

    # --- sample shared noise level and noise ---
    sigma = sample_sigma(model, P, device)               # [P]
    sigma_bc = sigma.view(-1, *([1] * (winners.ndim - 1)))  # [P, 1, 1, 1]
    noise = torch.randn_like(winners)                     # shared across w/l

    # --- noisy inputs (same noise + sigma for both) ---
    x_w_noisy = winners + sigma_bc * noise
    x_l_noisy = losers  + sigma_bc * noise

    # --- model predictions (LoRA active, with grad) ---
    model.requires_grad_(True)
    D_w = model.tweedie(x_w_noisy, sigma)
    D_l = model.tweedie(x_l_noisy, sigma)
    model.requires_grad_(False)

    eps_w = (x_w_noisy - D_w) / sigma_bc
    eps_l = (x_l_noisy - D_l) / sigma_bc

    model_mse_w = (eps_w - noise).pow(2).mean(dim=[1, 2, 3])  # [P]
    model_mse_l = (eps_l - noise).pow(2).mean(dim=[1, 2, 3])  # [P]

    # --- reference predictions (LoRA zeroed, no grad) ---
    D_w_ref = frozen_tweedie(model, lora_modules, x_w_noisy, sigma)
    D_l_ref = frozen_tweedie(model, lora_modules, x_l_noisy, sigma)

    eps_w_ref = (x_w_noisy - D_w_ref) / sigma_bc
    eps_l_ref = (x_l_noisy - D_l_ref) / sigma_bc

    ref_mse_w = (eps_w_ref - noise).pow(2).mean(dim=[1, 2, 3])  # [P]
    ref_mse_l = (eps_l_ref - noise).pow(2).mean(dim=[1, 2, 3])  # [P]

    # --- DPO loss ---
    # Lower MSE = higher log-likelihood, so the sign is:
    #   log π_θ(w)/π_θ(l) ∝ -(MSE_w - MSE_l)
    # DPO: L = -log σ(β * [log π_θ(w)/π_θ(l) - log π_ref(w)/π_ref(l)])
    model_diff = model_mse_w - model_mse_l
    ref_diff = ref_mse_w - ref_mse_l

    inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)
    loss = -F.logsigmoid(inside_term).mean()

    # implicit accuracy: how often does the model already prefer winner?
    with torch.no_grad():
        implicit_acc = (inside_term > 0).float().mean()

    return loss, implicit_acc


# ---------------------------------------------------------------------------
# Per-image DPO test-time training
# ---------------------------------------------------------------------------

def run_dpo_ttt(model, sampler, operator, measurement, gt, evaluator,
                lora_rank=4, lora_alpha=1.0, lr=1e-4, num_dpo_steps=100,
                num_candidates=6, beta_dpo=5000.0, grad_clip=1.0,
                verbose=True):
    """DPO-based test-time training for a single measurement.

    Steps:
      1. Generate N candidate reconstructions via DPS (different seeds)
      2. Rank by measurement consistency, form winner/loser pairs
      3. Inject LoRA, train DPO loss on these pairs
      4. Final DPS evaluation with finetuned model
      5. Remove LoRA

    Args:
        model:          DDPM diffusion model.
        sampler:        DPS sampler instance.
        operator:       Forward operator A.
        measurement:    Observed y, shape (1, …).
        gt:             Ground-truth image (for evaluation only).
        evaluator:      Evaluator instance.
        lora_rank:      LoRA rank.
        lora_alpha:     LoRA scaling.
        lr:             AdamW learning rate.
        num_dpo_steps:  Number of DPO gradient steps.
        num_candidates: How many DPS runs to generate pairs from.
        beta_dpo:       DPO temperature.
        grad_clip:      Max gradient norm.
        verbose:        Print progress.

    Returns:
        dict with 'sample_before', 'sample_after', 'metrics_before',
        'metrics_after', 'losses'.
    """
    device = next(model.parameters()).device

    # --- baseline (before TTT) ---
    with torch.no_grad():
        x_start = sampler.get_start(measurement.shape[0], model)
        sample_before = sampler.sample(model, x_start, operator, measurement,
                                       verbose=False)
        metrics_before = evaluator(gt, measurement, sample_before)

    # --- generate candidates and form pairs ---
    if verbose:
        print(f"  Generating {num_candidates} DPS candidates for pair selection...")
    candidates, cand_losses = generate_candidates(
        model, sampler, operator, measurement, num_candidates,
    )
    winners, losers = make_pairs(candidates, cand_losses)
    num_pairs = winners.shape[0]
    if verbose:
        print(f"  Formed {num_pairs} winner/loser pairs")
        print(f"  Winner loss range: {cand_losses[cand_losses.argsort()[:num_pairs]].tolist()}")
        print(f"  Loser  loss range: {cand_losses[cand_losses.argsort()[num_candidates - num_pairs:]].tolist()}")

    # --- inject LoRA ---
    lora_modules = apply_lora(model, rank=lora_rank, alpha=lora_alpha)
    lora_params = get_lora_params(lora_modules)
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)

    # --- DPO training loop ---
    losses = []
    pbar = tqdm.trange(num_dpo_steps, desc="DPO-TTT") if verbose else range(num_dpo_steps)

    for step in pbar:
        optimizer.zero_grad()

        loss, acc = dpo_loss(model, lora_modules, winners, losers,
                             beta_dpo=beta_dpo)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if verbose:
            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc.item():.2f}")

    # --- final evaluation with finetuned model ---
    with torch.no_grad():
        x_start = sampler.get_start(measurement.shape[0], model)
        sample_after = sampler.sample(model, x_start, operator, measurement,
                                      verbose=False)
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

    # --- DPO hyperparameters (with defaults) ---
    dpo_cfg = OmegaConf.to_container(args.get("dpo", {}), resolve=True)
    lora_rank = dpo_cfg.get("lora_rank", 4)
    lora_alpha = dpo_cfg.get("lora_alpha", 1.0)
    lr = dpo_cfg.get("lr", 1e-4)
    num_dpo_steps = dpo_cfg.get("num_dpo_steps", 100)
    num_candidates = dpo_cfg.get("num_candidates", 6)
    beta_dpo = dpo_cfg.get("beta_dpo", 5000.0)
    grad_clip = dpo_cfg.get("grad_clip", 1.0)

    # --- data ---
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # --- operator & measurement ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # --- sampler (must be DPS) ---
    sampler = get_sampler(**args.sampler,
                          mcmc_sampler_config=task_group.get("mcmc_sampler_config", None))

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

    # --- per-image DPO TTT loop ---
    all_before, all_after = [], []
    for img_idx in range(total_number):
        print(f"\n=== Image {img_idx + 1}/{total_number} ===")
        gt_i = images[img_idx: img_idx + 1]
        y_i = y[img_idx: img_idx + 1]

        result = run_dpo_ttt(
            model, sampler, operator, y_i, gt_i, evaluator,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lr=lr,
            num_dpo_steps=num_dpo_steps, num_candidates=num_candidates,
            beta_dpo=beta_dpo, grad_clip=grad_clip, verbose=True,
        )

        for name in result["metrics_before"]:
            before_val = result["metrics_before"][name].item()
            after_val = result["metrics_after"][name].item()
            print(f"  {name}: {before_val:.4f} -> {after_val:.4f}")

        all_before.append(result["sample_before"])
        all_after.append(result["sample_after"])

        # save samples
        img_dir = root / "samples"
        img_dir.mkdir(exist_ok=True)
        save_image(result["sample_after"] * 0.5 + 0.5,
                   str(img_dir / f"{img_idx:05d}_dpo.png"))
        save_image(result["sample_before"] * 0.5 + 0.5,
                   str(img_dir / f"{img_idx:05d}_baseline.png"))

    # --- aggregate metrics ---
    all_before = torch.cat(all_before, dim=0)
    all_after = torch.cat(all_after, dim=0)
    print("\n=== Aggregate Results ===")
    with torch.no_grad():
        results_before = evaluator(images, y, all_before)
        results_after = evaluator(images, y, all_after)
    for name in results_before:
        print(f"  {name}: baseline={results_before[name].item():.4f}  "
              f"DPO={results_after[name].item():.4f}")

    print(f"\nFinished DPO-TTT for {args.name}!")


if __name__ == "__main__":
    main()
