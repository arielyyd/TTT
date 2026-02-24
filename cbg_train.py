"""
CBG (Classifier-Based Guidance) training script.

Trains a small MeasurementPredictor network M_phi(x_t, sigma, y) to predict
the measurement residual  A(Tweedie(x_t, sigma)) - y.

The diffusion model is frozen; only M_phi's parameters are updated.
Because no gradients flow through the diffusion model, training can use
larger batch sizes than LoRA/DPS-based methods.

Usage:
    python cbg_train.py \
        data=demo-ffhq model=ffhq256ddpm task=gaussian_blur \
        sampler=edm_dps task_group=pixel \
        name=cbg_blur save_dir=./results gpu=0
"""

import json
import time
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import hydra
import logging
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchvision.utils import save_image

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from classifier import MeasurementPredictor, save_classifier

log = logging.getLogger(__name__)


def setup_logging(log_path):
    """Set up logging to both file and stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Sigma sampling (same VP distribution as dpo_finetune.py)
# ---------------------------------------------------------------------------

def sample_sigma(model, batch_size, device):
    """Sample noise levels from the VP training distribution."""
    precond = model.model   # VPPrecond
    eps = 1e-5
    t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps
    return precond.sigma(t)


# ---------------------------------------------------------------------------
# Operator output-shape helper
# ---------------------------------------------------------------------------

def infer_operator_output(operator, resolution=256, device='cuda'):
    """Run operator on a dummy image to determine output shape."""
    dummy = torch.zeros(1, 3, resolution, resolution, device=device)
    with torch.no_grad():
        out = operator(dummy)
    return out.shape[1], (out.shape[2], out.shape[3])   # channels, (H, W)


def _norm01(x):
    """[-1,1] -> [0,1] for visualization."""
    return (x * 0.5 + 0.5).clamp(0, 1)


def _resize_to(x, target_hw):
    """Resize tensor to target (H, W) for grid alignment."""
    if x.shape[-2:] != target_hw:
        return F.interpolate(x, size=target_hw, mode='bilinear',
                             align_corners=False)
    return x


def _save_sample_viz(classifier, model, operator, val_images, val_y,
                     epoch, root, device, n_images=2):
    """Save diagnostic grid: y | pred | target at multiple sigma levels."""
    classifier.eval()
    sigma_levels = [0.01, 1.0, 40.0]
    n_sigma = len(sigma_levels)
    n_img = min(n_images, len(val_images))

    rows = []
    for img_i in range(n_img):
        x0 = val_images[img_i:img_i+1]
        y_i = val_y[img_i:img_i+1]

        for sig_val in sigma_levels:
            sigma = torch.tensor([sig_val], device=device)
            sigma_bc = sigma.view(1, 1, 1, 1)
            eps = torch.randn_like(x0)
            x_noisy = x0 + sigma_bc * eps

            with torch.no_grad():
                x0hat = model.tweedie(x_noisy, sigma).clamp(-1, 1)
                y_hat = operator(x0hat)
                target = y_hat - y_i
                pred = classifier(x_noisy, sigma, y_i)

            # Resize y, pred, target to image resolution for viz
            hw = x0.shape[-2:]
            y_viz = _resize_to(y_i, hw)
            pred_viz = _resize_to(pred, hw)
            target_viz = _resize_to(target, hw)

            # Normalize for display: shift to [0,1]
            def _viz(t):
                t = t[:, :3]  # take first 3 channels
                mn, mx = t.min(), t.max()
                if mx - mn > 1e-8:
                    return (t - mn) / (mx - mn)
                return t * 0 + 0.5

            rows.append(torch.cat([_viz(y_viz), _viz(pred_viz),
                                   _viz(target_viz)], dim=0))

    if rows:
        grid = torch.cat(rows, dim=0)
        save_image(grid, str(root / f"sample_viz_epoch{epoch}.png"),
                   nrow=3, padding=2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="default.yaml")
def main(args: DictConfig):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")
    device = f"cuda:{args.gpu}"

    # --- output dir ---
    root = Path(args.save_dir) / args.name
    root.mkdir(parents=True, exist_ok=True)
    setup_logging(str(root / "cbg_training.log"))

    log.info(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # --- CBG config ---
    cfg = OmegaConf.to_container(args.get("cbg", {}), resolve=True)
    lr            = cfg.get("lr", 1e-4)
    weight_decay  = cfg.get("weight_decay", 1e-4)
    batch_size    = cfg.get("batch_size", 8)
    num_epochs    = cfg.get("num_epochs", 50)
    base_channels = cfg.get("base_channels", 64)
    channel_mult  = cfg.get("channel_mult", [1, 2, 4, 4])
    emb_dim       = cfg.get("emb_dim", 256)
    attn_heads    = cfg.get("attn_heads", 4)
    grad_clip     = cfg.get("grad_clip", 1.0)
    save_every    = cfg.get("save_every", 10)
    val_fraction  = cfg.get("val_fraction", 0.1)
    train_pct     = cfg.get("train_pct", 100)

    # --- data ---
    dataset = get_dataset(**args.data)
    num_images = len(dataset)
    if train_pct < 100:
        num_images = max(1, int(num_images * train_pct / 100))
    images = dataset.get_data(num_images, 0)

    # --- train / val split ---
    num_val = max(1, int(num_images * val_fraction))
    num_train = num_images - num_val
    perm = np.random.permutation(num_images)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:]
    train_images = images[train_idx]
    val_images = images[val_idx]

    log.info(f"Dataset: {num_images} images  (train={num_train}, val={num_val})")

    # --- operator & measurements ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    train_y = operator.measure(train_images)
    val_y = operator.measure(val_images)

    # --- diffusion model (frozen) ---
    model = get_model(**args.model)
    model.requires_grad_(False)

    # --- infer operator output shape ---
    out_channels, out_size = infer_operator_output(operator, device=device)
    log.info(f"Operator output: channels={out_channels}, size={out_size}")

    # --- build classifier ---
    classifier = MeasurementPredictor(
        in_channels=3,
        y_channels=out_channels,
        out_channels=out_channels,
        out_size=out_size,
        base_channels=base_channels,
        emb_dim=emb_dim,
        channel_mult=channel_mult,
        attn_heads=attn_heads,
    ).to(device)

    num_params = sum(p.numel() for p in classifier.parameters())
    log.info(f"MeasurementPredictor: {num_params/1e6:.2f}M parameters")

    # --- optimizer & scheduler ---
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # --- save config ---
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    # --- training ---
    t_start = time.time()
    step_losses = []
    epoch_train_losses = []
    epoch_val_losses = []
    grad_norms = []
    best_val_loss = float('inf')

    log.info(f"\n{'='*60}")
    log.info(f"CBG Training: {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
    log.info(f"  train_pct={train_pct}, base_channels={base_channels}, "
             f"channel_mult={channel_mult}")
    log.info(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()
        classifier.train()
        order = np.random.permutation(num_train)
        epoch_loss = 0.0
        num_batches = 0
        epoch_sigma_stats = []
        epoch_target_stats = []

        pbar = tqdm.tqdm(range(0, num_train, batch_size),
                         desc=f"Epoch {epoch}/{num_epochs}")

        for start in pbar:
            idx = order[start: start + batch_size]
            x0 = train_images[idx]                       # [B, 3, 256, 256]
            y_batch = train_y[idx]                        # [B, C_y, H_y, W_y]
            B = x0.shape[0]

            # 1. Sample sigma & create noisy images
            sigma = sample_sigma(model, B, device)
            sigma_bc = sigma.view(-1, 1, 1, 1)
            eps = torch.randn_like(x0)
            x_noisy = x0 + sigma_bc * eps                # [B, 3, 256, 256]

            # Track sigma distribution
            epoch_sigma_stats.append(sigma.detach().cpu().numpy())

            # 2. Compute target (frozen diffusion model)
            with torch.no_grad():
                x0hat = model.tweedie(x_noisy, sigma).clamp(-1, 1)
                y_hat = operator(x0hat)                   # A(Tweedie(x_t))
                target = y_hat - y_batch                  # residual

            # Track target magnitude
            epoch_target_stats.append(target.detach().norm().item())

            # 3. Classifier prediction
            pred = classifier(x_noisy, sigma, y_batch)

            # 4. Loss & update
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), grad_clip).item()
            grad_norms.append(gn)
            optimizer.step()

            loss_val = loss.item()
            step_losses.append(loss_val)
            epoch_loss += loss_val
            num_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.6f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        epoch_train_losses.append(avg_train_loss)

        # --- validation ---
        classifier.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for start in range(0, num_val, batch_size):
                x0 = val_images[start: start + batch_size]
                y_batch = val_y[start: start + batch_size]
                B = x0.shape[0]

                sigma = sample_sigma(model, B, device)
                sigma_bc = sigma.view(-1, 1, 1, 1)
                eps = torch.randn_like(x0)
                x_noisy = x0 + sigma_bc * eps

                x0hat = model.tweedie(x_noisy, sigma).clamp(-1, 1)
                y_hat = operator(x0hat)
                target = y_hat - y_batch

                pred = classifier(x_noisy, sigma, y_batch)
                val_loss += F.mse_loss(pred, target).item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        epoch_val_losses.append(avg_val_loss)

        lr_scheduler.step()
        epoch_time = time.time() - t_epoch

        # --- diagnostics ---
        all_sigmas = np.concatenate(epoch_sigma_stats)
        avg_gnorm = np.mean(grad_norms[-num_batches:]) if num_batches > 0 else 0
        avg_target_mag = np.mean(epoch_target_stats) if epoch_target_stats else 0

        log.info(f"  Epoch {epoch:3d} | train_loss={avg_train_loss:.6f} "
                 f"| val_loss={avg_val_loss:.6f} "
                 f"| lr={optimizer.param_groups[0]['lr']:.2e} "
                 f"| grad_norm={avg_gnorm:.2f} "
                 f"| sigma=[{all_sigmas.min():.3f}, {all_sigmas.mean():.3f}, {all_sigmas.max():.3f}] "
                 f"| target_mag={avg_target_mag:.3f} "
                 f"| time={epoch_time:.0f}s")

        # --- save best ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_classifier(classifier, str(root / "classifier_best.pt"),
                            metadata={"epoch": epoch,
                                      "val_loss": avg_val_loss})
            log.info(f"  -> New best val_loss={avg_val_loss:.6f}, saved.")

        # --- periodic checkpoint + sample viz ---
        if epoch % save_every == 0 or epoch == 1:
            save_classifier(classifier, str(root / f"classifier_epoch{epoch}.pt"),
                            metadata={"epoch": epoch,
                                      "train_loss": avg_train_loss,
                                      "val_loss": avg_val_loss})
            try:
                _save_sample_viz(classifier, model, operator,
                                 val_images, val_y, epoch, root, device)
            except Exception as e:
                log.warning(f"  Sample viz failed: {e}")

        # --- progress.json ---
        progress = {
            "status": "training",
            "epoch": epoch,
            "total_epochs": num_epochs,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "elapsed_sec": int(time.time() - t_start),
        }
        with open(str(root / "progress.json"), "w") as f:
            json.dump(progress, f, indent=2)

    # --- save final ---
    save_classifier(classifier, str(root / "classifier_final.pt"),
                    metadata={"epochs": num_epochs,
                              "operator": task_group.operator.name,
                              "final_train_loss": epoch_train_losses[-1],
                              "final_val_loss": epoch_val_losses[-1]})

    # --- loss curves ---
    n_plots = 3 if grad_norms else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))

    ax = axes[0]
    ax.plot(step_losses, alpha=0.3, label="per step")
    if len(step_losses) > 20:
        window = max(len(step_losses) // 20, 5)
        smoothed = np.convolve(step_losses, np.ones(window) / window,
                               mode='valid')
        ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                label=f"smooth (w={window})")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Per-step training loss")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[1]
    epochs_arr = range(1, len(epoch_train_losses) + 1)
    ax.plot(epochs_arr, epoch_train_losses, "o-", label="train")
    ax.plot(epochs_arr, epoch_val_losses, "s-", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE loss")
    ax.set_title("Per-epoch loss")
    ax.legend()
    ax.set_yscale("log")

    if grad_norms and n_plots == 3:
        ax = axes[2]
        ax.plot(grad_norms, alpha=0.4, linewidth=0.5, label="per step")
        if len(grad_norms) > 10:
            window = max(len(grad_norms) // 20, 5)
            smoothed = np.convolve(grad_norms, np.ones(window)/window,
                                   mode='valid')
            ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                    linewidth=1.5, label=f"smoothed (w={window})")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Gradient norm")
        ax.set_title("Gradient norms (pre-clip)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(str(root / "cbg_training_loss.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # --- save training history ---
    history = {
        "step_losses": step_losses,
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_losses": epoch_val_losses,
        "grad_norms": grad_norms,
        "best_val_loss": best_val_loss,
    }
    with open(str(root / "cbg_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # --- final progress ---
    progress = {
        "status": "done",
        "epoch": num_epochs,
        "total_epochs": num_epochs,
        "train_loss": epoch_train_losses[-1],
        "val_loss": epoch_val_losses[-1],
        "best_val_loss": best_val_loss,
        "elapsed_sec": int(time.time() - t_start),
    }
    with open(str(root / "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)

    log.info(f"\nTraining complete!")
    log.info(f"  Best val loss: {best_val_loss:.6f}")
    log.info(f"  Classifier saved: {root / 'classifier_best.pt'}")
    log.info(f"  Loss curve:       {root / 'cbg_training_loss.png'}")
    log.info(f"  Progress:         {root / 'progress.json'}")


if __name__ == "__main__":
    main()
