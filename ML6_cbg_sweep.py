"""
Hyperparameter sweep for CBG (Classifier-Based Guidance) training.

Searches over learning rates, batch sizes, and base channel widths.
Phase 1: grid sweep on full data (each config creates an independent network).
Phase 2: full training with the best config (lowest val loss).

Mirrors ML6_training_loop_direct_finetuning.py but simpler: no LoRA
apply/remove cycle — each config creates and discards an independent
MeasurementPredictor network.

Usage:
  python ML6_cbg_sweep.py \
      data=demo-ffhq model=ffhq256ddpm task=gaussian_blur \
      sampler=edm_dps task_group=pixel \
      name=cbg_sweep_blur save_dir=./results gpu=0 \
      "+cbg.lr_list=[1e-4,5e-4,1e-3]" \
      "+cbg.batch_size_list=[4,8]" \
      "+cbg.base_channels_list=[32,64]" \
      +cbg.num_epochs=10 +cbg.full_num_epochs=100 +cbg.train_pct=80
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
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from classifier import MeasurementPredictor, save_classifier
from cbg_train import sample_sigma, infer_operator_output

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
# Single training config
# ---------------------------------------------------------------------------

def train_one_cbg_config(model, operator, train_images, train_y,
                         val_images, val_y, *, out_channels, out_size,
                         lr, batch_size, base_channels, channel_mult,
                         emb_dim, attn_heads, num_epochs, weight_decay,
                         grad_clip, seed, device):
    """Train a fresh MeasurementPredictor with one hyperparameter config.

    Returns:
        dict with step_losses, epoch_train_losses, epoch_val_losses,
        best_val_loss, best_state_dict
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_train = len(train_images)
    num_val = len(val_images)

    # Build fresh classifier
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

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    step_losses = []
    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = float('inf')
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        order = np.random.permutation(num_train)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_train, batch_size):
            idx = order[start: start + batch_size]
            x0 = train_images[idx]
            y_batch = train_y[idx]
            B = x0.shape[0]

            sigma = sample_sigma(model, B, device)
            sigma_bc = sigma.view(-1, 1, 1, 1)
            eps = torch.randn_like(x0)
            x_noisy = x0 + sigma_bc * eps

            with torch.no_grad():
                x0hat = model.tweedie(x_noisy, sigma).clamp(-1, 1)
                y_hat = operator(x0hat)
                target = y_hat - y_batch

            pred = classifier(x_noisy, sigma, y_batch)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            optimizer.step()

            loss_val = loss.item()
            step_losses.append(loss_val)
            epoch_loss += loss_val
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        epoch_train_losses.append(avg_train_loss)

        # Validation
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone()
                               for k, v in classifier.state_dict().items()}

    # Clean up to free GPU memory
    del classifier, optimizer, lr_scheduler
    torch.cuda.empty_cache()

    return {
        "step_losses": step_losses,
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_losses": epoch_val_losses,
        "best_val_loss": best_val_loss,
        "best_state_dict": best_state_dict,
        "num_params": num_params,
    }


# ---------------------------------------------------------------------------
# Sweep visualization
# ---------------------------------------------------------------------------

def save_sweep_snapshot(root, all_results, num_epochs):
    """Save current sweep results incrementally (after each config)."""
    summary = {}
    for tag, res in all_results.items():
        summary[tag] = {
            "epoch_train_losses": res["epoch_train_losses"],
            "epoch_val_losses": res["epoch_val_losses"],
            "step_losses": res["step_losses"],
            "best_val_loss": res["best_val_loss"],
            "num_params": res["num_params"],
        }
    with open(str(root / "sweep_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if len(all_results) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Per-step loss
    ax = axes[0]
    for tag, res in all_results.items():
        ax.plot(res["step_losses"], alpha=0.7, label=tag)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("MSE loss")
    ax.set_title("Per-step loss")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # Per-epoch val loss
    ax = axes[1]
    for tag, res in all_results.items():
        n = len(res["epoch_val_losses"])
        ax.plot(range(1, n + 1), res["epoch_val_losses"], "o-", label=tag)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE loss")
    ax.set_title("Per-epoch val loss")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # Final best val loss bar chart
    ax = axes[2]
    tags = list(all_results.keys())
    final_losses = [all_results[t]["best_val_loss"] for t in tags]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tags)))
    ax.bar(range(len(tags)), final_losses, color=colors)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Best val loss")
    ax.set_title("Best val loss by config")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(str(root / "sweep_results.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
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
    setup_logging(str(root / "training.log"))

    log.info(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # --- CBG sweep config ---
    cfg = OmegaConf.to_container(args.get("cbg", {}), resolve=True)

    # Architecture defaults
    channel_mult  = cfg.get("channel_mult", [1, 2, 4, 4])
    emb_dim       = cfg.get("emb_dim", 256)
    attn_heads    = cfg.get("attn_heads", 4)
    weight_decay  = cfg.get("weight_decay", 1e-4)
    grad_clip     = cfg.get("grad_clip", 1.0)
    val_fraction  = cfg.get("val_fraction", 0.1)
    train_pct     = cfg.get("train_pct", 100)

    # Sweep grid
    lr_list            = cfg.get("lr_list", [1e-4, 5e-4, 1e-3])
    batch_size_list    = cfg.get("batch_size_list", [4, 8])
    base_channels_list = cfg.get("base_channels_list", [32, 64])
    num_epochs         = cfg.get("num_epochs", 10)
    full_num_epochs    = cfg.get("full_num_epochs", 100)

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

    log.info(f"Dataset: {num_images} images (train={num_train}, val={num_val})")

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

    # --- save config ---
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    # ===================================================================
    # Phase 1: Sweep
    # ===================================================================
    configs = list(itertools.product(lr_list, batch_size_list,
                                     base_channels_list))
    all_results = {}

    log.info(f"\n{'='*60}")
    log.info(f"Hyperparameter sweep: {len(configs)} configs")
    log.info(f"  LRs: {lr_list}")
    log.info(f"  Batch sizes: {batch_size_list}")
    log.info(f"  Base channels: {base_channels_list}")
    log.info(f"  Data: {num_train} train, {num_val} val, {num_epochs} epochs")
    log.info(f"{'='*60}\n")

    for idx, (lr, bs, bc) in enumerate(configs):
        tag = f"lr{lr}_bs{bs}_ch{bc}"
        log.info(f"\n[{idx+1}/{len(configs)}] {tag}")
        t0 = time.time()

        result = train_one_cbg_config(
            model, operator, train_images, train_y,
            val_images, val_y,
            out_channels=out_channels, out_size=out_size,
            lr=lr, batch_size=bs, base_channels=bc,
            channel_mult=channel_mult, emb_dim=emb_dim,
            attn_heads=attn_heads, num_epochs=num_epochs,
            weight_decay=weight_decay, grad_clip=grad_clip,
            seed=args.seed, device=device)

        elapsed = time.time() - t0
        all_results[tag] = result
        log.info(f"  best_val_loss={result['best_val_loss']:.6f}, "
                 f"params={result['num_params']/1e6:.2f}M, "
                 f"time={elapsed:.0f}s")

        # Save snapshot after each config
        save_sweep_snapshot(root, all_results, num_epochs)
        log.info(f"  Snapshot saved ({len(all_results)}/{len(configs)} done)")

    # --- summary table ---
    log.info(f"\n{'='*70}")
    log.info(f"{'Config':<30} {'Best Val Loss':>14} {'Params':>10}")
    log.info(f"{'-'*70}")
    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]["best_val_loss"])
    for tag, res in ranked:
        log.info(f"{tag:<30} {res['best_val_loss']:>14.6f} "
                 f"{res['num_params']/1e6:>8.2f}M")
    log.info(f"{'-'*70}")
    best_tag = ranked[0][0]
    log.info(f"Best config: {best_tag} "
             f"(val_loss={ranked[0][1]['best_val_loss']:.6f})")

    # ===================================================================
    # Phase 2: Full training with best config
    # ===================================================================
    parts = best_tag.split("_")
    best_lr = float(parts[0].replace("lr", ""))
    best_bs = int(parts[1].replace("bs", ""))
    best_bc = int(parts[2].replace("ch", ""))

    log.info(f"\n{'='*60}")
    log.info(f"Phase 2: Full training with best config")
    log.info(f"  Config: lr={best_lr}, batch_size={best_bs}, "
             f"base_channels={best_bc}")
    log.info(f"  Data: {num_train} train, {num_val} val, "
             f"{full_num_epochs} epochs")
    log.info(f"{'='*60}\n")

    t0 = time.time()
    full_result = train_one_cbg_config(
        model, operator, train_images, train_y,
        val_images, val_y,
        out_channels=out_channels, out_size=out_size,
        lr=best_lr, batch_size=best_bs, base_channels=best_bc,
        channel_mult=channel_mult, emb_dim=emb_dim,
        attn_heads=attn_heads, num_epochs=full_num_epochs,
        weight_decay=weight_decay, grad_clip=grad_clip,
        seed=args.seed, device=device)
    full_time = time.time() - t0

    # Save final classifier from best state
    if full_result["best_state_dict"] is not None:
        classifier = MeasurementPredictor(
            in_channels=3,
            y_channels=out_channels,
            out_channels=out_channels,
            out_size=out_size,
            base_channels=best_bc,
            emb_dim=emb_dim,
            channel_mult=channel_mult,
            attn_heads=attn_heads,
        ).to(device)
        classifier.load_state_dict(full_result["best_state_dict"])
        save_classifier(classifier, str(root / "classifier_final.pt"),
                        metadata={
                            "best_config": best_tag,
                            "lr": best_lr,
                            "batch_size": best_bs,
                            "base_channels": best_bc,
                            "epochs": full_num_epochs,
                            "best_val_loss": full_result["best_val_loss"],
                            "operator": task_group.operator.name,
                        })
        del classifier

    # Plot full training loss curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(full_result["step_losses"], alpha=0.3, label="per step")
    sl = full_result["step_losses"]
    if len(sl) > 10:
        window = max(len(sl) // 20, 5)
        smoothed = np.convolve(sl, np.ones(window)/window, mode='valid')
        ax1.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                 label=f"smooth (w={window})")
    ax1.set_xlabel("Optimizer step")
    ax1.set_ylabel("MSE loss")
    ax1.set_title(f"Full training: lr={best_lr}, bs={best_bs}, ch={best_bc}")
    ax1.legend()
    ax1.set_yscale("log")

    n_ep = len(full_result["epoch_val_losses"])
    ax2.plot(range(1, n_ep + 1), full_result["epoch_train_losses"],
             "o-", label="train")
    ax2.plot(range(1, n_ep + 1), full_result["epoch_val_losses"],
             "s-", label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg MSE loss")
    ax2.set_title("Per-epoch loss")
    ax2.legend()
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(str(root / "full_training_loss.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    log.info(f"\nFull training complete! ({full_time:.0f}s)")
    log.info(f"  Best val loss: {full_result['best_val_loss']:.6f}")
    log.info(f"  Classifier saved: {root / 'classifier_final.pt'}")
    log.info(f"  Loss curve: {root / 'full_training_loss.png'}")


if __name__ == "__main__":
    main()
