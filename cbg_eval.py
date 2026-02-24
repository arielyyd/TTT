"""
Evaluation script for CBG (Classifier-Based Guidance).

Compares three approaches on the same test images:
  1. DPS baseline  — standard DPS with gradient through diffusion model
  2. CBG-guided    — CBGDPS with trained classifier (cheaper gradient)
  3. Plain diffusion — no guidance at all (sanity check)

Same starting noise (RNG reset) for fair comparison.

Usage:
    python cbg_eval.py \
        data=test-ffhq model=ffhq256ddpm \
        sampler=edm_dps task=gaussian_blur task_group=pixel \
        +cbg_eval.classifier_path=results/cbg_blur/classifier_best.pt \
        name=eval_cbg_blur save_dir=./results gpu=0
"""

import json
import time
import yaml
import torch
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
from sampler import get_sampler, DPS, CBGDPS
from classifier import load_classifier


# ---------------------------------------------------------------------------
# Plain diffusion sampling (no guidance)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plain_diffusion_sample(model, scheduler, x_start, verbose=False):
    """PF-ODE Euler sampling without any guidance."""
    sigma_steps = scheduler.sigma_steps
    num_steps = len(sigma_steps) - 1
    pbar = tqdm.trange(num_steps, desc="Plain diffusion") if verbose else range(num_steps)
    xt = x_start

    for step in pbar:
        sigma = sigma_steps[step]
        sigma_next = sigma_steps[step + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        x0hat = model.tweedie(xt / st, sigma)
        score = (x0hat - xt / st) / sigma ** 2
        deriv = dst / st * xt - st * dsigma * sigma * score
        xt = xt + dt * deriv

    return xt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(x):
    """[-1,1] -> [0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)


def resize_y(y, target_shape):
    """Resize measurement to match image shape for visualization."""
    if y.shape != target_shape:
        return torch.nn.functional.interpolate(
            y, size=target_shape[-2:], mode='bilinear', align_corners=False,
        )
    return y


# ---------------------------------------------------------------------------
# Main
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

    # --- eval config ---
    eval_cfg = OmegaConf.to_container(args.get("cbg_eval", {}), resolve=True)
    classifier_path = eval_cfg.get("classifier_path", None)
    if classifier_path is None:
        raise ValueError("Must provide +cbg_eval.classifier_path=<path>")
    skip_baseline = eval_cfg.get("skip_baseline", False)
    skip_plain = eval_cfg.get("skip_plain", True)
    cbg_guidance_scale = eval_cfg.get("guidance_scale", 1.0)

    # --- data ---
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # --- operator & measurement ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # --- DPS sampler (for baseline) ---
    sampler = get_sampler(**args.sampler,
                          mcmc_sampler_config=task_group.get(
                              "mcmc_sampler_config", None))
    assert isinstance(sampler, DPS), \
        "cbg_eval expects a DPS-type sampler for baseline comparison"

    # --- CBG sampler ---
    classifier = load_classifier(classifier_path)
    cbg_sampler = CBGDPS(
        scheduler_config=OmegaConf.to_container(
            args.sampler.scheduler_config, resolve=True),
        guidance_scale=cbg_guidance_scale,
        classifier=classifier,
    )

    # --- model ---
    model = get_model(**args.model)

    # --- evaluator ---
    eval_fn_list = [get_eval_fn(name) for name in args.eval_fn_list]
    evaluator = Evaluator(eval_fn_list)

    # --- output dirs ---
    root = Path(args.save_dir) / args.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(exist_ok=True)
    (root / "comparisons").mkdir(exist_ok=True)

    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    timing = {}

    # ===================================================================
    # Phase 1: DPS baseline
    # ===================================================================
    if skip_baseline:
        print(f"\n{'='*60}")
        print(f"Phase 1: SKIPPED (skip_baseline=True)")
        print(f"{'='*60}")
        all_baseline = None
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: DPS baseline ({total_number} images)")
        print(f"{'='*60}")

        t0 = time.time()
        all_baseline = []
        for img_idx in tqdm.trange(total_number, desc="DPS baseline"):
            y_i = y[img_idx: img_idx + 1]
            x_start = sampler.get_start(1, model)
            x_hat = sampler.sample(model, x_start, operator, y_i,
                                   verbose=False)
            all_baseline.append(x_hat)

        all_baseline = torch.cat(all_baseline, dim=0)
        timing["dps_baseline_sec"] = time.time() - t0
        print(f"  DPS baseline: {timing['dps_baseline_sec']:.1f}s "
              f"({timing['dps_baseline_sec']/total_number:.1f}s/image)")

    # ===================================================================
    # Phase 2: CBG-guided sampling
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: CBG-guided ({total_number} images)")
    print(f"Classifier: {classifier_path}")
    print(f"Guidance scale: {cbg_guidance_scale}")
    print(f"{'='*60}")

    # Reset RNG for fair comparison
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    all_cbg = []
    for img_idx in tqdm.trange(total_number, desc="CBG-guided"):
        y_i = y[img_idx: img_idx + 1]
        x_start = cbg_sampler.get_start(1, model)
        x_hat = cbg_sampler.sample(model, x_start, operator, y_i,
                                   verbose=False)
        all_cbg.append(x_hat)

    all_cbg = torch.cat(all_cbg, dim=0)
    timing["cbg_sec"] = time.time() - t0
    print(f"  CBG: {timing['cbg_sec']:.1f}s "
          f"({timing['cbg_sec']/total_number:.1f}s/image)")

    # ===================================================================
    # Phase 3: Plain diffusion (no guidance, sanity check)
    # ===================================================================
    if skip_plain:
        print(f"\n{'='*60}")
        print(f"Phase 3: SKIPPED (skip_plain=True)")
        print(f"{'='*60}")
        all_plain = None
    else:
        print(f"\n{'='*60}")
        print(f"Phase 3: Plain diffusion ({total_number} images)")
        print(f"{'='*60}")

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        t0 = time.time()
        all_plain = []
        scheduler = sampler.scheduler
        for img_idx in tqdm.trange(total_number, desc="Plain diffusion"):
            x_start = sampler.get_start(1, model)
            x_hat = plain_diffusion_sample(model, scheduler, x_start)
            all_plain.append(x_hat)

        all_plain = torch.cat(all_plain, dim=0)
        timing["plain_sec"] = time.time() - t0
        print(f"  Plain: {timing['plain_sec']:.1f}s "
              f"({timing['plain_sec']/total_number:.1f}s/image)")

    # ===================================================================
    # Phase 4: Compute metrics and save results
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Phase 4: Computing metrics")
    print(f"{'='*60}")

    methods = {"cbg": all_cbg}
    if all_baseline is not None:
        methods["dps"] = all_baseline
    if all_plain is not None:
        methods["plain"] = all_plain

    all_metrics = []
    metric_names = None

    for img_idx in range(total_number):
        gt_i = images[img_idx: img_idx + 1]
        y_i = y[img_idx: img_idx + 1]
        img_metrics = {"image_idx": img_idx}

        for method_name, all_results in methods.items():
            result_i = all_results[img_idx: img_idx + 1]

            with torch.no_grad():
                m = evaluator(gt_i, y_i, result_i)
                mc = operator.loss(result_i, y_i)
                m["meas_l2"] = mc.mean()

            if metric_names is None:
                metric_names = list(m.keys())

            for name in metric_names:
                img_metrics[f"{name}_{method_name}"] = m[name].item()

        all_metrics.append(img_metrics)

        # save individual samples
        for method_name, all_results in methods.items():
            result_i = all_results[img_idx: img_idx + 1]
            save_image(norm(result_i),
                       str(root / "samples" / f"{img_idx:05d}_{method_name}.png"))

        # comparison grid: gt | y | dps | cbg [| plain]
        y_resized = resize_y(y_i, gt_i.shape)
        grid_items = [norm(gt_i), norm(y_resized)]
        for method_name in ["dps", "cbg", "plain"]:
            if method_name in methods:
                grid_items.append(norm(methods[method_name][img_idx:img_idx+1]))
        grid = torch.cat(grid_items, dim=0)
        save_image(grid,
                   str(root / "comparisons" / f"{img_idx:05d}.png"),
                   nrow=len(grid_items), padding=2)

    # --- aggregate statistics ---
    print(f"\n{'='*70}")
    print(f"Results | {total_number} images")
    print(f"{'='*70}")

    method_names = list(methods.keys())
    header = f"{'metric':<10}"
    for mn in method_names:
        header += f" {mn:>18}"
    print(header)
    print(f"{'-'*70}")

    aggregate = {}
    for name in metric_names:
        row = f"{name:<10}"
        agg = {}
        for mn in method_names:
            vals = np.array([m[f"{name}_{mn}"] for m in all_metrics])
            row += f" {vals.mean():>7.4f} +/- {vals.std():<6.4f}"
            agg[f"{mn}_mean"] = float(vals.mean())
            agg[f"{mn}_std"] = float(vals.std())
        # Add delta stats for CBG vs DPS
        if "dps" in methods:
            cbg_vals = np.array([m[f"{name}_cbg"] for m in all_metrics])
            dps_vals = np.array([m[f"{name}_dps"] for m in all_metrics])
            delta = cbg_vals - dps_vals
            agg["delta_mean"] = float(delta.mean())
            agg["delta_std"] = float(delta.std())
            agg["delta_min"] = float(delta.min())
            agg["delta_max"] = float(delta.max())
        print(row)
        aggregate[name] = agg
    print(f"{'-'*70}")

    # improvement counts (CBG vs DPS)
    if "dps" in methods:
        print("\nCBG vs DPS improvement:")
        for name in metric_names:
            delta = np.array([aggregate[name]["delta_mean"]])
            cbg_vals = np.array([m[f"{name}_cbg"] for m in all_metrics])
            dps_vals = np.array([m[f"{name}_dps"] for m in all_metrics])
            delta = cbg_vals - dps_vals
            improved = (delta > 0).sum() if name in ("psnr", "ssim") \
                else (delta < 0).sum()
            print(f"  {name}: {improved}/{total_number} images improved "
                  f"(delta mean={delta.mean():+.4f}, "
                  f"std={delta.std():.4f}, "
                  f"range=[{delta.min():+.4f}, {delta.max():+.4f}])")

    # --- save full comparison grid ---
    y_resized_all = resize_y(y, images.shape)
    grid_items = [norm(images), norm(y_resized_all)]
    for mn in ["dps", "cbg", "plain"]:
        if mn in methods:
            grid_items.append(norm(methods[mn]))
    full_grid = torch.cat(grid_items, dim=0)
    save_image(full_grid, str(root / "full_comparison.png"),
               nrow=total_number, padding=2)

    # --- save metrics to JSON ---
    output = {
        "classifier_path": str(classifier_path),
        "cbg_guidance_scale": cbg_guidance_scale,
        "num_images": total_number,
        "methods": method_names,
        "timing": timing,
        "aggregate": aggregate,
        "per_image": all_metrics,
    }
    with open(str(root / "metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    # --- timing summary ---
    if timing:
        print(f"\nTiming summary:")
        for phase, secs in timing.items():
            print(f"  {phase}: {secs:.1f}s ({secs/total_number:.1f}s/image)")
        if "dps_baseline_sec" in timing and "cbg_sec" in timing:
            speedup = timing["dps_baseline_sec"] / max(timing["cbg_sec"], 1e-6)
            print(f"  CBG speedup vs DPS: {speedup:.1f}x")

    print(f"\nResults saved to {root}")
    print(f"  - Per-image comparisons: {root / 'comparisons'}")
    print(f"  - Individual samples:    {root / 'samples'}")
    print(f"  - Full grid:             {root / 'full_comparison.png'}")
    print(f"  - Metrics:               {root / 'metrics.json'}")


if __name__ == "__main__":
    main()
