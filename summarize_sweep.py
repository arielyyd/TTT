"""Summarize online TTT sweep results.

Usage:
    python3 summarize_sweep.py results/sweep
"""

import json
import sys
import csv
from pathlib import Path


def load_run(run_dir):
    """Load metrics.json from a run directory, return dict or None."""
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    with open(mpath) as f:
        return json.load(f)


def main():
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sweep")
    if not sweep_dir.exists():
        print(f"ERROR: {sweep_dir} not found")
        sys.exit(1)

    # --- Collect baseline from baseline run ---
    baseline_data = load_run(sweep_dir / "baseline")
    baseline_agg = None
    if baseline_data and "aggregate" in baseline_data:
        agg = baseline_data["aggregate"]
        baseline_agg = {}
        for metric, vals in agg.items():
            if "baseline_mean" in vals:
                baseline_agg[metric] = vals["baseline_mean"]

    # --- Collect all runs ---
    rows = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        data = load_run(run_dir)
        if data is None:
            continue

        config = data.get("config", {})
        agg = data.get("aggregate", {})

        row = {
            "name": run_dir.name,
            "draft_k": config.get("draft_k", "?"),
            "rounds": config.get("num_draft_rounds", "?"),
            "rank": config.get("lora_rank", "?"),
        }

        for metric in ["psnr", "ssim", "lpips", "meas_l2"]:
            if metric in agg:
                row[f"{metric}"] = agg[metric].get("ttt_mean", None)

        rows.append(row)

    if not rows:
        print("No results found.")
        sys.exit(0)

    # --- Sort by PSNR descending ---
    rows.sort(key=lambda r: r.get("psnr", 0) or 0, reverse=True)

    # --- Print table ---
    print()
    print("=" * 80)
    print("Online TTT Sweep Summary")
    print("=" * 80)

    if baseline_agg:
        print(f"\nDPS Baseline:  ", end="")
        parts = []
        for m, v in baseline_agg.items():
            parts.append(f"{m}={v:.4f}")
        print("  ".join(parts))

    header = f"{'rank':<6} {'name':<22} {'k':>4} {'R':>4} {'rnk':>4} " \
             f"{'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'meas_l2':>10}"
    print()
    print(header)
    print("-" * len(header))

    for i, row in enumerate(rows):
        psnr = row.get("psnr")
        ssim = row.get("ssim")
        lpips = row.get("lpips")
        meas = row.get("meas_l2")
        marker = " *" if i == 0 else ""
        print(f"  {i+1:<4} {row['name']:<22} {row['draft_k']:>4} "
              f"{row['rounds']:>4} {row['rank']:>4} "
              f"{psnr:>8.4f} {ssim:>8.4f} {lpips:>8.4f} "
              f"{meas:>10.4f}{marker}")

    print("-" * len(header))
    if baseline_agg and "psnr" in baseline_agg and rows[0].get("psnr"):
        delta = rows[0]["psnr"] - baseline_agg["psnr"]
        print(f"\nBest config: {rows[0]['name']}  "
              f"(PSNR delta vs DPS: {delta:+.4f})")

    # --- Save CSV ---
    csv_path = sweep_dir / "summary.csv"
    fields = ["name", "draft_k", "rounds", "rank", "psnr", "ssim", "lpips", "meas_l2"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
