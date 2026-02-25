#!/bin/bash
# Sweep: draft_k × num_draft_rounds × lora_rank
# 27 configs × 10 images, single GPU serial
#
# Usage: bash sweep_online_ttt.sh [GPU_ID]
#   GPU_ID defaults to 0

set -uo pipefail

GPU=${1:-0}
SWEEP_DIR="results/sweep"
mkdir -p "$SWEEP_DIR"

COMMON="+data=test-imagenet data.end_id=10 \
+model=imagenet256ddpm +sampler=edm_dps \
+task=gaussian_blur task_group=pixel \
+ttt.lr=1e-3 +ttt.guidance_scale=1.0 \
+ttt.lambda_buffer=1.0 +ttt.buffer_batch_size=4 \
save_dir=$SWEEP_DIR gpu=$GPU"

echo "========================================"
echo "  Online TTT Sweep"
echo "  Grid: draft_k={1,10,50} × rounds={1,3,5} × rank={4,16,64}"
echo "  Data: 10 images, gaussian_blur"
echo "  GPU:  $GPU"
echo "  Started: $(date)"
echo "========================================"

# --- Step 1: DPS baseline (run once with skip_baseline=false) ---
echo ""
echo ">>> [0/27] DPS Baseline"
python3 run_online_ttt.py $COMMON \
    +ttt.draft_k=1 +ttt.num_draft_rounds=1 +ttt.lora_rank=4 \
    +ttt.skip_baseline=false \
    name=baseline \
    2>&1 | tee "$SWEEP_DIR/baseline.log"

# --- Step 2: Grid sweep (skip_baseline=true) ---
DRAFT_KS=(1 10 50)
ROUNDS=(1 3 5)
RANKS=(4 16 64)

TOTAL=$(( ${#DRAFT_KS[@]} * ${#ROUNDS[@]} * ${#RANKS[@]} ))
COUNT=0

for K in "${DRAFT_KS[@]}"; do
  for R in "${ROUNDS[@]}"; do
    for RANK in "${RANKS[@]}"; do
      COUNT=$((COUNT + 1))
      NAME="k${K}_r${R}_rank${RANK}"

      # Skip the baseline config (already ran above)
      if [ "$K" = "1" ] && [ "$R" = "1" ] && [ "$RANK" = "4" ]; then
        echo ">>> [$COUNT/$TOTAL] $NAME — skipped (already in baseline run)"
        continue
      fi

      echo ""
      echo ">>> [$COUNT/$TOTAL] $NAME"
      python3 run_online_ttt.py $COMMON \
          +ttt.draft_k=$K +ttt.num_draft_rounds=$R +ttt.lora_rank=$RANK \
          +ttt.skip_baseline=true \
          name="$NAME" \
          2>&1 | tee "$SWEEP_DIR/${NAME}.log" || {
        echo ">>> FAILED: $NAME (continuing)"
      }
    done
  done
done

echo ""
echo "========================================"
echo "  Sweep complete: $(date)"
echo "  $COUNT/$TOTAL configs attempted"
echo "========================================"

# --- Step 3: Summarize ---
echo ""
echo ">>> Generating summary..."
python3 summarize_sweep.py "$SWEEP_DIR"
