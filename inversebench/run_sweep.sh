#!/bin/bash
# HP sweep: train_pct x lambda_kl (y-conditioned LoRA, y_channels=4)
# Fixed: lr=3e-3, lora_rank=64, num_steps=50, method=direct, y_channels=4
# 4 runs: (train_pct=10,20) x (lambda_kl=0.0, 0.01)
# Runs 2 at a time on GPUs 0+1
#
# Usage: bash run_sweep.sh

set -e

IB_DIR="$HOME/projects/ttt/InverseBench"
cd "$IB_DIR"
source .venv/bin/activate

mkdir -p exps/ttt

COMMON="problem=inv-scatter pretrain=inv-scatter +ttt.method=direct +ttt.lr=3e-3 +ttt.lora_rank=64 +ttt.num_epochs=10 +ttt.diffusion_scheduler_config.num_steps=50 +ttt.y_channels=4"

echo "=== HP Sweep: train_pct x lambda_kl (y-conditioned, ych=4) ==="
echo "  Fixed: lr=3e-3, rank=64, num_steps=50, direct, y_channels=4"
echo "  Started: $(date)"
echo ""

# --- Batch 1: train_pct=10, kl=0.0 (GPU 0) + train_pct=10, kl=0.01 (GPU 1) ---

echo ">>> Batch 1: train_pct=10 x kl={0.0, 0.01}"

CUDA_VISIBLE_DEVICES=0 python train_ttt.py $COMMON \
    +ttt.train_pct=10 +ttt.lambda_kl=0.0 \
    2>&1 | tee exps/ttt/sweep_cond_pct10_kl0.0.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python train_ttt.py $COMMON \
    +ttt.train_pct=10 +ttt.lambda_kl=0.01 \
    2>&1 | tee exps/ttt/sweep_cond_pct10_kl0.01.log &
PID2=$!

wait $PID1
echo ">>> pct=10 kl=0.0 finished"
wait $PID2
echo ">>> pct=10 kl=0.01 finished"

# --- Batch 2: train_pct=20, kl=0.0 (GPU 0) + train_pct=20, kl=0.01 (GPU 1) ---
echo ""
echo ">>> Batch 2: train_pct=20 x kl={0.0, 0.01}"

CUDA_VISIBLE_DEVICES=0 python train_ttt.py $COMMON \
    +ttt.train_pct=20 +ttt.lambda_kl=0.0 \
    2>&1 | tee exps/ttt/sweep_cond_pct20_kl0.0.log &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python train_ttt.py $COMMON \
    +ttt.train_pct=20 +ttt.lambda_kl=0.01 \
    2>&1 | tee exps/ttt/sweep_cond_pct20_kl0.01.log &
PID4=$!

wait $PID3
echo ">>> pct=20 kl=0.0 finished"
wait $PID4
echo ">>> pct=20 kl=0.01 finished"

echo ""
echo "=== Sweep complete: $(date) ==="
echo "Results in exps/ttt/"
