#!/bin/bash
# LR sweep for TTT-LoRA direct method on inv-scatter
# Runs 3 learning rates: 1e-3, 3e-3, 1e-2
# Uses GPU 0 and GPU 1 in parallel
#
# Usage: bash inversebench/run_sweep.sh

set -e

IB_DIR="$HOME/projects/ttt/InverseBench"
cd "$IB_DIR"
source .venv/bin/activate

mkdir -p exps/ttt

echo "=== LR Sweep: direct method, train_pct=10, num_steps=50, K=1 ==="
echo "Started: $(date)"
echo ""

# Run lr=1e-3 and lr=3e-3 in parallel on GPUs 0 and 1
echo ">>> Launching lr=1e-3 on GPU 0 and lr=3e-3 on GPU 1..."

CUDA_VISIBLE_DEVICES=0 python train_ttt.py \
    problem=inv-scatter pretrain=inv-scatter \
    +ttt.method=direct \
    +ttt.train_pct=10 \
    +ttt.lr=1e-3 \
    +ttt.lora_rank=64 \
    +ttt.num_epochs=10 \
    +ttt.diffusion_scheduler_config.num_steps=50 \
    2>&1 | tee exps/ttt/sweep_lr1e-3.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python train_ttt.py \
    problem=inv-scatter pretrain=inv-scatter \
    +ttt.method=direct \
    +ttt.train_pct=10 \
    +ttt.lr=3e-3 \
    +ttt.lora_rank=64 \
    +ttt.num_epochs=10 \
    +ttt.diffusion_scheduler_config.num_steps=50 \
    2>&1 | tee exps/ttt/sweep_lr3e-3.log &
PID2=$!

# Wait for both to finish
wait $PID1
echo ">>> lr=1e-3 finished"
wait $PID2
echo ">>> lr=3e-3 finished"

# Run lr=1e-2 on GPU 0
echo ""
echo ">>> Launching lr=1e-2 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train_ttt.py \
    problem=inv-scatter pretrain=inv-scatter \
    +ttt.method=direct \
    +ttt.train_pct=10 \
    +ttt.lr=1e-2 \
    +ttt.lora_rank=64 \
    +ttt.num_epochs=10 \
    +ttt.diffusion_scheduler_config.num_steps=50 \
    2>&1 | tee exps/ttt/sweep_lr1e-2.log

echo ""
echo "=== Sweep complete: $(date) ==="
echo "Check results in exps/ttt/inverse-scatter-linear_direct_10pct*/"
