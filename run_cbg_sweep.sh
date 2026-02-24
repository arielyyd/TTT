#!/bin/bash
# Run CBG hyperparameter sweep + evaluation
# Usage: bash run_cbg_sweep.sh

cd "$(dirname "$0")"
source .venv/bin/activate

# Phase 1: HP sweep + full training
python ML6_cbg_sweep.py \
    data=demo-ffhq model=ffhq256ddpm \
    task=gaussian_blur sampler=edm_dps task_group=pixel \
    name=cbg_sweep_blur save_dir=./results gpu=0 \
    "+cbg.lr_list=[1e-4,5e-4,1e-3]" \
    "+cbg.batch_size_list=[4,8]" \
    "+cbg.base_channels_list=[32,64]" \
    +cbg.num_epochs=10 \
    +cbg.full_num_epochs=100 \
    +cbg.train_pct=80

# Phase 2: Evaluation
python cbg_eval.py \
    data=test-ffhq model=ffhq256ddpm \
    task=gaussian_blur sampler=edm_dps task_group=pixel \
    name=eval_cbg_blur save_dir=./results gpu=0 \
    +cbg_eval.classifier_path=results/cbg_sweep_blur/classifier_final.pt
