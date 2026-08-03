#!/usr/bin/env bash
set -euo pipefail

echo "===== A0: fixed fusion, 60 epochs ====="
python train_qc_fixed_fusion.py \
  --epoch 60 --batch_size 128 --random_seed 2024 \
  2>&1 | tee qc_a0_fixed_full.log

echo "===== A1: current gate, 60 epochs ====="
python train_qc_current_gate.py \
  --epoch 60 --batch_size 128 --random_seed 2024 \
  2>&1 | tee qc_a1_current_full.log

echo "===== A2: quality router without ranking, 60 epochs ====="
python train_qc_router_no_rank.py \
  --epoch 60 --batch_size 128 --random_seed 2024 \
  2>&1 | tee qc_a2_router_no_rank_full.log

echo "===== A3: full quality-calibrated router, 60 epochs ====="
python train_qc_router.py \
  --epoch 60 --batch_size 128 --random_seed 2024 \
  2>&1 | tee qc_a3_router_full.log

echo "===== All QCRouter full runs completed ====="
