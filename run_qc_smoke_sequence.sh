#!/usr/bin/env bash
set -euo pipefail

echo "===== A0: fixed fusion ====="
python train_qc_fixed_fusion.py \
  --epoch 5 --batch_size 128 --random_seed 9999 \
  2>&1 | tee qc_a0_fixed_smoke.log

echo "===== A1: current gate ====="
python train_qc_current_gate.py \
  --epoch 5 --batch_size 128 --random_seed 9999 \
  2>&1 | tee qc_a1_current_smoke.log

echo "===== A2: quality router without ranking ====="
python train_qc_router_no_rank.py \
  --epoch 5 --batch_size 128 --random_seed 9999 \
  2>&1 | tee qc_a2_router_no_rank_smoke.log

echo "===== A3: full quality-calibrated router ====="
python train_qc_router.py \
  --epoch 5 --batch_size 128 --random_seed 9999 \
  2>&1 | tee qc_a3_router_smoke.log

echo "===== All QCRouter smoke runs completed ====="
