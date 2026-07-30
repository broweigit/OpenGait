#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON_BIN="${PYTHON_BIN:-/home/browei/miniconda3/envs/gait_new/bin/python}"
MASTER_PORT="${MASTER_PORT:-29873}"
CFG="configs/biggergait/OFFICIAL_REBUTTAL_T2A3_LEARNED_MASK_CCGR_FINETUNE_6000.yaml"
LOG="train_OFFICIAL_REBUTTAL_T2A3_LEARNED_MASK_CCGR_FINETUNE_6000.log"

"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port "${MASTER_PORT}" \
  opengait/main.py \
  --cfgs "${CFG}" \
  --phase train \
  --log_to_file \
  2>&1 | tee "${LOG}"

