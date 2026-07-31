#!/usr/bin/env bash
set -euo pipefail

task_gpus="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
task_nproc="${NPROC_PER_NODE:-4}"
task_port="${MASTER_PORT:-29707}"
task_cfg="configs/biggergait/OFFICIAL_REBUTTAL_T2A4_T2B2_T2C3_T2D4_MAIN_CCPG_TRAIN.yaml"
task_log="${CONSOLE_LOG:-official_rebuttal_t2a4_main_ccpg_train_console.log}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

CUDA_VISIBLE_DEVICES="${task_gpus}" \
python -m torch.distributed.launch \
  --nproc_per_node="${task_nproc}" \
  --master_port="${task_port}" \
  opengait/main.py \
  --cfgs "${task_cfg}" \
  --phase train \
  --log_to_file 2>&1 | tee "${task_log}"
