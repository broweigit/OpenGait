#!/usr/bin/env bash
set -euo pipefail

task_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
task_nproc="${NPROC_PER_NODE:-8}"
task_port="${MASTER_PORT:-29705}"
task_backend="${OPENGAIT_DIST_BACKEND:-gloo}"
task_cfg="configs/biggergait/OFFICIAL_REBUTTAL_R2Q3_A4_CCPG2CCGR_SPARSE_HYPERPARAM_EXTREME_10K_TEST.yaml"
task_log="${CONSOLE_LOG:-official_rebuttal_r2q3_a4_ccpg2ccgr_sparse_hyperparam_extreme_10k_console.log}"

CUDA_VISIBLE_DEVICES="${task_gpus}" \
OPENGAIT_DIST_BACKEND="${task_backend}" \
python -m torch.distributed.launch \
  --nproc_per_node="${task_nproc}" \
  --master_port="${task_port}" \
  opengait/main.py \
  --cfgs "${task_cfg}" \
  --phase test \
  --log_to_file 2>&1 | tee "${task_log}"
