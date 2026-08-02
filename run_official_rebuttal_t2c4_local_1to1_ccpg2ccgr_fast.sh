#!/usr/bin/env bash
set -euo pipefail

task_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
task_nproc="${NPROC_PER_NODE:-4}"
task_port="${MASTER_PORT:-29707}"
task_backend="${OPENGAIT_DIST_BACKEND:-gloo}"
task_cfg="configs/biggergait/OFFICIAL_REBUTTAL_T2C4_LOCAL_1TO1_CCPG2CCGR_FAST_TEST.yaml"
task_log="${CONSOLE_LOG:-official_rebuttal_t2c4_local_1to1_ccpg2ccgr_fast_console.log}"

CUDA_VISIBLE_DEVICES="${task_gpus}" \
OPENGAIT_DIST_BACKEND="${task_backend}" \
python -m torch.distributed.launch \
  --nproc_per_node="${task_nproc}" \
  --master_port="${task_port}" \
  opengait/main.py \
  --cfgs "${task_cfg}" \
  --phase test \
  --log_to_file 2>&1 | tee "${task_log}"
