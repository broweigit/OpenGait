#!/usr/bin/env bash
set -euo pipefail

task_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
task_nproc="${NPROC_PER_NODE:-4}"
task_port="${MASTER_PORT:-29709}"
task_cfg="configs/biggergait/OFFICIAL_REBUTTAL_R2Q2_A4_CCGR_TEMPORAL_SHAPE_SCALE_MEAN_FAST_TEST.yaml"
task_log="${CONSOLE_LOG:-official_rebuttal_r2q2_temporal_shape_scale_mean_fast_console.log}"

# This host mixes Ada and A40 GPUs. Avoid cross-generation CUDA P2P during
# DDP initialization of the 1.35B-parameter SAM3D model.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

CUDA_VISIBLE_DEVICES="${task_gpus}" \
python -m torch.distributed.launch \
  --nproc_per_node="${task_nproc}" \
  --master_port="${task_port}" \
  opengait/main.py \
  --cfgs "${task_cfg}" \
  --phase test \
  --log_to_file 2>&1 | tee "${task_log}"
