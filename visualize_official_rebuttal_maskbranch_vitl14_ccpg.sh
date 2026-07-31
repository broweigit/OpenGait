#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/browei/miniconda3/envs/gait_new/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" \
  misc/visualize_dinov2l_mask_branch_ccpg.py \
  --mask-checkpoint pretrained_LVMs/MaskBranch_vitl14.pt \
  --output-dir visual_assets/mask_branch_vitl14_ccpg_check \
  --device cuda:0 \
  "$@"
