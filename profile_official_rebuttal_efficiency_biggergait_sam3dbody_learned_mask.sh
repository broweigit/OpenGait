#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/home/browei/miniconda3/envs/gait_new/bin/python}"
MASTER_PORT="${MASTER_PORT:-29873}"
CFG="configs/biggergait/OFFICIAL_REBUTTAL_EFFICIENCY_BIGGERGAIT_SAM3DBODY_LEARNED_MASK_PROFILE.yaml"
LOG="profile_OFFICIAL_REBUTTAL_EFFICIENCY_BIGGERGAIT_SAM3DBODY_LEARNED_MASK.log"

"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port "${MASTER_PORT}" \
  opengait/main.py \
  --cfgs "${CFG}" \
  --phase train \
  --log_to_file \
  2>&1 | tee "${LOG}"

echo
echo "Profile complete:"
echo "  grep '\[ProfileStats\]' ${LOG}"
echo "JSON:"
echo "  output/CCPG/BiggerGait__SAM3DBody_LearnedMask_ProfileStats_Gaitbase_Share/"
echo "  OFFICIAL_REBUTTAL_EFFICIENCY_BIGGERGAIT_SAM3DBODY_LEARNED_MASK_PROFILE/profile_stats.json"
