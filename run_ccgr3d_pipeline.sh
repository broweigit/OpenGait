#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_ccgr3d_pipeline.sh protocol
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh preprocess
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh mask
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh teacher
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh student
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh test
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_ccgr3d_pipeline.sh cache
#   CUDA_VISIBLE_DEVICES=0 bash run_ccgr3d_pipeline.sh adapter

stage="${1:-help}"
nproc="${NPROC_PER_NODE:-8}"
port="${MASTER_PORT:-29631}"
python_bin="${PYTHON_BIN:-python}"

case "$stage" in
  protocol)
    "$python_bin" misc/prepare_ccgr3d_protocol.py
    ;;
  preprocess)
    "$python_bin" misc/preprocess_ccgr3d_sam3d_body_decoder_dataset.py \
      --gpus "$nproc" --master-port "$port" --chunk-size 32 --save-mode minimal
    ;;
  visualize-mask)
    "$python_bin" misc/visualize_ccgr3d_mask_targets.py
    ;;
  mask)
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$port" \
      misc/train_ccgr3d_dinov2s_mask_branch.py --steps 1000 --batch-size 64
    ;;
  teacher)
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$port" \
      opengait/main.py --cfgs configs/biggergait/PUPPETGAIT_CCGR3D_DINOV2S_TEACHER_TRAIN.yaml \
      --phase train --log_to_file
    ;;
  student)
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$port" \
      opengait/main.py --cfgs configs/biggergait/BIGGERGAIT_CCGR3D_DINOV2S_BASE_TRAIN.yaml \
      --phase train --log_to_file
    ;;
  test)
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$port" \
      opengait/main.py --cfgs configs/biggergait/PUPPETGAIT_CCGR3D_DINOV2S_TEACHER_TRAIN.yaml \
      --phase test --log_to_file
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$((port + 1))" \
      opengait/main.py --cfgs configs/biggergait/BIGGERGAIT_CCGR3D_DINOV2S_BASE_TRAIN.yaml \
      --phase test --log_to_file
    ;;
  cache)
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$port" \
      opengait/main.py --cfgs configs/biggergait/PUPPETGAIT_CCGR3D_DINOV2S_TEACHER_CACHE_TRAIN.yaml \
      --phase test --log_to_file
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$((port + 1))" \
      opengait/main.py --cfgs configs/biggergait/BIGGERGAIT_CCGR3D_DINOV2S_BASE_CACHE_TRAIN.yaml \
      --phase test --log_to_file
    torchrun --standalone --nproc_per_node="$nproc" --master_port="$((port + 2))" \
      opengait/main.py --cfgs configs/biggergait/BIGGERGAIT_CCGR3D_DINOV2S_BASE_CACHE_TEST.yaml \
      --phase test --log_to_file
    ;;
  adapter)
    "$python_bin" misc/train_ccgr3d_view_adapter.py
    "$python_bin" misc/evaluate_ccgr3d_view_adapter.py
    ;;
  *)
    echo "stage must be one of: protocol preprocess visualize-mask mask teacher student test cache adapter" >&2
    exit 2
    ;;
esac
