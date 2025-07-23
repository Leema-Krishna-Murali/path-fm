#!/usr/bin/env bash
# -------------------------------------------------------------
# Launch a 4-GPU DINOv2 training run and keep a full log.
# -------------------------------------------------------------
set -euo pipefail

##### 0.  Logging boilerplate #################################
LOG_ROOT="/home/paul/pathologyDino/logs"
mkdir -p "${LOG_ROOT}"
NOW=$(date +'%Y%m%d_%H%M%S')
LOGFILE="${LOG_ROOT}/vitl16_4gpu_${NOW}.log"

# Mirror stdout+stderr to both terminal and file
exec > >(tee -a "${LOGFILE}") 2>&1
echo "▶  Logging to ${LOGFILE}"
set -x                                   # echo every command

##### 1.  Environment #########################################
source /home/paul/.bashrc
source /home/paul/dinov2/venv/bin/activate
cd /home/paul/pathologyDino

export PYTHONPATH="/home/paul/pathologyDino${PYTHONPATH:+:$PYTHONPATH}"

# Pathology training configuration
DATASTR="SlideDataset:root=/data/TCGA"
export WANDB_PROJECT=dinov2_pathology
export WANDB_NAME=vitl16_pathology_30ep_pretrained
echo $WANDB_API_KEY $WANDB_PROJECT $WANDB_NAME

# Use pre-trained ViT-L/14 with registers weights - memory optimized
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc_per_node=4 dinov2/train/train.py \
  --config-file dinov2/configs/train/pathology.yaml \
  --output-dir "${LOG_ROOT}/${WANDB_NAME}" \
  train.dataset_path="${DATASTR}" \
  --no-resume \
  student.pretrained_weights="/data/dino_pretrained/dinov2_vitl14_reg4_pretrain.pth" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# # ImageNet configuration
# DATASTR="ImageNet:split=TRAIN:root=/data/imagenet_2012:extra=/data/imagenet_2012"
# export WANDB_PROJECT=dinov2
# export WANDB_NAME=vitl16_h100_30ep
# echo $WANDB_API_KEY $WANDB_PROJECT $WANDB_NAME

# CUDA_VISIBLE_DEVICES=4,5 \
# torchrun --standalone --nproc_per_node=2 dinov2/train/train.py \
#   --config-file dinov2/configs/train/vitl16_h100_30ep.yaml \
#   --output-dir "${LOG_ROOT}" \
#   train.dataset_path="${DATASTR}"
