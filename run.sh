#!/bin/bash
source .venv/bin/activate

# Set paths and parameters
# S3_ROOT="s3://sophont/paul/data/omezarr-test"
S3_ROOT="/data/omezarr-test"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Run training with multiple GPUs
CUDA_VISIBLE_DEVICES=1 torchrun \
  --master_port=30011 \
  --nproc_per_node=1 \
  dinov2/train/train.py \
  --config-file ./dinov2/configs/train/vits14_reg4.yaml \
  --output-dir ./output_zarr_streaming \
  train.dataset_path="SlideDatasetZarr:root=${S3_ROOT}"