#!/bin/bash
source .venv/bin/activate

# Set paths and parameters
S3_ROOT="s3://tcga-omezarr"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Run training with multiple GPUs
CUDA_VISIBLE_DEVICES=6,7 torchrun \
  --master_port=30014 \
  --nproc_per_node=2 \
  dinov2/train/train.py \
  --config-file ./dinov2/configs/train/vits14_reg4.yaml \
  --output-dir ./output_zarr_streaming \
  train.dataset_path="SlideDatasetZarr:root=${S3_ROOT}"