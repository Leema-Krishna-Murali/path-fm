#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=34001 --nproc_per_node=1 dinov2/train/train.py --config-file ./dinov2/configs/train/vits14_reg4.yaml --output-dir ./output_pretrained_on_test train.dataset_path=pathology:root=/teamspace/studios/this_studio/tcga/


# Multi-node launch: 4 nodes × 8 GPUs per node
# Usage: run this same script on each node with the SAME RDZV_HOST/RDZV_PORT/RDZV_ID
# Example:
#   RDZV_HOST=node0.example.com RDZV_PORT=34118 RDZV_ID=dinov2-tcga-run1 \
#   bash run.sh
# use hostname -I | awk '{print $1}' to find your HOST url

# Training config
CONFIG_FILE=${CONFIG_FILE:-"./dinov2/configs/train/vitg14_reg4.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output_pretrained_on_test"}
DATASET_PATH=${DATASET_PATH:-"s3://tcga-12tb-litdata/"}

# Distributed config
NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
RDZV_HOST=${RDZV_HOST:?Set RDZV_HOST to the hostname/IP of node 0}
RDZV_PORT=${RDZV_PORT:-34118}
RDZV_ID=${RDZV_ID:-dinov2-tcga}

# Make all 8 GPUs visible on each node
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Clean output directory only from the rendezvous host to avoid races
if [[ "$(hostname)" == "$RDZV_HOST" ]]; then
  rm -rf "$OUTPUT_DIR"
fi

uv run \
  torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${RDZV_HOST}:${RDZV_PORT} \
    --rdzv_id=${RDZV_ID} \
    dinov2/train/train.py \
      --config-file ${CONFIG_FILE} \
      --output-dir ${OUTPUT_DIR} \
      train.dataset_path=${DATASET_PATH}
