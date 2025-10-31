#!/usr/bin/env bash
set -euo pipefail

# Worker node configuration
export MASTER_ADDR="10.128.0.10"  # <-- Replace this with master node IP
export MASTER_PORT=29500

export NNODES=2
export NPROC_PER_NODE=8

# Determine which worker node this is 
# add 1 to whatever Lightning's assigned NODE RANK is since
# we are using the Studio node as NODE RANK 0
export NODE_RANK=$(( NODE_RANK + 1 ))

# Training config (must match master node! double-check your run.sh script!)
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="./output_pretrained_on_test"
DATASET_PATH="s3://tcga-12tb-litdata/"

# Set Python path for imports
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[Worker Node ${NODE_RANK}] Joining training..."
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

uv run torchrun \
  --nnodes ${NNODES} \
  --nproc_per_node ${NPROC_PER_NODE} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  dinov2/train/train.py \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    train.dataset_path="${DATASET_PATH}"