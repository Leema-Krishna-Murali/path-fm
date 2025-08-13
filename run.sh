#!/bin/bash
source .venv/bin/activate

# --- Load Cloudflare R2 credentials from rclone remote "sophont" ---
set -euo pipefail

# 1) Locate rclone.conf
if [ -n "${RCLONE_CONFIG:-}" ] && [ -f "$RCLONE_CONFIG" ]; then
  RCLONE_CONF="$RCLONE_CONFIG"
else
  RCLONE_CONF="$(rclone config file 2>/dev/null | sed -n 's/.*: //p' | tail -n1 || true)"
  [ -f "$RCLONE_CONF" ] || RCLONE_CONF="$HOME/.config/rclone/rclone.conf"
  [ -f "$RCLONE_CONF" ] || RCLONE_CONF="$HOME/.rclone.conf"
  [ -f "$RCLONE_CONF" ] || RCLONE_CONF="/etc/rclone.conf"
fi
[ -f "$RCLONE_CONF" ] || { echo "rclone config not found"; exit 1; }

# 2) Helper: read a key's value from the [sophont] section
get_rclone_conf_key() {
  local key="$1"
  sed -n "/^\[sophont\]/,/^\[/{ s/^[[:space:]]*$key[[:space:]]*=[[:space:]]*//p; }" "$RCLONE_CONF" | head -n1
}

# Extract credentials
AWS_ACCESS_KEY_ID="$(get_rclone_conf_key access_key_id || true)"
AWS_SECRET_ACCESS_KEY="$(get_rclone_conf_key secret_access_key || true)"
R2_ENDPOINT_URL="$(get_rclone_conf_key endpoint || true)"

# Some configs use alternate names
[ -n "$AWS_ACCESS_KEY_ID" ]     || AWS_ACCESS_KEY_ID="$(get_rclone_conf_key access_key || true)"
[ -n "$AWS_SECRET_ACCESS_KEY" ] || AWS_SECRET_ACCESS_KEY="$(get_rclone_conf_key secret_key || true)"

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$R2_ENDPOINT_URL" ]; then
  echo "Failed to read access_key_id / secret_access_key / endpoint from [sophont] in $RCLONE_CONF" >&2
  exit 1
fi

export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY R2_ENDPOINT_URL
export S3_ENDPOINT_URL="$R2_ENDPOINT_URL"
export AWS_DEFAULT_REGION=auto

echo "Using rclone config: $RCLONE_CONF"
echo "R2 endpoint: $R2_ENDPOINT_URL"

# Set paths and parameters
S3_ROOT="s3://tcga-zarr"
CACHE_GB=100.0  # Local cache size in GB
PATCH_SIZE=224  # Patch size for training

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Run training with multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --master_port=30001 \
  --nproc_per_node=2 \
  dinov2/train/train.py \
  --config-file ./dinov2/configs/train/vits14_reg4.yaml \
  --output-dir ./output_zarr_streaming \
  train.dataset_path="pathology_zarr:root=${S3_ROOT}:cache_gb=${CACHE_GB}:patch_size=${PATCH_SIZE}"