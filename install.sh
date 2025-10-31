#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps the project using uv, installing the managed Python,
# dependencies described in pyproject.toml, and GPU-aware PyTorch wheels.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.10.12"
DEFAULT_INSTALLER_URL="https://astral.sh/uv/install.sh"
DEFAULT_DOWNLOAD_MIRROR="https://wheelnext.astral.sh"

# Ensure `~/.local/bin` is considered when checking for uv.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  INSTALLER_URL="${UV_INSTALLER_URL:-$DEFAULT_INSTALLER_URL}"
  DOWNLOAD_MIRROR="${UV_INSTALLER_DOWNLOAD_URL:-$DEFAULT_DOWNLOAD_MIRROR}"
  echo "uv not found on PATH; installing from ${INSTALLER_URL}..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "${INSTALLER_URL}" | INSTALLER_DOWNLOAD_URL="${DOWNLOAD_MIRROR}" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${INSTALLER_URL}" | INSTALLER_DOWNLOAD_URL="${DOWNLOAD_MIRROR}" sh
  else
    echo "Neither curl nor wget is available; please install one and re-run." >&2
    exit 1
  fi
  hash -r
fi

cd "${PROJECT_ROOT}"

# Make sure the pinned interpreter is available so `uv sync` does not prompt.
if ! uv python list --only-installed | grep -q "${PYTHON_VERSION}"; then
  echo "Installing Python ${PYTHON_VERSION} via uv..."
  uv python install "${PYTHON_VERSION}"
fi

# Always install torch, torchvision, and xformers via the `accelerated` extra.
SYNC_ARGS=(--extra accelerated)
if [[ -f "${PROJECT_ROOT}/uv.lock" ]]; then
  SYNC_ARGS+=(--locked)
else
  echo "uv.lock not found; generating a fresh lockfile with this sync."
fi

echo "Synchronizing environment with uv (PyTorch backend auto-detected)..."
UV_TORCH_BACKEND=auto uv sync "${SYNC_ARGS[@]}"

source .venv/bin/activate

echo "Environment ready. Activate it with 'source .venv/bin/activate' when needed."
echo "By default wandb logging is enabled, remember to run 'wandb init' before training."
