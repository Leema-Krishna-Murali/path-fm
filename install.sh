#!/usr/bin/env bash
set -euo pipefail

# Ensure uv is installed and on PATH
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi
  # The installer typically places uv in ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
  hash -r
fi

uv sync
source .venv/bin/activate
uv pip install -e .
# cp _utils.py .venv/lib/python3.10/site-packages/eva/core/models/wrappers/
uv pip install torch==2.7.1 torchvision==0.22.1 xformers --torch-backend=auto
uv pip install 'kaiko-eva[vision]'

echo "Installation complete! Python environment is setup in .venv."
echo "Run 'wandb init' to setup your wandb credentials before training."