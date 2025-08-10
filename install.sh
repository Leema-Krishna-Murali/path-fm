pip install uv
uv sync
source .venv/bin/activate
uv pip install -e .
uv pip install 'kaiko-eva[vision]'
cp _utils.py .venv/lib/python3.10/site-packages/eva/core/models/wrappers/
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

