pip3 install uv
uv venv dino_env
source dino_env/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
