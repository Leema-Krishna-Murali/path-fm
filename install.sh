pip3 install uv
uv venv dino_env
source dino_env/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv pip install openslide-python openslide-bin opencv-python scikit-image einops matplotlib opencv-python
uv pip install 'kaiko-eva[vision]'
cp _utils.py ./dino_env/lib/python3.11/site-packages/eva/core/models/wrappers/
