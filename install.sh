pip3 install uv
uv venv dino_env
source dino_env/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
python3 download_model.sh
#This needs to be run here because otherwise you get a package naming issue.
