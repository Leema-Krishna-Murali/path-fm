# Pathology DinoV2

**[SophontAI](https://sophontai.com/)**

In this repository, following a plethora of works before us, we apply DINO(V2) to the pathology space.
If you are interested in helping out, check the open Issues.

## Installation

Clone the repository, cd into it, then run the provided installation script. 
```shell
pip install uv
uv sync
source .venv/bin/activate
cp _utils.py .venv/lib/python3.10/site-packages/eva/core/models/wrappers/
```
This will create a virtual environment with all necessary packages pre-installed called "pathologydino", located as a .venv folder in the same directory as path-fm.

## Training

We provide a script, run.sh.
This will activate the venv created above, and run training on a single node. 
If you modified the venv directory, you will need to make that change in run.sh also.

```shell
bash run.sh
```

By default, we make only 4 GPUs visible, and run on those 4. If you want to change the indexes, modify the numbers after "CUDA_VISIBLE_DEVICES=0,1,2,3".

If you change the number of GPUs, you will need to change the value of "--nproc_per_node=4" to properly reflect this.

By default, we use a vits, with 4 registers. This is reflected in the config. 

Output will be saved into the directory specificed by "--output_dir". Ensure that this directory does not contain any old files from training runs, or the code will attempt to resume instead.


## Evaluation

At this time, we use [Kaiko-Eva](https://github.com/kaiko-ai/eva) for evaluation.
In order to test the [Bach](https://arxiv.org/abs/1808.04277) dataset, you will run:
```
eva predict fit --config dinov2/eval_config.yaml
```
Please modify the checkpoint_path to match the checkpoint you wish to test.
Trained checkpoints will be found in output_dir/eval/training_X.