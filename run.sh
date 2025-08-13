source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=30005 --nproc_per_node=2 dinov2/train/train.py --config-file ./dinov2/configs/train/vits14_reg4.yaml --output-dir ./output_pretrained_on_test train.dataset_path=pathology:root=/data/litTCGA/

