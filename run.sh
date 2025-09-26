source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=32001 --nproc_per_node=4 dinov2/train/train.py --config-file ./dinov2/configs/train/vits14_reg4.yaml --output-dir ./output_pretrained_on_test2 train.dataset_path=pathology:root=/data/TCGA/


