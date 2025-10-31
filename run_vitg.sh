source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 torchrun --master_port=54570 --nproc_per_node=1 dinov2/train/train.py --config-file ./dinov2/configs/train/vitg14_reg4.yaml --output-dir ./output_pretrained_giant_hed_lowlr_replication train.dataset_path=pathology:root=/data/TCGA/
