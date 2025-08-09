source .venv/bin/activate
# CUDA_VISIBLE_DEVICES=1,2 torchrun --master_port=34005 --nproc_per_node=2 dinov2/train/train.py --config-file ./dinov2/configs/train/vits14_reg4.yaml --output-dir ./output_pretrained_on_test train.dataset_path=pathology:root=/data/TCGA/

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 dinov2/train/train.py --config-file ./dinov2/configs/train/vits14_reg4.yaml --output-dir ./output_pretrained_on_test_zarr train.dataset_path=pathology_zarr:root=/data/TCGA_zarr/