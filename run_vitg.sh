source .venv/bin/activate
#python3 ./dinov2/run/train/train.py --nodes 1 --config-file ./dinov2/configs/train/vits14_reg4.yaml --output-dir ./output train.dataset_path=pathology:root=/data/inet_images/
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=54570 --nproc_per_node=4 dinov2/train/train.py --config-file ./dinov2/configs/train/vitg14_reg4.yaml --output-dir ./output_pretrained_giant train.dataset_path=pathology:root=/data/TCGA/



