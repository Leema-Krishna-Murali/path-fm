CUDA_VISIBLE_DEVICES=0 uv run \
  torchrun --master_port=34001 --nproc_per_node=1 \
  dinov2/train/train.py \
  --config-file ./dinov2/configs/train/vits14_reg4.yaml \
  --output-dir ./output_pretrained_on_test