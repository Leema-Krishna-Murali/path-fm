source .venv/bin/activate

# Disable NVLink SHARP in NCCL; current driver stack throws `cudaErrorInvalidValue`
# when NCCL tries to bring NVLS up on >2 GPUs. Forcing it off keeps multi-GPU init stable.
export NCCL_NVLS_ENABLE=0

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --master_port=54570 --nproc_per_node=7 dinov2/train/train.py --config-file ./dinov2/configs/train/vitg14_reg4.yaml --output-dir ./output_pretrained_giant_hed_lowlr_replication train.dataset_path=pathology:root=/data/TCGA/
