# Evaluate a specific checkpoint
cd /home/paul/dinov2
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # Use different GPU

python -m dinov2.eval.linear \
    --config-file /home/paul/dinov2_runs/vitl16_h100_30ep/config.yaml \
    --pretrained-weights /home/paul/dinov2_runs/vitl16_h100_30ep/checkpoints/iter_002000/teacher.pth \
    --output-dir /home/paul/dinov2_runs/vitl16_h100_30ep/eval/iter_002000 \
    --epochs 10 \
    --batch-size 256
