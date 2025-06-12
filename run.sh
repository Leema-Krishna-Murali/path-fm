source dino_env/bin/activate
python3 ./dinov2/run/train/train.py --nodes 1 --config-file ./dinov2/configs/train/vits14_reg4.yaml --output-dir ./output train.dataset_path=pathology:root=/data/inet_images/

