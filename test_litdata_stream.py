#!/usr/bin/env python3
"""Quick sanity script to stream every patch from a LitData dataset once.

The script walks the dataset sequentially (no shuffling) and reports how many
samples were read along with the total wall-clock time. Each sample's image
bytes are decoded into a Pillow image to ensure the underlying payload is fully
materialized.
"""

import argparse
import time
from io import BytesIO
from pathlib import Path

import litdata as ld
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate through a LitData dataset once and report timing."
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/TCGA_lit_sample30",
        help="Directory containing the LitData shards to read.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print a progress message every N samples (0 disables).",
    )
    return parser.parse_args()


def decode_item(item: dict) -> dict:
    """Decode the raw image bytes to ensure we pay the full IO/CPU cost."""
    with Image.open(BytesIO(item["image_bytes"])) as image:
        image.load()
    return item


def main() -> None:
    from tqdm import tqdm 
    
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_root}")

    dataset = ld.StreamingDataset(
        input_dir=str(dataset_root),
        shuffle=False,
        drop_last=False,
        seed=0,
        transform=decode_item,
    )

    sample_count = 0
    start_time = time.perf_counter()

    for sample in tqdm(dataset):
        sample_count += 1
        if args.log_every and sample_count % args.log_every == 0:
            print(f"Loaded {sample_count:,d} patches...", flush=True)

    elapsed = time.perf_counter() - start_time

    print(f"Total patches read: {sample_count:,d}")
    print(f"Total elapsed time: {elapsed:.2f} seconds")
    if sample_count:
        print(f"Average time per patch: {elapsed / sample_count:.6f} seconds")
        print(f"Patches per second: {sample_count / elapsed:.2f}")


if __name__ == "__main__":
    main()
