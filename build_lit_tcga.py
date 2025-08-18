#!/usr/bin/env python3
"""
Convert TCGA SVS dataset to LitData optimized format for efficient streaming during training.
Following the Midnight paper approach for multi-scale tile extraction.
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import multiprocessing as mp

import numpy as np
import cv2
from PIL import Image
import openslide
from openslide import OpenSlide
import boto3  
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from litdata import optimize
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TileConfig:
    """Configuration for tile extraction following Midnight paper."""
    tile_size: int = 256  # Fixed tile size in pixels
    magnifications: List[float] = None  # μm/px magnifications
    foreground_threshold: float = 0.4  # 40% foreground area threshold
    hsv_threshold: bool = True  # Apply HSV color space thresholding
    hsv_ranges: Dict[str, Tuple[int, int]] = None
    pixel_threshold: float = 0.6  # 60% of pixels must pass HSV threshold
    
    def __post_init__(self):
        if self.magnifications is None:
            # Midnight paper magnifications: 2, 1, 0.5, 0.25 μm/px
            self.magnifications = [2.0, 1.0, 0.5, 0.25]
        
        if self.hsv_ranges is None:
            # HSV ranges from Midnight paper
            self.hsv_ranges = {
                'hue': (90, 180),
                'saturation': (8, 255),
                'value': (103, 255)
            }


def _extract_tiles_fn(slide_path: str, config: TileConfig, tiles_per_mag: int):
    """LitData optimize() worker: given one slide path, yield many tile dicts."""
    extractor = TileExtractor(config)
    tiles = extractor.extract_tiles_from_slide(slide_path, tiles_per_mag)
    for t in tiles:
        # LitData serializes PIL Images out of the box
        t["image"] = Image.fromarray(t["image"])
        yield t


class TileExtractor:
    """Extract tiles from WSI following Midnight paper methodology."""
    
    def __init__(self, config: TileConfig):
        self.config = config
        
    def get_tissue_mask(self, slide: OpenSlide, level: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Build a tissue mask from a downsampled level.
        Default: use the coarsest level (level_count - 1) for speed and safety.
        """
        if level is None:
            safe_level = slide.level_count - 1               # coarsest is always valid
        else:
            safe_level = max(0, min(level, slide.level_count - 1))  # clamp to valid range

        dims = slide.level_dimensions[safe_level]
        thumbnail = slide.read_region((0, 0), safe_level, dims).convert('RGB')
        thumbnail_np = np.array(thumbnail)

        gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
        otsu_thresh, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

        return tissue_mask, otsu_thresh
    
    def check_foreground_threshold(self, tile: np.ndarray, slide_threshold: float = None) -> bool:
        """
        Check if tile meets foreground area threshold.
        Uses a consistent threshold across all tiles from the same slide.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        
        if slide_threshold is not None:
            # Use the threshold computed from the whole slide
            _, binary = cv2.threshold(gray, slide_threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            # Fallback: use a fixed threshold of 200 for background detection, which is very bright
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate foreground ratio
        foreground_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        foreground_ratio = foreground_pixels / total_pixels
        
        return foreground_ratio >= self.config.foreground_threshold
    
    def apply_hsv_threshold(self, tile: np.ndarray) -> bool:
        """Apply HSV color space threshold from Midnight paper."""
        if not self.config.hsv_threshold:
            return True
            
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        
        # Create mask for pixels within HSV ranges
        mask = (
            (hsv[:, :, 0] >= self.config.hsv_ranges['hue'][0]) & 
            (hsv[:, :, 0] <= self.config.hsv_ranges['hue'][1]) &
            (hsv[:, :, 1] >= self.config.hsv_ranges['saturation'][0]) & 
            (hsv[:, :, 1] <= self.config.hsv_ranges['saturation'][1]) &
            (hsv[:, :, 2] >= self.config.hsv_ranges['value'][0]) & 
            (hsv[:, :, 2] <= self.config.hsv_ranges['value'][1])
        )
        
        # Check if enough pixels pass the threshold
        pixel_ratio = np.sum(mask) / mask.size
        return pixel_ratio >= self.config.pixel_threshold
    
    def get_available_magnifications(self, slide: OpenSlide) -> Dict[float, int]:
        """Map desired magnifications (μm/px) to the nearest valid slide level,
        skipping targets that are outside the slide's available μm/px range.
        """
        # Try to read base MPP (μm/px at level 0)
        try:
            base_mpp = float(slide.properties.get(
                openslide.PROPERTY_NAME_MPP_X,
                slide.properties.get('aperio.MPP', 0.25)  # fallback
            ))
        except Exception:
            base_mpp = 0.25  # conservative default (40x)

        # Build (level, level_mpp) list
        levels_mpp = []
        for lvl in range(slide.level_count):
            down = float(slide.level_downsamples[lvl])
            levels_mpp.append((lvl, base_mpp * down))

        # Determine available μm/px range
        available_mpps = [mpp for _, mpp in levels_mpp]
        min_mpp, max_mpp = min(available_mpps), max(available_mpps)

        mapping: Dict[float, int] = {}

        for target in self.config.magnifications:
            # If outside the available μm/px range, skip it
            eps = 0.02  # 2% tolerance for metadata rounding
            if target < min_mpp * (1 - eps) or target > max_mpp * (1 + eps):
                logger.debug(
                    f"Skipping target {target:.3f} μm/px: outside available "
                    f"[{min_mpp:.3f}, {max_mpp:.3f}] μm/px for this slide."
                )
                continue

            # Otherwise, choose the nearest level
            best_level, best_mpp = min(levels_mpp, key=lambda x: abs(x[1] - target))
            mapping[target] = best_level

        return mapping
    
    def extract_tiles_from_slide(self, slide_path: str, max_tiles_per_mag: int = 100) -> List[Dict[str, Any]]:
        """Extract tiles from a single slide at multiple magnifications."""
        tiles_data = []
        
        try:
            slide = OpenSlide(slide_path)
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            return tiles_data
        
        # Get tissue mask and threshold
        mask_level = min(2, slide.level_count - 1)
        tissue_mask, otsu_thresh = self.get_tissue_mask(slide, level=mask_level)
        
        # Get magnification to level mapping
        mag_map = self.get_available_magnifications(slide)
        
        # Extract tiles for each magnification
        for target_mag, level in mag_map.items():
            downsample = slide.level_downsamples[level]
            
            # Tile size in level 0 coordinates corresponds to the physical area we want to capture
            tile_size_level0 = int(self.config.tile_size * downsample)
            
            # Generate random positions
            n_attempts = 0
            n_extracted = 0
            max_attempts = max_tiles_per_mag * 20  # Allow multiple attempts
            
            while n_extracted < max_tiles_per_mag and n_attempts < max_attempts:
                n_attempts += 1
                
                try:
                    x = random.randint(0, slide.dimensions[0] - tile_size_level0)
                    y = random.randint(0, slide.dimensions[1] - tile_size_level0)
                except ValueError:
                    logger.warning(f"Slide {Path(slide_path).name} dimensions are smaller than tile size at level {level}.")
                    break
                
                # Check if position is in tissue region using a downsampled tissue mask
                mask_downsample = slide.level_downsamples[mask_level]
                
                mask_x = int(x / mask_downsample)
                mask_y = int(y / mask_downsample)
                mask_size = int(tile_size_level0 / mask_downsample)
                
                # Define mask boundaries
                mask_x_end = min(mask_x + mask_size, tissue_mask.shape[1])
                mask_y_end = min(mask_y + mask_size, tissue_mask.shape[0])
                
                if mask_x >= tissue_mask.shape[1] or mask_y >= tissue_mask.shape[0]:
                    continue
                
                mask_region = tissue_mask[mask_y:mask_y_end, mask_x:mask_x_end]
                if mask_region.size == 0 or np.sum(mask_region > 0) / mask_region.size < self.config.foreground_threshold:
                    continue
                
                # Read a 256x256 region directly from the specified level.
                # The `location` is in level 0 coordinates, and OpenSlide handles the conversion.
                tile = slide.read_region(
                    location=(x, y), 
                    level=level, 
                    size=(self.config.tile_size, self.config.tile_size)
                )
                
                tile = tile.convert('RGB')
                tile_np = np.array(tile)
                
                # Apply thresholds
                if not self.check_foreground_threshold(tile_np, otsu_thresh):
                    continue
                    
                if not self.apply_hsv_threshold(tile_np):
                    continue
                
                # Valid tile found
                tile_data = {
                    'image': tile_np,
                    'slide_path': slide_path,
                    'position': (x, y),
                    'level': level,
                    'magnification': target_mag,
                    'slide_id': Path(slide_path).stem
                }
                
                tiles_data.append(tile_data)
                n_extracted += 1
                
            logger.debug(f"Extracted {n_extracted} tiles at {target_mag} μm/px from {Path(slide_path).name}")
        
        slide.close()
        return tiles_data


def find_svs_files(root_dir: str, exclude_file: Optional[str] = None) -> List[str]:
    """Find all SVS files in directory tree, excluding bad files if specified."""
    root_path = Path(root_dir)
    logger.info("Searching for SVS files, this may take a while for large datasets...")
    svs_files = [str(p) for p in root_path.rglob("*.svs")] + [str(p) for p in root_path.rglob("*.SVS")]
    
    excluded = set()
    if exclude_file and Path(exclude_file).exists():
        with open(exclude_file, 'r') as f:
            excluded.update(line.strip() for line in f if line.strip() and not line.startswith('#'))
    
    valid_files = [f for f in svs_files if f not in excluded]
    logger.info(f"Found {len(valid_files)} total valid SVS files.")
    return valid_files

def get_processed_slides(s3_client: BaseClient, bucket: str, key: str) -> Set[str]:
    """Reads the list of processed slides from the log file in S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        processed = set(line.strip() for line in content.splitlines() if line.strip())
        logger.info(f"Resuming. Found {len(processed)} previously processed slides in log file.")
        return processed
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info("No existing log file found. Starting a fresh run.")
            return set()
        else:
            logger.error(f"Error reading log file from S3: {e}")
            raise

def update_processed_slides(s3_client: BaseClient, bucket: str, key: str, all_processed_paths: Set[str]):
    """Writes the updated list of processed slides to the log file in S3."""
    logger.info(f"Updating log file with {len(all_processed_paths)} total processed slides.")
    content = "\n".join(sorted(list(all_processed_paths)))
    s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'))
    logger.info("Log file successfully updated.")

def create_litdata_dataset(
    input_dir: str, output_dir: str, config: TileConfig,
    tiles_per_mag: int, num_workers: int, exclude_file: Optional[str]
):
    is_s3_output = output_dir.startswith("s3://")
    s3_client = None
    log_bucket, log_key = None, None

    if is_s3_output:
        logger.info("Outputting to S3-compatible storage.")
        s3_client = boto3.client("s3")
        parsed_url = urlparse(output_dir)
        log_bucket = parsed_url.netloc
        prefix = parsed_url.path.lstrip('/')
        if prefix and not prefix.endswith('/'): prefix += '/'
        log_key = f"{prefix}_processed_slides.log"
    else:
        logger.error("Resumption is only supported for S3 outputs. Please provide an s3:// URI.")
        return

    # --- RESUMING FROM INTERRUPT LOGIC ---
    all_slide_paths = find_svs_files(input_dir, exclude_file)
    already_processed = get_processed_slides(s3_client, log_bucket, log_key)
    
    # Make sure both sides are comparable strings (absolute is safer)
    norm = lambda p: os.path.abspath(p)
    already_processed = set(map(norm, already_processed))
    slides_to_process = [p for p in map(norm, all_slide_paths) if p not in already_processed]

    if not slides_to_process:
        logger.info("All slides have already been processed. Nothing to do.")
        return
    
    logger.info(f"Starting processing for {len(slides_to_process)} remaining slides out of {len(all_slide_paths)} total.")
    
    # Upload config only if it's the first run
    if not already_processed:
        config_key = f"{prefix}config.json"
        s3_client.put_object(Bucket=log_bucket, Key=config_key, Body=json.dumps(asdict(config), indent=2))
        logger.info(f"Uploaded config to s3://{log_bucket}/{config_key}")

    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, 8)
    
    logger.info("Starting LitData optimization (slide list + per-slide extractor)...")

    # Decide mode: first successful run uses overwrite, later runs append
    mode = "overwrite" if not already_processed else "append"

    # OPTIONAL: batch slides to reduce blast radius on failure (e.g., 200 slides per call)
    batch_size = 200
    batches = [slides_to_process[i:i+batch_size] for i in range(0, len(slides_to_process), batch_size)]

    for i, batch in enumerate(batches):
        # first batch of a brand new dataset: overwrite; otherwise append
        batch_mode = "overwrite" if (mode == "overwrite" and i == 0) else "append"
        
        try:
            optimize(
                fn=partial(_extract_tiles_fn, config=config, tiles_per_mag=tiles_per_mag),
                inputs=batch,
                output_dir=output_dir,
                num_workers=num_workers,
                chunk_bytes="128MB",
                keep_data_ordered=False,   # helps avoid end-of-run stalls
                mode=batch_mode,
            )
        except Exception as e:
            # Do NOT update logs/metadata on failure; leave state consistent so you can resume
            logger.exception(f"Batch {i} failed before finalization; no progress recorded for this batch.")
            raise

        newly = set(map(os.path.abspath, batch))
        fully_processed = already_processed.union(newly)
        update_processed_slides(s3_client, log_bucket, log_key, set(map(str, fully_processed)))
        already_processed = fully_processed
    
        metadata = {
            "num_slides": len(fully_processed),
            "tiles_per_magnification": tiles_per_mag,
            "magnifications": config.magnifications,
            "tile_size": config.tile_size,
            "last_updated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "batch_index": i,
            "total_batches": len(batches),
        }
        metadata_key = f"{prefix}metadata.json"
        s3_client.put_object(Bucket=log_bucket, Key=metadata_key, Body=json.dumps(metadata, indent=2))
        logger.info(f"Batch {i} committed. Processed slides so far: {len(fully_processed)}")

    logger.info("Processing finished.")

def main():
    parser = argparse.ArgumentParser(description="Convert TCGA SVS dataset to LitData format, saving to local or S3 storage.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/TCGA",
        help="Input directory containing TCGA SVS files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="s3://tcga-litdata",
        help="Output directory for LitData dataset (e.g., /path/to/dir or s3://bucket/prefix)"
    )
    parser.add_argument(
        "--tiles-per-mag",
        type=int,
        default=1, #8445 for approximately 384 million tiles (8445 = 384M / (#_of_slides x #_of_levels))
        help="Number of tiles to extract per magnification level"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--exclude-file",
        type=str,
        default="baddata.txt",
        help="File containing paths of SVS files to exclude"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Tile size in pixels (default: 256)"
    )
    parser.add_argument(
        "--no-hsv-threshold",
        action="store_true",
        help="Disable HSV color space thresholding"
    )
    
    args = parser.parse_args()
    
    config = TileConfig(
        tile_size=args.tile_size,
        hsv_threshold=not args.no_hsv_threshold
    )
    
    create_litdata_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config,
        tiles_per_mag=args.tiles_per_mag,
        num_workers=args.num_workers,
        exclude_file=args.exclude_file
    )


if __name__ == "__main__":
    main()