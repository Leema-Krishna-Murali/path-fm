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
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
import cv2
from PIL import Image
import openslide
from openslide import OpenSlide
import torch
from litdata import optimize
from litdata.streaming import StreamingDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _identity_fn(tile_data):
    """Identity function for LitData optimization - returns data as-is."""
    return tile_data


@dataclass
class TileConfig:
    """Configuration for tile extraction following Midnight paper."""
    tile_size: int = 256  # Fixed tile size in pixels
    magnifications: List[float] = None  # µm/px magnifications
    foreground_threshold: float = 0.4  # 40% foreground area threshold
    hsv_filter: bool = True  # Apply HSV color space filtering
    hsv_ranges: Dict[str, Tuple[int, int]] = None
    pixel_threshold: float = 0.6  # 60% of pixels must pass HSV filter
    
    def __post_init__(self):
        if self.magnifications is None:
            # Midnight paper magnifications: 2, 1, 0.5, 0.25 µm/px
            self.magnifications = [2.0, 1.0, 0.5, 0.25]
        
        if self.hsv_ranges is None:
            # HSV ranges from Midnight paper
            self.hsv_ranges = {
                'hue': (90, 180),
                'saturation': (8, 255),
                'value': (103, 255)
            }


class TileExtractor:
    """Extract tiles from WSI following Midnight paper methodology."""
    
    def __init__(self, config: TileConfig):
        self.config = config
        
    def get_tissue_mask(self, slide: OpenSlide, level: int = 2) -> np.ndarray:
        """Generate tissue mask for the slide."""
        # Get thumbnail for tissue detection
        dims = slide.level_dimensions[level]
        thumbnail = slide.read_region((0, 0), level, dims)
        thumbnail = thumbnail.convert('RGB')
        thumbnail_np = np.array(thumbnail)
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
        _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        
        return tissue_mask
    
    def check_foreground_threshold(self, tile: np.ndarray) -> bool:
        """Check if tile meets foreground area threshold."""
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_ratio = np.sum(binary > 0) / binary.size
        return foreground_ratio >= self.config.foreground_threshold
    
    def apply_hsv_filter(self, tile: np.ndarray) -> bool:
        """Apply HSV color space filter from Midnight paper."""
        if not self.config.hsv_filter:
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
        
        # Check if enough pixels pass the filter
        pixel_ratio = np.sum(mask) / mask.size
        return pixel_ratio >= self.config.pixel_threshold
    
    def get_available_magnifications(self, slide: OpenSlide) -> Dict[float, int]:
        """Map desired magnifications to available slide levels."""
        # Get slide magnification from properties
        try:
            slide_mpp = float(slide.properties.get(
                openslide.PROPERTY_NAME_MPP_X,
                slide.properties.get('aperio.MPP', 0.25)
            ))
        except:
            slide_mpp = 0.25  # Default to 40x (0.25 µm/px)
        
        magnification_map = {}
        for target_mag in self.config.magnifications:
            # Find closest available level
            best_level = 0
            min_diff = float('inf')
            
            for level in range(slide.level_count):
                downsample = slide.level_downsamples[level]
                level_mpp = slide_mpp * downsample
                diff = abs(level_mpp - target_mag)
                
                if diff < min_diff:
                    min_diff = diff
                    best_level = level
            
            magnification_map[target_mag] = best_level
            
        return magnification_map
    
    def extract_tiles_from_slide(self, slide_path: str, max_tiles_per_mag: int = 100) -> List[Dict[str, Any]]:
        """Extract tiles from a single slide at multiple magnifications."""
        tiles_data = []
        
        try:
            slide = OpenSlide(slide_path)
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            return tiles_data
        
        # Get tissue mask
        tissue_mask = self.get_tissue_mask(slide)
        
        # Get magnification to level mapping
        mag_map = self.get_available_magnifications(slide)
        
        # Extract tiles for each magnification
        for target_mag, level in mag_map.items():
            level_dims = slide.level_dimensions[level]
            downsample = slide.level_downsamples[level]
            
            # Calculate tile size at this level
            tile_size_level0 = int(self.config.tile_size * downsample)
            
            # Generate random positions
            n_attempts = 0
            n_extracted = 0
            max_attempts = max_tiles_per_mag * 10  # Allow multiple attempts
            
            while n_extracted < max_tiles_per_mag and n_attempts < max_attempts:
                n_attempts += 1
                
                # Random position in level 0 coordinates
                x = random.randint(0, slide.dimensions[0] - tile_size_level0)
                y = random.randint(0, slide.dimensions[1] - tile_size_level0)
                
                # Check if position is in tissue region
                mask_x = int(x / slide.level_downsamples[2])
                mask_y = int(y / slide.level_downsamples[2])
                mask_size = int(tile_size_level0 / slide.level_downsamples[2])
                
                if mask_x + mask_size >= tissue_mask.shape[1] or mask_y + mask_size >= tissue_mask.shape[0]:
                    continue
                    
                mask_region = tissue_mask[mask_y:mask_y+mask_size, mask_x:mask_x+mask_size]
                if np.mean(mask_region) < 128:  # Not enough tissue
                    continue
                
                # Extract tile
                tile = slide.read_region((x, y), level, (self.config.tile_size, self.config.tile_size))
                tile = tile.convert('RGB')
                tile_np = np.array(tile)
                
                # Apply filters
                if not self.check_foreground_threshold(tile_np):
                    continue
                    
                if not self.apply_hsv_filter(tile_np):
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
                
            logger.info(f"Extracted {n_extracted} tiles at {target_mag} µm/px from {Path(slide_path).name}")
        
        slide.close()
        return tiles_data


def process_slide_batch(slide_paths: List[str], config: TileConfig, tiles_per_mag: int = 100) -> List[Dict[str, Any]]:
    """Process a batch of slides."""
    extractor = TileExtractor(config)
    all_tiles = []
    
    for slide_path in slide_paths:
        tiles = extractor.extract_tiles_from_slide(slide_path, tiles_per_mag)
        all_tiles.extend(tiles)
    
    return all_tiles


def tile_generator(slide_paths: List[str], config: TileConfig, tiles_per_mag: int = 100):
    """Generator that yields tiles for LitData optimization."""
    extractor = TileExtractor(config)
    
    for slide_path in tqdm(slide_paths, desc="Processing slides"):
        try:
            tiles = extractor.extract_tiles_from_slide(slide_path, tiles_per_mag)
            for tile_data in tiles:
                # Convert numpy array to PIL Image for better compatibility
                tile_data['image'] = Image.fromarray(tile_data['image'])
                yield tile_data
        except Exception as e:
            logger.error(f"Error processing {slide_path}: {e}")
            continue


def find_svs_files(root_dir: str, exclude_file: Optional[str] = None) -> List[str]:
    """Find all SVS files in directory tree, excluding bad files if specified."""
    root_path = Path(root_dir)
    svs_files = list(root_path.rglob("*.svs"))
    
    # Load exclusion list if provided
    excluded = set()
    if exclude_file and Path(exclude_file).exists():
        with open(exclude_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    excluded.add(line)
    
    # Filter out excluded files
    valid_files = []
    for svs_file in svs_files:
        if str(svs_file) not in excluded:
            valid_files.append(str(svs_file))
        else:
            logger.info(f"Excluding: {svs_file}")
    
    logger.info(f"Found {len(valid_files)} valid SVS files (excluded {len(excluded)})")
    return valid_files


def create_litdata_dataset(
    input_dir: str,
    output_dir: str,
    config: TileConfig,
    tiles_per_mag: int = 100,
    num_workers: int = None,
    exclude_file: Optional[str] = None
):
    """Create LitData optimized dataset from TCGA SVS files."""
    
    # Find all SVS files
    slide_paths = find_svs_files(input_dir, exclude_file)
    
    if not slide_paths:
        logger.error(f"No SVS files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(slide_paths)} slides")
    logger.info(f"Output directory: {output_dir}")
    
    # Set number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count() - 1, 8)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = Path(output_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Optimize dataset using LitData
    logger.info("Starting LitData optimization...")
    
    # Create generator that yields processed tiles
    def data_generator():
        return tile_generator(slide_paths, config, tiles_per_mag)
    
    # Create list of inputs (we need to convert generator to list for litdata)
    all_tiles = list(data_generator())
    
    optimize(
        fn=_identity_fn,
        inputs=all_tiles,
        output_dir=output_dir,
        num_workers=num_workers,
        chunk_bytes="64MB",
        mode="overwrite"
    )
    
    logger.info(f"Dataset optimization complete! Output saved to {output_dir}")
    
    # Create metadata file
    metadata = {
        'num_slides': len(slide_paths),
        'tiles_per_magnification': tiles_per_mag,
        'magnifications': config.magnifications,
        'tile_size': config.tile_size,
        'total_tiles_estimate': len(slide_paths) * tiles_per_mag * len(config.magnifications)
    }
    
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert TCGA SVS dataset to LitData format")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/TCGA_test",
        help="Input directory containing TCGA SVS files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/litTCGA",
        help="Output directory for LitData dataset"
    )
    parser.add_argument(
        "--tiles-per-mag",
        type=int,
        default=100,
        help="Number of tiles to extract per magnification level"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
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
        "--no-hsv-filter",
        action="store_true",
        help="Disable HSV color space filtering"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = TileConfig(
        tile_size=args.tile_size,
        hsv_filter=not args.no_hsv_filter
    )
    
    # Run conversion
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