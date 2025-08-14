#!/usr/bin/env python3
"""
convert_tcga_to_zarr.py
High-performance TCGA to OME-Zarr converter with direct R2 writes.
Implements Midnight CPath specifications.
"""

import os
import sys
import json
import logging
import hashlib
import numpy as np
import zarr
import numcodecs
from numcodecs import Zstd
import s3fs
import pyvips
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Midnight CPath specifications
MIDNIGHT_RESOLUTIONS = {
    '0.25': 0.25,  # 0.25 μm/px (40x equivalent)
    '0.5': 0.5,    # 0.5 μm/px (20x equivalent)
    '1.0': 1.0,    # 1.0 μm/px (10x equivalent)
    '2.0': 2.0,    # 2.0 μm/px (5x equivalent)
}

class DirectR2ZarrConverter:
    """
    High-performance SVS to OME-Zarr converter with direct R2 writes.
    Implements Midnight CPath specifications.
    """
    
    def __init__(
        self,
        local_root: str,
        s3_bucket: str = "tcga-omezarr",
        s3_prefix: str = "",
        chunk_size: int = 512,
        compression_level: int = 3,
        max_workers: int = None,
        batch_size: int = 16
    ):
        self.local_root = Path(local_root)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        self.max_workers = max_workers or mp.cpu_count()
        self.batch_size = batch_size
        
        # Setup S3 filesystem for R2
        self.fs = self._setup_s3fs()
        
        # Compressor - using Zstd instead of Blosc
        self.compressor = Zstd(level=compression_level)
        
        logger.info(f"Initialized converter: bucket={s3_bucket}, workers={self.max_workers}")
    
    def _setup_s3fs(self) -> s3fs.S3FileSystem:
        """Setup S3 filesystem with R2 configuration."""
        endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        if not endpoint_url:
            raise ValueError("R2_ENDPOINT_URL environment variable not set")
        
        return s3fs.S3FileSystem(
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=endpoint_url,
            client_kwargs={
                'region_name': 'auto',
                'use_ssl': True
            },
            config_kwargs={
                'max_pool_connections': 50,
                'retries': {
                    'max_attempts': 3,
                    'mode': 'adaptive'
                }
            }
        )
    
    def find_svs_files(self) -> List[Path]:
        """Find all SVS files in the TCGA directory."""
        svs_files = []
        for svs_path in self.local_root.rglob("*.svs"):
            if "logs" not in str(svs_path):
                svs_files.append(svs_path)
        
        logger.info(f"Found {len(svs_files)} SVS files")
        return sorted(svs_files)
    
    def get_slide_info(self, svs_path: Path) -> Dict:
        """Get slide information and calculate pyramid levels."""
        try:
            # Open slide with pyvips
            slide = pyvips.Image.new_from_file(str(svs_path), access='sequential')
            
            # Get base properties
            width = slide.width
            height = slide.height
            
            # Try to get pixel spacing from metadata
            # Default to 0.25 μm/px (40x) if not found
            base_mpp = 0.25
            
            # Try to read from vips metadata
            try:
                mpp_x = slide.get('openslide.mpp-x')
                if mpp_x:
                    base_mpp = float(mpp_x)
            except:
                pass
            
            # Calculate pyramid levels for Midnight CPath resolutions
            levels = []
            for res_name, target_mpp in MIDNIGHT_RESOLUTIONS.items():
                downsample = target_mpp / base_mpp
                if downsample >= 1.0:  # Only include if downsampling
                    level_width = int(width / downsample)
                    level_height = int(height / downsample)
                    
                    levels.append({
                        'name': res_name,
                        'mpp': target_mpp,
                        'downsample': downsample,
                        'width': level_width,
                        'height': level_height
                    })
            
            # Always include the base resolution
            if not levels or levels[0]['downsample'] > 1.0:
                levels.insert(0, {
                    'name': 'base',
                    'mpp': base_mpp,
                    'downsample': 1.0,
                    'width': width,
                    'height': height
                })
            
            return {
                'width': width,
                'height': height,
                'base_mpp': base_mpp,
                'levels': levels,
                'slide': slide
            }
            
        except Exception as e:
            logger.error(f"Failed to get slide info for {svs_path}: {e}")
            return None
    
    def create_ome_zarr_metadata(self, slide_info: Dict, slide_name: str) -> Dict:
        """Create OME-NGFF v0.4 compliant metadata."""
        datasets = []
        
        for i, level in enumerate(slide_info['levels']):
            # Coordinate transformations for this level
            transforms = [
                {
                    "type": "scale",
                    "scale": [
                        level['mpp'],  # Z (or C) dimension scaling
                        level['mpp'],  # Y dimension scaling  
                        level['mpp']   # X dimension scaling
                    ]
                }
            ]
            
            datasets.append({
                "path": str(i),
                "coordinateTransformations": transforms
            })
        
        # Main multiscales metadata
        multiscales = [{
            "version": "0.4",
            "name": slide_name,
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": datasets,
            "type": "gaussian",
            "metadata": {
                "method": "pyvips",
                "version": "1.0",
                "kwargs": {"base_mpp": slide_info['base_mpp']}
            }
        }]
        
        # OMERO rendering metadata
        omero = {
            "id": 1,
            "name": slide_name,
            "version": "0.4",
            "channels": [
                {
                    "active": True,
                    "coefficient": 1,
                    "color": "FF0000",
                    "family": "linear",
                    "inverted": False,
                    "label": "Red",
                    "window": {"end": 255, "max": 255, "min": 0, "start": 0}
                },
                {
                    "active": True,
                    "coefficient": 1,
                    "color": "00FF00",
                    "family": "linear",
                    "inverted": False,
                    "label": "Green",
                    "window": {"end": 255, "max": 255, "min": 0, "start": 0}
                },
                {
                    "active": True,
                    "coefficient": 1,
                    "color": "0000FF",
                    "family": "linear",
                    "inverted": False,
                    "label": "Blue",
                    "window": {"end": 255, "max": 255, "min": 0, "start": 0}
                }
            ],
            "rdefs": {
                "defaultT": 0,
                "defaultZ": 0,
                "model": "color"
            }
        }
        
        return {
            "multiscales": multiscales,
            "omero": omero,
            "_creator": {
                "name": "DirectR2ZarrConverter",
                "version": "2.0"
            }
        }
    
    def process_slide_direct(self, svs_path: Path) -> bool:
        """Process a single slide with direct R2 writes."""
        try:
            # Generate S3 path
            relative_path = svs_path.relative_to(self.local_root)
            case_id = relative_path.parts[0]
            slide_name = svs_path.stem
            
            if self.s3_prefix:
                s3_path = f"{self.s3_bucket}/{self.s3_prefix}/{case_id}/{slide_name}.zarr"
            else:
                s3_path = f"{self.s3_bucket}/{case_id}/{slide_name}.zarr"
            
            logger.info(f"Processing {slide_name} -> s3://{s3_path}")
            
            # Check if already exists
            if self.fs.exists(f"{s3_path}/.zgroup"):
                logger.info(f"Skipping {slide_name} - already exists")
                return True
            
            # Get slide information
            slide_info = self.get_slide_info(svs_path)
            if not slide_info:
                return False
            
            # Create FSStore for direct S3 writes
            store = zarr.storage.FSStore(s3_path, fs=self.fs)
            
            # Create root group
            root = zarr.group(store=store, overwrite=True)
            
            # Process each pyramid level
            slide = slide_info['slide']
            
            for i, level in enumerate(slide_info['levels']):
                logger.info(f"  Level {i} ({level['name']}): {level['width']}x{level['height']} @ {level['mpp']} μm/px")
                
                # Create array for this level
                arr = root.create_dataset(
                    name=str(i),
                    shape=(3, level['height'], level['width']),  # CYX order
                    chunks=(3, min(self.chunk_size, level['height']), min(self.chunk_size, level['width'])),
                    dtype='u1',
                    compressor=self.compressor,
                    overwrite=True
                )
                
                # Resize slide if needed
                if level['downsample'] > 1.0:
                    resized = slide.resize(1.0 / level['downsample'])
                else:
                    resized = slide
                
                # Process in chunks for memory efficiency
                chunk_h = min(self.chunk_size * 4, level['height'])  # Process larger chunks
                chunk_w = min(self.chunk_size * 4, level['width'])
                
                with tqdm(total=(level['height'] * level['width']) // (chunk_h * chunk_w), 
                         desc=f"Level {i}", leave=False) as pbar:
                    
                    for y in range(0, level['height'], chunk_h):
                        for x in range(0, level['width'], chunk_w):
                            h = min(chunk_h, level['height'] - y)
                            w = min(chunk_w, level['width'] - x)
                            
                            # Extract region
                            region = resized.crop(x, y, w, h)
                            
                            # Convert to numpy (RGB)
                            region_np = np.ndarray(
                                buffer=region.write_to_memory(),
                                dtype=np.uint8,
                                shape=(h, w, region.bands)
                            )
                            
                            # Ensure RGB and transpose to CYX
                            if region_np.shape[2] > 3:
                                region_np = region_np[:, :, :3]
                            region_np = np.transpose(region_np, (2, 0, 1))
                            
                            # Write directly to R2
                            arr[:, y:y+h, x:x+w] = region_np
                            
                            pbar.update(1)
            
            # Write metadata
            metadata = self.create_ome_zarr_metadata(slide_info, slide_name)
            root.attrs.update(metadata)
            
            logger.info(f"Successfully converted {slide_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {svs_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def convert_all_parallel(self):
        """Convert all slides in parallel with direct R2 writes."""
        svs_files = self.find_svs_files()
        
        if not svs_files:
            logger.warning("No SVS files found")
            return
        
        # Process in parallel
        success_count = 0
        failed_files = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_svs = {
                executor.submit(self.process_slide_direct, svs_path): svs_path 
                for svs_path in svs_files
            }
            
            # Process results with progress bar
            with tqdm(total=len(svs_files), desc="Converting slides") as pbar:
                for future in as_completed(future_to_svs):
                    svs_path = future_to_svs[future]
                    try:
                        success = future.result(timeout=600)  # 10 min timeout per slide
                        if success:
                            success_count += 1
                        else:
                            failed_files.append(svs_path)
                    except Exception as e:
                        logger.error(f"Exception processing {svs_path}: {e}")
                        failed_files.append(svs_path)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': success_count,
                        'failed': len(failed_files)
                    })
        
        # Report results
        logger.info(f"\nConversion complete:")
        logger.info(f"  Successful: {success_count}/{len(svs_files)}")
        logger.info(f"  Failed: {len(failed_files)}")
        
        if failed_files:
            logger.error("Failed files:")
            for f in failed_files[:10]:  # Show first 10
                logger.error(f"  - {f}")
            if len(failed_files) > 10:
                logger.error(f"  ... and {len(failed_files) - 10} more")

def main():
    """Conversion script entry point."""
    parser = argparse.ArgumentParser(
        description='Convert TCGA to OME-Zarr following Midnight CPath specs'
    )
    parser.add_argument('--local-root', default='/data/TCGA',
                        help='Root directory containing SVS files')
    parser.add_argument('--s3-bucket', default='tcga-omezarr',
                        help='S3 bucket name')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Chunk size for zarr arrays')
    parser.add_argument('--compression-level', type=int, default=3,
                        help='Zstd compression level (1-22)')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum parallel workers')
    
    args = parser.parse_args()
    
    # Check R2 credentials
    required_vars = ['R2_ENDPOINT_URL', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing = [v for v in required_vars if not os.environ.get(v)]
    
    if missing:
        print(f"Error: Missing environment variables: {missing}")
        print("Please set R2 credentials first")
        sys.exit(1)
    
    # Run conversion
    converter = DirectR2ZarrConverter(
        local_root=args.local_root,
        s3_bucket=args.s3_bucket,
        chunk_size=args.chunk_size,
        compression_level=args.compression_level,
        max_workers=args.max_workers
    )
    
    converter.convert_all_parallel()


if __name__ == "__main__":
    main()