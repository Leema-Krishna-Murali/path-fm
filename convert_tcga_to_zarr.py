#!/usr/bin/env python3
"""
convert_tcga_to_omezarr.py
Converts TCGA SVS files to OME-Zarr format and uploads to R2.
"""

import os
import sys
import json
import logging
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import zarr
import boto3
from botocore.config import Config
import tifffile
import pyvips # requires libvips-dev installed
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SVSToOMEZarrConverter:
    """Converts SVS files to OME-Zarr format with pyramidal structure."""
    
    def __init__(
        self,
        local_root: str,
        s3_bucket: str,
        s3_prefix: str = "",
        temp_dir: str = "/tmp/zarr_conversion",
        tile_size: int = 512,
        compression: str = "zstd",
        compression_level: int = 5,
        max_workers: int = 4
    ):
        self.local_root = Path(local_root)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.tile_size = tile_size
        self.compression = compression
        self.compression_level = compression_level
        self.max_workers = max_workers
        
        # Setup S3 client for R2
        self.s3_client = self._setup_s3_client()
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
    def _setup_s3_client(self):
        """Setup S3 client with R2 configuration."""
        endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        if not endpoint_url:
            raise ValueError("R2_ENDPOINT_URL environment variable not set")
        
        return boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 10, 'mode': 'adaptive'}
            )
        )
    
    def _ensure_bucket_exists(self):
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"Bucket {self.s3_bucket} exists")
        except:
            try:
                self.s3_client.create_bucket(Bucket=self.s3_bucket)
                logger.info(f"Created bucket {self.s3_bucket}")
            except Exception as e:
                logger.error(f"Failed to create bucket: {e}")
                raise
    
    def find_svs_files(self) -> List[Path]:
        """Find all SVS files in the local TCGA directory."""
        svs_files = []
        for svs_path in self.local_root.rglob("*.svs"):
            # Skip files in logs directories
            if "logs" not in str(svs_path):
                svs_files.append(svs_path)
        
        logger.info(f"Found {len(svs_files)} SVS files")
        return sorted(svs_files)
    
    def get_zarr_key(self, svs_path: Path) -> str:
        """Generate S3 key for the zarr store."""
        # Extract case ID from path
        relative_path = svs_path.relative_to(self.local_root)
        case_id = relative_path.parts[0]  # First directory is case ID
        slide_name = svs_path.stem  # Filename without extension
        
        if self.s3_prefix:
            return f"{self.s3_prefix}/{case_id}/{slide_name}.zarr"
        return f"{case_id}/{slide_name}.zarr"
    
    def convert_svs_to_zarr(self, svs_path: Path) -> Optional[Path]:
        """
        Convert a single SVS file to OME-Zarr format.
        Uses pyvips for efficient pyramidal processing.
        """
        try:
            logger.info(f"Converting {svs_path.name}")
            
            # Create temporary zarr directory
            zarr_name = svs_path.stem + ".zarr"
            temp_zarr_path = self.temp_dir / zarr_name
            
            # Clean up any existing temp directory
            if temp_zarr_path.exists():
                shutil.rmtree(temp_zarr_path)
            
            # Open slide with pyvips
            slide = pyvips.Image.new_from_file(str(svs_path), access='sequential')
            
            # Get dimensions
            width = slide.width
            height = slide.height
            
            # Create zarr store
            store = zarr.DirectoryStore(str(temp_zarr_path))
            root = zarr.group(store=store, overwrite=True)
            
            # Calculate pyramid levels
            levels = []
            current_width, current_height = width, height
            level = 0
            
            while current_width > self.tile_size or current_height > self.tile_size:
                levels.append({
                    'level': level,
                    'width': current_width,
                    'height': current_height,
                    'downsample': 2 ** level
                })
                current_width //= 2
                current_height //= 2
                level += 1
                
                # Stop at reasonable minimum size
                if current_width < 256 or current_height < 256:
                    break
            
            # Add final level
            levels.append({
                'level': level,
                'width': current_width,
                'height': current_height,
                'downsample': 2 ** level
            })
            
            # Process each pyramid level
            datasets_metadata = []
            for level_info in levels:
                level_idx = level_info['level']
                level_width = level_info['width']
                level_height = level_info['height']
                downsample = level_info['downsample']
                
                logger.info(f"  Level {level_idx}: {level_width}x{level_height} (downsample={downsample})")
                
                # Create level array
                arr = root.create_dataset(
                    name=f"level{level_idx}",
                    shape=(level_height, level_width, 3),
                    chunks=(min(self.tile_size, level_height), 
                           min(self.tile_size, level_width), 3),
                    dtype='u1',
                    compressor=zarr.Blosc(
                        cname=self.compression, 
                        clevel=self.compression_level, 
                        shuffle=zarr.Blosc.SHUFFLE
                    )
                )
                
                # Resize image if needed
                if downsample > 1:
                    resized = slide.resize(1.0 / downsample)
                else:
                    resized = slide
                
                # Convert to numpy array in tiles to manage memory
                for y in range(0, level_height, self.tile_size):
                    for x in range(0, level_width, self.tile_size):
                        tile_width = min(self.tile_size, level_width - x)
                        tile_height = min(self.tile_size, level_height - y)
                        
                        # Extract tile
                        tile = resized.crop(x, y, tile_width, tile_height)
                        
                        # Convert to numpy array (RGB)
                        tile_np = np.ndarray(
                            buffer=tile.write_to_memory(),
                            dtype=np.uint8,
                            shape=(tile_height, tile_width, tile.bands)
                        )
                        
                        # Write to zarr
                        arr[y:y+tile_height, x:x+tile_width, :] = tile_np[:, :, :3]  # Ensure RGB
                
                # Add to metadata
                datasets_metadata.append({
                    "path": f"level{level_idx}",
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [float(downsample), float(downsample), 1.0]
                    }]
                })
            
            # Write OME-NGFF metadata
            multiscales = [{
                "version": "0.4",
                "name": svs_path.stem,
                "axes": [
                    {"name": "y", "type": "space", "unit": "pixel"},
                    {"name": "x", "type": "space", "unit": "pixel"},
                    {"name": "c", "type": "channel"}
                ],
                "datasets": datasets_metadata
            }]
            
            # Add rendering metadata
            omero = {
                "channels": [
                    {
                        "color": "FF0000",
                        "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                        "label": "Red",
                        "active": True
                    },
                    {
                        "color": "00FF00",
                        "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                        "label": "Green",
                        "active": True
                    },
                    {
                        "color": "0000FF",
                        "window": {"start": 0, "end": 255, "min": 0, "max": 255},
                        "label": "Blue",
                        "active": True
                    }
                ]
            }
            
            root.attrs['multiscales'] = multiscales
            root.attrs['omero'] = omero
            
            logger.info(f"Successfully converted {svs_path.name}")
            return temp_zarr_path
            
        except Exception as e:
            logger.error(f"Failed to convert {svs_path}: {e}")
            return None
    
    def upload_zarr_to_s3(self, local_zarr_path: Path, s3_key: str) -> bool:
        """Upload zarr store to S3."""
        try:
            logger.info(f"Uploading to s3://{self.s3_bucket}/{s3_key}")
            
            # Upload all files in the zarr store
            for file_path in local_zarr_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_zarr_path)
                    object_key = f"{s3_key}/{relative_path}"
                    
                    # Upload file
                    self.s3_client.upload_file(
                        str(file_path),
                        self.s3_bucket,
                        object_key,
                        ExtraArgs={'ContentType': 'application/octet-stream'}
                    )
            
            logger.info(f"Successfully uploaded {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {s3_key}: {e}")
            return False
    
    def process_single_slide(self, svs_path: Path) -> bool:
        """Process a single slide: convert and upload."""
        # Generate S3 key
        s3_key = self.get_zarr_key(svs_path)
        
        # Check if already exists in S3
        try:
            self.s3_client.head_object(
                Bucket=self.s3_bucket,
                Key=f"{s3_key}/.zgroup"
            )
            logger.info(f"Skipping {svs_path.name} - already exists in S3")
            return True
        except:
            pass  # Does not exist, proceed with conversion
        
        # Convert to zarr
        temp_zarr_path = self.convert_svs_to_zarr(svs_path)
        if temp_zarr_path is None:
            return False
        
        # Upload to S3
        success = self.upload_zarr_to_s3(temp_zarr_path, s3_key)
        
        # Clean up temp files
        if temp_zarr_path.exists():
            shutil.rmtree(temp_zarr_path)
        
        return success
    
    def convert_all(self, resume: bool = True):
        """Convert all SVS files to OME-Zarr format."""
        svs_files = self.find_svs_files()
        
        if not svs_files:
            logger.warning("No SVS files found")
            return
        
        # Track progress
        success_count = 0
        failed_files = []
        
        # Process files with progress bar
        with tqdm(total=len(svs_files), desc="Converting slides") as pbar:
            for svs_path in svs_files:
                success = self.process_single_slide(svs_path)
                
                if success:
                    success_count += 1
                else:
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
            for f in failed_files:
                logger.error(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description='Convert TCGA SVS files to OME-Zarr')
    parser.add_argument('--local-root', default='/data/TCGA',
                        help='Root directory containing SVS files')
    parser.add_argument('--s3-bucket', default='tcga-omezarr',
                        help='S3 bucket name')
    parser.add_argument('--s3-prefix', default='',
                        help='S3 prefix for zarr stores')
    parser.add_argument('--temp-dir', default='/tmp/zarr_conversion',
                        help='Temporary directory for conversion')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Tile size for chunking')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum parallel workers')
    parser.add_argument('--resume', action='store_true',
                        help='Resume conversion, skip existing files')
    
    args = parser.parse_args()
    
    # Ensure R2 credentials are set
    required_env_vars = ['R2_ENDPOINT_URL', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please source your R2 credentials first")
        sys.exit(1)
    
    # Create converter
    converter = SVSToOMEZarrConverter(
        local_root=args.local_root,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        temp_dir=args.temp_dir,
        tile_size=args.tile_size,
        max_workers=args.max_workers
    )
    
    # Run conversion
    converter.convert_all(resume=args.resume)


if __name__ == "__main__":
    main()