# dinov2/data/datasets/slide_dataset_zarr.py
"""
Enhanced OME-Zarr dataset with Midnight CPath paper specifications:
- Multi-resolution sampling (2, 1, 0.5, 0.25 µm/px)
- HSV-based tissue filtering
- 40% foreground threshold
- Online random patching
"""

import os
import random
import logging
import hashlib
import threading
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import zarr
import fsspec
import s3fs
from skimage.color import rgb2hsv

from .extended import ExtendedVisionDataset

logger = logging.getLogger(__name__)

# Suppress noisy logs
for logger_name in ['aiobotocore.credentials', 'botocore.credentials', 's3fs']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


class DirectR2Store:
    """Direct-to-R2 Zarr store using FSStore."""
    
    def __init__(self, s3_path: str, cache_metadata: bool = True):
        """
        Args:
            s3_path: Full S3 path to zarr store (s3://bucket/path)
            cache_metadata: Whether to cache metadata locally
        """
        self.s3_path = s3_path
        
        # Setup S3 filesystem
        self.fs = s3fs.S3FileSystem(
            endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            use_ssl=True,
            client_kwargs={
                "connect_timeout": 30,
                "read_timeout": 60,
                "retries": {"max_attempts": 5, "mode": "adaptive"}
            }
        )
        
        # Create FSStore for direct R2 access
        self.store = zarr.storage.FSStore(
            self.s3_path,
            fs=self.fs,
            mode='r',
            dimension_separator='/',
            check=False
        )
        
        # Open zarr group
        self.group = zarr.open_group(self.store, mode='r')
        
        # Cache metadata if requested
        self._metadata_cache = {} if cache_metadata else None
        if cache_metadata:
            self._cache_metadata()
    
    def _cache_metadata(self):
        """Cache all metadata locally for faster access."""
        try:
            # Cache multiscales
            self._metadata_cache['multiscales'] = self.group.attrs.get('multiscales', [])
            
            # Cache array info
            self._metadata_cache['arrays'] = {}
            for key in self.group.keys():
                if not key.startswith('.'):
                    arr = self.group[key]
                    self._metadata_cache['arrays'][key] = {
                        'shape': arr.shape,
                        'chunks': arr.chunks,
                        'dtype': str(arr.dtype)
                    }
        except Exception as e:
            logger.warning(f"Failed to cache metadata: {e}")


class SlideDatasetZarr(ExtendedVisionDataset):
    """
    OME-Zarr dataset with Midnight CPath paper specifications.
    
    Features:
    - Multi-resolution sampling (2, 1, 0.5, 0.25 µm/px)
    - HSV-based tissue filtering (≥60% pixels in specified ranges)
    - 40% foreground area threshold
    - 256×256 tile size
    - Direct R2 streaming with intelligent caching
    """
    
    def __init__(
        self,
        root: str,
        tile_size: int = 256,  # Midnight uses 256×256
        foreground_threshold: float = 0.4,  # 40% foreground
        hsv_filter: bool = True,
        hsv_ranges: Dict[str, Tuple[int, int]] = None,
        target_mpp: List[float] = None,  # [2.0, 1.0, 0.5, 0.25] µm/px
        cache_gb: float = 1000.0,
        prefetch_workers: int = 4,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.root = str(root)
        self.tile_size = tile_size
        self.foreground_threshold = foreground_threshold
        self.hsv_filter = hsv_filter
        
        # HSV ranges from Midnight paper
        self.hsv_ranges = hsv_ranges or {
            'hue': (90, 180),      # Hue range [90, 180]
            'saturation': (8, 255),  # Saturation range [8, 255]
            'value': (103, 255)     # Value range [103, 255]
        }
        
        # Target microns per pixel (magnifications)
        self.target_mpp = target_mpp or [2.0, 1.0, 0.5, 0.25]
        
        # Setup direct R2 access
        self.zarr_stores = self._find_zarr_stores()
        if len(self.zarr_stores) == 0:
            raise ValueError(f"No zarr stores found in {self.root}")
        
        logger.info(f"Found {len(self.zarr_stores)} zarr stores")
        logger.info(f"Target magnifications (µm/px): {self.target_mpp}")
        logger.info(f"Tile size: {self.tile_size}×{self.tile_size}")
        logger.info(f"HSV filtering: {self.hsv_filter}")
        
        # Store handles cache
        self._store_cache = {}
        self._cache_lock = threading.Lock()
        
        # Prefetch executor
        self.prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_workers)
        
        # Statistics
        self.stats = {
            'tiles_extracted': 0,
            'tiles_accepted': 0,
            'tiles_rejected_foreground': 0,
            'tiles_rejected_hsv': 0
        }
    
    def _find_zarr_stores(self) -> List[str]:
        """Find all zarr stores in the S3 path."""
        stores = []
        
        # Setup S3 filesystem
        fs = s3fs.S3FileSystem(
            endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        
        # Parse S3 path
        if not self.root.startswith("s3://"):
            raise ValueError(f"Root must be an S3 path, got: {self.root}")
        
        path_without_scheme = self.root.replace("s3://", "")
        
        try:
            # Find all .zarr directories
            items = fs.find(path_without_scheme, maxdepth=None, withdirs=True)
            
            for item in items:
                if item.endswith(".zarr/.zgroup"):
                    zarr_path = item.replace("/.zgroup", "")
                    if not zarr_path.startswith("s3://"):
                        zarr_path = f"s3://{zarr_path}"
                    stores.append(zarr_path)
            
            return sorted(list(set(stores)))
            
        except Exception as e:
            logger.error(f"Failed to list zarr stores: {e}")
            return []
    
    def _get_store(self, store_path: str) -> DirectR2Store:
        """Get or create a DirectR2Store for a zarr store."""
        with self._cache_lock:
            if store_path not in self._store_cache:
                self._store_cache[store_path] = DirectR2Store(
                    store_path, 
                    cache_metadata=True
                )
            return self._store_cache[store_path]
    
    def _select_resolution_level(self, store: DirectR2Store) -> Optional[Tuple[str, float]]:
        """
        Select resolution level closest to target magnifications.
        Returns (level_name, estimated_mpp).
        """
        if not store._metadata_cache:
            return None
        
        arrays = store._metadata_cache.get('arrays', {})
        if not arrays:
            return None
        
        # Get multiscales metadata
        multiscales = store._metadata_cache.get('multiscales', [])
        if not multiscales:
            # Fallback: use any available level
            level_names = [k for k in arrays.keys() if not k.startswith('.')]
            if level_names:
                return (level_names[0], 1.0)
            return None
        
        # Extract scale information
        datasets = multiscales[0].get('datasets', [])
        
        # Choose random target magnification
        target_mpp = random.choice(self.target_mpp)
        
        # Find closest level
        best_level = None
        best_diff = float('inf')
        
        for dataset in datasets:
            path = dataset['path']
            transforms = dataset.get('coordinateTransformations', [])
            
            # Extract scale factor
            scale = 1.0
            for transform in transforms:
                if transform.get('type') == 'scale':
                    # Assume scale is uniform in X/Y
                    scale = transform['scale'][0] if len(transform['scale']) > 0 else 1.0
            
            # Estimate MPP (assuming base level is ~0.25 µm/px)
            estimated_mpp = 0.25 * scale
            
            diff = abs(estimated_mpp - target_mpp)
            if diff < best_diff:
                best_diff = diff
                best_level = (path, estimated_mpp)
        
        return best_level
    
    def _check_foreground(self, tile: np.ndarray) -> bool:
        """
        Check if tile has sufficient foreground (non-background) content.
        Background is typically very bright (white).
        """
        # Convert to grayscale
        gray = np.mean(tile, axis=2)
        
        # Background is typically > 230 in grayscale
        foreground_mask = gray < 230
        foreground_ratio = np.mean(foreground_mask)
        
        return foreground_ratio >= self.foreground_threshold
    
    def _check_hsv_filter(self, tile: np.ndarray) -> bool:
        """
        Apply HSV filter from Midnight paper.
        Accept tile if ≥60% of pixels are in specified HSV ranges.
        """
        if not self.hsv_filter:
            return True
        
        # Convert to HSV (values in [0,1])
        hsv = rgb2hsv(tile / 255.0)
        
        # Convert to degrees/255 scale
        h = hsv[:, :, 0] * 180  # Hue in [0, 180]
        s = hsv[:, :, 1] * 255  # Saturation in [0, 255]
        v = hsv[:, :, 2] * 255  # Value in [0, 255]
        
        # Check if pixels are in specified ranges
        mask = (
            (h >= self.hsv_ranges['hue'][0]) & (h <= self.hsv_ranges['hue'][1]) &
            (s >= self.hsv_ranges['saturation'][0]) & (s <= self.hsv_ranges['saturation'][1]) &
            (v >= self.hsv_ranges['value'][0]) & (v <= self.hsv_ranges['value'][1])
        )
        
        # Calculate percentage of pixels in range
        pixels_in_range = np.mean(mask)
        
        return pixels_in_range >= 0.6  # 60% threshold
    
    def _extract_random_tile(self, store_path: str) -> Optional[np.ndarray]:
        """
        Extract a random tile following Midnight paper specifications.
        """
        try:
            store = self._get_store(store_path)
            
            # Select resolution level
            level_info = self._select_resolution_level(store)
            if not level_info:
                return None
            
            level_name, estimated_mpp = level_info
            
            # Get array
            arr = store.group[level_name]
            shape = arr.shape
            
            # Assuming shape is (height, width, channels) or (channels, height, width)
            if len(shape) == 3:
                if shape[0] <= 4:  # Channels first
                    h, w = shape[1], shape[2]
                    channel_axis = 0
                else:  # Channels last
                    h, w = shape[0], shape[1]
                    channel_axis = 2
            else:
                return None
            
            ts = self.tile_size
            
            if h < ts or w < ts:
                return None
            
            # Try multiple times to find a valid tile
            max_attempts = 20  # More attempts for stricter filtering
            
            for attempt in range(max_attempts):
                # Random position (uniform sampling)
                y = random.randint(0, h - ts)
                x = random.randint(0, w - ts)
                
                # Extract tile
                if channel_axis == 0:
                    tile_data = arr[:3, y:y+ts, x:x+ts]
                    tile = np.transpose(tile_data, (1, 2, 0))
                else:
                    tile = arr[y:y+ts, x:x+ts, :3]
                
                tile = np.asarray(tile, dtype=np.uint8)
                
                self.stats['tiles_extracted'] += 1
                
                # Check foreground threshold
                if not self._check_foreground(tile):
                    self.stats['tiles_rejected_foreground'] += 1
                    continue
                
                # Check HSV filter
                if not self._check_hsv_filter(tile):
                    self.stats['tiles_rejected_hsv'] += 1
                    continue
                
                # Tile accepted
                self.stats['tiles_accepted'] += 1
                
                # Log statistics periodically
                if self.stats['tiles_extracted'] % 100 == 0:
                    self._log_stats()
                
                return tile
            
            # No valid tile found after max attempts
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract tile from {store_path}: {e}")
            return None
    
    def _log_stats(self):
        """Log extraction statistics."""
        total = self.stats['tiles_extracted']
        if total > 0:
            accept_rate = self.stats['tiles_accepted'] / total
            fg_reject_rate = self.stats['tiles_rejected_foreground'] / total
            hsv_reject_rate = self.stats['tiles_rejected_hsv'] / total
            
            logger.info(f"Tile extraction stats: "
                       f"Accepted: {accept_rate:.2%}, "
                       f"Rejected (foreground): {fg_reject_rate:.2%}, "
                       f"Rejected (HSV): {hsv_reject_rate:.2%}")
    
    def __len__(self) -> int:
        # Return a large number for online patching
        # Each epoch will sample this many tiles
        return len(self.zarr_stores) * 1000  # Adjust multiplier as needed
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a tile with online random patching as in Midnight paper.
        Returns (augmented_dict, None) for DINOv2 training.
        """
        # Select random store (WSI)
        store_idx = random.randint(0, len(self.zarr_stores) - 1)
        store_path = self.zarr_stores[store_idx]
        
        # Extract random tile with filtering
        tile = self._extract_random_tile(store_path)
        
        if tile is None:
            # Fallback: create dummy tile
            tile = np.ones((self.tile_size, self.tile_size, 3), dtype=np.uint8) * 255
        
        # Convert to PIL Image
        img = Image.fromarray(tile, mode='RGB')
        
        # Apply DINOv2 augmentations
        if self.transforms is not None:
            return self.transforms(img), None
        
        raise RuntimeError("SlideDatasetZarr requires DataAugmentationDINO transform")