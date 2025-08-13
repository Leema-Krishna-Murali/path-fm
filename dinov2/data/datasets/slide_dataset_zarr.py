"""
Dataset following Midnight CPath paper specifications:
- 256x256 tiles
- Online patching from random positions
- 40% foreground threshold
- HSV filtering for tissue detection
"""

import os
import random
import logging
import hashlib
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import zarr
import fsspec
import s3fs
import cv2

from .extended import ExtendedVisionDataset

logger = logging.getLogger(__name__)

class SlideDatasetZarr(ExtendedVisionDataset):
    """
    OME-Zarr dataset with Midnight CPath paper specifications.
    
    Paper specifications:
    - Tile size: 256x256
    - Resolutions: 2, 1, 0.5, and 0.25 µm/px
    - Foreground threshold: 40%
    - HSV filtering for tissue quality
    """
    
    def __init__(
        self,
        root: str,
        tile_size: int = 256,  # Midnight paper uses 256x256
        foreground_threshold: float = 0.4,  # 40% as per paper
        hsv_tissue_ratio: float = 0.6,  # 60% as per paper
        cache_gb: float = 1000.0,
        preferred_resolutions: List[float] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.root = str(root)
        self.tile_size = tile_size
        self.foreground_threshold = foreground_threshold
        self.hsv_tissue_ratio = hsv_tissue_ratio
        
        # Preferred resolutions (µm/px) as per Midnight paper
        if preferred_resolutions is None:
            self.preferred_resolutions = [0.25, 0.5, 1.0, 2.0]
        else:
            self.preferred_resolutions = preferred_resolutions
        
        # HSV ranges from Midnight paper
        # Hue: [90, 180], Saturation: [8, 255], Value: [103, 255]
        self.hsv_lower = np.array([90, 8, 103])
        self.hsv_upper = np.array([180, 255, 255])
        
        # Setup caching
        self.chunk_cache = ChunkCache(max_size_gb=cache_gb)
        
        # Setup S3 filesystem
        self.fs = self._setup_s3fs()
        
        # Find all zarr stores
        self.zarr_stores = self._find_zarr_stores()
        
        if len(self.zarr_stores) == 0:
            raise ValueError(f"No zarr stores found in {self.root}")
        
        logger.info(f"Found {len(self.zarr_stores)} zarr stores (Midnight mode)")
        
        # Metadata cache
        self.metadata_cache = {}
        
        # Statistics
        self.accepted_tiles = 0
        self.rejected_tiles = 0
    
    def _setup_s3fs(self) -> s3fs.S3FileSystem:
        """Setup S3 filesystem with R2 configuration."""
        endpoint = os.environ.get("R2_ENDPOINT_URL")
        
        client_kwargs = {
            "endpoint_url": endpoint,
            "connect_timeout": 30,
            "read_timeout": 60,
            "retries": {
                "max_attempts": 5,
                "mode": "adaptive"
            }
        } if endpoint else {}
        
        return s3fs.S3FileSystem(
            anon=False,
            client_kwargs=client_kwargs,
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            use_ssl=True,
            default_block_size=5*1024*1024,
            config_kwargs={'max_pool_connections': 50}
        )
    
    def _find_zarr_stores(self) -> List[str]:
        """Find all zarr stores in the S3 path."""
        stores = []
        
        if not self.root.startswith("s3://"):
            raise ValueError(f"Root must be an S3 path, got: {self.root}")
        
        path_without_scheme = self.root.replace("s3://", "")
        
        try:
            items = self.fs.find(path_without_scheme, maxdepth=None, withdirs=True)
            
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
    
    def _check_foreground_ratio(self, tile: np.ndarray) -> bool:
        """
        Check if tile has sufficient foreground (non-background) content.
        Background is typically very bright (close to white).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        
        # Background is typically > 230 in grayscale
        foreground_mask = gray < 230
        foreground_ratio = np.mean(foreground_mask)
        
        return foreground_ratio >= self.foreground_threshold
    
    def _check_hsv_filter(self, tile: np.ndarray) -> bool:
        """
        Apply HSV filter as per Midnight paper.
        A tile is accepted if ≥60% of pixels have:
        - Hue in [90, 180]
        - Saturation in [8, 255]
        - Value in [103, 255]
        """
        # Convert to HSV
        # OpenCV uses H: 0-179, S: 0-255, V: 0-255
        # Paper uses H: 0-360, so we need to scale
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        
        # Scale hue values for OpenCV (paper range [90, 180] in 360 scale)
        # In OpenCV scale (0-179): [45, 90]
        hsv_lower_cv = np.array([45, 8, 103])
        hsv_upper_cv = np.array([90, 255, 255])
        
        # Create mask for pixels within the range
        mask = cv2.inRange(hsv, hsv_lower_cv, hsv_upper_cv)
        
        # Calculate ratio of pixels within range
        tissue_ratio = np.sum(mask > 0) / (tile.shape[0] * tile.shape[1])
        
        return tissue_ratio >= self.hsv_tissue_ratio
    
    def _is_valid_tile(self, tile: np.ndarray) -> bool:
        """
        Check if tile passes all quality filters:
        1. Sufficient foreground (40%)
        2. HSV tissue filter (60% pixels in range)
        """
        # Check foreground ratio
        if not self._check_foreground_ratio(tile):
            return False
        
        # Check HSV filter
        if not self._check_hsv_filter(tile):
            return False
        
        return True
    
    def _get_random_tile(self, store_path: str) -> Optional[np.ndarray]:
        """
        Extract a random tile following Midnight paper's online patching.
        Samples uniformly at random from arbitrary positions.
        """
        try:
            # Get metadata
            metadata = self._get_store_metadata(store_path)
            if not metadata or not metadata.get('levels'):
                return None
            
            # Choose resolution level based on paper's preferences
            available_levels = metadata['levels'].keys()
            
            # Try to find levels matching paper's resolutions
            selected_level = None
            for pref_res in self.preferred_resolutions:
                level_name = f"res_{pref_res:.2f}um"
                if level_name in available_levels:
                    selected_level = level_name
                    break
            
            # Fallback to any available level
            if selected_level is None:
                selected_level = list(available_levels)[0]
            
            level_info = metadata['levels'][selected_level]
            shape = level_info['shape']
            h, w = shape[0], shape[1]
            
            if h < self.tile_size or w < self.tile_size:
                return None
            
            # Open array with caching
            store_id = hashlib.md5(store_path.encode()).hexdigest()[:8]
            mapper = fsspec.get_mapper(
                f"{store_path}/{selected_level}",
                anon=False,
                client_kwargs={"endpoint_url": os.environ.get("R2_ENDPOINT_URL")}
            )
            cached_store = CachedZarrStore(mapper, self.chunk_cache, store_id)
            arr = zarr.open_array(cached_store, mode='r')
            
            # Online patching: try multiple random positions
            max_attempts = 20  # More attempts to find good tiles
            
            for attempt in range(max_attempts):
                # Sample uniformly at random position
                y = random.randint(0, h - self.tile_size)
                x = random.randint(0, w - self.tile_size)
                
                # Extract tile
                tile = np.asarray(
                    arr[y:y+self.tile_size, x:x+self.tile_size, :],
                    dtype=np.uint8
                )
                
                # Check if tile passes quality filters
                if self._is_valid_tile(tile):
                    self.accepted_tiles += 1
                    return tile
                else:
                    self.rejected_tiles += 1
            
            # If no valid tile found after max_attempts, return None
            return None
            
        except Exception as e:
            logger.error(f"Failed to get tile from {store_path}: {e}")
            return None
    
    def _get_store_metadata(self, store_path: str) -> Dict[str, Any]:
        """Get metadata for a zarr store."""
        if store_path in self.metadata_cache:
            return self.metadata_cache[store_path]
        
        try:
            mapper = fsspec.get_mapper(
                store_path,
                anon=False,
                client_kwargs={"endpoint_url": os.environ.get("R2_ENDPOINT_URL")}
            )
            
            group = zarr.open_group(mapper, mode='r')
            
            multiscales = group.attrs.get('multiscales', [])
            if not multiscales:
                return {}
            
            metadata = {
                'multiscales': multiscales[0],
                'levels': {}
            }
            
            # Get info for each level
            for dataset in multiscales[0].get('datasets', []):
                level_path = dataset['path']
                if level_path in group:
                    arr = group[level_path]
                    metadata['levels'][level_path] = {
                        'shape': arr.shape,
                        'chunks': arr.chunks,
                        'dtype': str(arr.dtype),
                        'resolution': dataset.get('resolution_um_px', None)
                    }
            
            self.metadata_cache[store_path] = metadata
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to get metadata for {store_path}: {e}")
            return {}
    
    def __len__(self) -> int:
        return len(self.zarr_stores)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get item following Midnight paper's approach.
        Returns (augmented_dict, None) for DINOv2 training.
        """
        # Try multiple stores if needed to get a valid tile
        max_store_attempts = 5
        
        for store_attempt in range(max_store_attempts):
            # Select a random store (online patching from arbitrary WSIs)
            store_idx = random.randint(0, len(self.zarr_stores) - 1)
            store_path = self.zarr_stores[store_idx]
            
            # Get random tile with quality filtering
            tile = self._get_random_tile(store_path)
            
            if tile is not None:
                # Convert to PIL Image
                img = Image.fromarray(tile, mode='RGB')
                
                # Apply DINOv2 augmentations
                if self.transforms is not None:
                    return self.transforms(img), None
                else:
                    raise RuntimeError("SlideDatasetZarr requires DataAugmentationDINO transform")
        
        # If no valid tile found, return a dummy tile
        # This should be rare with proper data
        logger.warning(f"Could not find valid tile after {max_store_attempts} attempts")
        dummy_tile = np.ones((self.tile_size, self.tile_size, 3), dtype=np.uint8) * 255
        img = Image.fromarray(dummy_tile, mode='RGB')
        
        if self.transforms is not None:
            return self.transforms(img), None
        else:
            raise RuntimeError("SlideDatasetZarr requires DataAugmentationDINO transform")
    
    def log_stats(self):
        """Log tile acceptance statistics."""
        total = self.accepted_tiles + self.rejected_tiles
        if total > 0:
            accept_rate = self.accepted_tiles / total
            logger.info(f"Tile stats - Accepted: {self.accepted_tiles}, "
                       f"Rejected: {self.rejected_tiles}, "
                       f"Accept rate: {accept_rate:.2%}")


class ChunkCache:
    """LRU cache for Zarr chunks."""
    
    def __init__(self, max_size_gb: float = 1000.0):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cache = OrderedDict()
        self.size_map = {}
        self.current_size = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[bytes]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: bytes):
        size = len(value)
        
        with self.lock:
            if key in self.cache:
                self.current_size -= self.size_map[key]
                del self.cache[key]
                del self.size_map[key]
            
            while self.current_size + size > self.max_size_bytes and self.cache:
                evict_key, _ = self.cache.popitem(last=False)
                self.current_size -= self.size_map[evict_key]
                del self.size_map[evict_key]
            
            self.cache[key] = value
            self.size_map[key] = size
            self.current_size += size


class CachedZarrStore:
    """Zarr store wrapper with chunk caching."""
    
    def __init__(self, base_store, cache: ChunkCache, store_id: str):
        self.base_store = base_store
        self.cache = cache
        self.store_id = store_id
    
    def __getitem__(self, key):
        cache_key = f"{self.store_id}:{key}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        value = self.base_store[key]
        self.cache.put(cache_key, value)
        return value
    
    def __contains__(self, key):
        return key in self.base_store
    
    def keys(self):
        return self.base_store.keys()