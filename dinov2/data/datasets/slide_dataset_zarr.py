# Copyright (c) 2025 SophontAI
from pathlib import Path
import os
import random
import numpy as np
from PIL import Image
import zarr
import fsspec
import s3fs
import logging
import hashlib
import threading
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
import time
from concurrent.futures import ThreadPoolExecutor
import pickle

from .extended import ExtendedVisionDataset

logger = logging.getLogger(__name__)

# Suppress the repetitive aiobotocore credentials messages
logging.getLogger('aiobotocore.credentials').setLevel(logging.WARNING)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
logging.getLogger('s3fs').setLevel(logging.WARNING)


class ChunkCacheStore:
    """
    A Zarr store wrapper that caches individual chunks locally.
    This leverages Zarr's chunk-based design for efficient partial access.
    """
    
    def __init__(self, remote_store, cache_dir: Path, max_cache_gb: float = 1000.0):
        self.remote_store = remote_store
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = max_cache_gb * (1024**3)
        
        # LRU cache for chunks
        self._chunk_access = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_size = 0
        
        # Calculate current cache size
        self._init_cache_size()
        
    def _init_cache_size(self):
        """Calculate initial cache size from existing files"""
        self._cache_size = sum(
            f.stat().st_size 
            for f in self.cache_dir.rglob("*") 
            if f.is_file()
        )
    
    def _get_chunk_cache_path(self, key: str) -> Path:
        """Get local cache path for a chunk key"""
        # Zarr chunk keys look like: "level0/0.0.0" or "level0/.zarray"
        safe_key = key.replace("/", "_")
        return self.cache_dir / safe_key
    
    def __getitem__(self, key):
        """Get a chunk, using cache if available"""
        cache_path = self._get_chunk_cache_path(key)
        
        with self._cache_lock:
            # Update LRU
            if key in self._chunk_access:
                del self._chunk_access[key]
            self._chunk_access[key] = time.time()
        
        # Check cache first
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return f.read()
        
        # Fetch from remote
        try:
            data = self.remote_store[key]
        except KeyError:
            raise KeyError(key)
        
        # Cache the chunk
        self._cache_chunk(key, data)
        return data
    
    def _cache_chunk(self, key: str, data: bytes):
        """Cache a chunk with LRU eviction if needed"""
        chunk_size = len(data)
        
        with self._cache_lock:
            # Check if we need to evict
            while self._cache_size + chunk_size > self.max_cache_bytes and self._chunk_access:
                # Evict oldest
                oldest_key, _ = self._chunk_access.popitem(last=False)
                oldest_path = self._get_chunk_cache_path(oldest_key)
                if oldest_path.exists():
                    evicted_size = oldest_path.stat().st_size
                    oldest_path.unlink()
                    self._cache_size -= evicted_size
            
            # Write to cache
            cache_path = self._get_chunk_cache_path(key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(data)
            self._cache_size += chunk_size
    
    def __contains__(self, key):
        """Check if key exists in remote store"""
        return key in self.remote_store
    
    def keys(self):
        """Return keys from remote store"""
        return self.remote_store.keys()


class SmartZarrCache:
    """
    Intelligent caching system that:
    1. Caches individual Zarr chunks (not entire stores)
    2. Prefetches chunks likely to be needed
    3. Uses persistent cache across runs
    """
    
    def __init__(self, cache_dir: Path, max_cache_gb: float = 1000.0):
        self.cache_base = Path(cache_dir)
        self.cache_base.mkdir(parents=True, exist_ok=True)
        self.max_cache_gb = max_cache_gb
        
        # Per-store cache directories
        self._store_caches: Dict[str, ChunkCacheStore] = {}
        self._cache_lock = threading.Lock()
        
        # Prefetch executor
        self.prefetch_executor = ThreadPoolExecutor(max_workers=8)
        
        # S3 filesystem with increased timeout and retry settings
        endpoint = os.environ.get("R2_ENDPOINT_URL")
        client_kwargs = {
            "endpoint_url": endpoint,
            "connect_timeout": 30,  # Increased connection timeout
            "read_timeout": 60,     # Increased read timeout
            "retries": {
                "max_attempts": 5,
                "mode": "adaptive"
            }
        } if endpoint else {
            "connect_timeout": 30,
            "read_timeout": 60,
            "retries": {
                "max_attempts": 5,
                "mode": "adaptive"
            }
        }
        
        self.fs = s3fs.S3FileSystem(
            anon=False,
            client_kwargs=client_kwargs,
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            use_ssl=True,
            default_block_size=5*1024*1024,  # 5MB blocks
            config_kwargs={'max_pool_connections': 50}
        )
        
        logger.info(f"Smart cache initialized with {max_cache_gb:.1f}GB limit")
    
    def get_zarr_array(self, s3_path: str, array_name: str, 
                       prefetch_region: Optional[Tuple[slice, ...]] = None) -> zarr.Array:
        """
        Get a Zarr array with chunk-level caching.
        
        Args:
            s3_path: S3 path to zarr store
            array_name: Name of array (e.g., "level0")
            prefetch_region: Optional region to prefetch in background
        """
        
        # Get or create cache store for this zarr store
        cache_key = f"{s3_path}/{array_name}"
        
        with self._cache_lock:
            if cache_key not in self._store_caches:
                # Create cache directory for this store/array
                store_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
                cache_dir = self.cache_base / store_hash
                
                # Open remote store
                remote_mapper = fsspec.get_mapper(
                    f"{s3_path}/{array_name}",
                    anon=False,
                    client_kwargs={"endpoint_url": os.environ.get("R2_ENDPOINT_URL")} if os.environ.get("R2_ENDPOINT_URL") else {}
                )
                
                # Wrap with cache
                cache_store = ChunkCacheStore(
                    remote_mapper, 
                    cache_dir,
                    max_cache_gb=self.max_cache_gb / 10  # Divide space among stores
                )
                self._store_caches[cache_key] = cache_store
        
        # Open array with cached store
        arr = zarr.open_array(self._store_caches[cache_key], mode='r')
        
        # Prefetch chunks if requested
        if prefetch_region is not None:
            self._prefetch_chunks(arr, prefetch_region, cache_key)
        
        return arr
    
    def _prefetch_chunks(self, arr: zarr.Array, region: Tuple[slice, ...], cache_key: str):
        """Prefetch chunks that will be needed for a region"""
        
        def prefetch_task():
            try:
                # Determine which chunks are needed for this region
                chunk_coords = []
                for dim_slice, dim_chunks, dim_size in zip(region, arr.chunks, arr.shape):
                    # Convert slice to chunk indices
                    start = dim_slice.start if dim_slice.start is not None else 0
                    stop = dim_slice.stop if dim_slice.stop is not None else dim_size
                    
                    start_chunk = start // dim_chunks
                    stop_chunk = (stop - 1) // dim_chunks
                    chunk_coords.append(range(start_chunk, stop_chunk + 1))
                
                # Prefetch each chunk
                import itertools
                store = self._store_caches[cache_key]
                for chunk_idx in itertools.product(*chunk_coords):
                    chunk_key = ".".join(map(str, chunk_idx))
                    if chunk_key not in store._chunk_access:  # Not in cache
                        try:
                            # This will trigger caching
                            _ = store[chunk_key]
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Prefetch error (non-critical): {e}")
        
        # Submit prefetch task
        self.prefetch_executor.submit(prefetch_task)


class SlideDatasetZarr(ExtendedVisionDataset):
    """
    Zarr dataset optimized for chunk-based caching from R2.
    Returns (sample_dict, target) where sample_dict is produced by self.transform
    and has 'global_crops'/'local_crops' keys as expected by DINOv2 collate.
    """
    def __init__(self, root, patch_size: int = 224, 
                 cache_dir: str = "/tmp/zarr_chunk_cache",
                 cache_gb: float = 1000.0, 
                 prefetch_radius: int = 512,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.root = str(root)
        self.patch_size = int(patch_size)
        self.prefetch_radius = prefetch_radius  # How much around the patch to prefetch
        
        self.is_s3 = self.root.startswith("s3://")
        
        if self.is_s3:
            # Initialize smart chunk cache
            self.cache = SmartZarrCache(Path(cache_dir), max_cache_gb=cache_gb)
            
            # List zarr stores with retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    fs = self.cache.fs
                    path_without_scheme = self.root.replace("s3://", "")
                    parts = path_without_scheme.split("/", 1)
                    bucket = parts[0]
                    prefix = parts[1] if len(parts) > 1 else ""
                    
                    logger.info(f"Listing S3 bucket: {bucket}, prefix: {prefix} (attempt {attempt + 1}/{max_retries})")
                    
                    search_path = f"{bucket}/{prefix}" if prefix else bucket
                    
                    all_items = []
                    # Use fs.find for recursive search of .zarr directories
                    items = fs.find(search_path, maxdepth=None, withdirs=True)
                    
                    for item in items:
                        if item.endswith(".zarr") or ".zarr/" in item:
                            # Get the zarr store root path
                            if ".zarr/" in item:
                                zarr_root = item.split(".zarr/")[0] + ".zarr"
                            else:
                                zarr_root = item
                            
                            if not zarr_root.startswith("s3://"):
                                zarr_root = f"s3://{zarr_root}"
                            
                            if zarr_root not in all_items:
                                all_items.append(zarr_root)
                    
                    self.groups = sorted(all_items)
                    logger.info(f"Successfully found {len(self.groups)} zarr stores")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.error(f"Failed to list S3 contents (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error("All retry attempts failed")
                        self.groups = []
            
            # Cache to store array metadata (shapes, levels)
            self._metadata_cache = {}
            
        else:
            # Local filesystem - no caching needed
            self.cache = None
            p = Path(self.root)
            self.groups = sorted([str(x) for x in p.rglob("*.zarr") if x.is_dir()])
        
        logger.info(f"[Zarr] stores found: {len(self.groups)}")
        
        if len(self.groups) == 0:
            logger.warning(f"No zarr stores found in {self.root}")
            # Create a dummy entry to prevent crashes
            self._use_dummy = True
        else:
            self._use_dummy = False
    
    def _get_store_metadata(self, zpath: str) -> Dict:
        """Get metadata about arrays in a zarr store"""
        if zpath in self._metadata_cache:
            return self._metadata_cache[zpath]
        
        # Fetch just the metadata files (.zarray) to discover structure
        metadata = {}
        
        if self.is_s3:
            fs = self.cache.fs
            fs_path = zpath.replace("s3://", "")
            
            # List arrays in the store
            try:
                items = fs.ls(fs_path, detail=False)
                for item in items:
                    name = item.split("/")[-1]
                    if not name.startswith("."):
                        # Check if it's an array by looking for .zarray
                        zarray_path = f"{item}/.zarray"
                        if fs.exists(zarray_path):
                            # Read array metadata
                            import json
                            zarray_data = json.loads(fs.cat(zarray_path))
                            metadata[name] = {
                                'shape': tuple(zarray_data['shape']),
                                'chunks': tuple(zarray_data['chunks']),
                                'dtype': zarray_data['dtype']
                            }
            except Exception as e:
                logger.warning(f"Failed to get metadata for {zpath}: {e}")
        
        self._metadata_cache[zpath] = metadata
        return metadata
    
    def __len__(self) -> int:
        # Always return a valid length, even if no data
        if self._use_dummy or len(self.groups) == 0:
            return 1
        return len(self.groups)
    
    def _get_dummy_sample(self):
        """Return a dummy sample when no real data is available"""
        img = Image.new('RGB', (self.patch_size, self.patch_size), color='white')
        if self.transform is None:
            raise RuntimeError("SlideDatasetZarr requires a transform that returns "
                               "a DINOv2-style dict with 'global_crops'/'local_crops'.")
        return self.transform(img), None
    
    def __getitem__(self, index):
        """
        Returns (sample_dict, None).
        sample_dict comes from self.transform(img) and must contain
        'global_crops' and 'local_crops' keys for DINOv2.
        """
        # Return dummy sample if no data
        if self._use_dummy or len(self.groups) == 0:
            return self._get_dummy_sample()
        
        # Safe modulo to prevent division by zero
        index = index % max(1, len(self.groups))
        
        try:
            zpath = self.groups[index]
            
            # Get store metadata
            metadata = self._get_store_metadata(zpath)
            if not metadata:
                logger.warning(f"No metadata found for {zpath}, using dummy sample")
                return self._get_dummy_sample()
            
            # Find level arrays
            level_arrays = {k: v for k, v in metadata.items() if k.startswith("level")}
            if not level_arrays:
                logger.warning(f"No level arrays found in {zpath}, using dummy sample")
                return self._get_dummy_sample()
            
            # Choose a level (prefer lower resolutions for speed)
            level_names = sorted(level_arrays.keys())
            if len(level_names) > 1:
                # Prefer level 1 or 2 (downsampled)
                weights = [1.0 if "0" in name else 2.0 for name in level_names]
                level_name = random.choices(level_names, weights=weights)[0]
            else:
                level_name = level_names[0]
            
            level_info = level_arrays[level_name]
            h, w = level_info['shape'][:2]
            ps = self.patch_size
            
            # Choose random position
            if h < ps or w < ps:
                y0 = max(0, (h - ps) // 2)
                x0 = max(0, (w - ps) // 2)
            else:
                y0 = random.randint(0, h - ps)
                x0 = random.randint(0, w - ps)
            
            # Define the region we need
            y1 = min(y0 + ps, h)
            x1 = min(x0 + ps, w)
            
            if self.is_s3 and self.cache:
                # Define prefetch region (slightly larger than needed)
                pf_y0 = max(0, y0 - self.prefetch_radius)
                pf_y1 = min(h, y1 + self.prefetch_radius)
                pf_x0 = max(0, x0 - self.prefetch_radius)
                pf_x1 = min(w, x1 + self.prefetch_radius)
                
                prefetch_region = (slice(pf_y0, pf_y1), slice(pf_x0, pf_x1), slice(None))
                
                # Get array with caching
                arr = self.cache.get_zarr_array(zpath, level_name, prefetch_region)
            else:
                # Local access
                grp = zarr.open_group(zpath, mode='r')
                arr = grp[level_name]
            
            # Extract patch (this will use cached chunks when available)
            patch_data = arr[y0:y1, x0:x1, :]
            patch = np.asarray(patch_data, dtype=np.uint8)
            
            # Pad if necessary
            if patch.shape[0] < ps or patch.shape[1] < ps:
                padded = np.zeros((ps, ps, 3), dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded
            
            img = Image.fromarray(patch, mode="RGB")
            if self.transform is None:
                raise RuntimeError("SlideDatasetZarr requires a transform that returns "
                                   "a DINOv2-style dict with 'global_crops'/'local_crops'.")
            return self.transform(img), None
            
        except Exception as e:
            logger.error(f"Error accessing item {index}: {e}")
            return self._get_dummy_sample()