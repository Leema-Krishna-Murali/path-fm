# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, Dict, Optional, List, Set
from pathlib import Path
from PIL import Image
import zarr
from zarr.storage import KVStore
import fsspec
import s3fs
import os
import random
import numpy as np
import logging
import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .extended import ExtendedVisionDataset

logger = logging.getLogger(__name__)

# Suppress noisy credentials messages
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
logging.getLogger('s3fs').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)


class ChunkCacheStore:
    """
    A simple MutableMapping-compatible store that caches chunks locally.
    Implements the minimal interface needed for Zarr compatibility.
    """
    
    def __init__(self, remote_store, cache_dir: Path, max_cache_gb: float):
        self.remote_store = remote_store
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = max_cache_gb * (1024**3)
        self._chunk_access = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_size = 0
        
        # Pre-calculate existing cache size
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                self._cache_size += f.stat().st_size
    
    def _get_chunk_cache_path(self, key: str) -> Path:
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / safe_key
    
    def __getitem__(self, key):
        cache_path = self._get_chunk_cache_path(key)
        
        # Update LRU tracking
        with self._cache_lock:
            if key in self._chunk_access:
                self._chunk_access.move_to_end(key)
        
        # Check local cache first
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except Exception:
                pass
        
        # Fetch from remote
        try:
            data = self.remote_store[key]
        except KeyError:
            raise KeyError(key)
        
        # Cache the chunk
        self._cache_chunk(key, data)
        return data
    
    def _cache_chunk(self, key: str, data: bytes):
        chunk_size = len(data)
        
        with self._cache_lock:
            # Evict old chunks if necessary
            while self._cache_size + chunk_size > self.max_cache_bytes and self._chunk_access:
                oldest_key, _ = self._chunk_access.popitem(last=False)
                oldest_path = self._get_chunk_cache_path(oldest_key)
                if oldest_path.exists():
                    try:
                        evicted_size = oldest_path.stat().st_size
                        oldest_path.unlink()
                        self._cache_size -= evicted_size
                    except Exception:
                        pass
            
            # Write new chunk
            cache_path = self._get_chunk_cache_path(key)
            try:
                with open(cache_path, 'wb') as f:
                    f.write(data)
                self._cache_size += chunk_size
                self._chunk_access[key] = None
            except Exception as e:
                logger.warning(f"Failed to cache chunk {key}: {e}")
    
    def __setitem__(self, key, value):
        # Not used for read-only access, but required for MutableMapping
        raise NotImplementedError("Read-only store")
    
    def __delitem__(self, key):
        # Not used for read-only access, but required for MutableMapping
        raise NotImplementedError("Read-only store")
    
    def __contains__(self, key):
        return key in self.remote_store
    
    def __iter__(self):
        return iter(self.remote_store)
    
    def __len__(self):
        return len(self.remote_store)
    
    def keys(self):
        return self.remote_store.keys()
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class SlideDatasetZarr(ExtendedVisionDataset):
    """
    Optimized dataset for loading patches from OME-Zarr slides on S3.
    
    Key optimizations:
    - Direct zarr access (faster than Dask)
    - Intelligent chunk-level caching
    - Parallel metadata fetching
    - Optional warmup for predictable access patterns
    """
    
    def __init__(self, 
                 root: str, 
                 manifest_path: str = "manifest.txt",
                 cache_dir: str = "~/.zarr_chunk_cache",
                 cache_gb: float = 100.0,
                 warmup_cache: bool = False,  # Disabled by default for faster startup
                 warmup_steps: int = 100,     # Reduced default warmup
                 patch_size: int = 224,
                 num_workers_metadata: int = 16,
                 num_workers_warmup: int = 32,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.root = root
        self.patch_size = patch_size
        self.is_s3 = self.root.startswith("s3://")
        
        logger.info(f"Initializing SlideDatasetZarr")
        logger.info(f"  Manifest: {manifest_path}")
        logger.info(f"  Cache: {cache_gb}GB at {cache_dir}")
        logger.info(f"  Warmup: {warmup_cache} ({warmup_steps} steps if enabled)")
        
        # Load manifest
        try:
            with open(manifest_path, 'r') as f:
                self.zarr_files = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Manifest file not found at: {manifest_path}")
            raise
        
        if not self.zarr_files:
            raise ValueError(f"No files listed in manifest: {manifest_path}")
        
        logger.info(f"Loaded {len(self.zarr_files)} Zarr stores from manifest")
        
        # Setup caching
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_gb = cache_gb
        self._store_caches = {}
        self._store_cache_lock = threading.Lock()
        
        # Setup S3 connection
        if self.is_s3:
            endpoint_url = os.environ.get("R2_ENDPOINT_URL")
            
            # S3fs configuration
            self.s3fs_kwargs = {
                "key": os.environ.get("AWS_ACCESS_KEY_ID"),
                "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
                "client_kwargs": {"endpoint_url": endpoint_url} if endpoint_url else {},
                "config_kwargs": {
                    'max_pool_connections': 50,
                    'retries': {'max_attempts': 3, 'mode': 'adaptive'}
                }
            }
            
            self.fs = s3fs.S3FileSystem(**self.s3fs_kwargs)
            logger.info(f"S3 filesystem initialized (endpoint: {endpoint_url or 'default'})")
        
        # Metadata cache
        self._slide_meta_cache = {}
        self._slide_meta_lock = threading.Lock()
        
        # Prefetch metadata for faster first access
        self._prefetch_metadata(num_workers=num_workers_metadata)
        
        # Optional warmup
        if self.is_s3 and warmup_cache:
            self._warmup_cache(warmup_steps, num_workers=num_workers_warmup)
    
    def _prefetch_metadata(self, num_workers: int = 16):
        """Prefetch metadata for all slides in parallel."""
        logger.info(f"Prefetching metadata for {len(self.zarr_files)} slides...")
        
        def fetch_meta(path):
            try:
                if self.is_s3:
                    mapper = fsspec.get_mapper(path, **self.s3fs_kwargs)
                else:
                    mapper = path
                
                # Try consolidated metadata first (faster)
                try:
                    group = zarr.open_consolidated(mapper, mode='r')
                except (KeyError, ValueError):
                    group = zarr.open_group(mapper, mode='r')
                
                # Extract level information
                levels = []
                for key in sorted(group.array_keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                    arr = group[key]
                    if len(arr.shape) == 3:
                        levels.append((key, (arr.shape[0], arr.shape[1], arr.shape[2])))
                
                return path, levels
            except Exception as e:
                logger.debug(f"Failed to load metadata for {path}: {e}")
                return path, []
        
        # Fetch in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(fetch_meta, path) for path in self.zarr_files]
            
            completed = 0
            for future in as_completed(futures):
                path, levels = future.result()
                if levels:
                    level_arrays, level_shapes = zip(*levels)
                    with self._slide_meta_lock:
                        self._slide_meta_cache[path] = (list(level_arrays), list(level_shapes))
                else:
                    with self._slide_meta_lock:
                        self._slide_meta_cache[path] = ([], [])
                
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"  Metadata fetched: {completed}/{len(self.zarr_files)}")
        
        valid_slides = sum(1 for _, (arrays, _) in self._slide_meta_cache.items() if arrays)
        logger.info(f"Metadata prefetch complete: {valid_slides}/{len(self.zarr_files)} valid slides")
    
    def _get_zarr_array(self, s3_path: str, array_name: str) -> zarr.Array:
        """Get a Zarr array with caching."""
        cache_key = f"{s3_path}/{array_name}"
        
        with self._store_cache_lock:
            if cache_key not in self._store_caches:
                if self.is_s3:
                    # Create cache directory for this array
                    store_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
                    cache_dir = self.cache_dir / store_hash
                    
                    # Get remote store
                    full_path = f"{s3_path}/{array_name}"
                    remote_mapper = fsspec.get_mapper(full_path, **self.s3fs_kwargs)
                    
                    # Wrap in caching store
                    cache_store = ChunkCacheStore(remote_mapper, cache_dir, self.cache_gb)
                    
                    # Wrap in KVStore for Zarr compatibility
                    kv_store = KVStore(cache_store)
                    self._store_caches[cache_key] = kv_store
                else:
                    # Local filesystem, no caching needed
                    self._store_caches[cache_key] = f"{s3_path}/{array_name}"
        
        return zarr.open_array(self._store_caches[cache_key], mode='r')
    
    def _warmup_cache(self, num_steps: int, num_workers: int = 32):
        """Optionally warmup cache with expected access patterns."""
        logger.info(f"Warming up cache with {num_steps} sample accesses...")
        
        valid_slides = [path for path, (arrays, _) in self._slide_meta_cache.items() if arrays]
        if not valid_slides:
            logger.warning("No valid slides for warmup")
            return
        
        # Generate sample accesses
        warmup_tasks = []
        rng = random.Random(42)  # Deterministic for reproducibility
        
        for _ in range(num_steps):
            path = rng.choice(valid_slides)
            arrays, shapes = self._slide_meta_cache[path]
            
            if not arrays:
                continue
            
            level_idx = rng.randint(0, len(arrays) - 1)
            level_name = arrays[level_idx]
            height, width = shapes[level_idx]
            
            if height > self.patch_size and width > self.patch_size:
                y = rng.randint(0, height - self.patch_size)
                x = rng.randint(0, width - self.patch_size)
                warmup_tasks.append((path, level_name, y, x))
        
        # Execute warmup in parallel
        def warmup_access(task):
            path, level_name, y, x = task
            try:
                arr = self._get_zarr_array(path, level_name)
                # Just access the data to trigger caching
                _ = arr[y:y+self.patch_size, x:x+self.patch_size, :3]
                return True
            except Exception:
                return False
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(warmup_access, task) for task in warmup_tasks]
            
            success = 0
            for future in as_completed(futures):
                if future.result():
                    success += 1
        
        logger.info(f"Cache warmup complete: {success}/{len(warmup_tasks)} successful accesses")
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a random patch from a random slide."""
        
        # Select a slide
        if hasattr(self, '_last_valid_slides'):
            # Use cached valid slides for efficiency
            path = random.choice(self._last_valid_slides)
        else:
            path = self.zarr_files[index % len(self.zarr_files)]
        
        # Get metadata
        with self._slide_meta_lock:
            level_arrays, level_shapes = self._slide_meta_cache[path]
        
        # Cache valid slides
        if not hasattr(self, '_last_valid_slides'):
            self._last_valid_slides = [p for p, (a, _) in self._slide_meta_cache.items() if a]
        
        # Select a random level
        level_idx = random.randint(0, len(level_arrays) - 1)
        level_name = level_arrays[level_idx]
        channels, height, width = level_shapes[level_idx] # RGB, height, width
        
        if height <= self.patch_size or width <= self.patch_size:
            print(f"\nERROR height={height} width={width} patch_size={self.patch_size}")
            err

        # Random patch location
        y = random.randint(0, height - self.patch_size)
        x = random.randint(0, width - self.patch_size)
        
        # Load the patch
        arr = self._get_zarr_array(path, level_name)
        patch_data = arr[:, y:y+self.patch_size, x:x+self.patch_size]
        
        # Convert to numpy and ensure correct shape
        patch = np.asarray(patch_data, dtype=np.uint8).transpose(1, 2, 0)
        
        # Convert to PIL Image
        img = Image.fromarray(patch, mode="RGB")
        
        # Apply transforms
        if self.transforms is not None:
            return self.transforms(img, None)
        return img, None
    
    def __len__(self) -> int:
        return len(self.zarr_files)