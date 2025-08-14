# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, Dict, Optional, List, Set
from pathlib import Path
from PIL import Image
import zarr
import fsspec
import s3fs
import os
import random
import numpy as np
import cv2
import logging
import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import itertools
import math
from botocore.config import Config

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
        # Zarr chunk keys look like: "0/0.0.0" or "0/.zarray"
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

        self.global_budget_bytes = int(max_cache_gb * (1024**3))
        self.allowed_store_prefixes = set()   # active window (s3://.../.zarr paths)
        self.enforce_window = True           # if True, __getitem__ will restrict to cached window
        
        # Per-store cache directories
        self._store_caches: Dict[str, ChunkCacheStore] = {}
        self._cache_lock = threading.Lock()
        
        # Prefetch executor
        self.prefetch_executor = ThreadPoolExecutor(max_workers=8)
        
        # S3 filesystem with minimal configuration for compatibility
        endpoint = os.environ.get("R2_ENDPOINT_URL")
        client_kwargs = {"endpoint_url": endpoint} if endpoint else {}

        # Many parallel connections and aggressive retries:
        config_kwargs = {
            "retries": {"max_attempts": 10, "mode": "adaptive"},
            "max_pool_connections": 256,   # bump if NIC allows
        }
        self.fs = s3fs.S3FileSystem(
            anon=False,
            client_kwargs=client_kwargs,
            config_kwargs=config_kwargs,
            key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            use_ssl=True,
            default_block_size=5 * 1024 * 1024,
        )
        
        logger.info(f"Smart cache initialized with {max_cache_gb:.1f}GB limit")

    # ---------- global cache helpers ----------
    def _cache_size_bytes(self) -> int:
        total = 0
        for root, _, files in os.walk(self.cache_base):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except FileNotFoundError:
                    pass
        return total

    def clear_all(self):
        try:
            import shutil
            shutil.rmtree(self.cache_base)
        except FileNotFoundError:
            pass
        self.cache_base.mkdir(parents=True, exist_ok=True)

    # ---------- prefetch one array (all chunks) ----------
    def _prefetch_array_all_chunks(self, store_path: str, array_name: str, max_workers: int = 16):
        """
        Ensure *all chunks* of s3://.../.zarr/<array_name> are brought into the local chunk cache.
        """
        # Get a cached array (wraps ChunkCacheStore)
        arr = self.get_zarr_array(store_path, array_name, prefetch_region=None)
        cache_key = f"{store_path}/{array_name}"
        store = self._store_caches[cache_key]  # ChunkCacheStore

        # Cache metadata first
        try:
            _ = store[".zarray"]
        except KeyError:
            pass
        try:
            _ = store[".zattrs"]
        except KeyError:
            pass

        # Build the chunk coordinate grid
        shape = arr.shape
        chunks = arr.chunks
        ndim = len(shape)
        grid = [range((shape[d] + chunks[d] - 1) // chunks[d]) for d in range(ndim)]
        keys = [".".join(map(str, idx)) for idx in itertools.product(*grid)]

        # Pull chunks concurrently (bytes go into ChunkCacheStore)
        def pull(k: str):
            try:
                _ = store[k]
            except KeyError:
                # Some tails may be missing in sparse stores; ignore
                pass
            except Exception:
                pass

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for _ in ex.map(pull, keys):
                pass

    # ---------- warmup many stores until budget is hit ----------
    def warmup_cache(self, zarr_store_paths, arrays=("0","1","2","3","4","5"),
                    headroom_ratio: float = 0.98, max_workers: int = 16):
        """
        Prefetch stores (all chunks for selected arrays) until ~budget is reached.
        Returns the list of stores fully covered in this window.
        """
        target = int(self.global_budget_bytes * headroom_ratio)
        cached_window = []

        for spath in zarr_store_paths:
            # Ensure s3:// scheme + .zarr
            p = spath if spath.startswith("s3://") else f"s3://{spath}"
            if not p.endswith(".zarr"):
                # accept store folder path that endswith .zarr after we normalize
                pass

            # Discover which requested arrays actually exist
            try:
                g = self.get_zarr_group(p)   # metadata only (no chunk fetch)
            except Exception:
                continue

            present = []
            for a in arrays:
                try:
                    if a in g and isinstance(g[a], zarr.Array):
                        present.append(a)
                except Exception:
                    continue

            # Prefetch all chunks for present arrays
            for a in present:
                self._prefetch_array_all_chunks(p, a, max_workers=max_workers)
                if self._cache_size_bytes() >= target:
                    cached_window.append(p.rstrip("/"))
                    self.allowed_store_prefixes = set(cached_window)
                    return cached_window

            cached_window.append(p.rstrip("/"))

        self.allowed_store_prefixes = set(cached_window)
        return cached_window

    def warmup_by_plan(self, plan: Dict[str, Dict[str, set]],
                       headroom_ratio: float = 0.98,
                       max_workers: int = 128):
        """
        plan: { "s3://.../store.zarr": { "0": {chunk_keys...}, "1": {...}, ... } }
        Fetch only these chunk keys into the local chunk cache.
        """
        target = int(self.global_budget_bytes * headroom_ratio)

        from concurrent.futures import ThreadPoolExecutor

        for store_path, arrays in plan.items():
            for array_name, keys in arrays.items():
                if not keys:
                    continue
                # Ensure the per-array cache store exists
                _ = self.get_zarr_array(store_path, array_name)
                store = self._store_caches[f"{store_path}/{array_name}"]

                def pull(k: str):
                    try:
                        _ = store[k]
                    except Exception:
                        pass

                # High concurrency for many small GETs
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for _ in ex.map(pull, list(keys)):
                        pass

                if self._cache_size_bytes() >= target:
                    # Build allowed set so dataset can keep sampling inside the warm cache
                    self.allowed_store_prefixes = set(plan.keys())
                    return

        self.allowed_store_prefixes = set(plan.keys())

    # ---------- rotate to next window ----------
    def rotate_cache_window(self, next_store_paths, arrays=("0","1","2","3","4","5"),
                            headroom_ratio: float = 0.98, max_workers: int = 16):
        """
        Clears the cache dir and warms it with the next set of stores.
        """
        self.clear_all()
        return self.warmup_cache(next_store_paths, arrays=arrays,
                                headroom_ratio=headroom_ratio, max_workers=max_workers)
    
    def get_zarr_array(self, s3_path: str, array_name: str, 
                       prefetch_region: Optional[Tuple[slice, ...]] = None) -> zarr.Array:
        """
        Get a Zarr array with chunk-level caching.
        
        Args:
            s3_path: S3 path to zarr store
            array_name: Name of array (e.g., "0", "1", "2")
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
                    max_cache_gb=self.max_cache_gb
                )
                self._store_caches[cache_key] = cache_store
        
        # Open array with cached store (wrap in KVStore for Zarr v3.1)
        from zarr.storage import KVStore
        store = self._store_caches[cache_key]
        wrapped_store = KVStore(store)
        arr = zarr.open_array(wrapped_store, mode='r')
        
        # Prefetch chunks if requested
        if prefetch_region is not None:
            self._prefetch_chunks(arr, prefetch_region, cache_key)
        
        return arr
    
    def get_zarr_group(self, s3_path: str) -> zarr.Group:
        path = s3_path if s3_path.startswith("s3://") else f"s3://{s3_path}"
        mapper = fsspec.get_mapper(
            path,
            anon=False,
            client_kwargs={"endpoint_url": os.environ.get("R2_ENDPOINT_URL")} if os.environ.get("R2_ENDPOINT_URL") else {}
        )
        from zarr.storage import KVStore
        wrapped = KVStore(mapper)

        # This structure is robust across zarr versions.
        # It first tries the fast consolidated method. If that fails for any
        # reason (e.g., an old zarr version causing a TypeError, or no
        # .zmetadata file causing a KeyError), it falls back to the standard,
        # non-consolidated method.
        try:
            # Ideal path for modern zarr with consolidated metadata
            return zarr.open_group(wrapped, mode='r', consolidated=True)
        except Exception:
            # Fallback for older zarr versions or non-consolidated stores
            try:
                return zarr.open_group(wrapped, mode='r')
            except zarr.errors.GroupNotFoundError:
                # If it's still not found, it might be a Zarr v3 array/group at the root
                try:
                    root = zarr.open(wrapped, mode='r')
                    if isinstance(root, zarr.Group):
                        return root
                    # It opened but it's not a group (e.g., just an array), so it's not found
                    raise zarr.errors.GroupNotFoundError(path)
                except Exception:
                    # Re-raise the original, more informative error if the v3 check fails
                    raise zarr.errors.GroupNotFoundError(path)
    
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
    Zarr-based pathology dataset that mirrors SlideDataset but uses zarr stores.
    Compatible with DINOv2's dataset string format.
    """
    def __init__(self, root, cache_dir: str = "~/.zarr_chunk_cache",
                 cache_gb: float = 1000.0, prefetch_radius: int = 0,
                 warmup_cache: bool = True,
                 restrict_to_cached_window: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        
        # Handle root parameter - it might come as a Path or string
        self.root = str(root)
        self.zarr_files = []
        self.restrict_to_cached_window = restrict_to_cached_window
        
        # Debug logging
        logger.info(f"SlideDatasetZarr initialized with root: {self.root}")
        
        self.prefetch_radius = prefetch_radius
        self.is_s3 = self.root.startswith("s3://")
        print(f"\nroot: {self.root}")
        if True:#self.is_s3:
            # Initialize smart chunk cache
            self.cache = SmartZarrCache(Path(cache_dir), max_cache_gb=cache_gb)
            
            # Parse bucket & prefix (kept for logging)
            path_wo_scheme = self.root.replace("s3://", "")
            parts = path_wo_scheme.split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            logger.info(f"Listing S3 bucket: {bucket}, prefix: {prefix}")

            def _ensure_s3(p: str) -> str:
                return p if p.startswith("s3://") else f"s3://{p}"

            def _is_dir(entry: dict, name: str) -> bool:
                # s3fs uses type='directory' for prefixes; fall back to trailing slash just in case
                t = (entry.get("type") or "").lower()
                storage = (entry.get("StorageClass") or "").upper()
                return t == "directory" or storage == "DIRECTORY" or name.endswith("/")

            start = _ensure_s3(self.root.rstrip("/"))
            queue: List[Tuple[str, int]] = [(start, 0)]
            seen: Set[str] = set()
            max_depth = 6
            zarr_paths: Set[str] = set()

            while queue:
                base, depth = queue.pop(0)
                if base in seen:
                    continue
                seen.add(base)

                try:
                    entries = self.cache.fs.ls(base, detail=True)
                except Exception as e:
                    logger.warning(f"ls failed at {base}: {e}")
                    continue

                for e in entries:
                    # s3fs detail dicts have 'name'; some envs also expose 'Key'
                    raw = e.get("name") or e.get("Key") or ""
                    if not raw:
                        continue
                    full = _ensure_s3(raw).rstrip("/")

                    # If it *is* a Zarr store, record it and don't descend into it
                    if full.endswith(".zarr"):
                        zarr_paths.add(full)
                        continue

                    # Otherwise, descend only into directories and only to a bounded depth
                    if _is_dir(e, full) and depth < max_depth:
                        queue.append((full, depth + 1))

            self.zarr_files = sorted(zarr_paths)
            print("\nFound this many files", len(self.zarr_files))
            print(f"self.zarr_files: {self.zarr_files[:5]} ...")
        else:
            # Local filesystem - no caching needed
            self.cache = None
            folder_path = Path(self.root)
            self.zarr_files = sorted([str(x) for x in folder_path.rglob("*.zarr") if x.is_dir()])
        
        print("\nFound this many files", len(self.zarr_files))
        print(f"self.zarr_files: {self.zarr_files}")

        # ----------- Warm up the cache window (once) -----------
        if self.is_s3 and self.cache and warmup_cache and len(self.zarr_files) > 0:
            # Decide how many steps you'll run (rough but OK):
            approx_steps = 2000     # e.g., 2000 iterations
            batch_size = 32         # your real batch size

            logger.info("Building prefetch plan...")
            plan = self.build_prefetch_plan(steps=approx_steps, batch_size=batch_size, seed=42)

            logger.info("Warming chunks by plan (high concurrency)...")
            self.cache.warmup_by_plan(plan, headroom_ratio=0.98, max_workers=min(128, (os.cpu_count() or 64)*4))

            # Optional: lock sampling to the plan for 100% cache hits
            self.use_plan_for_sampling(plan, seed=42)
            self.cache.enforce_window = True  # keep sampling inside the cached set
            self.cached_window = list(plan.keys())
            logger.info(f"Cached window size: {len(self.cached_window)} stores; "
                        f"cache bytes ~{self.cache._cache_size_bytes()/1e9:.2f} GB")
    
    def _load_level_meta(self, path: str):
        """Return (levels_present, level_shapes) from _slide_cache or by opening metadata once."""
        cache_key = f"slide_metadata_{path}"
        if hasattr(self, "_slide_cache") and cache_key in self._slide_cache:
            level_arrays, image_levels, level_shapes = self._slide_cache[cache_key]
            return level_arrays, level_shapes
        # Fallback: open group once to populate cache
        if self.is_s3 and self.cache:
            image = self.cache.get_zarr_group(path)
        else:
            image = zarr.open_group(path, mode="r")
        level_arrays, level_shapes = [], []
        for level in ["0","1","2","3","4","5"]:
            if level in image:
                arr = image[level]
                level_arrays.append(level)
                if len(arr.shape) >= 2:
                    level_shapes.append((arr.shape[-2], arr.shape[-1]))  # (H,W)
                else:
                    level_shapes.append(arr.shape)
            else:
                break
        if not hasattr(self, "_slide_cache"):
            self._slide_cache = {}
        self._slide_cache[cache_key] = (level_arrays, len(level_arrays), level_shapes)
        return level_arrays, level_shapes

    def _chunk_keys_for_roi(self, store_path: str, level: str,
                            y0: int, x0: int, h: int, w: int,
                            channel_axis_last: bool = True) -> List[str]:
        """
        Compute chunk keys intersecting ROI for s3://.../.zarr/<level>.
        """
        arr = self.cache.get_zarr_array(store_path, level)
        H, W = arr.shape[0], arr.shape[1] if channel_axis_last else (arr.shape[1], arr.shape[2])
        y1, x1 = min(H, y0 + h), min(W, x0 + w)
        cy, cx, cc = arr.chunks  # expect (y_chunk, x_chunk, c_chunk) for OME-Zarr
        # Which chunk indices intersect the ROI?
        y_chunks = range(y0 // cy, (y1 - 1) // cy + 1)
        x_chunks = range(x0 // cx, (x1 - 1) // cx + 1)
        # Channels: usually small (3) so grab all present chunks
        c_chunks = range(0, math.ceil(arr.shape[2] / cc)) if channel_axis_last and len(arr.shape) == 3 else [0]
        return [f"{iy}.{ix}.{ic}" for iy in y_chunks for ix in x_chunks for ic in c_chunks]

    def build_prefetch_plan(self,
                            steps: int,
                            batch_size: int,
                            seed: int = 0,
                            patch_size: int = 224) -> Dict[str, Dict[str, set]]:
        """
        Decide exactly which chunks a run will read and return:
        { store_path: { level_name: set(chunk_keys) } }
        """
        rng = random.Random(seed)
        plan: Dict[str, Dict[str, set]] = {}

        # Estimate how many samples (patches) we'll actually draw
        total_samples = steps * batch_size
        nstores = len(self.zarr_files)
        if nstores == 0:
            return plan

        # Round-robin across stores so we don't hammer a single one
        for i in range(total_samples):
            store_path = self.zarr_files[i % nstores]
            levels, shapes = self._load_level_meta(store_path)
            if not levels:
                continue

            # Pick a level (bias toward coarser if that matches patch resolution better, optional)
            level = levels[min(rng.randrange(len(levels)), len(levels) - 1)]
            idx = levels.index(level)
            H, W = shapes[idx]

            # Pick a random top-left inside bounds
            if H <= patch_size or W <= patch_size:
                x = max(0, (W - patch_size) // 2)
                y = max(0, (H - patch_size) // 2)
            else:
                x = rng.randrange(0, W - patch_size)
                y = rng.randrange(0, H - patch_size)

            keys = self._chunk_keys_for_roi(store_path, level, y, x, patch_size, patch_size, channel_axis_last=True)
            d = plan.setdefault(store_path, {})
            s = d.setdefault(level, set())
            s.update(keys)

        return plan

    def use_plan_for_sampling(self, plan: Dict[str, Dict[str, set]], seed: int = 0, patch_size: int = 224):
        """
        Optional: lock dataset sampling to the planned patches so every read
        hits the warm cache. Creates a deterministic list of (store, level, y, x).
        """
        rng = random.Random(seed)
        seq = []
        for store_path, levels in plan.items():
            # Reconstruct approximate ROIs by sampling one pixel in each planned chunk
            for level, keys in levels.items():
                # We approximate by sampling a random point inside each chunk
                arr = self.cache.get_zarr_array(store_path, level)
                cy, cx, cc = arr.chunks
                for k in keys:
                    iy, ix, _ = map(int, k.split("."))
                    y = min(iy * cy, arr.shape[0] - patch_size)
                    x = min(ix * cx, arr.shape[1] - patch_size)
                    seq.append((store_path, level, y, x))
        rng.shuffle(seq)
        self._planned_seq = seq
        self._plan_idx = 0

    def get_all(self, index):
        path = self.zarr_files[index]
        
        if self.is_s3 and self.cache:
            # Get group through cache
            image = self.cache.get_zarr_group(path)
        else:
            # Open from local filesystem
            image = zarr.open_group(path, mode='r')
        
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        debug = False
        
        if hasattr(self, "_planned_seq") and self._planned_seq:
            # Deterministic planned patch
            store_path, level_name, y, x = self._planned_seq[self._plan_idx % len(self._planned_seq)]
            self._plan_idx += 1
            path = store_path
            # Get array via cache (no extra metadata round-trips)
            arr = self.cache.get_zarr_array(path, level_name)
            patch_size = 224
            height, width = arr.shape[0], arr.shape[1]
            channel_axis = 2
            # Extract patch straight away
            try:
                patch = arr[y:y+patch_size, x:x+patch_size, :3]
                patch = np.asarray(patch, dtype=np.uint8)
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded
                res = Image.fromarray(patch, mode="RGB")
                if self.transforms is not None:
                    return self.transforms(res, None)
                return res, None
            except Exception as e:
                # Fallback to normal path on any error
                pass

        path = self.zarr_files[index]

        if (self.is_s3 and self.cache and self.restrict_to_cached_window and
            self.cache.enforce_window and self.cached_window):

            # If this index points outside the active window, swap to a cached path.
            if path not in self.cache.allowed_store_prefixes:
                # Replace with a random cached store (keeps training fast)
                path = random.choice(self.cached_window)
        
        # Ultra-fast metadata caching
        level_arrays, level_shapes = self._load_level_meta(path)
        image_levels = len(level_arrays)
        
        if debug:
            print("This many image levels", image_levels)
            print("This dim", level_shapes)
        
        if image_levels == 0:
            raise RuntimeError(f"No level arrays found in {path}")
        
        # Pick a random level
        level_idx = random.randint(0, image_levels - 1)
        level_name = level_arrays[level_idx]
        if debug:
            print("picked", level_name)
        
        patch_size = 224
        
        # Use cached dimensions - no need to access zarr metadata again
        height, width = level_shapes[level_idx]
        channel_axis = 2  # Assume channels last for our data
        
        if debug:
            print("these dims", (width, height))
        
        # Try to find a valid patch
        i = 0
        while True:
            if debug:
                print("start loop", flush=True)
            i = i + 1
            if i == 100:
                raise RuntimeError(f"Couldn't find matching item in slide {path}")
            
            if debug:
                print("iteration", i)
            
            # Random position (matching slide_dataset.py logic exactly)
            if height <= patch_size or width <= patch_size:
                x = max(0, (width - patch_size) // 2)
                y = max(0, (height - patch_size) // 2)
            else:
                x = random.randint(0, width - patch_size)
                y = random.randint(0, height - patch_size)
            
            if debug:
                print("Reading this", path, x, y, level_name)
            
            try:
                # Get the array with caching and prefetching if using S3
                if self.is_s3 and self.cache:
                    # Define prefetch region (slightly larger than needed)
                    pf_y0 = max(0, y - self.prefetch_radius)
                    pf_y1 = min(height, y + patch_size + self.prefetch_radius)
                    pf_x0 = max(0, x - self.prefetch_radius)
                    pf_x1 = min(width, x + patch_size + self.prefetch_radius)
                    
                    prefetch_region = (slice(pf_y0, pf_y1), slice(pf_x0, pf_x1), slice(None))
                    
                    # Get array with caching
                    arr = self.cache.get_zarr_array(path, level_name, prefetch_region)
                else:
                    group = zarr.open_group(path, mode='r')
                    arr = group[level_name]

                # Extract patch from zarr array
                patch_data = arr[y:y+patch_size, x:x+patch_size, :3]
                patch = np.asarray(patch_data, dtype=np.uint8)
                
                # Pad if necessary
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded
                
                # Convert to PIL Image
                patch = Image.fromarray(patch, mode="RGB")
                
            except Exception as e:
                print("failed on path", path, x, y, level_name, "Error:", e)
                continue
            
            if True:  # Skip HSV filtering for now (matching slide_dataset.py behavior)
                res = patch
                break
                
            res = self.hsv(patch, patch_size)
            print("have result", path, x, y, level_name)
            if res is None:
                pass
            else:
                break
        
        # The transform used is a torchvision StandardTransform.
        # This means that it takes as input two things, and runs two different transforms on both.
        if self.transforms is not None:
            return self.transforms(res, None)
        return res, None
    
    def hsv(self, tile_rgb, patch_size):
        tile = np.array(tile_rgb)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        min_ratio = .6
        
        lower_bound = np.array([90, 8, 103])
        upper_bound = np.array([180, 255, 255])
        
        mask = cv2.inRange(tile, lower_bound, upper_bound)
        
        ratio = np.count_nonzero(mask) / mask.size
        if ratio > min_ratio:
            return tile_rgb
        else:
            return None
    
    def __len__(self) -> int:
        return len(self.zarr_files)