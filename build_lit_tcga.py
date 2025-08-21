#!/usr/bin/env python3
"""
Convert TCGA SVS dataset to LitData-optimized format for efficient streaming during training (S3 only).

Goals:
- Streamlined, efficient extraction of valid tissue patches to S3
- Balanced sampling across magnification levels (equal per-mag totals)
- Flexible budgeting by either total tiles or approximate data size

Notes:
- We use simple, robust heuristics (tissue mask + HSV gating) rather than
  strictly mirroring any single prior work. S3-compatible endpoints (e.g., R2)
  are supported via AWS_ENDPOINT_URL.
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
# Delay importing litdata.optimize to avoid heavy imports and sandbox issues
from io import BytesIO
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
            # Default magnifications in μm/px: 2, 1, 0.5, 0.25
            self.magnifications = [2.0, 1.0, 0.5, 0.25]
        
        if self.hsv_ranges is None:
            # Default HSV ranges tuned for H&E-like tissue foreground
            self.hsv_ranges = {
                'hue': (90, 180),
                'saturation': (8, 255),
                'value': (103, 255)
            }


def _extract_tiles_fn(
    item: Any,
    config: TileConfig,
    tiles_per_mag: int = None,
    per_slide_mag_quota: Dict[str, Dict[float, int]] = None,
    stats_bucket: Optional[str] = None,
    stats_prefix: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """LitData optimize() worker: given one task item, yield many tile dicts.

    The `item` can be either:
    - a string slide path (backwards compatible), or
    - a tuple of (slide_path, task_quota_dict[, task_id]) where task_quota_dict
      is a per-magnification quota for this task chunk, and task_id is an
      optional unique identifier used to write per-chunk stats without
      overwriting.

    Honors a per-slide, per-magnification quota if provided; otherwise uses a
    uniform `tiles_per_mag` across available magnifications.
    """
    # Parse task input
    slide_path: str
    task_quota: Optional[Dict[float, int]] = None
    task_id: Optional[str] = None
    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], dict):
        slide_path = item[0]
        task_quota = item[1]
        if len(item) >= 3 and isinstance(item[2], str):
            task_id = item[2]
    elif isinstance(item, str):
        slide_path = item
    else:
        raise ValueError(f"Unsupported task item type: {type(item)}")

    extractor = TileExtractor(config)
    # Resolve quota mapping
    quota_map = None
    if task_quota:
        # Explicit per-task quota takes precedence
        quota_map = task_quota
    elif per_slide_mag_quota is not None:
        # normalize to absolute path for stable lookup
        abs_path = os.path.abspath(slide_path)
        quota_map = per_slide_mag_quota.get(abs_path)

    counts_by_mag: Dict[float, int] = {}
    for t in extractor.extract_tiles_from_slide(slide_path, max_tiles_per_mag=tiles_per_mag, per_mag_quota=quota_map):
        m = float(t.get("magnification", -1))
        if m >= 0:
            counts_by_mag[m] = counts_by_mag.get(m, 0) + 1
        # LitData serializes PIL Images out of the box
        t["image"] = Image.fromarray(t["image"])
        yield t

    # Write stats after yielding all tiles for this task (slide or chunk)
    if stats_bucket and stats_prefix is not None:
        try:
            endpoint_url = os.getenv("AWS_ENDPOINT_URL")
            s3c = boto3.client("s3", endpoint_url=endpoint_url)
            slide_id = Path(slide_path).stem
            import hashlib
            slide_abs = os.path.abspath(slide_path)
            sid = hashlib.sha1(slide_abs.encode("utf-8")).hexdigest()[:16]
            # If chunked, append task_id to avoid overwrites and allow proper aggregation
            if task_id:
                stats_key = f"{stats_prefix}{run_id}/{slide_id}_{sid}_{task_id}.json"
            else:
                stats_key = f"{stats_prefix}{run_id}/{slide_id}_{sid}.json"
            payload = {
                "slide_id": slide_id,
                "slide_path": os.path.abspath(slide_path),
                "run_id": run_id,
                "counts_by_mag": {str(k): int(v) for k, v in counts_by_mag.items()},
                "total": int(sum(counts_by_mag.values())),
            }
            s3c.put_object(Bucket=stats_bucket, Key=stats_key, Body=json.dumps(payload).encode("utf-8"))
        except Exception as e:
            logger.warning(f"Failed to write stats for {slide_path}: {e}")


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
    
    def extract_tiles_from_slide(
        self,
        slide_path: str,
        max_tiles_per_mag: int = 100,
        per_mag_quota: Optional[Dict[float, int]] = None,
    ):
        """Yield tiles from a single slide at multiple magnifications.

        If `per_mag_quota` is provided, it determines the requested number of
        tiles for each magnification (μm/px). Otherwise, `max_tiles_per_mag`
        is used uniformly for all magnifications available in this slide.
        """

        try:
            slide = OpenSlide(slide_path)
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            return

        try:
            # Get tissue mask and threshold
            mask_level = min(2, slide.level_count - 1)
            tissue_mask, otsu_thresh = self.get_tissue_mask(slide, level=mask_level)

            # Get magnification to level mapping
            mag_map = self.get_available_magnifications(slide)

            # Extract tiles for each magnification (iterate in random order to reduce bias)
            items = list(mag_map.items())
            random.shuffle(items)
            for target_mag, level in items:
                downsample = slide.level_downsamples[level]

                # Tile size in level 0 coordinates corresponds to the physical area we want to capture
                tile_size_level0 = int(self.config.tile_size * downsample)

                # Generate random positions
                n_attempts = 0
                n_extracted = 0
                # Determine target quota for this magnification
                target_quota = (
                    int(per_mag_quota.get(target_mag, 0)) if per_mag_quota is not None else int(max_tiles_per_mag)
                )
                if target_quota <= 0:
                    continue
                # Allow multiple attempts; cap to avoid unbounded sweeps
                max_attempts = max(target_quota * 25, 500)

                while n_extracted < target_quota and n_attempts < max_attempts:
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

                    yield tile_data
                    n_extracted += 1

                logger.debug(f"Extracted {n_extracted} tiles at {target_mag} μm/px from {Path(slide_path).name}")
        finally:
            try:
                slide.close()
            except Exception:
                pass


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


def parse_size_to_bytes(size_str: str) -> int:
    """Parse human-friendly size strings like '200GB', '50g', '1.5TB', '300MB', '2e9'."""
    s = size_str.strip().lower()
    if not s:
        raise ValueError("Empty size string")
    # Scientific notation
    try:
        if 'e' in s:
            return int(float(s))
    except ValueError:
        pass
    # Unit map (powers of 2 and 10 are both common; use powers of 2 for conservative estimate)
    units = {
        'b': 1,
        'kb': 1024,
        'mb': 1024**2,
        'gb': 1024**3,
        'tb': 1024**4,
    }
    # Extract number and unit
    num = ''
    unit = ''
    for ch in s:
        if (ch.isdigit() or ch == '.' or ch == '+'):  # allow decimals
            num += ch
        else:
            unit += ch
    if not num:
        raise ValueError(f"Invalid size '{size_str}'")
    if not unit:
        # bytes by default
        return int(float(num))
    unit = unit.strip()
    # Normalize unit variants
    aliases = {
        'k': 'kb', 'm': 'mb', 'g': 'gb', 't': 'tb'
    }
    unit = aliases.get(unit, unit)
    if unit not in units:
        raise ValueError(f"Unknown unit '{unit}' in size '{size_str}'")
    return int(float(num) * units[unit])


def estimate_tiles_for_target_size(
    target_size_bytes: int,
    tile_size: int,
    avg_bytes_per_tile: int = None,
) -> int:
    """Estimate tile count for a target size budget.

    - By default, use an empirical lower-bound estimate: uncompressed RGB bytes
      (tile_size^2 * 3). LitData typically compresses images, so this is a
      conservative estimate. Callers can override via `avg_bytes_per_tile`.
    """
    if avg_bytes_per_tile is None or avg_bytes_per_tile <= 0:
        avg_bytes_per_tile = tile_size * tile_size * 3  # uncompressed RGB
    if target_size_bytes <= 0:
        raise ValueError("target_size_bytes must be positive")
    return max(int(target_size_bytes // avg_bytes_per_tile), 1)


def _list_all_keys(s3_client: BaseClient, bucket: str, prefix: str) -> List[str]:
    """List all keys under an S3 prefix, handling pagination."""
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) :
            keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def read_cumulative_counts_from_stats(
    s3_client: BaseClient,
    bucket: str,
    stats_prefix: str,
    magnifications: List[float],
) -> Dict[float, int]:
    """Aggregate cumulative per-magnification counts from per-slide stats JSONs."""
    cum: Dict[float, int] = {float(m): 0 for m in magnifications}
    try:
        keys = _list_all_keys(s3_client, bucket, stats_prefix)
        for k in keys:
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=k)
                data = json.loads(obj["Body"].read().decode("utf-8"))
                counts = data.get("counts_by_mag", {})
                for m_str, v in counts.items():
                    try:
                        m = float(m_str)
                        if m in cum:
                            cum[m] += int(v)
                    except Exception:
                        continue
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    continue
                else:
                    raise
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return cum
        else:
            raise
    return cum


def allocate_across_magnifications(
    cumulative: Dict[float, int],
    budget: int,
    magnifications: List[float],
) -> Dict[float, int]:
    """Allocate a tile budget across magnifications to equalize cumulative counts.

    Water-filling algorithm: raises the lowest counts first until the budget is exhausted.
    Returns per-mag allocation for the current run.
    """
    mags = list(magnifications)
    # Prepare vectors
    cum = np.array([int(cumulative.get(m, 0)) for m in mags], dtype=np.int64)
    M = len(mags)
    if M == 0 or budget <= 0:
        return {m: 0 for m in mags}

    # Sort by cumulative counts
    order = np.argsort(cum)
    sorted_cum = cum[order].astype(np.float64)

    T = float(budget)
    levels = sorted_cum.copy()

    # Water-filling
    for i in range(M - 1):
        gap = levels[i + 1] - levels[i]
        cost = gap * (i + 1)
        if T >= cost and cost > 0:
            levels[: i + 1] += gap
            T -= cost
        else:
            # Distribute remaining T evenly among first i+1
            inc = T / (i + 1)
            levels[: i + 1] += inc
            T = 0.0
            break
    if T > 0:
        # Distribute leftover evenly across all
        inc = T / M
        levels += inc
        T = 0.0

    # Compute real-valued deltas and then quantize to integers
    deltas = levels - sorted_cum
    # Floor to integers first
    deltas_int = np.floor(deltas).astype(np.int64)
    # Distribute remainder based on largest fractional parts
    remainder = int(round(budget - int(deltas_int.sum())))
    if remainder > 0:
        frac = deltas - deltas_int
        idx = np.argsort(-frac)  # descending by fractional part
        for j in idx[:remainder]:
            deltas_int[j] += 1
    elif remainder < 0:
        # Should rarely happen, but guard by removing from smallest fractional parts
        frac = deltas - deltas_int
        idx = np.argsort(frac)  # ascending
        for j in idx[: (-remainder)]:
            if deltas_int[j] > 0:
                deltas_int[j] -= 1

    # Map back to original mag order
    alloc_sorted = deltas_int
    alloc = np.zeros_like(alloc_sorted)
    alloc[order] = alloc_sorted

    return {m: int(a) for m, a in zip(mags, alloc.tolist())}


def pilot_estimate_avg_tile_bytes(
    slide_paths: List[str],
    config: TileConfig,
    total_samples: int = 256,
    max_slides: int = 16,
) -> Optional[int]:
    """Run a small pilot extraction to estimate average tile bytes.

    - Distributes the sample target equally across magnifications.
    - Encodes tiles in-memory and measures encoded byte size.
    - Returns None if no tiles could be sampled.
    """
    if total_samples <= 0:
        return None

    mags = list(config.magnifications)
    per_mag_needed = {m: total_samples // len(mags) for m in mags}
    # distribute remainder
    remainder = total_samples % len(mags)
    for m in mags[:remainder]:
        per_mag_needed[m] += 1

    extractor = TileExtractor(config)
    rng_slides = slide_paths.copy()
    random.shuffle(rng_slides)
    rng_slides = rng_slides[: max_slides]

    total_bytes = 0
    total_count = 0

    for sp in rng_slides:
        # Quick check which mags this slide supports
        try:
            slide = OpenSlide(sp)
        except Exception:
            continue
        try:
            mag_map = extractor.get_available_magnifications(slide)
        finally:
            try:
                slide.close()
            except Exception:
                pass

        # Build a small quota for this slide based on remaining needs
        per_mag_quota = {m: min(per_mag_needed.get(m, 0), 8) for m in mags if m in mag_map and per_mag_needed.get(m, 0) > 0}
        if not per_mag_quota:
            continue

        for td in extractor.extract_tiles_from_slide(sp, per_mag_quota=per_mag_quota):
            img = Image.fromarray(td["image"])  # already RGB
            bio = BytesIO()
            img.save(bio, format="PNG", optimize=True)
            size = bio.tell()
            total_bytes += size
            total_count += 1
            m = float(td.get("magnification", -1))
            if m in per_mag_needed and per_mag_needed[m] > 0:
                per_mag_needed[m] -= 1
        # Early stop if done
        if all(v <= 0 for v in per_mag_needed.values()):
            break

    if total_count == 0:
        logger.warning("Pilot sampling produced zero tiles; using default uncompressed estimate.")
        return None
    est = int(total_bytes // total_count)
    logger.info(f"Pilot estimated avg tile bytes: {est} from {total_count} samples.")
    return est

def compute_per_slide_mag_quotas(
    slide_paths: List[str],
    config: TileConfig,
    total_tiles: Optional[int],
    default_tiles_per_mag: Optional[int],
    per_mag_totals: Optional[Dict[float, int]] = None,
) -> Tuple[Dict[str, Dict[float, int]], Dict[float, int]]:
    """Compute per-slide, per-magnification quotas to balance totals across magnifications.

    Returns:
    - per_slide_quota: {abs_slide_path: {magnification_um_per_px: tiles}}
    - per_mag_total: {magnification_um_per_px: total_tiles_budgeted}

    Logic:
    - If total_tiles is provided, split equally across magnifications (rounded down),
      then distribute to slides that support each mag as evenly as possible.
    - If default_tiles_per_mag is provided (and total_tiles is None), assign that
      value to each slide for each supported mag.
    """
    # Normalize paths
    slide_paths = [os.path.abspath(p) for p in slide_paths]
    per_slide_quota: Dict[str, Dict[float, int]] = {p: {} for p in slide_paths}

    # Pre-scan which magnifications each slide supports (fast metadata read only)
    extractor = TileExtractor(config)
    supported_by_mag: Dict[float, List[str]] = {m: [] for m in config.magnifications}
    for p in slide_paths:
        slide = None
        try:
            slide = OpenSlide(p)
            mag_map = extractor.get_available_magnifications(slide)
            for m in config.magnifications:
                if m in mag_map:
                    supported_by_mag[m].append(p)
        except Exception as e:
            logger.warning(f"Skipping slide due to OpenSlide error during pre-scan: {p} ({e})")
        finally:
            try:
                if slide is not None:
                    slide.close()
            except Exception:
                pass

    per_mag_total: Dict[float, int] = {}
    if per_mag_totals is not None:
        # Use provided per-mag totals directly
        for m in config.magnifications:
            per_mag_total[m] = int(per_mag_totals.get(m, 0))
    elif total_tiles is not None:
        # Split total equally across mags
        num_mags = len(config.magnifications)
        if num_mags == 0:
            raise ValueError("No magnifications configured")
        base = total_tiles // num_mags
        remainder = total_tiles % num_mags
        for idx, m in enumerate(config.magnifications):
            per_mag_total[m] = base + (1 if idx < remainder else 0)
    else:
        # Default per-mag totals derive from per-slide default_tiles_per_mag
        if default_tiles_per_mag is None:
            raise ValueError("Either total_tiles, per_mag_totals, or default_tiles_per_mag must be provided")
        for m in config.magnifications:
            per_mag_total[m] = default_tiles_per_mag * len(supported_by_mag[m])

    # Distribute per-mag totals as evenly as possible over supporting slides
    for m in config.magnifications:
        slides = supported_by_mag[m]
        if not slides:
            continue
        total_m = per_mag_total[m]
        base = total_m // len(slides)
        rem = total_m % len(slides)
        # Randomize order for fair distribution of remainders
        shuffled = slides.copy()
        random.shuffle(shuffled)
        for i, p in enumerate(shuffled):
            quota = base + (1 if i < rem else 0)
            if quota > 0:
                per_slide_quota[p][m] = quota

    return per_slide_quota, per_mag_total

def create_litdata_dataset(
    input_dir: str,
    output_dir: str,
    config: TileConfig,
    tiles_per_mag: Optional[int],
    num_workers: int,
    exclude_file: Optional[str],
    total_tiles: Optional[int] = None,
    target_size: Optional[str] = None,
    avg_tile_bytes: Optional[int] = None,
    pilot_sample: int = 0,
    pilot_slides: int = 16,
    progress_chunk: int = 256,
):
    if not output_dir.startswith("s3://"):
        raise ValueError("--output-dir must be an s3:// URI (S3 or S3-compatible endpoint)")

    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    if endpoint_url:
        logger.info(f"Using custom S3 endpoint: {endpoint_url}")
    s3_client = boto3.client("s3", endpoint_url=endpoint_url)
    parsed_url = urlparse(output_dir)
    log_bucket = parsed_url.netloc
    prefix = parsed_url.path.lstrip('/')
    if prefix and not prefix.endswith('/'): prefix += '/'
    log_key = f"{prefix}_processed_slides.log"
    stats_prefix = f"{prefix}_stats/"
    run_id = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # --- RESUMING FROM INTERRUPT LOGIC ---
    all_slide_paths = find_svs_files(input_dir, exclude_file)
    if not all_slide_paths:
        logger.warning("No SVS files found under input-dir; nothing to do.")
        return
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

    # Compute budgets/quotas
    computed_total_tiles: Optional[int] = None
    per_slide_mag_quota: Optional[Dict[str, Dict[float, int]]] = None
    per_mag_total: Optional[Dict[float, int]] = None
    cumulative_counts: Optional[Dict[float, int]] = None
    avg_bytes_used: Optional[int] = None

    if target_size is not None:
        try:
            _ = parse_size_to_bytes(target_size)
        except Exception as e:
            raise ValueError(f"Failed to parse --target-size '{target_size}': {e}")

    # If using --target-size and no avg_tile_bytes provided, run pilot
    if target_size is not None and avg_tile_bytes is None and pilot_sample and pilot_sample > 0:
        try:
            est = pilot_estimate_avg_tile_bytes(
                slides_to_process, config, total_samples=pilot_sample, max_slides=pilot_slides
            )
            if est is not None and est > 0:
                avg_tile_bytes = est
        except Exception as e:
            logger.warning(f"Pilot sampling failed: {e}")

    # Determine final total_tiles
    if total_tiles is not None:
        computed_total_tiles = None
        final_total_tiles = int(total_tiles)
    elif target_size is not None:
        target_size_bytes = parse_size_to_bytes(target_size)
        computed_total_tiles = estimate_tiles_for_target_size(target_size_bytes, config.tile_size, avg_tile_bytes)
        final_total_tiles = int(computed_total_tiles)
        avg_bytes_used = avg_tile_bytes or config.tile_size**2 * 3
        logger.info(f"Target size {target_size} -> approx {final_total_tiles} tiles (avg {avg_bytes_used} B/tile)")
    else:
        final_total_tiles = None

    # Compute quotas
    if final_total_tiles is not None:
        cumulative_counts = read_cumulative_counts_from_stats(s3_client, log_bucket, stats_prefix, config.magnifications)
        logger.info("Cumulative per-mag so far: " + 
                    ", ".join([f"{m}: {cumulative_counts.get(m,0)}" for m in config.magnifications]))
        per_mag_total = allocate_across_magnifications(cumulative_counts, final_total_tiles, config.magnifications)
        per_slide_mag_quota, _ = compute_per_slide_mag_quotas(
            slides_to_process, config, None, None, per_mag_totals=per_mag_total
        )
        logger.info("This run per-mag allocations: " + 
                    ", ".join([f"{m} μm/px: {per_mag_total[m]}" for m in config.magnifications]))
        tiles_per_mag = None
    else:
        if tiles_per_mag is None:
            raise ValueError("Provide either --tiles-per-mag or --total-tiles or --target-size.")
        per_slide_mag_quota, per_mag_total = compute_per_slide_mag_quotas(
            slides_to_process, config, None, tiles_per_mag
        )
        logger.info(f"Using uniform tiles_per_mag={tiles_per_mag} across supported magnifications.")

    # (Simplified) Pilot already handled above; quotas computed based on final_total_tiles or tiles_per_mag

    # Decide mode: first successful run uses overwrite, later runs append
    mode = "overwrite" if not already_processed else "append"

    # OPTIONAL: batch slides to reduce blast radius on failure (e.g., 200 slides per call)
    batch_size = 200
    batches = [slides_to_process[i:i+batch_size] for i in range(0, len(slides_to_process), batch_size)]

    def _build_task_inputs(batch_slides: List[str]) -> List[tuple]:
        """Build smaller per-slide task chunks so progress advances more frequently.

        Returns a list of items acceptable by _extract_tiles_fn: either
        (slide_path, per_task_quota_dict, task_id) tuples when chunking is enabled,
        or plain slide paths if chunking is disabled.
        """
        if not per_slide_mag_quota or progress_chunk is None or progress_chunk <= 0:
            return list(batch_slides)

        items: List[tuple] = []
        for sp in batch_slides:
            abs_sp = os.path.abspath(sp)
            qdict = per_slide_mag_quota.get(abs_sp, {})
            if not qdict:
                # No explicit per-mag quotas for this slide; fall back to single task
                items.append(sp)
                continue
            # Create per-mag chunks
            task_counter = 0
            for m, total in qdict.items():
                remaining = int(total)
                chunk_idx = 0
                while remaining > 0:
                    take = min(progress_chunk, remaining)
                    remaining -= take
                    task_id = f"m{str(m).replace('.', 'p')}-{chunk_idx:04d}"
                    items.append((sp, {m: take}, task_id))
                    chunk_idx += 1
                    task_counter += 1
        return items

    for i, batch in enumerate(batches):
        # first batch of a brand new dataset: overwrite; otherwise append
        batch_mode = "overwrite" if (mode == "overwrite" and i == 0) else "append"
        
        try:
            # Lazy import here so help/pilot can run in restricted envs
            from litdata import optimize
            task_inputs = _build_task_inputs(batch)
            if isinstance(task_inputs, list):
                logger.info(f"Batch {i}: scheduling {len(task_inputs)} task(s) across {len(batch)} slide(s) [progress-chunk={progress_chunk}].")
            optimize(
                fn=partial(
                    _extract_tiles_fn,
                    config=config,
                    tiles_per_mag=tiles_per_mag,
                    per_slide_mag_quota=per_slide_mag_quota,
                    stats_bucket=log_bucket,
                    stats_prefix=stats_prefix,
                    run_id=run_id,
                ),
                inputs=task_inputs,
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

        # Recompute cumulative counts after this batch and write metadata
        cumulative_counts = read_cumulative_counts_from_stats(s3_client, log_bucket, stats_prefix, config.magnifications)
        cumulative_total = int(sum(cumulative_counts.values()))

        metadata = {
            "num_slides": len(fully_processed),
            "tiles_per_magnification": tiles_per_mag,
            "magnifications": config.magnifications,
            "tile_size": config.tile_size,
            "last_updated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "batch_index": i,
            "total_batches": len(batches),
            "per_mag_total_budget_run": per_mag_total,
            "cumulative_per_mag_counts": {str(k): int(v) for k, v in cumulative_counts.items()} if cumulative_counts else None,
            "cumulative_total_tiles": cumulative_total,
            "avg_tile_bytes_estimate": avg_bytes_used,
            "run_id": run_id,
        }
        metadata_key = f"{prefix}metadata.json"
        s3_client.put_object(Bucket=log_bucket, Key=metadata_key, Body=json.dumps(metadata, indent=2))
        logger.info(f"Batch {i} committed. Processed slides so far: {len(fully_processed)}")

    logger.info("Processing finished.")

def main():
    parser = argparse.ArgumentParser(description="Convert TCGA SVS dataset to LitData format, saving to S3.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/TCGA",
        help="Input directory containing TCGA SVS files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="s3://tcga-13tb-litdata",
        help="S3 output directory for LitData dataset (must be s3://bucket/prefix)"
    )
    group_budget = parser.add_mutually_exclusive_group()
    group_budget.add_argument(
        "--tiles-per-mag",
        type=int,
        help="Uniform number of tiles to extract per supported magnification per slide"
    )
    group_budget.add_argument(
        "--total-tiles",
        type=int,
        help="Total desired number of tiles across the dataset (distributed equally across magnifications and slides)"
    )
    group_budget.add_argument(
        "--target-size",
        type=str,
        default="10GB",
        help="Approximate target dataset size (e.g., 200GB, 50M, 1.5TB)."
    )
    parser.add_argument(
        "--avg-tile-bytes",
        type=int,
        default=None,
        help="Average bytes per tile when using --target-size (default: uncompressed RGB or pilot estimate)."
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
    # Pilot sampling 
    parser.add_argument(
        "--pilot-sample",
        type=int,
        default=0,
        help="Pilot: number of tiles to sample for estimating avg bytes (0 disables)."
    )
    parser.add_argument(
        "--pilot-slides",
        type=int,
        default=4,
        help="Pilot: maximum slides to touch when estimating."
    )
    parser.add_argument(
        "--progress-chunk",
        type=int,
        default=256,
        help="Tiles per task chunk to make progress bar more responsive (0 disables chunking)."
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
        exclude_file=args.exclude_file,
        total_tiles=args.total_tiles,
        target_size=args.target_size,
        avg_tile_bytes=args.avg_tile_bytes,
        pilot_sample=args.pilot_sample,
        pilot_slides=args.pilot_slides,
        progress_chunk=args.progress_chunk,
    )


if __name__ == "__main__":
    main()
