#!/usr/bin/env python3
"""
Revised: Convert TCGA SVS dataset to LitData format with faster masking and sampling.

Key improvements:
- Coarsest-level tissue mask by default (safer, cheaper), configurable.
- Grid-based candidate generation using integral image for fast area checks.
- Optional low-res HSV precheck on coarsest-level thumbnail to avoid expensive reads.
- OpenCV thread caps to avoid CPU oversubscription in multi-process runs.
- Expose litdata.optimize knobs: chunk_bytes, num_uploaders/downloaders, start_method.
- Incremental stats aggregation to avoid scanning historical runs every batch.

Notes:
- Magnification handling remains μm/px targets mapped to nearest slide level.
- Tile size is in pixels at the read level; location is in level-0 coordinates.
- Defaults tuned for S3 StreamingDataset training (chunk_bytes=64MB).
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
import time
import signal

import numpy as np
import cv2
from PIL import Image
import openslide
from openslide import OpenSlide
import boto3
from botocore.config import Config as BotoConfig
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from io import BytesIO
from functools import partial


# Conservative CPU threading defaults to avoid oversubscription across workers
try:
    _opencv_threads = int(os.getenv("OPENCV_NUM_THREADS", "1"))
    cv2.setNumThreads(max(_opencv_threads, 1))
except Exception:
    pass
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"


# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quiet noisy boto logs at INFO that spam endpoint/credentials discovery
for _name in ("botocore", "boto3", "s3transfer"):
    try:
        logging.getLogger(_name).setLevel(logging.WARNING)
    except Exception:
        pass


# Cached, process-local S3 client to avoid repeated endpoint/credential scanning
_S3_SESSION = None
_S3_CLIENT = None


def get_s3_client() -> BaseClient:
    global _S3_SESSION, _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    # Prefer explicit env credentials to avoid provider-chain scans
    session_kwargs = dict(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "auto",
    )
    _S3_SESSION = boto3.session.Session(**session_kwargs)
    cfg = BotoConfig(
        s3={"addressing_style": "auto"},
        retries={
            "max_attempts": int(os.getenv("AWS_MAX_ATTEMPTS", "5")),
            "mode": os.getenv("AWS_RETRY_MODE", "standard"),
        },
        max_pool_connections=int(os.getenv("AWS_MAX_POOL_CONNECTIONS", "64")),
        connect_timeout=int(os.getenv("AWS_CONNECT_TIMEOUT", "10")),
        read_timeout=int(os.getenv("AWS_READ_TIMEOUT", "300")),
        tcp_keepalive=True,
    )
    _S3_CLIENT = _S3_SESSION.client("s3", endpoint_url=endpoint_url, config=cfg)
    return _S3_CLIENT


@dataclass
class TileConfig:
    tile_size: int = 256
    magnifications: List[float] = None  # μm/px targets
    foreground_threshold: float = 0.4   # fraction of tissue mask inside tile
    hsv_threshold: bool = True
    hsv_ranges: Dict[str, Tuple[int, int]] = None
    pixel_threshold: float = 0.6        # fraction of HSV-accepted pixels

    # Revised controls
    mask_level: Optional[str] = "coarsest"  # "coarsest" or int as string
    grid_sampling: bool = True
    grid_stride: float = 1.0            # stride factor relative to tile window at mask level
    grid_jitter: bool = True
    grid_jitter_frac: float = 0.3
    lowres_hsv_precheck: bool = True
    # Informativeness gate (fast, conservative)
    informativeness: bool = True
    min_gray_std: float = 5.0

    def __post_init__(self):
        if self.magnifications is None:
            self.magnifications = [2.0, 1.0, 0.5, 0.25]
        if self.hsv_ranges is None:
            self.hsv_ranges = {
                'hue': (90, 180),
                'saturation': (8, 255),
                'value': (103, 255)
            }


def _integral_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Sum over region using integral image (expects shape (H+1, W+1))."""
    x2 = x + w
    y2 = y + h
    return int(ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x])


class TileExtractor:
    """Fast tile extractor using coarsest mask + grid candidates + optional low-res HSV precheck."""

    def __init__(self, config: TileConfig):
        self.config = config

    # -------- Low-res reference (mask + optional HSV) --------
    def _pick_mask_level(self, slide: OpenSlide) -> int:
        ml = self.config.mask_level
        if ml is None or ml == "coarsest":
            return slide.level_count - 1
        try:
            lvl = int(ml)
            return max(0, min(lvl, slide.level_count - 1))
        except Exception:
            return slide.level_count - 1

    def build_lowres_reference(self, slide: OpenSlide) -> Dict[str, Any]:
        level = self._pick_mask_level(slide)
        dims = slide.level_dimensions[level]
        thumb = slide.read_region((0, 0), level, dims).convert('RGB')
        rgb = np.array(thumb)

        # Tissue mask from grayscale Otsu on coarsest
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        otsu_thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Integral image for fast area checks (use 1/0 instead of 255/0)
        mask01 = (mask > 0).astype(np.uint8)
        ii = cv2.integral(mask01, sdepth=cv2.CV_32S)

        hsv = None
        if self.config.hsv_threshold and self.config.lowres_hsv_precheck:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        return {
            'level': level,
            'dims': dims,
            'downsample': float(slide.level_downsamples[level]),
            'mask': mask,
            'mask_integral': ii,
            'otsu_thresh': float(otsu_thresh),
            'rgb': rgb,
            'hsv': hsv,
        }

    # -------- Utility checks --------
    def check_foreground_threshold(self, tile: np.ndarray, slide_threshold: float = None) -> bool:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        if slide_threshold is not None:
            _, binary = cv2.threshold(gray, slide_threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        foreground_ratio = float(np.sum(binary > 0)) / float(binary.size)
        return foreground_ratio >= self.config.foreground_threshold

    def apply_hsv_threshold(self, tile: np.ndarray) -> bool:
        if not self.config.hsv_threshold:
            return True
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        hr = self.config.hsv_ranges['hue']
        sr = self.config.hsv_ranges['saturation']
        vr = self.config.hsv_ranges['value']
        mask = (
            (hsv[:, :, 0] >= hr[0]) & (hsv[:, :, 0] <= hr[1]) &
            (hsv[:, :, 1] >= sr[0]) & (hsv[:, :, 1] <= sr[1]) &
            (hsv[:, :, 2] >= vr[0]) & (hsv[:, :, 2] <= vr[1])
        )
        pixel_ratio = float(np.sum(mask)) / float(mask.size)
        return pixel_ratio >= self.config.pixel_threshold

    def lowres_hsv_precheck_ok(self, lowres_hsv: np.ndarray) -> bool:
        hr = self.config.hsv_ranges['hue']
        sr = self.config.hsv_ranges['saturation']
        vr = self.config.hsv_ranges['value']
        mask = (
            (lowres_hsv[:, :, 0] >= hr[0]) & (lowres_hsv[:, :, 0] <= hr[1]) &
            (lowres_hsv[:, :, 1] >= sr[0]) & (lowres_hsv[:, :, 1] <= sr[1]) &
            (lowres_hsv[:, :, 2] >= vr[0]) & (lowres_hsv[:, :, 2] <= vr[1])
        )
        pixel_ratio = float(np.sum(mask)) / float(mask.size)
        return pixel_ratio >= self.config.pixel_threshold

    def is_informative(self, tile: np.ndarray) -> bool:
        """Conservative, fast informativeness check based on grayscale std.

        - Computes grayscale standard deviation and requires it to be above a
          small threshold. Keeps most tiles while filtering very flat patches.
        """
        if not self.config.informativeness:
            return True
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        std = float(gray.std())
        return std >= float(self.config.min_gray_std)

    def get_available_magnifications(self, slide: OpenSlide) -> Dict[float, int]:
        try:
            base_mpp = float(slide.properties.get(
                openslide.PROPERTY_NAME_MPP_X,
                slide.properties.get('aperio.MPP', 0.25)
            ))
        except Exception:
            base_mpp = 0.25
        levels_mpp = []
        for lvl in range(slide.level_count):
            down = float(slide.level_downsamples[lvl])
            levels_mpp.append((lvl, base_mpp * down))
        available_mpps = [mpp for _, mpp in levels_mpp]
        min_mpp, max_mpp = min(available_mpps), max(available_mpps)
        mapping: Dict[float, int] = {}
        for target in self.config.magnifications:
            eps = 0.02
            if target < min_mpp * (1 - eps) or target > max_mpp * (1 + eps):
                continue
            best_level, _ = min(levels_mpp, key=lambda x: abs(x[1] - target))
            mapping[target] = best_level
        return mapping

    # -------- Candidate grid generation --------
    def iter_candidates_level0(self,
                               slide: OpenSlide,
                               lowref: Dict[str, Any],
                               tile_size_level0: int) -> Tuple[int, int]:
        """Yield top-left candidates in level-0 coordinates using the mask integral image.

        - Compute window size on mask level: ws = ceil(tile_size_level0 / mask_downsample)
        - Grid stride: stride = max(1, round(grid_stride * ws))
        - Accept if tissue fraction >= foreground_threshold
        - Optional: low-res HSV precheck on the masked window
        """
        mask = lowref['mask']
        ii = lowref['mask_integral']
        mask_down = lowref['downsample']
        Hm, Wm = mask.shape

        ws = int(np.ceil(tile_size_level0 / mask_down))
        if ws <= 0:
            return
        stride = max(1, int(round(self.config.grid_stride * ws)))

        # Bounds to ensure we can map back to level-0 without overflow
        max_xm = max(0, Wm - ws)
        max_ym = max(0, Hm - ws)

        # Collect accepted windows, then shuffle for randomness
        accepted: List[Tuple[int, int]] = []

        # Fast scan using integral image
        area_total = ws * ws
        thr_pixels = int(np.ceil(self.config.foreground_threshold * area_total))

        for ym in range(0, max_ym + 1, stride):
            # Ensure integer arithmetic for integral image indexing
            for xm in range(0, max_xm + 1, stride):
                s = _integral_sum(ii, xm, ym, ws, ws)
                if s < thr_pixels:
                    continue
                # Optional low-res HSV precheck
                if self.config.hsv_threshold and self.config.lowres_hsv_precheck and lowref.get('hsv') is not None:
                    # Slice low-res HSV window
                    hsv_win = lowref['hsv'][ym:ym+ws, xm:xm+ws]
                    if hsv_win.size == 0:
                        continue
                    if not self.lowres_hsv_precheck_ok(hsv_win):
                        continue
                accepted.append((xm, ym))

        if not accepted:
            return
        random.shuffle(accepted)

        # Map to level-0 coordinates with optional jitter to avoid visible grid
        # Compute a jitter budget so we don't step out of accepted window area
        for xm, ym in accepted:
            x0 = int(xm * mask_down)
            y0 = int(ym * mask_down)
            if self.config.grid_jitter:
                # Jitter up to a fraction of the mask window in level-0 coords
                j = max(0, int(self.config.grid_jitter_frac * ws * mask_down))
                if j > 0:
                    jx = random.randint(0, j)
                    jy = random.randint(0, j)
                    x0 = min(x0 + jx, slide.dimensions[0] - tile_size_level0)
                    y0 = min(y0 + jy, slide.dimensions[1] - tile_size_level0)

            # Clamp inside bounds
            if x0 < 0 or y0 < 0:
                continue
            if x0 + tile_size_level0 > slide.dimensions[0] or y0 + tile_size_level0 > slide.dimensions[1]:
                continue
            yield (x0, y0)

    # -------- Main extraction --------
    def extract_tiles_from_slide(self,
                                 slide_path: str,
                                 max_tiles_per_mag: int = 100,
                                 per_mag_quota: Optional[Dict[float, int]] = None):
        try:
            slide = OpenSlide(slide_path)
        except Exception as e:
            logger.error(f"Failed to open slide {slide_path}: {e}")
            return

        try:
            lowref = self.build_lowres_reference(slide)
            mag_map = self.get_available_magnifications(slide)

            # Iterate magnifications in random order
            items = list(mag_map.items())
            random.shuffle(items)
            for target_mag, level in items:
                downsample = float(slide.level_downsamples[level])
                tile_size_level0 = int(self.config.tile_size * downsample)

                # Determine quota
                target_quota = int(per_mag_quota.get(target_mag, 0)) if per_mag_quota is not None else int(max_tiles_per_mag)
                if target_quota <= 0:
                    continue

                n_extracted = 0

                # Preferred: grid-based candidates
                candidate_iter = None
                if self.config.grid_sampling:
                    candidate_iter = self.iter_candidates_level0(slide, lowref, tile_size_level0)

                if candidate_iter is not None:
                    for (x, y) in candidate_iter:
                        # Read region at target level
                        tile = slide.read_region((x, y), level, (self.config.tile_size, self.config.tile_size)).convert('RGB')
                        tile_np = np.array(tile)
                        # Final checks (order: cheap grayscale gates first, HSV last)
                        if not self.check_foreground_threshold(tile_np, lowref['otsu_thresh']):
                            continue
                        if not self.is_informative(tile_np):
                            continue
                        if not self.apply_hsv_threshold(tile_np):
                            continue
                        yield {
                            'image': tile_np,
                            'slide_path': slide_path,
                            'position': (x, y),
                            'level': level,
                            'magnification': target_mag,
                            'slide_id': Path(slide_path).stem
                        }
                        n_extracted += 1
                        if n_extracted >= target_quota:
                            break

                # Fallback: limited random attempts if grid under-samples
                if n_extracted < target_quota:
                    attempts = 0
                    max_attempts = max((target_quota - n_extracted) * 10, 200)
                    while n_extracted < target_quota and attempts < max_attempts:
                        attempts += 1
                        try:
                            x = random.randint(0, slide.dimensions[0] - tile_size_level0)
                            y = random.randint(0, slide.dimensions[1] - tile_size_level0)
                        except ValueError:
                            break
                        # Coarsest mask quick gate
                        md = lowref['downsample']
                        xm = int(x / md)
                        ym = int(y / md)
                        ws = int(np.ceil(tile_size_level0 / md))
                        if xm < 0 or ym < 0 or xm + ws > lowref['mask'].shape[1] or ym + ws > lowref['mask'].shape[0]:
                            continue
                        s = _integral_sum(lowref['mask_integral'], xm, ym, ws, ws)
                        if s < self.config.foreground_threshold * (ws * ws):
                            continue
                        tile = slide.read_region((x, y), level, (self.config.tile_size, self.config.tile_size)).convert('RGB')
                        tile_np = np.array(tile)
                        if not self.check_foreground_threshold(tile_np, lowref['otsu_thresh']):
                            continue
                        if not self.is_informative(tile_np):
                            continue
                        if not self.apply_hsv_threshold(tile_np):
                            continue
                        yield {
                            'image': tile_np,
                            'slide_path': slide_path,
                            'position': (x, y),
                            'level': level,
                            'magnification': target_mag,
                            'slide_id': Path(slide_path).stem
                        }
                        n_extracted += 1
        finally:
            try:
                slide.close()
            except Exception:
                pass


# --------- Dataset helpers (mostly as in original, with minor improvements) ---------
def find_svs_files(root_dir: str, exclude_file: Optional[str] = None) -> List[str]:
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
    logger.info(f"Updating log file with {len(all_processed_paths)} total processed slides.")
    content = "\n".join(sorted(list(all_processed_paths)))
    s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'))
    logger.info("Log file successfully updated.")


def parse_size_to_bytes(size_str: str) -> int:
    s = size_str.strip().lower()
    if not s:
        raise ValueError("Empty size string")
    try:
        if 'e' in s:
            return int(float(s))
    except ValueError:
        pass
    units = {'b': 1, 'kb': 1024, 'mb': 1024**2, 'gb': 1024**3, 'tb': 1024**4}
    num = ''
    unit = ''
    for ch in s:
        if (ch.isdigit() or ch == '.' or ch == '+'):
            num += ch
        else:
            unit += ch
    if not num:
        raise ValueError(f"Invalid size '{size_str}'")
    if not unit:
        return int(float(num))
    unit = unit.strip()
    aliases = {'k': 'kb', 'm': 'mb', 'g': 'gb', 't': 'tb'}
    unit = aliases.get(unit, unit)
    if unit not in units:
        raise ValueError(f"Unknown unit '{unit}' in size '{size_str}'")
    return int(float(num) * units[unit])


def estimate_tiles_for_target_size(target_size_bytes: int, tile_size: int, avg_bytes_per_tile: int = None) -> int:
    if avg_bytes_per_tile is None or avg_bytes_per_tile <= 0:
        avg_bytes_per_tile = tile_size * tile_size * 3
    if target_size_bytes <= 0:
        raise ValueError("target_size_bytes must be positive")
    return max(int(target_size_bytes // avg_bytes_per_tile), 1)


def _list_all_keys(s3_client: BaseClient, bucket: str, prefix: str) -> List[str]:
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


def read_cumulative_counts_from_stats(s3_client: BaseClient,
                                      bucket: str,
                                      stats_prefix: str,
                                      magnifications: List[float]) -> Dict[float, int]:
    """Aggregate cumulative counts across all runs (expensive; call sparingly)."""
    cum: Dict[float, int] = {float(m): 0 for m in magnifications}
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
            if e.response["Error"].get("Code") == "NoSuchKey":
                continue
            else:
                raise
    return cum


def read_aggregate_counts_fast(s3_client: BaseClient,
                               bucket: str,
                               prefix: str,
                               magnifications: List[float]) -> Optional[Dict[float, int]]:
    """Try to read a single aggregate counts file to avoid large prefix scans.

    Returns a dict if present, otherwise None.
    """
    key = f"{prefix}aggregate_counts.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        out: Dict[float, int] = {float(k): int(v) for k, v in data.get("cumulative_per_mag_counts", {}).items()}
        # Ensure all requested mags present
        for m in magnifications:
            out.setdefault(float(m), 0)
        logger.info("Loaded baseline counts from aggregate_counts.json")
        return out
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise


def write_aggregate_counts(s3_client: BaseClient,
                           bucket: str,
                           prefix: str,
                           cumulative_counts: Dict[float, int],
                           run_id: str) -> None:
    key = f"{prefix}aggregate_counts.json"
    payload = {
        "updated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "cumulative_per_mag_counts": {str(k): int(v) for k, v in cumulative_counts.items()},
    }
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload, indent=2))


def read_counts_for_run(s3_client: BaseClient,
                        bucket: str,
                        stats_prefix: str,
                        run_id: str,
                        magnifications: List[float]) -> Dict[float, int]:
    prefix = f"{stats_prefix}{run_id}/"
    cum: Dict[float, int] = {float(m): 0 for m in magnifications}
    keys = _list_all_keys(s3_client, bucket, prefix)
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
            if e.response["Error"].get("Code") == "NoSuchKey":
                continue
            else:
                raise
    return cum


def read_counts_for_run_local(local_stats_dir: str,
                              magnifications: List[float]) -> Dict[float, int]:
    """Aggregate counts written locally by workers (1 small file per task)."""
    out: Dict[float, int] = {float(m): 0 for m in magnifications}
    p = Path(local_stats_dir)
    if not p.exists():
        return out
    for f in p.rglob("*.json"):
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            for m_str, v in data.get("counts_by_mag", {}).items():
                try:
                    m = float(m_str)
                    if m in out:
                        out[m] += int(v)
                except Exception:
                    continue
        except Exception:
            continue
    return out


def allocate_across_magnifications(cumulative: Dict[float, int], budget: int, magnifications: List[float]) -> Dict[float, int]:
    """Water-fill budget across magnifications to equalize totals.

    Fixes edge case when all cumulative counts are equal (or ties exist):
    we should advance to the next level rather than allocating the entire
    remaining budget to the first group.
    """
    mags = list(magnifications)
    cum = np.array([int(cumulative.get(m, 0)) for m in mags], dtype=np.int64)
    M = len(mags)
    if M == 0 or budget <= 0:
        return {m: 0 for m in mags}
    order = np.argsort(cum)
    sorted_cum = cum[order].astype(np.float64)
    T = float(budget)
    levels = sorted_cum.copy()
    # Raise the lowest groups to the next distinct level while we have budget
    for i in range(M - 1):
        gap = levels[i + 1] - levels[i]
        cost = gap * (i + 1)
        # If there is no gap (equal baselines), just move to the next group
        if cost <= 0:
            continue
        if T >= cost:
            levels[: i + 1] += gap
            T -= cost
        else:
            inc = T / (i + 1)
            levels[: i + 1] += inc
            T = 0.0
            break
    # If budget remains after equalizing all groups, distribute uniformly
    if T > 0:
        inc = T / M
        levels += inc
        T = 0.0
    # Convert fractional deltas to integer allocations with remainder handling
    deltas = levels - sorted_cum
    deltas_int = np.floor(deltas).astype(np.int64)
    remainder = int(round(budget - int(deltas_int.sum())))
    if remainder > 0:
        frac = deltas - deltas_int
        idx = np.argsort(-frac)
        for j in idx[:remainder]:
            deltas_int[j] += 1
    elif remainder < 0:
        frac = deltas - deltas_int
        idx = np.argsort(frac)
        for j in idx[: (-remainder)]:
            if deltas_int[j] > 0:
                deltas_int[j] -= 1
    alloc_sorted = deltas_int
    alloc = np.zeros_like(alloc_sorted)
    alloc[order] = alloc_sorted
    return {m: int(a) for m, a in zip(mags, alloc.tolist())}


def pilot_estimate_avg_tile_bytes(slide_paths: List[str], config: TileConfig, total_samples: int = 256, max_slides: int = 16) -> Optional[int]:
    if total_samples <= 0:
        return None
    mags = list(config.magnifications)
    per_mag_needed = {m: total_samples // len(mags) for m in mags}
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
        if all(v <= 0 for v in per_mag_needed.values()):
            break
    if total_count == 0:
        logger.warning("Pilot sampling produced zero tiles; using default uncompressed estimate.")
        return None
    est = int(total_bytes // total_count)
    logger.info(f"Pilot estimated avg tile bytes: {est} from {total_count} samples.")
    return est


def compute_per_slide_mag_quotas(slide_paths: List[str],
                                 config: TileConfig,
                                 total_tiles: Optional[int],
                                 default_tiles_per_mag: Optional[int],
                                 per_mag_totals: Optional[Dict[float, int]] = None) -> Tuple[Dict[str, Dict[float, int]], Dict[float, int]]:
    slide_paths = [os.path.abspath(p) for p in slide_paths]
    per_slide_quota: Dict[str, Dict[float, int]] = {p: {} for p in slide_paths}
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
        for m in config.magnifications:
            per_mag_total[m] = int(per_mag_totals.get(m, 0))
    elif total_tiles is not None:
        num_mags = len(config.magnifications)
        if num_mags == 0:
            raise ValueError("No magnifications configured")
        base = total_tiles // num_mags
        remainder = total_tiles % num_mags
        for idx, m in enumerate(config.magnifications):
            per_mag_total[m] = base + (1 if idx < remainder else 0)
    else:
        if default_tiles_per_mag is None:
            raise ValueError("Either total_tiles, per_mag_totals, or default_tiles_per_mag must be provided")
        for m in config.magnifications:
            per_mag_total[m] = default_tiles_per_mag * len(supported_by_mag[m])
    for m in config.magnifications:
        slides = supported_by_mag[m]
        if not slides:
            continue
        total_m = per_mag_total[m]
        base = total_m // len(slides)
        rem = total_m % len(slides)
        shuffled = slides.copy()
        random.shuffle(shuffled)
        for i, p in enumerate(shuffled):
            quota = base + (1 if i < rem else 0)
            if quota > 0:
                per_slide_quota[p][m] = quota
    return per_slide_quota, per_mag_total


def create_litdata_dataset(input_dir: str,
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
                           chunk_bytes: str = "64MB",
                           num_uploaders: Optional[int] = None,
                           num_downloaders: Optional[int] = None,
                           start_method: Optional[str] = None,
                           keep_data_ordered: bool = False,
                           stats_mode: str = "local",
                           local_stats_root: Optional[str] = None,
                           max_time_seconds: Optional[int] = None):
    if not output_dir.startswith("s3://"):
        raise ValueError("--output-dir must be an s3:// URI (S3 or S3-compatible endpoint)")
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    if endpoint_url:
        logger.info(f"Using custom S3 endpoint: {endpoint_url}")
    s3_client = get_s3_client()
    parsed_url = urlparse(output_dir)
    log_bucket = parsed_url.netloc
    prefix = parsed_url.path.lstrip('/')
    if prefix and not prefix.endswith('/'): prefix += '/'
    log_key = f"{prefix}_processed_slides.log"
    stats_prefix = f"{prefix}_stats/"
    run_id = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Local stats path (if using local aggregation)
    local_stats_dir: Optional[Path] = None
    if stats_mode == "local":
        base = Path(local_stats_root) if local_stats_root else (Path.cwd() / ".tcga_local_stats")
        local_stats_dir = base / run_id
        try:
            local_stats_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning(f"Failed to create local stats dir: {local_stats_dir}")

    # Resume logic
    all_slide_paths = find_svs_files(input_dir, exclude_file)
    if not all_slide_paths:
        logger.warning("No SVS files found under input-dir; nothing to do.")
        return
    already_processed = get_processed_slides(s3_client, log_bucket, log_key)
    norm = lambda p: os.path.abspath(p)
    already_processed = set(map(norm, already_processed))
    slides_to_process = [p for p in map(norm, all_slide_paths) if p not in already_processed]
    if not slides_to_process:
        logger.info("All slides have already been processed. Nothing to do.")
        return
    logger.info(f"Starting processing for {len(slides_to_process)} remaining slides out of {len(all_slide_paths)} total.")

    if not already_processed:
        config_key = f"{prefix}config.json"
        s3_client.put_object(Bucket=log_bucket, Key=config_key, Body=json.dumps(asdict(config), indent=2))
        logger.info(f"Uploaded config to s3://{log_bucket}/{config_key}")

    if num_workers is None:
        num_workers = min(max(mp.cpu_count() - 1, 1), 64)

    # Heuristic defaults for I/O concurrency if not provided
    if num_uploaders is None:
        # uploads dominate conversion; scale with workers, cap for stability
        num_uploaders = min(32, max(8, int(num_workers // 2)))
    if num_downloaders is None:
        # conversion is mostly local reads; keep a small number
        num_downloaders = 2

    # Budgeting
    computed_total_tiles: Optional[int] = None
    per_slide_mag_quota: Optional[Dict[str, Dict[float, int]]] = None
    per_mag_total: Optional[Dict[float, int]] = None
    avg_bytes_used: Optional[int] = None

    if target_size is not None:
        _ = parse_size_to_bytes(target_size)  # validate

    if target_size is not None and avg_tile_bytes is None and pilot_sample and pilot_sample > 0:
        try:
            est = pilot_estimate_avg_tile_bytes(slides_to_process, config, total_samples=pilot_sample, max_slides=pilot_slides)
            if est is not None and est > 0:
                avg_tile_bytes = est
        except Exception as e:
            logger.warning(f"Pilot sampling failed: {e}")

    if total_tiles is not None:
        final_total_tiles = int(total_tiles)
    elif target_size is not None:
        target_size_bytes = parse_size_to_bytes(target_size)
        computed_total_tiles = estimate_tiles_for_target_size(target_size_bytes, config.tile_size, avg_tile_bytes)
        final_total_tiles = int(computed_total_tiles)
        avg_bytes_used = avg_tile_bytes or config.tile_size**2 * 3
        logger.info(f"Target size {target_size} -> approx {final_total_tiles} tiles (avg {avg_bytes_used} B/tile)")
    else:
        final_total_tiles = None

    # Initial cumulative: try fast aggregate file, fall back to full scan
    baseline_cumulative_counts = read_aggregate_counts_fast(s3_client, log_bucket, prefix, config.magnifications)
    if baseline_cumulative_counts is None:
        baseline_cumulative_counts = read_cumulative_counts_from_stats(s3_client, log_bucket, stats_prefix, config.magnifications)

    if final_total_tiles is not None:
        per_mag_total = allocate_across_magnifications(baseline_cumulative_counts, final_total_tiles, config.magnifications)
        per_slide_mag_quota, _ = compute_per_slide_mag_quotas(slides_to_process, config, None, None, per_mag_totals=per_mag_total)
        logger.info("This run per-mag allocations: " + 
                    ", ".join([f"{m} μm/px: {per_mag_total[m]}" for m in config.magnifications]))
        tiles_per_mag = None
    else:
        if tiles_per_mag is None:
            raise ValueError("Provide either --tiles-per-mag or --total-tiles or --target-size.")
        per_slide_mag_quota, per_mag_total = compute_per_slide_mag_quotas(slides_to_process, config, None, tiles_per_mag)
        logger.info(f"Using uniform tiles_per_mag={tiles_per_mag} across supported magnifications.")

    # Mode selection
    mode = "overwrite" if not already_processed else "append"

    # Batch slides
    batch_size = 2500
    batches = [slides_to_process[i:i+batch_size] for i in range(0, len(slides_to_process), batch_size)]

    # Process batches
    for i, batch in enumerate(batches):
        batch_mode = "overwrite" if (mode == "overwrite" and i == 0) else "append"
        try:
            from litdata import optimize
            task_inputs: List[Any] = []
            # Build tasks; optionally chunk quotas per slide for smoother progress
            if not per_slide_mag_quota or progress_chunk is None or progress_chunk <= 0:
                task_inputs = list(batch)
            else:
                for sp in batch:
                    abs_sp = os.path.abspath(sp)
                    qdict = per_slide_mag_quota.get(abs_sp, {})
                    if not qdict:
                        task_inputs.append(sp)
                        continue
                    for m, total in qdict.items():
                        remaining = int(total)
                        chunk_idx = 0
                        while remaining > 0:
                            take = min(progress_chunk, remaining)
                            remaining -= take
                            task_id = f"m{str(m).replace('.', 'p')}-{chunk_idx:04d}"
                            task_inputs.append((sp, {m: take}, task_id))
                            chunk_idx += 1
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
                    stats_mode=stats_mode,
                    local_stats_dir=str(local_stats_dir) if local_stats_dir else None,
                    max_time_seconds=max_time_seconds,
                ),
                inputs=task_inputs,
                output_dir=output_dir,
                num_workers=num_workers,
                chunk_bytes=chunk_bytes,
                num_downloaders=num_downloaders,
                num_uploaders=num_uploaders,
                keep_data_ordered=keep_data_ordered,
                mode=batch_mode,
                start_method=start_method,
            )
        except Exception:
            logger.exception(f"Batch {i} failed before finalization; no progress recorded for this batch.")
            raise

        newly = set(map(os.path.abspath, batch))
        fully_processed = already_processed.union(newly)
        update_processed_slides(s3_client, log_bucket, log_key, set(map(str, fully_processed)))
        already_processed = fully_processed

        # Incremental cumulative: baseline + current run (so far)
        if stats_mode == "s3":
            run_counts = read_counts_for_run(s3_client, log_bucket, stats_prefix, run_id, config.magnifications)
        elif stats_mode == "local":
            run_counts = read_counts_for_run_local(str(local_stats_dir) if local_stats_dir else "", config.magnifications)
        else:
            run_counts = {float(m): 0 for m in config.magnifications}
        cumulative_counts = {m: int(baseline_cumulative_counts.get(m, 0)) + int(run_counts.get(m, 0)) for m in config.magnifications}
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
            "cumulative_per_mag_counts": {str(k): int(v) for k, v in cumulative_counts.items()},
            "cumulative_total_tiles": cumulative_total,
            "avg_tile_bytes_estimate": avg_bytes_used,
            "run_id": run_id,
        }
        metadata_key = f"{prefix}metadata.json"
        s3_client.put_object(Bucket=log_bucket, Key=metadata_key, Body=json.dumps(metadata, indent=2))
        # Also refresh aggregate counts for fast resume
        try:
            write_aggregate_counts(s3_client, log_bucket, prefix, cumulative_counts, run_id)
        except Exception:
            logger.warning("Failed to update aggregate_counts.json (non-fatal)")
        logger.info(f"Batch {i} committed. Processed slides so far: {len(fully_processed)}")

    logger.info("Processing finished.")


# --------- Worker entry for litdata.optimize ---------
def _extract_tiles_fn(item: Any,
                      config: TileConfig,
                      tiles_per_mag: int = None,
                      per_slide_mag_quota: Dict[str, Dict[float, int]] = None,
                      stats_bucket: Optional[str] = None,
                      stats_prefix: Optional[str] = None,
                      run_id: Optional[str] = None,
                      stats_mode: str = "s3",
                      local_stats_dir: Optional[str] = None,
                      max_time_seconds: Optional[int] = None):
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
    quota_map = None
    if task_quota:
        quota_map = task_quota
    elif per_slide_mag_quota is not None:
        abs_path = os.path.abspath(slide_path)
        quota_map = per_slide_mag_quota.get(abs_path)

    counts_by_mag: Dict[float, int] = {}

    # Per-task timeout handling
    use_alarm = False
    if max_time_seconds is not None and max_time_seconds > 0:
        if hasattr(signal, "SIGALRM"):
            use_alarm = True
            def _timeout_handler(signum, frame):
                raise TimeoutError("worker task exceeded max_time_seconds")
            prev_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            try:
                signal.alarm(int(max_time_seconds))
            except Exception:
                use_alarm = False
        deadline = time.monotonic() + float(max_time_seconds)
    else:
        deadline = None

    try:
        for t in extractor.extract_tiles_from_slide(slide_path, max_tiles_per_mag=tiles_per_mag, per_mag_quota=quota_map):
            if not use_alarm and deadline is not None and time.monotonic() >= deadline:
                logger.warning(f"Timeout reached while processing {slide_path}; stopping this task with partial results.")
                break
            m = float(t.get("magnification", -1))
            if m >= 0:
                counts_by_mag[m] = counts_by_mag.get(m, 0) + 1
            t["image"] = Image.fromarray(t["image"])  # LitData serializes PIL out of the box
            yield t
    except TimeoutError:
        logger.warning(f"Max time exceeded for slide: {slide_path}; emitted {sum(counts_by_mag.values())} tiles so far.")
    finally:
        if max_time_seconds and use_alarm:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)
            except Exception:
                pass

    # Per-task stats (slide or chunk)
    try:
        slide_id = Path(slide_path).stem
        import hashlib
        slide_abs = os.path.abspath(slide_path)
        sid = hashlib.sha1(slide_abs.encode("utf-8")).hexdigest()[:16]
        payload = {
            "slide_id": slide_id,
            "slide_path": os.path.abspath(slide_path),
            "run_id": run_id,
            "counts_by_mag": {str(k): int(v) for k, v in counts_by_mag.items()},
            "total": int(sum(counts_by_mag.values())),
        }
        if stats_mode == "s3" and stats_bucket and stats_prefix is not None:
            s3c = get_s3_client()
            if task_id:
                stats_key = f"{stats_prefix}{run_id}/{slide_id}_{sid}_{task_id}.json"
            else:
                stats_key = f"{stats_prefix}{run_id}/{slide_id}_{sid}.json"
            s3c.put_object(Bucket=stats_bucket, Key=stats_key, Body=json.dumps(payload).encode("utf-8"))
        elif stats_mode == "local" and local_stats_dir:
            try:
                Path(local_stats_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            if task_id:
                fname = f"{slide_id}_{sid}_{task_id}.json"
            else:
                fname = f"{slide_id}_{sid}.json"
            with open(str(Path(local_stats_dir) / fname), "w") as fh:
                json.dump(payload, fh)
    except Exception as e:
        logger.warning(f"Failed to record stats for {slide_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Revised: Convert TCGA SVS dataset to LitData (faster mask & sampling).")
    parser.add_argument("--input-dir", type=str, default="/data/TCGA", help="Input directory containing TCGA SVS files")
    parser.add_argument("--output-dir", type=str, required=False, default="s3://tcga-13tb-litdata", help="S3 output directory (s3://bucket/prefix)")

    group_budget = parser.add_mutually_exclusive_group()
    group_budget.add_argument("--tiles-per-mag", type=int, help="Uniform tiles per supported magnification per slide")
    group_budget.add_argument("--total-tiles", type=int, help="Total desired tiles across dataset (balanced across mags)")
    group_budget.add_argument("--target-size", type=str, default=None, help="Approx target dataset size (e.g., 200GB, 50M, 1.5TB)")
    parser.add_argument("--avg-tile-bytes", type=int, default=None, help="Avg bytes/tile when using --target-size; default estimates")

    parser.add_argument("--num-workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--exclude-file", type=str, default="baddata.txt", help="File with paths of SVS files to exclude")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels at read level")

    # Revised mask and sampling
    parser.add_argument("--mask-level", type=str, default="coarsest", help="Mask level: 'coarsest' or integer level index")
    parser.add_argument("--grid-sampling", action="store_true", default=True, help="Enable grid-based candidate sampling")
    parser.add_argument("--no-grid-sampling", dest="grid_sampling", action="store_false")
    parser.add_argument("--grid-stride", type=float, default=1.0, help="Stride factor relative to mask-window size (e.g., 1.0 or 0.5)")
    parser.add_argument("--grid-no-jitter", dest="grid_jitter", action="store_false", help="Disable jitter inside accepted windows")
    parser.add_argument("--lowres-hsv-precheck", action="store_true", default=True, help="Enable low-res HSV precheck on mask")
    parser.add_argument("--no-lowres-hsv-precheck", dest="lowres_hsv_precheck", action="store_false")
    parser.add_argument("--no-hsv-threshold", action="store_true", help="Disable HSV thresholding entirely (both low/high res)")
    # Informativeness gate (fast and conservative)
    parser.add_argument("--no-informativeness", dest="informativeness", action="store_false", help="Disable informativeness gate (grayscale std)")
    parser.set_defaults(informativeness=True)
    parser.add_argument("--min-gray-std", type=float, default=5.0, help="Minimum grayscale std to accept a tile (conservative)")

    # Pilot sampling
    parser.add_argument("--pilot-sample", type=int, default=0, help="Tiles to sample for estimating avg bytes (0 disables)")
    parser.add_argument("--pilot-slides", type=int, default=4, help="Max slides to touch for pilot estimate")

    # litdata.optimize knobs
    parser.add_argument("--chunk-bytes", type=str, default="64MB", help="Chunk size for litdata (e.g., 64MB)")
    parser.add_argument("--num-uploaders", type=int, default=None, help="Concurrent upload workers for litdata")
    parser.add_argument("--num-downloaders", type=int, default=None, help="Concurrent download workers for litdata")
    parser.add_argument("--start-method", type=str, default=None, help="Multiprocessing start method: fork, forkserver, or spawn")
    parser.add_argument("--keep-data-ordered", action="store_true", default=False, help="Keep data ordered in litdata (may stall at end)")

    parser.add_argument("--progress-chunk", type=int, default=1024, help="Tiles per task chunk to improve progress responsiveness")

    # Stats/telemetry controls (reduce S3 chatter)
    parser.add_argument("--stats-mode", choices=["s3", "local", "none"], default="local", help="Where to write per-task stats (local reduces S3 traffic)")
    parser.add_argument("--local-stats-root", type=str, default=None, help="Local folder for per-task stats when --stats-mode=local")

    # Task timeout to avoid stalls on the last item
    parser.add_argument("--max-time-seconds", type=int, default=600, help="Max seconds per task/slide before skipping (0 disables)")

    # Ensure grid jitter default to True unless explicitly disabled
    parser.set_defaults(grid_jitter=True)
    args = parser.parse_args()

    # Apply OpenCV/OMP threading caps early (inherit to workers)
    try:
        cv2.setNumThreads(max(int(os.getenv("OPENCV_NUM_THREADS", "1")), 1))
    except Exception:
        pass
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"

    config = TileConfig(
        tile_size=args.tile_size,
        hsv_threshold=not args.no_hsv_threshold,
        mask_level=args.mask_level,
        grid_sampling=args.grid_sampling,
        grid_stride=args.grid_stride,
        grid_jitter=bool(args.grid_jitter),
        lowres_hsv_precheck=args.lowres_hsv_precheck,
        informativeness=args.informativeness,
        min_gray_std=args.min_gray_std,
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
        chunk_bytes=args.chunk_bytes,
        num_uploaders=args.num_uploaders,
        num_downloaders=args.num_downloaders,
        start_method=args.start_method,
        keep_data_ordered=args.keep_data_ordered,
        stats_mode=args.stats_mode,
        local_stats_root=args.local_stats_root,
        max_time_seconds=args.max_time_seconds,
    )


if __name__ == "__main__":
    main()
