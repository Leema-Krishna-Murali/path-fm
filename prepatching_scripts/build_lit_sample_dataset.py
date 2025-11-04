#!/usr/bin/env python3
"""
Generate a LitData optimized dataset of TCGA patches described in sample_dataset_30.txt.

Each line in the input text file is expected to follow the format emitted by
create_sample_dataset_txt.py:

    /path/to/slide.svs <x> <y> <level>

The script opens every referenced slide once, extracts a 224x224 RGB patch at the
requested coordinates/level, and streams the samples into LitData .bin shards
(~256 MiB by default). It supports PNG/JPEG/raw outputs, task chunking, and
parallelization knobs tuned for high-core-count nodes. Results land in
/data/TCGA_lit_sample30/ unless overridden.
"""

from __future__ import annotations

import argparse
import atexit
import logging
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import os
import random
from pathlib import Path
from functools import partial
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from openslide import OpenSlide
from litdata import optimize


@dataclass(frozen=True)
class PatchSpec:
    slide_path: str
    x: int
    y: int
    level: int


try:
    _env_cache_limit = int(os.environ.get("LIT_MAX_OPEN_SLIDES", "16"))
except (TypeError, ValueError):
    _env_cache_limit = 16
DEFAULT_MAX_OPEN_SLIDES = max(1, _env_cache_limit)
MAX_OPEN_SLIDES = DEFAULT_MAX_OPEN_SLIDES
SLIDE_CACHE: "OrderedDict[str, OpenSlide]" = OrderedDict()


def prune_slide_cache() -> None:
    """Ensure the slide cache does not exceed MAX_OPEN_SLIDES."""
    if MAX_OPEN_SLIDES < 1:
        return
    while len(SLIDE_CACHE) > MAX_OPEN_SLIDES:
        old_path, old_slide = SLIDE_CACHE.popitem(last=False)
        try:
            old_slide.close()
        except Exception:
            logging.exception("Failed to close slide %s during cache prune.", old_path)


def set_max_open_slides(limit: int) -> None:
    """Update cache limit and prune if needed."""
    global MAX_OPEN_SLIDES
    limit = max(1, int(limit))
    if limit == MAX_OPEN_SLIDES:
        return
    MAX_OPEN_SLIDES = limit
    prune_slide_cache()


def ensure_cache_limit(limit: int) -> None:
    """Idempotent helper invoked inside workers."""
    try:
        set_max_open_slides(limit)
    except Exception:
        logging.exception("Unexpected failure updating slide cache limit.")


@atexit.register
def close_all_slides() -> None:
    """Close any slides left in the cache when the process exits."""
    while SLIDE_CACHE:
        path, slide = SLIDE_CACHE.popitem(last=False)
        try:
            slide.close()
        except Exception:
            logging.exception("Failed to close slide %s during interpreter shutdown.", path)


def get_slide(path: str) -> OpenSlide:
    slide = SLIDE_CACHE.get(path)
    if slide is None:
        slide = OpenSlide(path)
        SLIDE_CACHE[path] = slide
        SLIDE_CACHE.move_to_end(path, last=True)
        prune_slide_cache()
    else:
        SLIDE_CACHE.move_to_end(path, last=True)
    return slide


def parse_specs(spec_file: Path) -> List[PatchSpec]:
    specs: List[PatchSpec] = []
    with spec_file.open("r", encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid spec line {idx}: {line}")
            slide_path = parts[0]
            x, y, level = map(int, parts[1:4])
            specs.append(PatchSpec(slide_path, x, y, level))
    if not specs:
        raise ValueError(f"No patch specifications found in {spec_file}")
    logging.info("Parsed %d patch specifications from %s.", len(specs), spec_file)
    return specs


def group_by_slide(specs: Sequence[PatchSpec]) -> List[Tuple[str, List[PatchSpec]]]:
    """Group patch specs so each slide is opened only once."""
    grouped: "OrderedDict[str, List[PatchSpec]]" = OrderedDict()
    for spec in specs:
        grouped.setdefault(spec.slide_path, []).append(spec)
    return [(slide, patches) for slide, patches in grouped.items()]


def build_tasks(
    groups: Sequence[Tuple[str, Sequence[PatchSpec]]],
    max_patches_per_task: int,
) -> List[Tuple[str, List[PatchSpec], str]]:
    """Chunk grouped specs to improve parallel balance on large nodes."""
    tasks: List[Tuple[str, List[PatchSpec], str]] = []
    idx = 0
    for slide, patches in groups:
        chunk = len(patches) if max_patches_per_task <= 0 else max_patches_per_task
        for start in range(0, len(patches), chunk):
            chunk_specs = list(patches[start : start + chunk])
            task_id = f"{idx:08d}"
            tasks.append((slide, chunk_specs, task_id))
            idx += 1
    return tasks


def extract_patches(
    task: Tuple[str, Sequence[PatchSpec], str],
    tile_size: int,
    encoding: str,
    progress_dir: str,
    max_open_slides: int,
) -> Iterator[dict]:
    """Worker function invoked by litdata.optimize that yields patch samples."""
    slide_path, specs, task_id = task
    ensure_cache_limit(max_open_slides)
    slide = get_slide(slide_path)
    buf = BytesIO() if encoding in ("png", "jpeg") else None
    progress_path = Path(progress_dir) / f"{task_id}.done"

    for spec in specs:
        if spec.level < 0 or spec.level >= slide.level_count:
            raise ValueError(f"Level {spec.level} invalid for {slide_path}")
        region = slide.read_region(
            (spec.x, spec.y), spec.level, (tile_size, tile_size)
        ).convert("RGB")
        sample: dict = {
            "slide_path": slide_path,
            "x": spec.x,
            "y": spec.y,
            "level": spec.level,
            "tile_size": tile_size,
        }
        if encoding == "png":
            buf.seek(0)
            buf.truncate(0)
            region.save(buf, format="PNG", optimize=True)
            sample["image_bytes"] = buf.getvalue()
        elif encoding == "jpeg":
            buf.seek(0)
            buf.truncate(0)
            region.save(buf, format="JPEG", quality=95, optimize=True)
            sample["image_bytes"] = buf.getvalue()
        else:
            arr = np.asarray(region, dtype=np.uint8)
            sample["image_bytes"] = arr.tobytes()
            sample["image_shape"] = arr.shape
            sample["image_dtype"] = str(arr.dtype)
        sample["level_downsample"] = float(slide.level_downsamples[spec.level])
        yield sample
    progress_path.write_text("1\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LitData optimized dataset of TCGA patches listed in a text file."
        )
    )
    parser.add_argument(
        "--spec-file",
        type=Path,
        default=Path("sample_dataset_30.txt"),
        help="Path to the text file produced by create_sample_dataset_txt.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/TCGA_lit_sample30/"),
        help="Directory where LitData shards will be written (default: /data/TCGA_lit_sample30/).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Square tile size (pixels) to read from each slide.",
    )
    parser.add_argument(
        "--chunk-mb",
        type=float,
        default=256.0,
        help="Target shard size for litdata.optimize expressed in MiB.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes for litdata.optimize (default: auto based on CPUs).",
    )
    parser.add_argument(
        "--start-method",
        choices=("fork", "spawn", "forkserver"),
        default=None,
        help="Multiprocessing start method for litdata workers (default: library default).",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="Whether to overwrite or append to an existing LitData dataset.",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep samples in the same order as listed in the spec file.",
    )
    parser.add_argument(
        "--encoding",
        choices=("png", "jpeg", "raw"),
        default="png",
        help="Patch serialization format (png: smaller, jpeg/raw: faster).",
    )
    parser.add_argument(
        "--task-chunk-size",
        type=int,
        default=512,
        help="Max patches per task to balance multi-process workloads (<=0 disables chunking).",
    )
    parser.add_argument(
        "--shuffle-tasks",
        action="store_true",
        help="Shuffle task order (ignored when --keep-order is set).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that completed in a previous run using progress markers.",
    )
    parser.add_argument(
        "--max-open-slides",
        type=int,
        default=DEFAULT_MAX_OPEN_SLIDES,
        help=(
            "Maximum slides cached per worker before least-recently-used eviction; "
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    ensure_cache_limit(args.max_open_slides)

    if not args.spec_file.exists():
        raise SystemExit(f"Spec file not found: {args.spec_file}")

    specs = parse_specs(args.spec_file)
    if not specs:
        raise SystemExit("No valid patch specs to process. Aborting.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and args.mode == "overwrite":
        if any(args.output_dir.glob("*.bin")):
            raise ValueError(
                f"--resume cannot be combined with --mode overwrite when {args.output_dir} already contains shards. "
                "Use --mode append or clear the directory first."
            )

    grouped = group_by_slide(specs)
    tasks = build_tasks(grouped, args.task_chunk_size)
    progress_dir = args.output_dir / ".resume"
    progress_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for marker in progress_dir.iterdir():
            if marker.is_file():
                marker.unlink()
    done_ids = {marker.stem for marker in progress_dir.iterdir() if marker.is_file()}
    if done_ids:
        tasks = [task for task in tasks if task[2] not in done_ids]
    if args.shuffle_tasks and not args.keep_order:
        random.shuffle(tasks)
    if not tasks:
        logging.info("No tasks remaining after resume check.")
        return

    logging.info(
        "Prepared %d task(s) across %d slide(s) [chunk_size=%d, pending_patches=%d].",
        len(tasks),
        len(grouped),
        args.task_chunk_size,
        sum(len(task[1]) for task in tasks),
    )
    if done_ids:
        logging.info("Skipping %d previously completed task(s).", len(done_ids))

    chunk_bytes = int(max(args.chunk_mb, 1) * 1024 * 1024)
    cpu_count = os.cpu_count() or 1
    num_workers = args.num_workers
    if num_workers is None: num_workers = max(1, min(cpu_count, 64))
    logging.info(
        "Optimizing dataset into %s (chunk_bytes=%d, num_workers=%d, mode=%s, encoding=%s).",
        args.output_dir,
        chunk_bytes,
        num_workers,
        args.mode,
        args.encoding,
    )

    worker_fn = partial(
        extract_patches,
        tile_size=args.tile_size,
        encoding=args.encoding,
        progress_dir=str(progress_dir),
        max_open_slides=args.max_open_slides,
    )

    opt_kwargs = dict(
        fn=worker_fn,
        inputs=tasks,
        output_dir=str(args.output_dir),
        chunk_bytes=chunk_bytes,
        num_workers=num_workers,
        keep_data_ordered=args.keep_order,
        mode=args.mode,
    )
    if args.start_method is not None:
        opt_kwargs["start_method"] = args.start_method

    optimize(**opt_kwargs)
    logging.info("Dataset build complete.")


if __name__ == "__main__":
    main()

# python3 build_lit_sample_dataset.py \
#   --spec-file sample_dataset_30.txt \
#   --output-dir /data/TCGA_lit_sample30/ \
#   --encoding jpeg \
#   --shuffle-tasks \
#   --num-workers 16 \
#   --max-open-slides 1 \
#   --start-method spawn \
#   --mode overwrite \
#   --resume

# PS: Always set max-open-slides to 1 because you'll go out of memory otherwise;