#!/usr/bin/env python3
import argparse, os, sys, json, time, shutil, subprocess, uuid
from pathlib import Path
from typing import List, Optional
from subprocess import DEVNULL, PIPE
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# pip install openslide-python zarr numcodecs pillow
from openslide import OpenSlide
import zarr
from numcodecs import Blosc
from PIL import Image  # noqa: F401  (PIL used implicitly by OpenSlide conversion)

# ----------------------------- utils -----------------------------

def human(n: int) -> str:
    for unit in ['B','KiB','MiB','GiB','TiB','PiB']:
        if abs(n) < 1024: return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} EiB"

def bytes_from_tib(tib: float) -> int:
    # binary TiB
    return int(tib * (1024 ** 4))

def dir_size_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p, followlinks=False):
        for f in files:
            try:
                total += (Path(root)/f).stat().st_size
            except FileNotFoundError:
                pass
    return total

def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free

def list_svs(root: Path, exclude_file: Optional[Path]) -> List[Path]:
    # optional exclude list – one path per line (exact match or suffix match)
    excludes = set()
    if exclude_file and exclude_file.exists():
        for line in exclude_file.read_text().splitlines():
            s = line.strip()
            if s:
                excludes.add(s)
    svs = []
    for p in root.rglob("*.svs"):
        sp = str(p)
        if sp in excludes or any(sp.endswith(ex) for ex in excludes):
            continue
        svs.append(p)
    svs.sort()
    return svs

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2))

def run(cmd: List[str]) -> int:
    print(" ", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode

def rclone_copy_and_check(local_dir: Path, remote_prefix: str, retries: int = 3, transfers: int = 16) -> bool:
    """
    remote_prefix example: 'r2:tcga-zarr/TCGA' or 'sophont:tcga-zarr'
    """
    dest = f"{remote_prefix}/{local_dir.name}"
    for attempt in range(1, retries+1):
        print(f"[rclone] copy attempt {attempt}/{retries}: {local_dir} -> {dest}", flush=True)
        rc = run([
            "rclone", "copy", "--fast-list",
            "--transfers", str(transfers), "--checkers", "64",
            "--s3-chunk-size", "64M", "--s3-upload-concurrency", "8",
            str(local_dir), dest
        ])
        if rc != 0:
            print(f"[rclone] copy failed (rc={rc}); will retry", flush=True)
            continue
        print(f"[rclone] verifying with rclone check…", flush=True)
        rc2 = run(["rclone", "check", "--one-way", str(local_dir), dest])
        if rc2 == 0:
            print("[rclone] check OK", flush=True)
            return True
        else:
            print(f"[rclone] check failed (rc={rc2}); will retry copy", flush=True)
    return False

def append_state(state_path: Path, **record):
    with state_path.open("a") as f:
        f.write(json.dumps({**record, "ts": time.time()}) + "\n")

def load_uploaded_from_state(state_path: Path) -> set[str]:
    done = set()
    if state_path.exists():
        for line in state_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") in ("uploaded", "deleted_local"):
                s = rec.get("slide")
                if s:
                    done.add(s)
    return done

def list_remote_zarr_roots(remote_prefix: str) -> set[str]:
    """
    Return set of '<name>.zarr' directories that already exist under remote_prefix.
    Single call instead of per-slide probes.
    """
    proc = subprocess.run(
        ["rclone", "lsf", "--dirs-only", "--recursive", remote_prefix],
        stdout=PIPE, stderr=DEVNULL, text=True
    )
    if proc.returncode != 0:
        return set()
    names = set()
    for line in proc.stdout.splitlines():
        line = line.strip().rstrip("/")
        if line.endswith(".zarr"):
            names.add(line.split("/")[-1])  # '<slide>.zarr'
    return names

# ------------------------ conversion logic ------------------------

def convert_svs_to_zarr(
    src: Path,
    dst_root: Path,
    tile: int = 512,
    compressor_name: str = "zstd",
    clevel: int = 5,
    write_native_pyramid: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Convert a single SVS to a simple multiscale Zarr store:
      root/
        level0 (H x W x 3, uint8, chunks tile x tile x 3)
        level1 ...
        ...
        _complete.json
        _meta_source.json
    """
    name = src.stem
    out_dir = dst_root / f"{name}.zarr"
    tmp_dir = dst_root / f".{name}.zarr.tmp"

    if out_dir.exists() and (out_dir / "_complete.json").exists() and not overwrite:
        print(f"[skip] already complete: {out_dir}", flush=True)
        return out_dir

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)

    meta = {"source": str(src), "created": time.time(), "tile": tile}
    write_json(tmp_dir / "_meta_source.json", meta)

    comp = Blosc(cname=compressor_name, clevel=int(clevel), shuffle=Blosc.BITSHUFFLE)
    store = zarr.DirectoryStore(str(tmp_dir))
    root = zarr.group(store=store, overwrite=True)

    slide = OpenSlide(str(src))
    try:
        level_indices = list(range(slide.level_count)) if write_native_pyramid else [0]

        for L in level_indices:
            (w, h) = slide.level_dimensions[L]
            ds = root.create_dataset(
                f"level{L}", shape=(h, w, 3),
                chunks=(min(tile,h), min(tile,w), 3),
                dtype="uint8", compressor=comp
            )
            down = float(slide.level_downsamples[L])
            # Iterate tile grid
            for y0 in range(0, h, tile):
                hh = min(tile, h - y0)
                for x0 in range(0, w, tile):
                    ww = min(tile, w - x0)
                    # read_region expects BASE (level-0) coordinates; location must be scaled by `down`
                    base_x = int(x0 * down)
                    base_y = int(y0 * down)
                    img = slide.read_region((base_x, base_y), L, (ww, hh)).convert("RGB")
                    arr = np.asarray(img)
                    ds[y0:y0+hh, x0:x0+ww, :] = arr
    finally:
        slide.close()

    write_json(tmp_dir / "_complete.json", {"complete": True, "levels": level_indices})
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return out_dir

# -------------------- top-level worker (picklable) --------------------

def worker_one(svs_str: str,
               out_root_str: str,
               tile: int,
               compressor: str,
               clevel: int,
               remote_prefix: str) -> list[dict]:
    """
    Run a full per-slide pipeline in a subprocess:
      convert -> rclone copy -> rclone check -> delete local (on success)
    Returns a list of event dicts for logging in the parent.
    """
    svs = Path(svs_str)
    out_root = Path(out_root_str)
    events = []
    try:
        print(f"[{svs.stem}] convert start", flush=True)
        zarr_dir = convert_svs_to_zarr(
            src=svs, dst_root=out_root, tile=tile,
            compressor_name=compressor, clevel=clevel, write_native_pyramid=True
        )
        size = dir_size_bytes(zarr_dir)
        print(f"[{svs.stem}] convert done → {human(size)}; uploading…", flush=True)
        events.append({"event":"converted","slide":str(svs),"zarr":str(zarr_dir),"size":size})
        ok = rclone_copy_and_check(zarr_dir, remote_prefix)
        if ok:
            print(f"[{svs.stem}] upload+check OK; deleting local", flush=True)
            events.append({"event":"uploaded","slide":str(svs),"zarr":str(zarr_dir),"size":size})
            shutil.rmtree(zarr_dir, ignore_errors=True)
            events.append({"event":"deleted_local","slide":str(svs),"zarr":str(zarr_dir)})
        else:
            print(f"[{svs.stem}] upload/check FAILED; keeping local copy", flush=True)
            events.append({"event":"upload_failed","slide":str(svs),"zarr":str(zarr_dir),"size":size})
    except Exception as e:
        print(f"[{svs.stem}] ERROR: {e}", flush=True)
        events.append({"event":"error","slide":str(svs),"error":str(e)})
    return events

# --------------------------- main driver ---------------------------

def process(
    svs_paths: List[Path],
    out_root: Path,
    remote_prefix: str,
    max_batch_bytes: int,
    min_free_gib: float,
    per_slide_upload: bool,
    tile: int,
    compressor: str,
    clevel: int,
    exclude_file: Optional[Path],
    workers: int = 1,
):
    ensure_dir(out_root)
    state_path = out_root / "state.jsonl"
    batch: List[Path] = []
    batch_bytes = 0

    print(f"[plan] {len(svs_paths)} SVS files to process", flush=True)

    # Parallel per-slide pipeline
    if per_slide_upload and workers > 1:
        print(f"[per-slide] parallel mode with {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(worker_one, str(svs), str(out_root), tile, compressor, clevel, remote_prefix)
                for svs in svs_paths
            ]
            for idx, fut in enumerate(as_completed(futs), 1):
                evs = fut.result()
                for ev in evs:
                    append_state(state_path, **ev)
                # brief progress on completion of a slide
                last = evs[-1] if evs else {}
                slide_name = Path(last.get('slide','?')).stem if last else '?'
                print(f"[done {idx}/{len(futs)}] {slide_name} -> {last.get('event','?')}", flush=True)
        return

    # Sequential fallback (per-slide or batch)
    for i, svs in enumerate(svs_paths, 1):
        print(f"[{i}/{len(svs_paths)}] Converting: {svs}", flush=True)
        zarr_dir = convert_svs_to_zarr(
            src=svs, dst_root=out_root, tile=tile,
            compressor_name=compressor, clevel=clevel, write_native_pyramid=True
        )
        size = dir_size_bytes(zarr_dir)
        print(f"  -> wrote {zarr_dir} ({human(size)})", flush=True)
        append_state(state_path, slide=str(svs), zarr=str(zarr_dir), event="converted", size=size)

        if per_slide_upload:
            ok = rclone_copy_and_check(zarr_dir, remote_prefix)
            if ok:
                append_state(state_path, slide=str(svs), zarr=str(zarr_dir), event="uploaded", size=size)
                shutil.rmtree(zarr_dir, ignore_errors=True)
                append_state(state_path, slide=str(svs), zarr=str(zarr_dir), event="deleted_local")
            else:
                print("[error] Upload failed for slide; leaving local data in place. You can rerun.", flush=True)
            continue

        # batch mode
        batch.append(zarr_dir)
        batch_bytes += size
        fb = free_bytes(out_root)
        print(f"  [disk] free={human(fb)}   batch_now={human(batch_bytes)} / cap={human(max_batch_bytes)}", flush=True)

        if batch_bytes >= max_batch_bytes or fb < int(min_free_gib * (1024 ** 3)):
            batch_id = uuid.uuid4().hex[:8]
            print(f"[upload] Triggering batch upload ({batch_id}) of {len(batch)} stores, total {human(batch_bytes)}", flush=True)
            for p in batch:
                ok = rclone_copy_and_check(p, remote_prefix)
                if ok:
                    append_state(state_path, slide=None, zarr=str(p), event="uploaded")
                    shutil.rmtree(p, ignore_errors=True)
                    append_state(state_path, slide=None, zarr=str(p), event="deleted_local")
                else:
                    print(f"[warn] Upload failed for {p}; keeping it locally for retry.", flush=True)
            batch.clear()
            batch_bytes = 0

    # final flush for batch mode
    if not per_slide_upload and batch:
        print(f"[upload] Final batch of {len(batch)} stores", flush=True)
        for p in batch:
            ok = rclone_copy_and_check(p, remote_prefix)
            if ok:
                append_state(state_path, slide=None, zarr=str(p), event="uploaded")
                shutil.rmtree(p, ignore_errors=True)
                append_state(state_path, slide=None, zarr=str(p), event="deleted_local")
            else:
                print(f"[warn] Upload failed for {p}; keeping it locally for retry.", flush=True)

def main():
    ap = argparse.ArgumentParser(description="Convert SVS to Zarr in space-safe batches and push to R2")
    ap.add_argument("--svs-root", default="/data/TCGA", type=Path)
    ap.add_argument("--zarr-root", default="/data/TCGA_zarr", type=Path)
    ap.add_argument("--r2-remote-prefix", required=True,
                    help="rclone remote:bucket/prefix (e.g. r2:tcga-zarr/TCGA or sophont:tcga-zarr)")
    ap.add_argument("--batch-tib", type=float, default=2.0,
                    help="Max local batch size before upload (TiB, binary). Ignored in --per-slide mode.")
    ap.add_argument("--min-free-gib", type=float, default=200.0,
                    help="If free space drops below this, force an upload (GiB).")
    ap.add_argument("--per-slide", action="store_true",
                    help="Upload & delete each slide right after conversion (lowest space usage).")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--compressor", choices=["zstd","lz4","zlib"], default="zstd")
    ap.add_argument("--clevel", type=int, default=5)
    ap.add_argument("--exclude-file", type=Path, default=None,
                    help="Optional list of paths to skip (one per line).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel slides to process when --per-slide is set.")
    ap.add_argument("--skip-if-remote-exists", action="store_true",
                    help="Skip conversion if <remote>/<slide>.zarr already exists on R2.")
    args = ap.parse_args()

    svs_paths = list_svs(args.svs_root, args.exclude_file)
    if not svs_paths:
        print("No SVS files found. Check --svs-root.", file=sys.stderr)
        sys.exit(1)

    # Resume: drop slides we already uploaded previously
    state_path = args.zarr_root / "state.jsonl"
    already_done = load_uploaded_from_state(state_path)
    if already_done:
        before = len(svs_paths)
        svs_paths = [p for p in svs_paths if str(p) not in already_done]
        print(f"[resume] skipped {before - len(svs_paths)} slides from state.jsonl", flush=True)

    # Optional: skip if remote already has the .zarr (single scan; no per-slide calls)
    if args.skip_if_remote_exists:
        print("[resume] scanning remote once for existing *.zarr …", flush=True)
        existing = list_remote_zarr_roots(args.r2_remote_prefix)
        keep, skipped = [], 0
        for p in svs_paths:
            if f"{p.stem}.zarr" in existing:
                skipped += 1
                append_state(state_path, slide=str(p), event="skipped_remote_exists")
            else:
                keep.append(p)
        svs_paths = keep
        print(f"[resume] skipped {skipped} slides present on remote", flush=True)

    process(
        svs_paths=svs_paths,
        out_root=args.zarr_root,
        remote_prefix=args.r2_remote_prefix,
        max_batch_bytes=bytes_from_tib(args.batch_tib),
        min_free_gib=args.min_free_gib,
        per_slide_upload=args.per_slide,
        tile=args.tile,
        compressor=args.compressor,
        clevel=args.clevel,
        exclude_file=args.exclude_file,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()

# Example:
# python svs_to_zarr_r2.py \
#   --svs-root /data/TCGA \
#   --zarr-root /data/TCGA_zarr \
#   --r2-remote-prefix sophont:tcga-zarr \
#   --exclude-file baddata.txt \
#   --per-slide \
#   --tile 512 \
#   --workers 4 \
#   --skip-if-remote-exists
