#!/usr/bin/env python3
import argparse, os, json, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from openslide import OpenSlide, OpenSlideError
import zarr
from numcodecs import Blosc

BADLIST_DEFAULT = Path(__file__).resolve().parents[1] / "dinov2" / "data" / "datasets" / "WSILists" / "baddata.txt"

def bytes_from_gib(gib: float) -> int:
    return int(gib * (1024**3))  # 1 GiB

def list_svs(src_root: Path, badlist_path: Path | None):
    svs = sorted(src_root.rglob("*.svs"))
    bad = set()
    if badlist_path and badlist_path.exists():
        for line in badlist_path.read_text().splitlines():
            s = line.strip()
            if s:
                bad.add(s)
    # skip any path that contains a bad token
    return [p for p in svs if not any(b in str(p) for b in bad)]

def pick_subset_until_budget(paths, budget_bytes: int):
    # sort largest-first to minimize file-count when testing
    sized = [(p, p.stat().st_size) for p in paths]
    sized.sort(key=lambda t: t[1], reverse=True)
    subset, acc = [], 0
    for p, sz in sized:
        if acc >= budget_bytes:
            break
        subset.append(p)
        acc += sz
    return subset, acc

def _write_attrs(root, slide):
    props = slide.properties
    # level_downsamples may be tuple of str/float; normalize to float
    lds = [float(d) for d in slide.level_downsamples]
    ldim = [[int(w), int(h)] for (w,h) in slide.level_dimensions]
    mpp_x = props.get("openslide.mpp-x")
    mpp_y = props.get("openslide.mpp-y")
    try:
        mpp_x = float(mpp_x) if mpp_x is not None else None
    except: mpp_x = None
    try:
        mpp_y = float(mpp_y) if mpp_y is not None else None
    except: mpp_y = None
    root.attrs.update({
        "spec": "simple-wsi-zarr-v0",
        "axes": "yxc",
        "vendor": props.get("openslide.vendor", ""),
        "level_count": slide.level_count,
        "level_downsamples": lds,
        "level_dimensions": ldim,  # [[w,h], ...] in OpenSlide order
        "mpp": {"x": mpp_x, "y": mpp_y},
    })

def convert_one(svs_path: Path, dst_root: Path, levels: list[int] | None, chunk: int, clevel: int):
    try:
        slide = OpenSlide(str(svs_path))
    except OpenSlideError as e:
        return (str(svs_path), False, f"OpenSlideError: {e}")

    zarr_dir = dst_root / (svs_path.stem + ".zarr")
    zarr_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(zarr_dir))
    root = zarr.group(store=store)

    if "spec" not in root.attrs:
        _write_attrs(root, slide)

    if levels is None:
        levels = list(range(slide.level_count))  # default: keep all
    else:
        # filter to actually present levels
        levels = [L for L in levels if 0 <= L < slide.level_count]

    compressor = Blosc(cname="zstd", clevel=int(clevel), shuffle=Blosc.SHUFFLE)

    for L in levels:
        w, h = slide.level_dimensions[L]  # (w,h)
        ds = float(slide.level_downsamples[L])

        arr = root.require_dataset(
            f"level_{L}",
            shape=(h, w, 3),  # y, x, c
            chunks=(chunk, chunk, 3),
            dtype="u1",
            compressor=compressor,
            overwrite=False,
        )

        # tile loop in level space (x,y are LEVEL coordinates)
        for y0 in range(0, h, chunk):
            tile_h = min(chunk, h - y0)
            for x0 in range(0, w, chunk):
                tile_w = min(chunk, w - x0)
                # OpenSlide read_region takes BASE coordinates; convert
                base_x = int(x0 * ds)
                base_y = int(y0 * ds)
                region = slide.read_region((base_x, base_y), L, (tile_w, tile_h))  # RGBA
                tile = np.asarray(region, dtype=np.uint8)[..., :3]  # strip alpha
                arr[y0:y0 + tile_h, x0:x0 + tile_w, :] = tile

    return (str(svs_path), True, "")

def main():
    ap = argparse.ArgumentParser(description="Convert subset of TCGA SVS to Zarr")
    ap.add_argument("--src-root", type=Path, required=True, help="Root folder with TCGA .svs")
    ap.add_argument("--dst-root", type=Path, required=True, help="Output Zarr root")
    ap.add_argument("--budget-gb", type=float, default=100.0, help="How many GiB of *input SVS* to convert")
    ap.add_argument("--levels", type=str, default="0,1,2", help='Comma list (e.g. "0,1,2") or "all"')
    ap.add_argument("--chunk", type=int, default=512, help="Zarr chunk edge in pixels (y and x)")
    ap.add_argument("--clevel", type=int, default=5, help="Blosc/Zstd compression level 1..9")
    ap.add_argument("--workers", type=int, default=4, help="Parallel conversion processes")
    ap.add_argument("--badlist", type=Path, default=BADLIST_DEFAULT, help="baddata.txt to skip problematic slides")
    ap.add_argument("--dry-run", action="store_true", help="Only print selected files")
    args = ap.parse_args()

    args.dst_root.mkdir(parents=True, exist_ok=True)

    all_paths = list_svs(args.src_root, args.badlist)
    if not all_paths:
        print("No SVS found under", args.src_root)
        return

    budget_bytes = bytes_from_gib(args.budget_gb)
    subset, acc = pick_subset_until_budget(all_paths, budget_bytes)
    print(f"Selected {len(subset)} SVS files totalling {acc/1024**3:.2f} GiB of input SVS")

    (args.dst_root / "subset_manifest.txt").write_text("\n".join(str(p) for p in subset))

    if args.dry_run:
        return

    levels = None if args.levels.strip().lower() == "all" else [int(x) for x in args.levels.split(",") if x.strip()!=""]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, p, args.dst_root, levels, args.chunk, args.clevel) for p in subset]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Converting"):
            results.append(fut.result())

    # write a small report
    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]
    report = {
        "src_root": str(args.src_root),
        "dst_root": str(args.dst_root),
        "budget_gib": args.budget_gb,
        "levels": args.levels,
        "chunk": args.chunk,
        "clevel": args.clevel,
        "selected_count": len(subset),
        "succeeded": len(ok),
        "failed": len(bad),
        "failures": [{"path": r[0], "error": r[2]} for r in bad],
    }
    (args.dst_root / "conversion_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

#### Convert just a subset of tcga to zarr for test purposes ####
# python scripts/convert_tcga_to_zarr_subset.py \
#   --src-root /data/TCGA --dst-root /home/paul/TCGA_zarr \
#   --budget-gb 30 --levels 0,1,2 --chunk 512 --clevel 5 --workers 8
