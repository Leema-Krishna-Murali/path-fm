# Copyright (c) 2025 SophontAI
from pathlib import Path
import os
import random
import numpy as np
from PIL import Image
import zarr
import fsspec
import s3fs

from .extended import ExtendedVisionDataset

class SlideDatasetZarr(ExtendedVisionDataset):
    """
    Mirrors SlideDataset, but reads from Zarr groups created by convert_tcga_to_zarr_subset.py.
    Each slide is a DirectoryStore ending with .zarr, with arrays: level_0, level_1, ...
    Arrays are shaped (y, x, c) with c=3 (RGB).
    """
    def __init__(self, root, patch_size: int = 224, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.root = str(root)
        self.patch_size = int(patch_size)

        self.is_s3 = self.root.startswith("s3://")
        if self.is_s3:
            # R2 endpoint from env
            endpoint = os.environ.get("R2_ENDPOINT_URL")
            client_kwargs = {"endpoint_url": endpoint} if endpoint else {}
            self.fs = s3fs.S3FileSystem(anon=False, client_kwargs=client_kwargs)
            # s3fs.find expects "bucket/prefix" (no scheme)
            bucket_prefix = self.root.split("s3://", 1)[1].rstrip("/")
            candidates = self.fs.find(bucket_prefix)
            # keep only *.zarr "directories" (prefixes)
            self.groups = [
                f"s3://{p}" for p in candidates
                if p.endswith(".zarr") or p.endswith(".zarr/")
            ]
        else:
            p = Path(self.root)
            self.groups = sorted([str(x) for x in p.rglob("*.zarr") if x.is_dir()])
        print("[Zarr] stores found:", len(self.groups))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index):
        zpath = self.groups[index]
        if self.is_s3:
            mapper = fsspec.get_mapper(zpath)  # honors R2 endpoint via env above
            try:
                grp = zarr.open_consolidated(mapper, mode="r")
            except Exception:
                grp = zarr.open_group(mapper, mode="r")
        else:
            grp = zarr.open_group(zpath, mode="r")
        
        # collect available level arrays
        level_keys = [k for k in grp.array_keys() if k.startswith("level")]
        # Accept "level0" or "level_0" styles
        def _levnum(k):
            s = k.replace("level_", "level")
            return int(s.replace("level", ""))
        level_keys = sorted(level_keys, key=_levnum)
        assert level_keys, f"No level_* arrays in {zpath}"
        level_key = random.choice(level_keys)
        arr = grp[level_key]  # zarr array: (y,x,c)

        h, w, c = arr.shape
        ps = self.patch_size
        if h < ps or w < ps:
            # fallback to smallest possible center crop if slide is tiny
            y0 = max(0, (h - ps) // 2)
            x0 = max(0, (w - ps) // 2)
        else:
            y0 = random.randint(0, h - ps)
            x0 = random.randint(0, w - ps)

        patch = arr[y0:y0 + ps, x0:x0 + ps, :]  # numpy view (loads only required chunks)
        img = Image.fromarray(np.asarray(patch), mode="RGB")

        if self.transforms is not None:
            return self.transforms(img, None)
        return img, None
