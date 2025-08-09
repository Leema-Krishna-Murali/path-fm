# Copyright (c) 2025 SophontAI
from pathlib import Path
import random
import numpy as np
from PIL import Image
import zarr

from .extended import ExtendedVisionDataset

class SlideDatasetZarr(ExtendedVisionDataset):
    """
    Mirrors SlideDataset, but reads from Zarr groups created by convert_tcga_to_zarr_subset.py.
    Each slide is a DirectoryStore ending with .zarr, with arrays: level_0, level_1, ...
    Arrays are shaped (y, x, c) with c=3 (RGB).
    """
    def __init__(self, root, patch_size: int = 224, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.root = Path(root)
        self.patch_size = int(patch_size)
        # Discover .zarr directories
        self.groups = sorted([p for p in self.root.rglob("*.zarr") if p.is_dir()])
        print("Found this many Zarr groups", len(self.groups))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index):
        zpath = self.groups[index]
        grp = zarr.open_group(str(zpath), mode="r")
        # collect available level arrays
        level_keys = sorted([k for k in grp.array_keys() if k.startswith("level_")],
                            key=lambda k: int(k.split("_")[1]))
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
