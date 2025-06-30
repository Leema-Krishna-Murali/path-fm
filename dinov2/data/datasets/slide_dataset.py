# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple
from torchvision.datasets import VisionDataset
from .extended import ExtendedVisionDataset
from .decoders import TargetDecoder, ImageDataDecoder
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from PIL import Image
from openslide import OpenSlide#other options?
import random
import numpy as np
import cv2

class SlideDataset(ExtendedVisionDataset):
    def __init__(self, root, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        
        folder_path = Path(root)

        # Image extensions to look for
        image_extensions = {'.svs'}

        # Recursively find all image files
        self.image_files = [p for p in folder_path.rglob("*") if p.suffix.lower() in image_extensions]
        print("Found this many files", len(self.image_files))
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            path = self.image_files[index]
            image = OpenSlide(path)
            image_levels = image.level_count
            #print("This many image levels", image_levels)
            #print("This dim", image.level_dimensions)#((49933, 41465), (12483, 10366), (3120, 2591))

            #for key, value in image.properties.items():
            #    print(f"{key}: {value}")
            #Decide on a magnification. For testing purposes, it'll be always level 0
            level = 0
            patch_size = 224
            height = image.level_dimensions[level][1]
            width = image.level_dimensions[level][0]
            
            #print("starting hsv loop")
            i = 0
            while True:
                print(i)
                i = i + 1
                if i == 10000:
                    print("Couldn't find matching item in slide", path)
                    exit()
                x = random.randint(0, width - patch_size)
                y = random.randint(0, height - patch_size)
                patch = image.read_region((x, y), level=0, size=(patch_size, patch_size))

                # Convert to RGB (removes alpha channel)
                patch = patch.convert("RGB")
                res = self.hsv(patch, patch_size)
                if res == None:
                    pass
                else:
                    break
        except Exception as e:
            print("Crash")
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        #The transform used is a torchvision StandardTransform.
        #This means that it takes as input two things, and runs two different transforms on both.
        if self.transforms is not None:
            return self.transforms(res, None)
        print("returning patch")
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
            #print("accept this")
            #tile_rgb.show()
            return tile_rgb
        else:
            #print("reject")
            #tile_rgb.show()
            return None

    def __len__(self) -> int:
        return len(self.image_files)
