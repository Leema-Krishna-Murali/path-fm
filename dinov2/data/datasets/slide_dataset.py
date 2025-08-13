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
import random

# class SlideDataset(ExtendedVisionDataset):
#     def __init__(self, root, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)  # type: ignore
        
#         folder_path = Path(root)

#         # Image extensions to look for
#         image_extensions = {'.svs'}

#         # Recursively find all image files
#         self.image_files = [p for p in folder_path.rglob("*") if p.suffix.lower() in image_extensions]
#         print("Found this many files", len(self.image_files))
        

#     def get_all(self, index):

#         path = self.image_files[index]
#         image = OpenSlide(path)
        

#         #for level in range(0, image.level_count):
#         #    image.read_region((0,0), level = level, size=(224, 244))
#         return image, path

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         debug = False
#         if True:
#             path = self.image_files[index]
#             if debug:
#                 print(path)
#             image = OpenSlide(path)
#             image_levels = image.level_count
#             if debug:
#                 print("This many image levels", image_levels)
#                 print("This dim", image.level_dimensions)#((49933, 41465), (12483, 10366), (3120, 2591))

#             #for key, value in image.properties.items():
#             #    print(f"{key}: {value}")

#             level = random.randint(0, image_levels - 1)
#             if debug:
#                 print("picked", level)
#             patch_size = 224
#             height = image.level_dimensions[0][1]
#             width = image.level_dimensions[0][0]
#             if debug:
#                 print("these dims", image.level_dimensions[level])
#             if False:#debug saving all.
#                 full = image.read_region((0,0), level = level, size=(int(width), int(height)))
#                 full.save("full.png")
#                 print("saved full")

#             #read_region is based on the top left pixel in the level 0, not our current
#             i = 0
#             while True:
#                 if debug:
#                     print("start loop", flush = True)
#                 i = i + 1
#                 if i == 100:
#                     print("Couldn't find matching item in slide", path, flush = True)
#                     exit()
#                 if debug:
#                     print("iteration", i)
#                 #4403, 4645) 
#                 x = random.randint(0, width - patch_size)
#                 y = random.randint(0, height - patch_size)
#                 if debug:
#                     print("Reading this", path, x, y, level)
#                 try:
#                     patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))
#                 except:
#                     print("failed on path", path, x, y, level)
#                     exit()

#                 # Convert to RGB (removes alpha channel)
#                 patch = patch.convert("RGB")
                 
#                 if True:
#                     res = patch
#                     break
#                 res = self.hsv(patch, patch_size)
#                 print("have result", path, x, y, level)
#                 if res == None:
#                     pass
#                 else:
#                     break
#         #except Exception as e:
#         #    print("Crash", path)
#         #    print(e)
#         #    #raise RuntimeError(f"can not read image for sample {index}") from e
#         #    exit()
        
#         #The transform used is a torchvision StandardTransform.
#         #This means that it takes as input two things, and runs two different transforms on both.
#         if self.transforms is not None:
#             return self.transforms(res, None)
#         return res, None
        
#     def hsv(self, tile_rgb, patch_size):
        
#         #tile_rgb.save("tile.png")

#         tile = np.array(tile_rgb)
#         tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
#         min_ratio = .6
        
#         lower_bound = np.array([90, 8, 103])
#         upper_bound = np.array([180, 255, 255])

#         mask = cv2.inRange(tile, lower_bound, upper_bound)

#         ratio = np.count_nonzero(mask) / mask.size
#         if ratio > min_ratio:
#             #print("accept this")
#             #tile_rgb.show()
#             return tile_rgb
#         else:
#             #tile_rgb.show()
#             return None

#     def __len__(self) -> int:
#         return len(self.image_files)

# revised SlideDataset to handle LitData
class SlideDataset(ExtendedVisionDataset):
    def __init__(self, root, *args, **kwargs):
        # Use LitData streaming
        from litdata import StreamingDataset
        
        class LitDataWrapper(StreamingDataset):
            def __init__(self, data_dir, transform=None):
                super().__init__(input_dir=data_dir, shuffle=True)
                self.transforms = transform
            
            def __getitem__(self, index):
                # Convert numpy int to Python int to avoid LitData StreamingDataset issues
                if hasattr(index, 'item'):
                    index = int(index.item())
                else:
                    index = int(index)
                    
                item = super().__getitem__(index)
                image = item['image']
                
                # Convert to PIL if needed
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                
                if self.transforms is not None:
                    # DINOv2 expects the transform to return a dict, wrapped in a tuple
                    return self.transforms(image), None
                return image, None
        
        self._dataset = LitDataWrapper(root, kwargs.get('transform', None))
        self.image_files = ["dummy"] * len(self._dataset)  # For compatibility
    
    def __getitem__(self, index):
        if hasattr(self, '_dataset'):
            # Convert numpy int to Python int to avoid LitData StreamingDataset issues
            if hasattr(index, 'item'):
                index = int(index.item())
            else:
                index = int(index)
            return self._dataset[index]
        return self._orig[index]
    
    def __len__(self):
        if hasattr(self, '_dataset'):
            return len(self._dataset)
        return len(self._orig)