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


class TestVisionDataset(ExtendedVisionDataset):
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
            print(image)
            #Decide on a magnification
            #get h/w for it
            #Randomly select x, y
            #Grab the patch, filter by HSV
            #if it fails, try again...
            

            
            #image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        #The transform used is a torchvision StandardTransform.
        #This means that it takes as input two things, and runs two different transforms on both.
        if self.transforms is not None:
            print(image.size, path)#Debug only
            return self.transforms(image, None)

        #this just returns a class index, which we do not need.
#        target = self.get_target(index)
#        target = TargetDecoder(target).decode()

        #if self.transforms is not None:
        #    image, target = self.transforms(image, target)

        return image, None

    def __len__(self) -> int:
        return len(self.image_files)
