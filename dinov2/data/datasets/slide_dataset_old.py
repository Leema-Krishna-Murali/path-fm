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

class SlideDataset(ExtendedVisionDataset):
    def __init__(self, root, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        
        folder_path = Path(root)

        # Image extensions to look for
        image_extensions = {'.svs'}

        # Recursively find all image files
        self.image_files = [p for p in folder_path.rglob("*") if p.suffix.lower() in image_extensions]
        print("Found this many files", len(self.image_files))
        
        #Load dataset_listed, which contains our acceptable patch indexes
        #/data/TCGA/575011dc-d267-4cb7-9ba2-e4d4c3c60f75/TCGA-CV-6959-01Z-00-DX1.AEC54909-0A3A-43E2-966C-7410CD7488EF.svs 13568 48128 0
        #Format is path, index, index, level
        #
        """
        self.levels = [[],[],[],[]]
        self.used_levels = [[],[],[],[]]
        with open("patches_listed", "r") as f:
            for line in f.readlines():
                parts = line.split(" ")
                path = parts[0]
                x = path[1]
                y = path[2]
                level = path[3]
                
                #We don't count this
                if level >=4:
                    pass
                self.levels[level].append((path, x, y))
        """


    #Takes a pil version of the highest level.
    def pseudo_unet(self, img, comp_w, comp_h):
       
        crop_height = comp_h
        crop_width = comp_w

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        #210 is the invere for how strong a pixel is - 255 is pure white.
        #In our case, we have a lot of light data, which we *do* want to work with anyway
        if False:
            cv2.imwrite("binary.png", thresh)
            cv2.imwrite("gray.png", gray)
            #exit()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        valid_components = []
        for i in range(1, num_labels):
            # The bounding box is at stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
            # stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Filter out components that are too small
            #Since we could be grabbing a piece at 224x224, ??
            if w >= crop_width and h >= crop_height:
                if False:
                    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(thresh_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(str(i) + "bounding.png", thresh_color)
                valid_components.append({'x': x, 'y': y, 'w': w, 'h': h})

        if not valid_components:
            #print("No valid components found to crop from.")
            return None

        return valid_components

    def get_all(self, index):

        path = self.image_files[index]
        image = OpenSlide(path)
        

        #for level in range(0, image.level_count):
        #    image.read_region((0,0), level = level, size=(224, 244))
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        debug = False
        if True:
            path = self.image_files[index]
            if debug:
                print(path)
            image = OpenSlide(path)
            image_levels = image.level_count
            if False:
                print("This many image levels", image_levels)
                print("This dim", image.level_dimensions)#((49933, 41465), (12483, 10366), (3120, 2591))

            #for key, value in image.properties.items():
            #    print(f"{key}: {value}")

            level = random.randint(0, image_levels - 1)
            if debug:
                print("picked", level)
            patch_size = 224
            height = image.level_dimensions[0][1]
            width = image.level_dimensions[0][0]
            if debug:
                print("these dims", image.level_dimensions[level])


            #read_region is based on the top left pixel in the level 0, not our current level
            i = 0
            tHeight = image.level_dimensions[image_levels - 1][1]
            tWidth = image.level_dimensions[image_levels - 1][0]


            lowest = image.read_region((0,0), level = image_levels - 1, size=(int(tWidth), int(tHeight)))

            #Get a rough scale here
            tHeight = image.level_dimensions[-1][1]
            tWidth = image.level_dimensions[-1][0]

            #print("Widht stuff", width, tWidth, comp_w)
            
            comp_w = 224/2
            comp_h = 224/2
            connections = self.pseudo_unet(np.array(lowest), comp_w, comp_h)
            if connections == None:
                #print("Curated connections was unable to find anything", path)
                #For now, I think most of our curated are fine.
                #If we can't find anything, treat image normally
                pass
            while True:
                if debug:
                    print("start loop", flush = True)
                i = i + 1
                if debug:
                    print("iteration", i)
                
                #We select between 0 and maximum patches, then multiply by 224 to get the patch number
                if connections != None:
                    if True:
                        chosen_component = random.choice(connections)
                        comp_x, comp_y, comp_w, comp_h = chosen_component['x'], chosen_component['y'], chosen_component['w'], chosen_component['h']

                        #All selections are done with the smaller image/highest level
                        crop_width = 224
                        crop_height = 224
                        
                        tHeight = image.level_dimensions[-1][1]
                        tWidth = image.level_dimensions[-1][0]


                        #1782 499 61 50 2863 2582 91631 82629
                        #print(comp_x, comp_y, comp_w, comp_h, tWidth, tHeight, width, height)
                        #exit()


                        y = int(comp_y * height/tHeight)
                        x = int(comp_x * width/tWidth)
                        w = int(comp_w * width/tWidth)
                        h = int(comp_h * height/tHeight)

                        #print(comp_x, comp_y, comp_w, comp_h)
                        x = random.randint(x, x + w - crop_width)
                        y = random.randint(y, y + h - crop_height)
                        
                        #Just select anywhere in that region, but 224 away from the edge in the correct scale....

                        #So this x/y is at the smallest image, we need to scale it to the maximum image, approximately
                else:
                    x = random.randint(0, (width - patch_size)//224) * 224
                    y = random.randint(0, (height - patch_size)//224) * 224
                    #Check if we have selected this patch before

                #if True:
                #    print("Reading this", path, x, y, level)

                try:
                    patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))
                except:
                    print("Unable to read path:", path, x, y, level)
                    exit()

                # Convert to RGB (removes alpha channel)
                patch = patch.convert("RGB")
                 
                if False:#Skip HSV
                    res = patch
                    break

                res = self.hsv(patch, patch_size)
                if res == None:
                    #For now, if res ==None, we consider this a fail and exit
                    if False:
                        print("First pass failure", path)
                        print(x, y)
                        
                        print(comp_x, comp_y, comp_w, comp_h, tWidth, tHeight, width, height)
                        #patch.save("patch.png")#This is the patch that failed
                        #lowest.save("lowest.png")#This is the complete image
                        
                        print("This many image levels", image_levels)
                        print("we sleected level", level)
                        print("This dim", image.level_dimensions)

                        #Let's save the original image, and the patch suggested by the connected.

                        #exit()
                    pass
                else:
                    break
                if i == 1000:#If you can't find anything after 1k that's super duper sus but whatever
                    print("Couldn't find matching item in slide", path, flush = True)
                    res = patch
                    break


        #The transform used is a torchvision StandardTransform.
        #This means that it takes as input two things, and runs two different transforms on both.
        if self.transforms is not None:
            return self.transforms(res, None)
        return res, None
        
    def hsv(self, tile_rgb, patch_size):
        
        #tile_rgb.save("tile.png")


        tile = np.array(tile_rgb)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        min_ratio = .3
        
        lower_bound = np.array([90, 8, 103])
        upper_bound = np.array([180, 255, 255])

        mask = cv2.inRange(tile, lower_bound, upper_bound)

        ratio = np.count_nonzero(mask) / mask.size
        if ratio > min_ratio:
            #print("accept this")
            #tile_rgb.show()
            return tile_rgb
        else:
            #tile_rgb.show()
            #print("Ratio fail", ratio)
            return None

    def __len__(self) -> int:
        return len(self.image_files)
