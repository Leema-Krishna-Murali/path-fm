# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import List

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from skimage.color import rgb2hed, hed2rgb

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import random
import matplotlib.pyplot as plt
logger = logging.getLogger("dinov2")

import torchvision
import dinov2.distributed as distributed

class hed_mod(torch.nn.Module):

    def forward(self, img, label = None):

        if img !=None:
            #Convert image from RGB to HED.
            #Input shape is (3,size, size)
            #Convert to chanels last, then swap back
            img = torchvision.transforms.functional.pil_to_tensor(img)

            img = rearrange(img, 'c h w -> h w c')
            img_orig = img
            hed_image = rgb2hed(img)
            #Modify channels, each with random amount, between -.05 and .05
            mini = -.05
            maxi = .05
            total = maxi - mini

            if False:
                hed_image[..., 0] *= (1 + random.uniform(0, total) - maxi)#H
                hed_image[..., 1] *= (1 + random.uniform(0, total) - maxi)#E
                hed_image[..., 2] *= (1 + random.uniform(0, total) - maxi)#D
            else:
                hed_image[..., 0] += random.uniform(0, total) - maxi#H
                hed_image[..., 1] += random.uniform(0, total) - maxi#E
                hed_image[..., 2] += random.uniform(0, total) - maxi#D

            img = hed2rgb(hed_image)

            if False:#debug
                fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Adjust figsize as needed
                axes[0].imshow(img_orig)
                axes[0].set_title("Before")
                axes[0].axis('off') # Turn off axis ticks and labels for cleaner image display

                # Plot the "After" image on the second subplot
                axes[1].imshow(img)
                axes[1].set_title("After")
                axes[1].axis('off') # Turn off axis ticks and labels

                # Adjust layout to prevent titles from overlapping
                plt.tight_layout()

                # Set the overall figure title (optional)
                fig.suptitle("Image Comparison: Before and After HED Channel Modification", y=1.02) # y adjusts title position

                plt.show()

                exit()
            img = rearrange(img, 'h w c -> c h w')
            img = torch.from_numpy(img)
            img = torchvision.transforms.functional.to_pil_image(img)

        if label != None:
            label = rearrange(label, 'c h w -> h w c')
            hed_image = rgb2hed(label)
            #Modify channels
            hed_image[..., 0] += random.uniform(0, total) - maxi#H
            hed_image[..., 1] += random.uniform(0, total) - maxi#E
            hed_image[..., 2] += random.uniform(0, total) - maxi#D
            label = rearrange(label, 'h w c -> c h w')
            label = torch.from_numpy(label)

            return img, label

        return img



class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        # Debug/visualization options
        save_crops: bool = False,
        save_dir: str = None,
        save_first_n: int = None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # Saving/visualization configuration
        self.save_crops = save_crops
        self.save_first_n = save_first_n
        self.save_dir = (
            save_dir if save_dir is not None else os.path.join(os.getcwd(), "crops_debug")
        )
        self._save_counter = 0
        if self.save_crops:
            os.makedirs(self.save_dir, exist_ok=True)

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                # ),
                
                ## Replacing RandomResizedCrop by RandomCrop (inspired by Virchow2 ECT)
                transforms.RandomCrop(
                    global_crops_size
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                # ),

                ## Replacing RandomResizedCrop by RandomCrop (inspired by Virchow2 ECT)
                transforms.RandomResizedCrop(
                    local_crops_size
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                #transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        hed = hed_mod()

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])#Do we apply to everything?
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([hed, color_jittering, local_transfo_extra, self.normalize])

    def _denormalize(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """Convert normalized tensor back to [0,1] range for visualization."""
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=tensor_img.dtype, device=tensor_img.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=tensor_img.dtype, device=tensor_img.device).view(3, 1, 1)
        return (tensor_img * std + mean).clamp(0.0, 1.0)

    def _maybe_save_crops(self, global1: torch.Tensor, global2: torch.Tensor, local_crops: List[torch.Tensor]):
        if not self.save_crops:
            return
        if self.save_first_n is not None and self._save_counter >= self.save_first_n:
            return
        try:
            if hasattr(distributed, "is_main_process") and not distributed.is_main_process():
                return
        except Exception:
            # If distributed is not initialized, continue and save
            pass

        os.makedirs(self.save_dir, exist_ok=True)
        sample_id = f"{self._save_counter:06d}_{os.getpid()}"

        # Save separate images
        save_image(self._denormalize(global1), os.path.join(self.save_dir, f"{sample_id}_global_1.png"))
        save_image(self._denormalize(global2), os.path.join(self.save_dir, f"{sample_id}_global_2.png"))
        for i, local in enumerate(local_crops):
            save_image(self._denormalize(local), os.path.join(self.save_dir, f"{sample_id}_local_{i+1}.png"))

        # Save a quick grid for at-a-glance view (handles different sizes)
        denorm_imgs = [
            self._denormalize(global1).detach().cpu(),
            self._denormalize(global2).detach().cpu(),
            *[self._denormalize(x).detach().cpu() for x in local_crops],
        ]
        # Compute target size: max H/W across all images
        heights = [img.shape[-2] for img in denorm_imgs]
        widths = [img.shape[-1] for img in denorm_imgs]
        target_h = max(heights)
        target_w = max(widths)
        resized_imgs = []
        for img in denorm_imgs:
            if img.shape[-2] != target_h or img.shape[-1] != target_w:
                img = F.interpolate(img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
            resized_imgs.append(img)
        grid = make_grid(resized_imgs, nrow=2)
        save_image(grid, os.path.join(self.save_dir, f"{sample_id}_grid.png"))

        self._save_counter += 1

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        from torchvision.utils import save_image
        if False:#Saving
            save_image(global_crop_1, "global.png")
            save_image(global_crop_2, "global2.png")
            exit()
        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        if False:
            for i, local in enumerate(local_crops):
                save_image(local, str(i) + "local" + ".png")
            exit()
        output["local_crops"] = local_crops
        output["offsets"] = ()

        # Optionally save crops for visual verification
        self._maybe_save_crops(global_crop_1, global_crop_2, local_crops)

        return output
