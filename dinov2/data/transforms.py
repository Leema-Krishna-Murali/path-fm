# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision import transforms
import random
import cv2  # OpenCV is required for fast blur
import numpy as np
from PIL import Image


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian blur (torchvision) with probability ``p``.

    Note: ``transforms.RandomApply``'s ``p`` is the probability to apply
    the transform (not to keep the original). The previous code inverted
    this, causing much more blur than intended.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)


class FastGaussianBlur(torch.nn.Module):
    """
    Faster Gaussian blur using OpenCV for PIL images (no fallback).

    - Expects PIL.Image input (before ToTensor) and applies ``cv2.GaussianBlur``.
    - This path is typically faster than PIL/torchvision on CPU due to SIMD.

    Args:
        p: Probability to apply the blur.
        radius_min: Minimum sigma for Gaussian kernel.
        radius_max: Maximum sigma for Gaussian kernel.
        kernel_size: Odd kernel size, defaults to 9 (DINO default).
    """

    def __init__(
        self,
        *,
        p: float = 0.5,
        radius_min: float = 0.1,
        radius_max: float = 2.0,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size <= 1:
            raise ValueError("kernel_size must be an odd integer > 1")
        if radius_min <= 0 or radius_max <= 0 or radius_min > radius_max:
            raise ValueError("radius_min and radius_max must be positive with min <= max")
        self.p = float(p)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.kernel_size = int(kernel_size)

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        if not isinstance(img, Image.Image):
            raise TypeError("FastGaussianBlur expects a PIL.Image input before ToTensor")

        sigma = random.uniform(self.radius_min, self.radius_max)

        # Convert PIL -> numpy (RGB) without color conversion; Gaussian is color-agnostic
        np_img = np.array(img)
        if not np_img.flags.c_contiguous:
            np_img = np.ascontiguousarray(np_img)

        blurred = cv2.GaussianBlur(
            np_img,
            (self.kernel_size, self.kernel_size),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT101,
        )

        return Image.fromarray(blurred)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)
