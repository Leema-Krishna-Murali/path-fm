# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import cv2
import numpy as np
import random
import torch
import torchvision
from torchvision import transforms
from skimage import color as skimage_color
from skimage.color import rgb2hed, hed2rgb
from einops import rearrange
from PIL import Image

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")


class MacenkoNormalizer:
    """
    Macenko stain normalization for H&E histopathology images.
    Based on: "A method for normalizing histology slides for quantitative analysis"
    M. Macenko et al., ISBI 2009

    Reference implementation: https://github.com/EIDOSLAB/torchstain
    """

    def __init__(self):
        # Reference stain vectors (H&E)
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def _convert_rgb2od(self, img, Io=240, beta=0.15):
        """Convert RGB to optical density."""
        img = img.reshape((-1, 3)).astype(np.float64)
        OD = -np.log((img + 1) / Io)
        ODhat = OD[~np.any(OD < beta, axis=1)]
        return OD, ODhat

    def _find_HE(self, ODhat, eigvecs, alpha=1):
        """Find H&E stain vectors."""
        That = np.dot(ODhat, eigvecs)
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        vMin = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T

        return HE

    def _find_concentration(self, OD, HE):
        """Find stain concentrations."""
        Y = OD.T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        return C

    def _compute_matrices(self, img, Io=240, alpha=1, beta=0.15):
        """Compute stain matrices and concentrations."""
        h, w, c = img.shape
        OD, ODhat = self._convert_rgb2od(img, Io, beta)

        if ODhat.shape[0] < 10:
            return None, None, None

        cov_matrix = np.cov(ODhat.T)
        eigenvalues, eigvecs = np.linalg.eigh(cov_matrix)
        eigvecs = eigvecs[:, [1, 2]]

        HE = self._find_HE(ODhat, eigvecs, alpha)
        C = self._find_concentration(OD, HE)

        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, img, Io=240, alpha=1, beta=0.15):
        """Fit normalizer to a target image."""
        if isinstance(img, Image.Image):
            img = np.array(img)

        HE, _, maxC = self._compute_matrices(img, Io, alpha, beta)
        if HE is not None:
            self.HERef = HE
            self.maxCRef = maxC

    def normalize(self, img, Io=240, alpha=1, beta=0.15):
        """Normalize staining appearance of H&E image."""
        if isinstance(img, Image.Image):
            img = np.array(img)
            was_pil = True
        else:
            was_pil = False

        h, w, c = img.shape

        HE, C, maxC = self._compute_matrices(img, Io, alpha, beta)

        if HE is None:
            return Image.fromarray(img) if was_pil else img

        maxC = np.clip(maxC, 1e-6, None)
        C = C * (self.maxCRef / maxC).reshape(-1, 1)

        Inorm = Io * np.exp(-np.dot(self.HERef, C))
        Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
        Inorm = Inorm.T.reshape(h, w, c)

        return Image.fromarray(Inorm) if was_pil else Inorm


class MacenkoAugmentation(torch.nn.Module):
    """
    Macenko stain normalization as a data augmentation transform.
    Applied with configurable probability.
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
        self.normalizer = MacenkoNormalizer()

    def forward(self, img):
        if random.random() > self.probability:
            return img

        return self.normalizer.normalize(img)


class RandStainNA(torch.nn.Module):
    """
    RandStainNA: Random Stain Normalization and Augmentation.
    Bridges stain normalization and augmentation by constraining variable stain styles
    in a practicable range using virtual template generation.

    Based on: "RandStainNA: Learning Stain-Agnostic Features from Histology Slides
    by Bridging Stain Augmentation and Normalization" (MICCAI 2022)

    Reference: https://github.com/yiqings/RandStainNA
    """

    # Default statistics for LAB color space from CRC dataset
    # These can be overridden by providing a yaml_file
    DEFAULT_LAB_STATS = {
        'L': {'avg': {'mean': 158.033, 'std': 48.792}, 'std': {'mean': 36.899, 'std': 14.383}},
        'A': {'avg': {'mean': 151.187, 'std': 10.958}, 'std': {'mean': 8.134, 'std': 2.822}},
        'B': {'avg': {'mean': 116.812, 'std': 6.643}, 'std': {'mean': 6.129, 'std': 2.013}},
    }

    # Default statistics for HED color space
    DEFAULT_HED_STATS = {
        'H': {'avg': {'mean': 0.05, 'std': 0.02}, 'std': {'mean': 0.03, 'std': 0.01}},
        'E': {'avg': {'mean': 0.02, 'std': 0.01}, 'std': {'mean': 0.02, 'std': 0.008}},
        'D': {'avg': {'mean': 0.0, 'std': 0.005}, 'std': {'mean': 0.01, 'std': 0.005}},
    }

    def __init__(
        self,
        color_space='LAB',
        std_hyper=-0.3,
        distribution='normal',
        probability=1.0,
    ):
        super().__init__()

        assert distribution in ['normal', 'laplace', 'uniform'], \
            f"Unsupported distribution: {distribution}"
        assert color_space in ['LAB', 'HSV', 'HED'], \
            f"Unsupported color space: {color_space}"

        self.color_space = color_space
        self.std_hyper = std_hyper
        self.distribution = distribution
        self.probability = probability

        if color_space == 'LAB':
            stats = self.DEFAULT_LAB_STATS
            self.channels = ['L', 'A', 'B']
        elif color_space == 'HED':
            stats = self.DEFAULT_HED_STATS
            self.channels = ['H', 'E', 'D']
        else:
            stats = self.DEFAULT_LAB_STATS
            self.channels = ['H', 'S', 'V']

        self.channel_avgs_mean = [stats[c]['avg']['mean'] for c in self.channels]
        self.channel_avgs_std = [stats[c]['avg']['std'] for c in self.channels]
        self.channel_stds_mean = [stats[c]['std']['mean'] for c in self.channels]
        self.channel_stds_std = [stats[c]['std']['std'] for c in self.channels]

    def _getavgstd(self, image):
        """Get mean and std for each channel."""
        avgs = []
        stds = []
        for idx in range(image.shape[2]):
            avgs.append(np.mean(image[:, :, idx]))
            stds.append(np.std(image[:, :, idx]))
        return np.array(avgs), np.array(stds)

    def _normalize(self, img, img_avgs, img_stds, tar_avgs, tar_stds):
        """Normalize image to target statistics."""
        img_stds = np.clip(img_stds, 0.0001, 255)
        img = (img - img_avgs) * (tar_stds / img_stds) + tar_avgs

        if self.color_space in ['LAB', 'HSV']:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _generate_virtual_template(self):
        """Generate virtual template statistics based on distribution."""
        tar_avgs = []
        tar_stds = []

        if self.distribution == 'uniform':
            for idx in range(3):
                tar_avg = np.random.uniform(
                    low=self.channel_avgs_mean[idx] - 3 * self.channel_avgs_std[idx],
                    high=self.channel_avgs_mean[idx] + 3 * self.channel_avgs_std[idx],
                )
                tar_std = np.random.uniform(
                    low=self.channel_stds_mean[idx] - 3 * self.channel_stds_std[idx],
                    high=self.channel_stds_mean[idx] + 3 * self.channel_stds_std[idx],
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)
        else:
            if self.distribution == 'normal':
                np_distribution = np.random.normal
            else:
                np_distribution = np.random.laplace

            for idx in range(3):
                tar_avg = np_distribution(
                    loc=self.channel_avgs_mean[idx],
                    scale=self.channel_avgs_std[idx] * (1 + self.std_hyper),
                )
                tar_std = np_distribution(
                    loc=self.channel_stds_mean[idx],
                    scale=self.channel_stds_std[idx] * (1 + self.std_hyper),
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)

        return np.array(tar_avgs), np.array(tar_stds)

    def augment(self, img):
        """Apply stain augmentation."""
        if isinstance(img, Image.Image):
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            was_pil = True
        else:
            image = img
            was_pil = False

        # Color space conversion
        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'HED':
            image = skimage_color.rgb2hed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Generate virtual template and normalize
        tar_avgs, tar_stds = self._generate_virtual_template()
        img_avgs, img_stds = self._getavgstd(image)

        image = self._normalize(
            img=image,
            img_avgs=img_avgs,
            img_stds=img_stds,
            tar_avgs=tar_avgs,
            tar_stds=tar_stds,
        )

        # Convert back to BGR/RGB
        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.color_space == 'HED':
            nimg = skimage_color.hed2rgb(image)
            imin = nimg.min()
            imax = nimg.max()
            rsimg = (255 * (nimg - imin) / (imax - imin + 1e-8)).astype('uint8')
            image = cv2.cvtColor(rsimg, cv2.COLOR_RGB2BGR)

        # Convert back to RGB for PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if was_pil:
            return Image.fromarray(image)
        return image

    def forward(self, img):
        if random.random() > self.probability:
            return img
        return self.augment(img)


class hed_mod(torch.nn.Module):
    """
    HED color space augmentation for H&E stained histopathology images.
    Randomly perturbs Hematoxylin, Eosin, and DAB channels.
    """

    def __init__(self, probability=0.5, perturbation_range=0.05):
        super().__init__()
        self.probability = probability
        self.mini = -perturbation_range
        self.maxi = perturbation_range

    def forward(self, img, label=None):
        if random.random() > self.probability:
            return img

        if img is not None:
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = rearrange(img, 'c h w -> h w c')
            hed_image = rgb2hed(img)

            hed_image[..., 0] += random.uniform(self.mini, self.maxi)  # H
            hed_image[..., 1] += random.uniform(self.mini, self.maxi)  # E
            hed_image[..., 2] += random.uniform(self.mini, self.maxi)  # D

            hed_image = np.clip(hed_image, 0, 1)
            img = hed2rgb(hed_image)

            img = rearrange(img, 'h w c -> c h w')
            img = torch.from_numpy(img)
            img = torchvision.transforms.functional.to_pil_image(img)

        if label is not None:
            label = rearrange(label, 'c h w -> h w c')
            hed_image = rgb2hed(label)
            hed_image[..., 0] += random.uniform(self.mini, self.maxi)
            hed_image[..., 1] += random.uniform(self.mini, self.maxi)
            hed_image[..., 2] += random.uniform(self.mini, self.maxi)
            label = rearrange(label, 'h w c -> c h w')
            label = torch.from_numpy(label)
            return img, label

        return img


class RandomRotation90(torch.nn.Module):
    """
    Random 90-degree rotation augmentation for histopathology images.
    Pathology images are rotation-invariant, so we can apply 0, 90, 180, or 270 degree rotations.
    """

    def __init__(self):
        super().__init__()
        self.angles = [0, 90, 180, 270]

    def forward(self, img):
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        return transforms.functional.rotate(img, angle)


class DataAugmentationDINO(object):
    """
    Data augmentation pipeline for DINOv2 training on histopathology images.

    Includes pathology-specific augmentations:
    - RandStainNA: Stain normalization/augmentation in LAB color space
    - Macenko normalization: Classical stain normalization
    - HED augmentation: Color space perturbation
    - 90-degree rotations: Rotation invariance
    - Vertical and horizontal flips
    - Gaussian blur
    - Color jitter (no grayscale for H&E images)
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("Pathology-specific augmentations possible:")
        logger.info("  - RandStainNA (LAB color space)")
        logger.info("  - Macenko stain normalization")
        logger.info("  - HED color augmentation")
        logger.info("  - 90-degree rotations")
        logger.info("  - Vertical + horizontal flips")
        logger.info("  - No grayscale (preserving H&E color information)")
        logger.info("###################################")

        # Geometric augmentations with rotation and both flips
        self.geometric_augmentation_global = transforms.Compose(
            [
                # RandomRotation90(),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                # RandomRotation90(),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        # Normalization (ImageNet stats used by default)
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        # Pathology-specific stain augmentations
        randstainna = RandStainNA(
            color_space='LAB',
            std_hyper=-0.3,
            distribution='normal',
            probability=0.5,
        )

        macenko_aug = MacenkoAugmentation(probability=0.3)

        hed_aug = hed_mod(probability=0.5, perturbation_range=0.05)

        # Compose full augmentation pipelines
        # Order: Macenko -> RandStainNA -> HED -> ColorJitter -> Blur -> Normalize
        self.global_transfo1 = transforms.Compose([
            # macenko_aug,
            # randstainna,
            hed_aug,
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            self.normalize
        ])

        self.global_transfo2 = transforms.Compose([
            # macenko_aug,
            # randstainna,
            hed_aug,
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            self.normalize
        ])

        self.local_transfo = transforms.Compose([
            # macenko_aug,
            # randstainna,
            hed_aug,
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            self.normalize
        ])

    def __call__(self, image):
        output = {}

        # Global crops
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # Local crops
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
