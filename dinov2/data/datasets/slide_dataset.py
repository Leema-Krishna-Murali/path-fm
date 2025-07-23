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
from PIL import Image, ImageEnhance
from openslide import OpenSlide
import random
import numpy as np
import cv2
import os
from skimage.color import rgb2hed, hed2rgb

class SlideDataset(ExtendedVisionDataset):
    def __init__(self, root, patch_size=224, max_attempts=100, min_tissue_ratio=0.4, level=0, 
                 hed_augment=True, hed_intensity_range=0.05, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        
        folder_path = Path(root)

        # Image extensions to look for
        image_extensions = {'.svs'}

        # Recursively find all image files
        self.image_files = [p for p in folder_path.rglob("*") if p.suffix.lower() in image_extensions]
        self.patch_size = patch_size
        self.max_attempts = max_attempts
        self.min_tissue_ratio = min_tissue_ratio  
        self.level = level
        self.hed_augment = hed_augment
        self.hed_intensity_range = hed_intensity_range
        
        # Tile rejection tracking
        self.rejection_stats = {
            'total_attempts': 0,
            'successful_tiles': 0,
            'rejected_tiles': 0,
            'avg_attempts_per_tile': 0.0
        }
        self._rejection_log = []
        
        # Cache slide metadata to avoid repeated file opens
        self._slide_cache = {}
        self._preload_slide_info()
        
        print(f"Found {len(self.image_files)} .svs files")

    def _preload_slide_info(self):
        """Preload slide dimensions and properties to avoid repeated opens."""
        for path in self.image_files:
            try:
                with OpenSlide(str(path)) as slide:
                    self._slide_cache[str(path)] = {
                        'levels': slide.level_count,
                        'dimensions': slide.level_dimensions,
                        'level_downsamples': slide.level_downsamples
                    }
            except Exception as e:
                print(f"Warning: Could not load slide {path}: {e}")
                continue

    def _get_valid_slide_paths(self):
        """Get list of valid slide paths that can be opened."""
        return [p for p in self.image_files if str(p) in self._slide_cache]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.image_files[index]
        path_str = str(path)
        
        if path_str not in self._slide_cache:
            raise RuntimeError(f"Slide metadata not cached for {path}")
        
        slide_info = self._slide_cache[path_str]
        
        # Use specified level or default to level 0
        level = min(self.level, slide_info['levels'] - 1)
        width, height = slide_info['dimensions'][level]
        
        # Calculate actual patch size at this level
        level_downsample = slide_info['level_downsamples'][level]
        actual_patch_size = int(self.patch_size * level_downsample)
        
        with OpenSlide(path_str) as slide:
            valid_patch = None
            attempts = 0
            
            for attempt in range(self.max_attempts):
                attempts += 1
                self.rejection_stats['total_attempts'] += 1
                
                # Random coordinates at the base level
                base_x = random.randint(0, max(0, slide_info['dimensions'][0][0] - actual_patch_size))
                base_y = random.randint(0, max(0, slide_info['dimensions'][0][1] - actual_patch_size))
                
                # Read region at base level, then resize to target patch size
                patch = slide.read_region((base_x, base_y), 0, (actual_patch_size, actual_patch_size))
                
                # Convert to RGB and resize to target patch size
                patch = patch.convert("RGB")
                if actual_patch_size != self.patch_size:
                    patch = patch.resize((self.patch_size, self.patch_size), Image.LANCZOS)
                
                # Check tissue content
                if self._has_sufficient_tissue(patch):
                    valid_patch = patch
                    self.rejection_stats['successful_tiles'] += 1
                    break
                else:
                    self.rejection_stats['rejected_tiles'] += 1
            
            # Update average attempts calculation
            successful = self.rejection_stats['successful_tiles']
            total = self.rejection_stats['total_attempts']
            if successful > 0:
                self.rejection_stats['avg_attempts_per_tile'] = total / successful
            
            # Log rejection info for debugging
            self._rejection_log.append({
                'slide': str(path),
                'attempts': attempts,
                'success': valid_patch is not None
            })
            
            if valid_patch is None:
                # Fallback: return center crop if no suitable patch found
                center_x = max(0, (slide_info['dimensions'][0][0] - actual_patch_size) // 2)
                center_y = max(0, (slide_info['dimensions'][0][1] - actual_patch_size) // 2)
                valid_patch = slide.read_region((center_x, center_y), 0, (actual_patch_size, actual_patch_size))
                valid_patch = valid_patch.convert("RGB")
                if actual_patch_size != self.patch_size:
                    valid_patch = valid_patch.resize((self.patch_size, self.patch_size), Image.LANCZOS)
        
        # Apply HED augmentation if enabled
        if self.hed_augment:
            valid_patch = self._apply_hed_augmentation(valid_patch)
        
        if self.transforms is not None:
            return self.transforms(valid_patch, None)
        return valid_patch, None
        
    def _has_sufficient_tissue(self, patch: Image.Image) -> bool:
        """Check if patch has sufficient tissue content using HSV filtering."""
        tile = np.array(patch)
        hsv_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        
        # HSV range for tissue detection - excludes white background and adipose tissue
        # Values from pathology literature for tissue vs background separation
        lower_bound = np.array([0, 10, 0])    # Lower bound for tissue
        upper_bound = np.array([180, 255, 230])  # Upper bound to exclude white background
        
        mask = cv2.inRange(hsv_tile, lower_bound, upper_bound)
        tissue_ratio = np.count_nonzero(mask) / mask.size
        
        return tissue_ratio > self.min_tissue_ratio

    def _apply_hed_augmentation(self, patch: Image.Image) -> Image.Image:
        """Apply HED color space augmentation with intensity ratios [-0.05, 0.05]."""
        # Convert PIL to numpy
        img_np = np.array(patch).astype(np.float32) / 255.0
        
        # Convert RGB to HED color space using skimage
        hed = rgb2hed(img_np)
        
        # Apply random intensity adjustment for each HED channel (H, E, D)
        hed[:, :, 0] *= (1 + random.uniform(-self.hed_intensity_range, self.hed_intensity_range))
        hed[:, :, 1] *= (1 + random.uniform(-self.hed_intensity_range, self.hed_intensity_range))
        hed[:, :, 2] *= (1 + random.uniform(-self.hed_intensity_range, self.hed_intensity_range))
        
        # Convert HED back to RGB
        augmented = hed2rgb(hed)
        
        # Ensure values are in valid range
        augmented = np.clip(augmented, 0, 1)
        augmented = (augmented * 255).astype(np.uint8)
        
        return Image.fromarray(augmented)

    def __len__(self) -> int:
        return len(self.image_files)
    
    def get_rejection_stats(self):
        """Get tile rejection statistics."""
        return {
            'total_attempts': self.rejection_stats['total_attempts'],
            'successful_tiles': self.rejection_stats['successful_tiles'],
            'rejected_tiles': self.rejection_stats['rejected_tiles'],
            'avg_attempts_per_tile': self.rejection_stats['avg_attempts_per_tile'],
            'rejection_rate': (self.rejection_stats['rejected_tiles'] / 
                             max(1, self.rejection_stats['total_attempts'])) * 100
        }
    
    def print_rejection_summary(self):
        """Print a summary of tile rejection statistics."""
        stats = self.get_rejection_stats()
        print(f"\n=== Tile Rejection Statistics ===")
        print(f"Total attempts: {stats['total_attempts']:,}")
        print(f"Successful tiles: {stats['successful_tiles']:,}")
        print(f"Rejected tiles: {stats['rejected_tiles']:,}")
        print(f"Average attempts per successful tile: {stats['avg_attempts_per_tile']:.2f}")
        print(f"Rejection rate: {stats['rejection_rate']:.1f}%")
        print("=" * 35)
