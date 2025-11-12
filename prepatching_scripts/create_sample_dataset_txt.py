import cv2
import random
from openslide import OpenSlide
import numpy as np

# Initialize the dataset
dataset = SlideDataset("/data/TCGA")
output_filename = "sample_dataset_30.txt"

def hsv(tile_rgb, patch_size):
    """
    Checks if a given tile has a high concentration of tissue based on an HSV mask.
    """
    tile = np.array(tile_rgb)
    # Convert from RGB to HSV color space
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    min_ratio = .6

    # Define the color range for tissue in HSV
    lower_bound = np.array([90, 8, 103])
    upper_bound = np.array([180, 255, 255])

    # Create a mask for the specified color range
    mask = cv2.inRange(tile, lower_bound, upper_bound)

    # Calculate the ratio of tissue pixels
    ratio = np.count_nonzero(mask) / mask.size
    
    if ratio > min_ratio:
        return tile_rgb
    else:
        return None

# --- Main execution loop ---
finish = 3072 * 1000000
datas = dataset.image_files_svs

# Open the output file in write mode ('w')
# This will create the file if it doesn't exist or overwrite it if it does.
with open(output_filename, 'w') as f:
    print(f"Starting patch sampling. Output will be saved to {output_filename}")
    
    for e in range(0, finish):
        for i in range(0, len(datas)):
            path = datas[i]
            try:
                image = OpenSlide(path)
            except Exception as exc:
                print(f"Could not open {path}: {exc}")
                continue # Skip to the next image

            patch_size = 224
            
            # Iterate through each level of the slide
            for level in range(0, image.level_count):
                
                # Get dimensions for the current level being processed
                width, height = image.level_dimensions[level]
                
                # Ensure dimensions are valid for patch extraction
                if width < patch_size or height < patch_size:
                    continue

                tries = 0
                while True:
                    tries += 1
                    
                    # Randomly select a top-left coordinate for the patch
                    x = random.randint(0, width - patch_size)
                    y = random.randint(0, height - patch_size)
                    
                    # Read the region from the slide
                    patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))
                    
                    # Check if the patch contains enough tissue
                    res = hsv(patch, (patch_size, patch_size))
                    
                    if res is not None:
                        # If the patch is valid, write its info to the file
                        output_line = f"{path} {x} {y} {level}\n"
                        f.write(output_line)
                        break # Move to the next level/image
                
                    if tries >= 1000:
                        # If 1000 random patches at this level are invalid, move on
                        break

print("Done")
