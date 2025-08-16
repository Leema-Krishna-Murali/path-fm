from openslide import OpenSlide#other options?

import numpy as np


means = []
stds = []
i = 0
with open("sample_dataset_30.txt","r") as f:
    
    for line in f.readlines():
        path, x, y, level = line.split(" ")

        x = int(x)
        y = int(y)
        level = int(level)

        image = OpenSlide(path)

        patch_size = 224
            #read_region is based on the top left pixel in the level 0, not our current level
        img = image.read_region((x, y), level=level, size=(patch_size, patch_size)).convert("RGB")

        np_img = np.array(img).astype(np.float32) / 255.0
    # Calculate mean and std for the current image
        mean = np.mean(np_img, axis=(0, 1))
        std = np.std(np_img, axis=(0, 1))
                
        means.append(mean)
        stds.append(std)
        i += 1
        print(i)
        if i == 50000:
            break

print(np.mean(means, axis=0))
print(np.mean(stds, axis=0))
