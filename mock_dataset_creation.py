#!/usr/bin/env python3
"""
Mock Dataset Generator for AmCLR Testing

Creates a minimal mock version of the datasets required for the project
following the exact directory structure:

.
+--bimodal_exps (code)
+--clip_train (captions)
|  +--cc3m_train_subset.json
|  +--coco_val.json
+--datasets (images)
|  +--cc3m_subset_100k
|  +--mscoco_val
|  +--imagnet
|     +--val

The script generates:
- 128 random noise images for each dataset with correct dimensions:
  - CC3M and MSCOCO: 640x480
  - ImageNet: 224x224
- Caption JSON files matching the expected format with image_id fields
"""

import os
import json
import numpy as np
from PIL import Image

def create_noise_image(width, height, path):
    """
    Create a random noise image of specified dimensions and save it to the given path.
    
    Args:
        width (int): Width of the image in pixels
        height (int): Height of the image in pixels
        path (str): Save path for the image
    """
    noise = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, 'JPEG')

directories = [
    'clip_train',
    'datasets/cc3m_subset_100k',
    'datasets/mscoco_val',
    'datasets/imagenet/val'
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)

cc3m_captions = []
coco_captions = []
NUM_IMAGES = 128

for i in range(NUM_IMAGES):
    cc3m_filename = f'mock_cc3m_{i:03d}.jpg'
    coco_filename = f'mock_coco_{i:03d}.jpg'
    imagenet_filename = f'mock_imagenet_{i:03d}.jpg'
    
    create_noise_image(640, 480, f'datasets/cc3m_subset_100k/{cc3m_filename}')
    create_noise_image(640, 480, f'datasets/mscoco_val/{coco_filename}')
    create_noise_image(224, 224, f'datasets/imagenet/val/{imagenet_filename}')
    
    cc3m_captions.append({
        "image_id": cc3m_filename,
        "caption": f"A sample image description for image {i}",
        "image": cc3m_filename
    })
    
    coco_captions.append({
        "image_id": coco_filename,
        "caption": f"A mock COCO image caption for image {i}",
        "image": coco_filename
    })

with open('clip_train/cc3m_train_subset.json', 'w') as f:
    json.dump(cc3m_captions, f, indent=2)

with open('clip_train/coco_val.json', 'w') as f:
    json.dump(coco_captions, f, indent=2)

print(f"Mock datasets created successfully with {NUM_IMAGES} images per dataset!")