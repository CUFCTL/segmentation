"""Script to calculate the mean and standard deviation across an image dataset
for each RGB channel
"""
import numpy as np
from tqdm import tqdm
import glob
import os
from pathlib import Path
from PIL import Image

dataset_root = '../models/swiftnet/datasets/rellis/Rellis-3D-camera-split/'

# Get list of image paths by looping recursively
# str grabs the relative path from the Path() object
train_images_list = [str(path) for path in Path(dataset_root).rglob('train/rgb/*.*')]
test_images_list = [str(path) for path in Path(dataset_root).rglob('test/rgb/*.*')]
images_list = train_images_list + test_images_list
num_images = len(images_list)
print(f'Calculating the mean and standardard deviation from {num_images} images')

running_mean = 0
running_std = 0
for image in tqdm(images_list):
    img = np.array(Image.open(image))
    img_mean = np.mean(img, axis=(0, 1))
    img_std = np.std(img, axis=(0, 1))

    running_mean += img_mean
    running_std += img_std

dataset_mean = running_mean / num_images
dataset_std = running_std / num_images
print(f'Dataset Mean: {dataset_mean}')
print(f'Dataset Standard Deviation: {dataset_std}') # I think i need to square root this value###################
##################################################################################