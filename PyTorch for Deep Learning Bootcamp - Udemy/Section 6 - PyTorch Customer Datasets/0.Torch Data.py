import os

import torch
from torch import nn

# Our dataset is a subset of the Food101 dataset


import requests
import zipfile
from pathlib import Path


data_path  = (Path("/Users/benjaminbrooke/PycharmProjects/Python_PyTroch/PyTorch for Deep Learning Bootcamp - Udemy/Section 6 - PyTorch Customer Datasets/"))
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)




def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents."""

    for  dir_path, dirnames, filesnames in os.walk(dir_path):
        print(f"There  are {len(dirnames)}  directories and {len(filesnames)} images in '{dir_path}'.")

walk_through_dir(image_path)



train_dir = image_path / "train"
test_dir = image_path / "test"


print(train_dir,test_dir)








import random
from PIL import Image

# Set seed
random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")









import numpy as np
import matplotlib.pyplot as plt

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);


print(img_as_array)
"""
[[[  7  18  38]
  [  5  16  34]
  [  2  13  31]
  ...
  [ 18   7  11]
  [ 12   6   8]
  [  8   4   5]]

 [[  2  13  33]
  [  2  13  31]
  [  2  13  31]
  ...
  [ 18   9  10]
  [ 17  11  13]
  [ 16  12  13]]

 [[  0  11  31]
  [  1  12  30]
  [  3  14  32]
  ...
  [ 17   8   9]
  [ 16  10  12]
  [ 15  11  12]]

 ...

 [[226 249 255]
  [229 253 255]
  [229 255 254]
  ...
  [250 149  95]
  [240 139  83]
  [234 130  75]]

 [[220 247 254]
  [224 252 255]
  [226 255 253]
  ...
  [236 133  88]
  [195  88  42]
  [170  62  16]]

 [[218 247 255]
  [223 252 255]
  [224 254 254]
  ...
  [255 157 116]
  [208  99  60]
  [166  52  15]]]
"""


from torch.utils.data import DataLoader
from torchvision import datasets, transforms


data_transform = transforms.Compose([])













