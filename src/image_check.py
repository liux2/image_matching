#!/bin/bash
"""Check if there are any broken images."""

import os
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
folder_path = "./img"
extensions = []
for filee in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filee)
    print("** Path: {}  **".format(file_path), end="\r", flush=True)
    try:
        im = Image.open(file_path)
        rgb_im = im.convert("RGB")
    except UnidentifiedImageError:
        os.remove(file_path)
        print("Deleted: ", file_path)
    if filee.split(".")[1] not in extensions:
        extensions.append(filee.split(".")[1])
