#!/bin/bash
"""Check if there are any broken images, and index them."""
# util
import os
from PIL import Image, ImageFile, UnidentifiedImageError
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import numpy as np

# custom
from update import UpdateTable
from features import FeatureExtraction

# Declear
ImageFile.LOAD_TRUNCATED_IMAGES = True
folder_path = "/Users/anerypatel/Desktop/Spring 2022/CV-495/image_matching/img"
extensions = []

# Create database
engine = create_engine("sqlite:///imageMeta.db", echo=True)
meta = MetaData()
images = Table(
    "images",
    meta,
    Column("id", Integer, primary_key=True),
    Column("filename", String),
    Column("ORB_keypoint", String),
    Column("ORB_descriptor", String),
    Column("KAZE_keypoint", String),
    Column("KAZE_descriptor", String),
)
meta.create_all(engine)

# Checking images
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

# Indexing images
tb = UpdateTable()

for filee in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filee)
    conn = engine.connect()
    fin = conn.execute(images.insert().values(filename=file_path))
    ftr = FeatureExtraction(file_path)
    tb.add_by_filename(
        file_path, ftr.unpickle(ftr.keypoints), np.asarray(ftr.descriptors)
    )
