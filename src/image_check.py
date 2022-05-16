#!/bin/bash
"""Check if there are any broken images, and index them."""

import os
from PIL import Image, ImageFile, UnidentifiedImageError
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

# Declear
ImageFile.LOAD_TRUNCATED_IMAGES = True
folder_path = "./img"
extensions = []

# Create database
engine = create_engine("sqlite:///imageMeta.db", echo=True)
meta = MetaData()
images = Table(
    "images",
    meta,
    Column("id", Integer, primary_key=True),
    Column("filename", String),
    Column("feature", String),
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
for filee in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filee)
    conn = engine.connect()
    fin = conn.execute(images.insert().values(filename=file_path))
