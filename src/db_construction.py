#!/bin/bash
"""Check if there are any broken images, and index them."""
# util
import os
from PIL import Image, ImageFile, UnidentifiedImageError
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import numpy as np
import pandas as pd
import spacy

# custom
from update import UpdateTable
from features import FeatureExtraction

# Declear
ImageFile.LOAD_TRUNCATED_IMAGES = True
folder_path = "./img/images"
extensions = []

# Create database
engine = create_engine("sqlite:///imageMeta.db", echo=True)
meta = MetaData()
images = Table(
    "images",
    meta,
    Column("id", Integer, primary_key=True),
    Column("filename", String),
    Column("caption", String),
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

# Import captions
caps = pd.read_csv("./img/captions.csv")
nlp = spacy.load("en_core_web_trf")


def keywords(file_name, caps, nlp):
    """Extracting keywords."""
    keyword = []
    caps = caps.loc[caps["image"] == file_name]["caption"]
    for cap in caps:
        doc = nlp(cap)
        keyword.extend(
            [
                str(tok)
                for tok in doc
                if (tok.dep_ == "nsubj")
                if str(tok) not in keyword
            ]
        )
    return keyword


# Indexing images
tb = UpdateTable()


for filee in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filee)
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(
                images.insert(),
                {
                    "filename": file_path,
                    "caption": np.asarray(keywords(filee, caps, nlp)),
                },
            )
            # conn.execute(images.insert().values(filename=file_path))
            # conn.execute(
            #     images.insert().values(caption=np.asarray(keywords(filee, caps, nlp)))
            # )
    ftr = FeatureExtraction()
    orb_kps, orb_des = ftr.descriptor(file_path, "ORB")
    kaze_kps, kaze_des = ftr.descriptor(file_path, "KAZE")
    tb.add_by_filename(
        file_path, ftr.unpickle(orb_kps), orb_des, ftr.unpickle(kaze_kps), kaze_des,
    )
