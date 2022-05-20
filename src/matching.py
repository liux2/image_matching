"""
General idea:
1. Provide an input image
2. Construct feature extraction
3. Decide whether extract feature on the go or store it ahead in db
4. compare and return the index, maybe store it as a column in the db
Resources:
CBIR system intro https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=410145
similarity retrivval: https://ieeexplore.ieee.org/document/1529438
"""
import cv2
import os
import numpy as np
from update import UpdateTable
from features import FeatureExtraction

# import matplotlib.pyplot as plt


class SimilarityRetrival:
    """Retrive similar images."""


def main():
    a = UpdateTable()
    ftr = FeatureExtraction()
    key, des = a.get_by_filename("./img/3657209354_cde9bbd2c5.jpg")


if __name__ == "__main__":
    main()
