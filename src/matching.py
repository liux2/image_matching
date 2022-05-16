"""
General idea:
1. Provide an input image
2. Construct feature extraction
    1. the extraction stratigy can be: https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
    2. or use sift
3. Decide whether extract feature on the go or store it ahead in db
    1. we can: https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
4. compare and return the index, maybe store it as a column in the db
"""
import cv2
import os
import numpy as np
from update import UpdateTable

# import matplotlib.pyplot as plt


class FeatureExtraction:
    """Extract features from images"""

    def __init__(self, input_image):
        # Read image as cv2 numpy array
        self.input = cv2.imread(input_image, 1)


def main():
    a = UpdateTable()
    a.add(1, np.array([[1, 2, 3]]))
    print(a.get(1))


if __name__ == "__main__":
    main()
