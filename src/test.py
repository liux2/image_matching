from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import sqlalchemy as db
import sqlite3
import numpy as np
import io
import cv2
from features import FeatureExtraction
from update import UpdateTable
import matplotlib.pyplot as plt
import ntpath

a = UpdateTable()
ftr = FeatureExtraction()
key1, des1 = a.get_by_id(1)
key2, des2 = a.get_by_filename("./img/456512643_0aac2fa9ce.jpg")

key1 = ftr.pickle(key1)
key2 = ftr.pickle(key2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img1 = cv2.imread(a.get_filename(1), 1)
img2 = cv2.imread("./img/456512643_0aac2fa9ce.jpg", 1)

out = cv2.drawMatches(img1, key1, img2, key2, matches, None)
plt.imshow(out)
plt.show()
# bf = cv2.BFMatcher()
#
# matches = bf.knnMatch(des1, des2, k=2)
#
# good = []
# for m, n in matches:
#     # print(m.distance)
#     if m.distance < 0.75 * n.distance:
#         good.append([m])
#         a = len(good)
#         percent = (a * 100) / len(key2)
#         print("{} % similarity".format(percent))
#         if percent >= 75.00:
#             print("Match Found")
#         if percent < 75.00:
#             print("Match not Found")

# img1 = cv2.imread(a.get_filename(1), 1)
# img2 = cv2.imread(a.get_filename(2), 1)
# img3 = cv2.drawMatchesKnn(img1, key1, img2, key2, matches, None, flags=2)
# plt.imshow(img3)
# plt.show()

# imageresult = cv2.drawKeypoints(
#     cv2.imread("./img/3657209354_cde9bbd2c5.jpg", 1),
#     ftr.keypoints,
#     None,
#     color=(255, 0, 0),
#     flags=0,
# )
# cv2.imshow("ORB_image", imageresult)
# cv2.waitKey()
