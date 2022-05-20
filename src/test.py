from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import sqlite3
import numpy as np
import io
import cv2
from features import FeatureExtraction
from update import UpdateTable


a = UpdateTable()
ftr = FeatureExtraction("./img/3657209354_cde9bbd2c5.jpg")
print(type(ftr.unpickle(ftr.keypoints)[0]))
# print(ftr.pickle(ftr.unpickle(ftr.keypoints)))
a.add_by_filename(
    "./img/3657209354_cde9bbd2c5.jpg", np.asarray([1, 2, 3]), np.asarray([1, 2, 3]),
)

# b = UpdateTable()
# b.add_by_id(1, np.asarray([1, 2, 3]), np.asarray([1, 2, 3]))
# print(b.get_by_id(1))

# imageresult = cv2.drawKeypoints(
#     cv2.imread("./img/3657209354_cde9bbd2c5.jpg", 1),
#     ftr.keypoints,
#     None,
#     color=(255, 0, 0),
#     flags=0,
# )
# cv2.imshow("ORB_image", imageresult)
# cv2.waitKey()
