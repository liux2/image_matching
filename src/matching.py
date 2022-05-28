"""
General idea:
1. Provide an input image
4. compare and return the index, maybe store it as a column in the db
Resources:
CBIR system intro https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=410145
similarity retrivval: https://ieeexplore.ieee.org/document/1529438
"""
import cv2
import os
import shutil
import ntpath
import numpy as np
from update import UpdateTable
from features import FeatureExtraction


class SimilarityRetrival:
    """Retrive similar images."""

    def __init__(self, query_img):
        self.table = UpdateTable()
        self.feature = FeatureExtraction()
        self.candidates = self.get_candidates(query_img)

    def get_matches_ORB(self, query_img, target_image):
        """Get a list matching points."""
        key1, des1 = self.table.get_by_id(query_img)
        key1 = self.feature.pickle(key1)

        key2, des2 = self.table.get_by_id(target_image)
        key2 = self.feature.pickle(key2)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches, key2

    def decide_qualif(self, matches, key2):
        """Calculate the distance, decide whether a good candidate."""
        valid_points = []
        # matches = matches[: int(len(matches) * 0.90)]
        # for m, n in matches:
        #     if m.distance < 0.95 * n.distance:
        #         valid_points.append([m])
        percent = (len(matches) * 100) / len(key2)
        if percent >= 30.00:
            return True
        if percent < 30.00:
            return False

    def get_candidates(self, query_img):
        """Get a list of candidates."""
        candidates = []
        # Fetching ids
        s = self.table.images.select().with_only_columns(self.table.images.c.id)
        res = self.table.conn.execute(s).fetchall()
        for row in res:
            target_image = row[0]
            if query_img != target_image:
                matches, key2 = self.get_matches_ORB(query_img, target_image)
                if self.decide_qualif(matches, key2):
                    candidates.append(self.table.get_filename(target_image))
        return candidates


def main():
    """Pipelines."""
    query_img = 1
    table = UpdateTable()
    sim = SimilarityRetrival(query_img)

    # Clear files for new retrival.
    for filee in os.listdir("./results"):
        file_path = os.path.join("./results", filee)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

    shutil.copyfile(table.get_filename(query_img), "./results/input")
    for src in sim.candidates:
        shutil.copyfile(src, "./results/" + ntpath.basename(src))


if __name__ == "__main__":
    main()
