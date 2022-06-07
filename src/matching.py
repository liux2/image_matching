import cv2
import os
import shutil
import ntpath
import numpy as np
from update import UpdateTable
from features import FeatureExtraction
import argparse


class SimilarityRetrival:
    """Retrive similar images."""

    def __init__(self, query_img=1, method="ORB"):
        self.table = UpdateTable()
        self.feature = FeatureExtraction()
        self.candidates = self.get_candidates(query_img, method)

    def get_matches_ORB(self, query_img, target_image):
        """Get a list matching points."""
        key1, des1 = self.table.get_by_id(query_img, method="ORB")
        key1 = self.feature.pickle(key1)
        print(target_image)
        key2, des2 = self.table.get_by_id(target_image, method="ORB")
        key2 = self.feature.pickle(key2)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches, key2

    def get_matches_KAZE(self, query_img, target_image):
        """Get a list matching points."""
        key1, des1 = self.table.get_by_id(query_img, method="KAZE")
        key1 = self.feature.pickle(key1)

        key2, des2 = self.table.get_by_id(target_image, method="KAZE")
        key2 = self.feature.pickle(key2)

        bf = cv2.BFMatcher()
        return bf.knnMatch(des1, des2, k=2), key2

    def decide_qualif(self, matches, key2, method):
        """Calculate the distance, decide whether a good candidate."""
        if method == "ORB":
            percent = (len(matches) * 100) / len(key2)
            if percent >= 30.00:
                return True
            if percent < 30.00:
                return False
        elif method == "KAZE":
            valid_points = []
            for m, n in matches:
                if m.distance < 0.95 * n.distance:
                    valid_points.append([m])
            percent = (len(valid_points) * 100) / len(key2)
            if percent >= 75.00:
                return True
            if percent < 75.00:
                return False

    def filter_content(self, cap1, cap2):
        """Get whether the content matches"""
        if len(np.intersect1d(cap1, cap2)) != 0:
            return True
        else:
            return False

    def get_candidates(self, query_img, method):
        """Get a list of candidates."""
        candidates = []
        # Fetching ids
        s = self.table.images.select().with_only_columns(self.table.images.c.id)
        res = self.table.conn.execute(s).fetchall()
        for row in res:
            target_image = row[0]
            if query_img != target_image:
                if method == "ORB":
                    matches, key2 = self.get_matches_ORB(query_img, target_image)
                    # cap1, cap2 = (
                    #     self.table.get_caption(id=query_img),
                    #     self.table.get_caption(id=target_image),
                    # )
                    if self.decide_qualif(
                        matches, key2, "ORB"
                    ):  # and self.filter_content(cap1, cap2):
                        candidates.append(self.table.get_filename(target_image))
                elif method == "KAZE":
                    matches, key2 = self.get_matches_KAZE(query_img, target_image)
                    # cap1, cap2 = (
                    #     self.table.get_caption(id=query_img),
                    #     self.table.get_caption(id=target_image),
                    # )
                    if self.decide_qualif(matches, key2, "KAZE"):
                        # and self.filter_content(cap1, cap2):
                        candidates.append(self.table.get_filename(target_image))
        return candidates


def main():
    """Pipelines."""
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Relative path in terms of image_matching directory to the query image.",
    )
    parser.add_argument("-m", "--method", type=str, help="KAZE or ORB.")
    args = parser.parse_args()

    table = UpdateTable()
    query_id = table.get_id(args.file)
    sim = SimilarityRetrival(query_img=query_id, method=args.method)

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

    shutil.copyfile(args.file, "./results/input")

    for src in sim.candidates:
        shutil.copyfile(
            src, "./results/" + ntpath.basename(src),
        )


if __name__ == "__main__":
    main()
