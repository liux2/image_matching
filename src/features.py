import cv2
import numpy as np


class FeatureExtraction:
    """
    Extract features from images by using ORB.
    ORB is faster and more accurate than SIFT and SURF.
    """

    def __init__(self, input_image=""):
        # Read image as cv2 numpy array
        input = cv2.imread(input_image, 1)
        self.keypoints, self.descriptors = self.descriptor(input)

    def descriptor(self, image):
        """Generate descriptors."""
        ORB_object = cv2.ORB_create()
        keypoints = ORB_object.detect(image)
        return ORB_object.compute(image, keypoints)

    def unpickle(self, keypoint):
        """Unpack keypoints."""
        return np.asarray(
            [
                (
                    point.pt[0],
                    point.pt[1],
                    point.size,
                    point.angle,
                    point.response,
                    point.octave,
                    point.class_id,
                )
                for point in keypoint
            ]
        )

    def pickle(self, arr):
        """Pack array back to keypoint objects."""
        return tuple(
            [
                cv2.KeyPoint(
                    x=point[0],
                    y=point[1],
                    size=point[2],
                    angle=point[3],
                    response=point[4],
                    octave=int(point[5]),
                    class_id=int(point[6]),
                )
                for point in arr
            ]
        )
