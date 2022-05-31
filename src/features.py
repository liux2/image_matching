import cv2
import numpy as np


class FeatureExtraction:
    """
    Extract features from images by using ORB.
    ORB is faster and more accurate than SIFT and SURF.
    """

    def __init__(self):
        pass

    def descriptor(self, input_image, method):
        """Generate descriptors."""
        # Read image as cv2 numpy array
        image = cv2.imread(input_image, 1)
        if method == "ORB":
            ORB_object = cv2.ORB_create()
            keypoints = ORB_object.detect(image)
            return ORB_object.compute(image, keypoints)
        elif method == "KAZE":
            vector_size = 32
            alg = cv2.KAZE_create()
            kps = alg.detect(image)
            # kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            return alg.compute(image, kps)
            # Flatten all of them in one big vector - our feature vector
            # dsc = dsc.flatten()
            # Making descriptor of same size
            # Descriptor vector size is 64
            # needed_size = (vector_size * 64)
            # if dsc.size < needed_size:
            #     # if we have less the 32 descriptors then just adding zeros at the
            #     # end of our feature vector
            #     dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

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
