import cv2
import numpy as np


class CourtMapper:
    """
    Maps image coordinates
    to real basketball court coordinates.
    """

    def __init__(self):
        """
        Initialize the court mapper.
        """

        self.image_points = None

        self.court_points = None

        self.homography = None

    def set_reference_points(
        self,
        image_points,
        court_points,
    ):
        """
        Store matching image and
        court reference points.
        """

        self.image_points = np.array(
            image_points,
            dtype=np.float32,
        )

        self.court_points = np.array(
            court_points,
            dtype=np.float32,
        )

    def compute_homography(
        self,
    ):
        """
        Compute the homography matrix.
        """

        if (
            self.image_points is None
            or
            self.court_points is None
        ):
            raise ValueError(
                "Reference points have not been set."
            )

        self.homography, _ = cv2.findHomography(
            self.image_points,
            self.court_points,
        )

    def pixel_to_court(
        self,
        pixel,
    ):
        """
        Convert an image point
        into court coordinates.
        """
