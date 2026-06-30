
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
            or self.court_points is None
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

        if self.homography is None:

            raise ValueError(
                "Homography has not been computed."
            )

        point = np.array(
            [[pixel]],
            dtype=np.float32,
        )

        court_point = cv2.perspectiveTransform(
            point,
            self.homography,
        )

        return tuple(
            court_point[0][0]
        )

    def map_players(
        self,
        tracked_players,
    ):
        """
        Convert every tracked player
        into court coordinates.
        """

        mapped_players = []

        for player in tracked_players:

            mapped_position = self.pixel_to_court(
                player["position"],
            )

            mapped_player = player.copy()

            mapped_player[
                "court_position"
            ] = mapped_position

            mapped_players.append(
                mapped_player,
            )

        return mapped_players

    def map_ball(
        self,
        tracked_ball,
    ):
        """
        Convert the basketball
        into court coordinates.
        """

        if tracked_ball is None:
            return None

        mapped_ball = tracked_ball.copy()

        mapped_ball[
            "court_position"
        ] = self.pixel_to_court(
            tracked_ball["position"],
        )

        return mapped_ball
