
from tracker import centroid_distance


class BallTracker:
    """
    Tracks the basketball throughout a game.
    """

    def __init__(self):
        """
        Initialize the ball tracker.
        """

        # Current ball position.
        self.current_position = None

        # Previous ball position.
        self.last_position = None

        # History of ball positions.
        self.trajectory = []

        # Whether the ball is visible.
        self.detected = False

        # Frames where the ball was detected.
        self.frames_seen = 0

        # Total distance travelled.
        self.distance_pixels = 0.0

        # Current speed (pixels/frame).
        self.current_speed = 0.0

        # Maximum speed observed.
        self.max_speed = 0.0

    def update(
        self,
        tracked_ball,
    ):
        """
        Update the ball tracker for one frame.
        """

        # ----------------------------
        # Ball not detected
        # ----------------------------

        if tracked_ball is None:

            self.detected = False

            return

        # ----------------------------
        # Ball detected
        # ----------------------------

        self.detected = True

        self.frames_seen += 1

        self.last_position = self.current_position

        self.current_position = tracked_ball["position"]

        # ----------------------------
        # Calculate movement
        # ----------------------------

        if self.last_position is not None:

            movement = centroid_distance(
                self.last_position,
                self.current_position,
            )

            self.distance_pixels += movement

            self.current_speed = movement

            if movement > self.max_speed:

                self.max_speed = movement

        # ----------------------------
        # Store trajectory
        # ----------------------------

        self.trajectory.append(
            self.current_position,
        )

    def reset(
        self,
    ):
        """
        Reset all tracking data.
        """

        self.current_position = None

        self.last_position = None

        self.trajectory = []

        self.detected = False

        self.frames_seen = 0

        self.distance_pixels = 0.0

        self.current_speed = 0.0

        self.max_speed = 0.0
