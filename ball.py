
import cv2

from tracker import centroid_distance


class BallTracker:
    """
    Tracks the basketball throughout a game.
    """

    def __init__(self):
        # Current location of the ball.
        self.current_position = None

        # Previous location of the ball.
        self.last_position = None

        # Complete history of ball positions.
        self.trajectory = []

        # Whether the ball was detected
        # in the current frame.
        self.detected = False

        # Number of frames where the
        # ball has been detected.
        self.frames_seen = 0

        # Current movement speed
        # (pixels per frame).
        self.current_speed = 0.0

        # Fastest movement observed.
        self.max_speed = 0.0

    def update(
        self,
        ball_position,
    ):
        """
        Update the basketball state
        for the current frame.
        """

        # ----------------------------
        # Ball not detected
        # ----------------------------

        if ball_position is None:

            self.detected = False

            return

        # ----------------------------
        # Ball detected
        # ----------------------------

        self.detected = True

        self.frames_seen += 1

        # ----------------------------
        # Save previous position
        # ----------------------------

        self.last_position = self.current_position

        # ----------------------------
        # Save current position
        # ----------------------------

        self.current_position = ball_position
                # ----------------------------
        # Calculate movement
        # ----------------------------

        if self.last_position is not None:

            movement = centroid_distance(
                self.last_position,
                self.current_position,
            )

            # ----------------------------
            # Total distance
            # ----------------------------

            self.distance_pixels += movement

            # ----------------------------
            # Current speed
            # ----------------------------

            self.current_speed = movement

            # ----------------------------
            # Maximum speed
            # ----------------------------

            if movement > self.max_speed:

                self.max_speed = movement

        # ----------------------------
        # Store trajectory
        # ----------------------------

        self.trajectory.append(
            ball_position,
        )
        # ----------------------------
# Calculate movement
# ----------------------------

if self.last_position is not None:

    dx = (
        self.current_position[0]
        - self.last_position[0]
    )

    dy = (
        self.current_position[1]
        - self.last_position[1]
    )

    movement = (
        dx ** 2 + dy ** 2
    ) ** 0.5

    self.current_speed = movement

    if movement > self.max_speed:

        self.max_speed = movement
