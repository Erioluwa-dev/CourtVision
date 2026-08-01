"""
BallTracker (Phase 1 + Phase 3 - Ball Intelligence).

Extends the original ball tracker with:

Phase 1:
  * Interpolation when the ball is temporarily lost (occlusion,
    motion blur) using the last known velocity.

Phase 3:
  * Velocity estimation (speed, acceleration, direction) over a
    configurable window.
  * Smoothing via Kalman filter or exponential moving average.
  * Trajectory prediction for hidden-ball frames.

The class stays backward-compatible: update()/reset() and the
existing attributes (position, speed, trajectory, ...) keep their
meaning.
"""

import math
from collections import deque

import numpy as np

from . import config


class _KalmanFilter:
    """
    Minimal constant-velocity Kalman filter for the ball.
    State: [x, y, vx, vy]. No scipy / opencv dependency.
    """

    def __init__(self):
        self.initialized = False

        # State estimate and covariance.
        self.x = np.zeros(4)
        self.p = np.eye(4) * 100.0

        # Constant-velocity motion model.
        self.f = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Measurement model: we observe x and y.
        self.h = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        # Measurement noise (pixels).
        self.r = np.eye(2) * 5.0

        # Process noise (acceleration uncertainty).
        self.q = np.eye(4) * 1.0

    def initialize(self, x, y):
        self.x[:2] = (x, y)
        self.x[2:] = 0.0
        self.initialized = True

    def predict(self):
        """
        Advance the state by one frame; return predicted (x, y).
        """
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return float(self.x[0]), float(self.x[1])

    def correct(self, x, y):
        """
        Incorporate a measurement; return corrected (x, y).
        """
        z = np.array([x, y], dtype=float)

        y_residual = z - self.h @ self.x

        s = self.h @ self.p @ self.h.T + self.r
        gain = self.p @ self.h.T @ np.linalg.inv(s)

        self.x = self.x + gain @ y_residual
        self.p = (np.eye(4) - gain @ self.h) @ self.p

        return float(self.x[0]), float(self.x[1])

    def velocity(self):
        return float(self.x[2]), float(self.x[3])


class BallTracker:
    """
    Tracks the basketball throughout a game.
    """

    def __init__(self):
        """
        Initialize the ball tracker.
        """

        # Current ball position (smoothed when smoothing is on).
        self.current_position = None

        # Previous ball position.
        self.last_position = None

        # History of ball positions.
        self.trajectory = []

        # Whether the ball is visible.
        self.detected = False

        # Frames where the ball was detected.
        self.frames_seen = 0

        # Total distance travelled (detected frames only).
        self.distance_pixels = 0.0

        # Current speed (pixels/frame).
        self.current_speed = 0.0

        # Maximum speed observed.
        self.max_speed = 0.0

        # ----------------------------
        # Phase 1 - interpolation
        # ----------------------------

        # Consecutive frames the ball has been missing.
        self.lost_frames = 0

        # True when the current position was interpolated.
        self.interpolated = False

        # ----------------------------
        # Phase 3 - velocity & smoothing
        # ----------------------------

        # Recent velocity samples (vx, vy) for the rolling window.
        self._velocity_samples = deque(maxlen=config.BALL_VELOCITY_WINDOW)

        # Latest velocity estimate (pixels/frame).
        self.velocity_vector = (0.0, 0.0)

        # Latest acceleration estimate (pixels/frame^2).
        self.current_acceleration = 0.0

        self._kalman = _KalmanFilter()
        self._ema_position = None

    # --------------------------------------------------------
    # Phase 3 - estimation helpers
    # --------------------------------------------------------

    def _smooth(self, x, y):
        """
        Apply the configured smoothing method and return
        the smoothed position.
        """

        if config.BALL_SMOOTHING == "kalman":

            if not self._kalman.initialized:
                self._kalman.initialize(x, y)
                return x, y

            self._kalman.predict()
            return self._kalman.correct(x, y)

        # Exponential moving average fallback.
        if self._ema_position is None:
            self._ema_position = (x, y)
            return x, y

        alpha = config.BALL_EMA_ALPHA

        smoothed = (
            alpha * x + (1.0 - alpha) * self._ema_position[0],
            alpha * y + (1.0 - alpha) * self._ema_position[1],
        )

        self._ema_position = smoothed

        return smoothed

    def _update_velocity(self, position):
        """
        Feed a new smoothed position into the rolling velocity
        window and refresh speed / acceleration / direction.
        """

        if self.last_position is None:
            return

        vx = position[0] - self.last_position[0]
        vy = position[1] - self.last_position[1]

        self._velocity_samples.append((vx, vy))

        if len(self._velocity_samples) < 2:
            self.velocity_vector = (vx, vy)
            return

        # Average over the window to reduce jitter.
        mean_vx = sum(s[0] for s in self._velocity_samples) / len(self._velocity_samples)
        mean_vy = sum(s[1] for s in self._velocity_samples) / len(self._velocity_samples)

        last_velocity = self.velocity_vector

        self.velocity_vector = (mean_vx, mean_vy)

        speed = math.hypot(mean_vx, mean_vy)
        last_speed = math.hypot(*last_velocity)

        if last_speed > 0:
            self.current_acceleration = abs(speed - last_speed)
        else:
            self.current_acceleration = 0.0

        self.current_speed = speed

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

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

            self.lost_frames += 1
            self.interpolated = False

            # Interpolate a short gap using the last known velocity.
            if (
                self.last_position is not None
                and self.lost_frames <= config.BALL_MAX_INTERPOLATION_FRAMES
            ):

                vx, vy = self.velocity_vector

                predicted = (
                    self.current_position[0] + vx,
                    self.current_position[1] + vy,
                )

                self.last_position = self.current_position
                self.current_position = predicted

                self.interpolated = True

                self.trajectory.append(predicted)

            # Ball genuinely lost: drop the interpolated trail.
            elif self.lost_frames > config.BALL_MAX_INTERPOLATION_FRAMES:
                self.current_position = None
                self.last_position = None

            return

        # ----------------------------
        # Ball detected
        # ----------------------------

        self.detected = True

        self.frames_seen += 1

        self.lost_frames = 0

        self.interpolated = False

        self.last_position = self.current_position

        raw_position = tracked_ball["position"]

        # Apply smoothing (Kalman / EMA).
        self.current_position = self._smooth(*raw_position)

        # ----------------------------
        # Calculate movement
        # ----------------------------

        if self.last_position is not None:

            movement = math.hypot(
                self.current_position[0] - self.last_position[0],
                self.current_position[1] - self.last_position[1],
            )

            self.distance_pixels += movement

            if movement > self.max_speed:
                self.max_speed = movement

        self._update_velocity(self.current_position)

        # ----------------------------
        # Store trajectory
        # ----------------------------

        self.trajectory.append(self.current_position)

    def get_velocity(self):
        """
        Return the current (vx, vy) velocity in pixels/frame.
        """

        return self.velocity_vector

    def get_direction(self):
        """
        Return the ball's movement direction in radians.
        """

        vx, vy = self.velocity_vector

        return math.atan2(vy, vx)

    def predict_position(self, frames_ahead=5):
        """
        Predict where the ball will be in `frames_ahead` frames
        using the current velocity (constant-velocity model).
        """

        if self.current_position is None:
            return None

        vx, vy = self.velocity_vector

        return (
            self.current_position[0] + vx * frames_ahead,
            self.current_position[1] + vy * frames_ahead,
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

        self.lost_frames = 0

        self.interpolated = False

        self._velocity_samples.clear()

        self.velocity_vector = (0.0, 0.0)

        self.current_acceleration = 0.0

        self._kalman = _KalmanFilter()

        self._ema_position = None
