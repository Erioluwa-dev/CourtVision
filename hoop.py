"""
HoopTracker (Phase 1 - Rim tracking).

Tracks the basketball hoop / rim across frames and provides a
temporally smoothed position to dampen per-frame jitter from the
classical-CV detector.
"""


class HoopTracker:
    """
    Detects and tracks the basketball hoop.
    """

    def __init__(self, smooth_window=5):

        # Current hoop location
        self.current_position = None

        # Previous hoop location
        self.last_position = None

        # History of hoop positions
        self.positions = []

        # Whether the hoop was detected
        self.detected = False

        # Number of recent positions averaged for smoothing
        self.smooth_window = smooth_window

    def update(
        self,
        hoop_position,
    ):
        """
        Update the hoop location
        for the current frame.
        """

        # ----------------------------
        # Hoop not detected
        # ----------------------------

        if hoop_position is None:

            self.detected = False

            return

        # ----------------------------
        # Hoop detected
        # ----------------------------

        self.detected = True

        self.last_position = self.current_position

        self.current_position = hoop_position

        self.positions.append(hoop_position)

    def get_smoothed_position(
        self,
    ):
        """
        Return the average of the most recent hoop positions,
        or None when nothing has been detected yet.
        """

        if not self.positions:
            return None

        recent = self.positions[-self.smooth_window:]

        avg_x = sum(p[0] for p in recent) / len(recent)
        avg_y = sum(p[1] for p in recent) / len(recent)

        return (avg_x, avg_y)
