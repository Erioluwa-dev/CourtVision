
class ShotDetector:
    """
    Detects shot attempts.
    """

    def __init__(self):

        self.shots = []

        self.current_attempt = None

    def start_attempt(
        self,
        player_id,
        frame_number,
    ):
        """
        Begin tracking a shot attempt.
        """

        # Ignore if a shot is already active.
        if self.current_attempt is not None:
            return

        self.current_attempt = {
            "player": player_id,
            "start_frame": frame_number,
            "end_frame": None,
            "made": None,
        }

    def finish_attempt(
        self,
        frame_number,
        made,
    ):
        """
        Finish the current shot attempt.
        """

        # No shot is currently active.
        if self.current_attempt is None:
            return

        # Record when the shot ended.
        self.current_attempt["end_frame"] = frame_number

        # Record whether the shot was made.
        self.current_attempt["made"] = made

        # Save the completed attempt.
        self.shots.append(
            self.current_attempt,
        )

        # Reset for the next shot.
        self.current_attempt = None
