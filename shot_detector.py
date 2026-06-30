
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

        if self.current_attempt is None:
            return

        self.current_attempt["end_frame"] = frame_number
        self.current_attempt["made"] = made

        self.shots.append(
            self.current_attempt,
        )

        self.current_attempt = None

    def get_all_shots(
        self,
    ):
        """
        Return every detected shot.
        """

        return self.shots

    def total_shots(
        self,
    ):
        """
        Return the total number
        of shot attempts.
        """

        return len(
            self.shots,
        )
