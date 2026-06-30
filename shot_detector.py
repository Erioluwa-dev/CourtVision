
from tracker import centroid_distance


class ShotDetector:
    """
    Detects shot attempts.
    """

    def __init__(self):

        self.shots = []

        self.current_attempt = None

        self.last_possessor = None

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

    def latest_shot(
        self,
    ):
        """
        Return the newest shot.
        """

        if len(self.shots) == 0:
            return None

        return self.shots[-1]

    def update(
        self,
        possession_tracker,
        tracked_players,
        tracked_ball,
        frame_number,
    ):
        """
        Detect shot attempts.
        """

        shooter = (
            possession_tracker.get_current_player()
        )

        if shooter is None:
            return

        if tracked_ball is None:
            return

        # ----------------------------
        # Find the shooter
        # ----------------------------

        shooter_data = None

        for player in tracked_players:

            if player["id"] == shooter:

                shooter_data = player

                break

        if shooter_data is None:
            return

        # ----------------------------
        # Measure distance
        # ----------------------------

        distance = centroid_distance(
            shooter_data["position"],
            tracked_ball["position"],
        )

        SHOT_DISTANCE = 120

        # ----------------------------
        # Shot begins
        # ----------------------------

        if (
            distance > SHOT_DISTANCE
            and self.current_attempt is None
        ):

            self.start_attempt(
                shooter,
                frame_number,
            )

        # ----------------------------
        # Shot ends
        # ----------------------------

        current_possessor = (
            possession_tracker.get_current_player()
        )

        if (
            self.current_attempt is not None
            and current_possessor is not None
            and current_possessor != shooter
        ):

            self.finish_attempt(
                frame_number,
                made=False,
            )
