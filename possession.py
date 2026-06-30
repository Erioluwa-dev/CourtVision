
from tracker import centroid_distance


class PossessionTracker:
    """
    Determines which player
    currently possesses the basketball.
    """

    def __init__(self):

        self.current_player = None

        self.last_player = None

        self.history = []

    def update(
        self,
        tracked_players,
        tracked_ball,
    ):
        """
        Determine which player
        possesses the basketball.
        """

        # ----------------------------
        # No basketball detected
        # ----------------------------

        if tracked_ball is None:

            self.last_player = self.current_player

            self.current_player = None

            return

        # ----------------------------
        # Find closest player
        # ----------------------------

        closest_player = None

        closest_distance = float("inf")

        ball_position = tracked_ball["position"]

        for player in tracked_players:

            distance = centroid_distance(
                player["position"],
                ball_position,
            )

            if distance < closest_distance:

                closest_distance = distance

                closest_player = player["id"]

        # ----------------------------
        # Decide possession
        # ----------------------------

        POSSESSION_DISTANCE = 60

        self.last_player = self.current_player

        if closest_distance <= POSSESSION_DISTANCE:

            self.current_player = closest_player

        else:

            self.current_player = None

        # ----------------------------
        # Store possession history
        # ----------------------------

        self.history.append(
            self.current_player,
        )

    def get_current_player(
        self,
    ):
        """
        Return the player currently
        possessing the basketball.
        """

        return self.current_player

    def get_last_player(
        self,
    ):
        """
        Return the previous player
        who possessed the basketball.
        """

        return self.last_player

    def get_history(
        self,
    ):
        """
        Return the complete
        possession history.
        """

        return self.history
