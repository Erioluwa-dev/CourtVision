
class CommentaryEngine:
    """
    Generates basketball commentary
    from CourtVision analytics.
    """

    def __init__(self):
        """
        Initialize the commentary engine.
        """

        self.events = []

    def add_event(
        self,
        text,
    ):
        """
        Store a new commentary event.
        """

        self.events.append(
            text,
        )

    def get_events(
        self,
    ):
        """
        Return every commentary event.
        """

        return self.events

    def latest_event(
        self,
    ):
        """
        Return the newest commentary event.
        """

        if len(self.events) == 0:
            return None

        return self.events[-1]

    def update_possession(
        self,
        possession_tracker,
    ):
        """
        Generate commentary whenever
        possession changes.
        """

        current_player = (
            possession_tracker.get_current_player()
        )

        last_player = (
            possession_tracker.get_last_player()
        )

        # Nobody currently has possession.
        if current_player is None:
            return

        # First possession of the game.
        if last_player is None:

            commentary = (
                f"Player {current_player} gains possession."
            )

        # Possession changed.
        elif current_player != last_player:

            commentary = (
                f"Player {current_player} receives the ball from "
                f"Player {last_player}."
            )

        else:
            return

        if self.latest_event() != commentary:

            self.add_event(
                commentary,
            )

    def update_passes(
        self,
        pass_detector,
    ):
        """
        Generate commentary whenever
        a new pass is detected.
        """

        passes = (
            pass_detector.get_all_passes()
        )

        if len(passes) == 0:
            return

        latest_pass = passes[-1]

        commentary = (
            f"Player {latest_pass['from']} "
            f"passes to Player "
            f"{latest_pass['to']}."
        )

        if self.latest_event() != commentary:

            self.add_event(
                commentary,
            )

    def update_shots(
        self,
        shot_detector,
    ):
        """
        Generate commentary whenever
        a shot attempt finishes.
        """

        latest_shot = (
            shot_detector.latest_shot()
        )

        if latest_shot is None:
            return

        player = latest_shot["player"]

        if latest_shot["made"]:

            commentary = (
                f"Player {player} scores!"
            )

        else:

            commentary = (
                f"Player {player} attempts a shot."
            )

        if self.latest_event() != commentary:

            self.add_event(
                commentary,
            )

    def clear_events(
        self,
    ):
        """
        Remove every stored
        commentary event.
        """

        self.events.clear()
