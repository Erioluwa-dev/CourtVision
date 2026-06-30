
class PassDetector:
    """
    Detects completed passes
    between players.
    """

    def __init__(self):

        self.last_possessor = None

        self.current_possessor = None

        self.passes = []

    def update(
        self,
        current_possessor,
    ):
        """
        Detect completed passes.
        """

        # ----------------------------
        # Save previous possessor
        # ----------------------------

        self.last_possessor = self.current_possessor

        # ----------------------------
        # Update current possessor
        # ----------------------------

        self.current_possessor = current_possessor

        # ----------------------------
        # Nobody has possession
        # ----------------------------

        if self.current_possessor is None:

            return

        # ----------------------------
        # First possession
        # ----------------------------

        if self.last_possessor is None:

            return

        # ----------------------------
        # Same player still has ball
        # ----------------------------

        if self.last_possessor == self.current_possessor:

            return

        # ----------------------------
        # A pass has occurred
        # ----------------------------

        self.record_pass(
            self.last_possessor,
            self.current_possessor,
        )

    def record_pass(
        self,
        from_player,
        to_player,
    ):
        """
        Store a completed pass.
        """

        pass_event = {
            "from": from_player,
            "to": to_player,
        }

        self.passes.append(
            pass_event,
        )

    def get_all_passes(
        self,
    ):
        """
        Return every detected pass.
        """

        return self.passes

    def total_passes(
        self,
    ):
        """
        Return the number
        of detected passes.
        """

        return len(
            self.passes,
        )
