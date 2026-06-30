
class HoopTracker:
    """
    Detects and tracks the basketball hoop.
    """

    def __init__(self):

        # Current hoop location
        self.current_position = None

        # Previous hoop location
        self.last_position = None

        # History of hoop positions
        self.positions = []

        # Whether the hoop was detected
        self.detected = False

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

        # ----------------------------
        # Save previous position
        # ----------------------------

        self.last_position = self.current_position

        # ----------------------------
        # Save current position
        # ----------------------------

        self.current_position = hoop_position

        # ----------------------------
        # Store history
        # ----------------------------

        self.positions.append(
            hoop_position,
        )
