
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
