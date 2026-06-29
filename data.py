
class MatchData:
    """
    Stores every frame of a basketball game.
    """

    def __init__(self):
        self.frames = []

    def add_frame(
        self,
        frame_number,
        tracked_players,
    ):
        """
        Store one processed frame.
        """

        frame_data = {
            "frame": frame_number,
            "players": tracked_players,
        }

        self.frames.append(
            frame_data
        )

    def get_frame(
        self,
        frame_number,
    ):
        """
        Return one frame.
        """

        for frame in self.frames:

            if frame["frame"] == frame_number:
                return frame

        return None

    def get_all_frames(
        self,
    ):
        """
        Return the entire match.
        """

        return self.frames
