"""
MatchData (Phase 1 - per-frame output).

Stores a structured record for every processed frame: players,
ball, rim and court state, plus possession at that frame.
"""


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
        tracked_ball=None,
        rim_position=None,
        court_lines=None,
        possession_player=None,
    ):
        """
        Store one processed frame.

        Args:
            frame_number: pipeline frame counter.
            tracked_players: CourtVision player dicts.
            tracked_ball: CourtVision ball dict or None.
            rim_position: rim centre (x, y) or None.
            court_lines: detected court line segments or None.
            possession_player: current possessor id or None.
        """

        frame_data = {
            "frame": frame_number,
            "players": tracked_players,
            "ball": tracked_ball,
            "rim": rim_position,
            "court_lines": court_lines,
            "possession": possession_player,
        }

        self.frames.append(frame_data)

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
