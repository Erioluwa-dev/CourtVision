"""
MatchData (Phase 1 - per-frame output).

Stores a structured record for every processed frame: players,
ball, rim and court state, plus possession at that frame.

Memory-bounded: a full game at 30fps would otherwise hold ~86k
frame dicts in RAM. When spool_path is set every frame is appended
to a JSONL file as it arrives, and only every sample_every-th frame
is kept in memory.
"""

import json


class MatchData:
    """
    Stores every frame of a basketball game.
    """

    def __init__(
        self,
        sample_every=1,
        spool_path=None,
    ):
        self.frames = []
        self.sample_every = max(1, sample_every)
        self.spool_path = spool_path

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

        if self.spool_path:

            with open(self.spool_path, "a") as f:
                json.dump(frame_data, f)
                f.write("\n")

        if frame_number % self.sample_every == 0:
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
        Return the entire match (in-memory sample).
        """

        return self.frames

    def save(
        self,
        path,
    ):
        """
        Write the in-memory frames to a JSON file.
        """

        with open(path, "w") as f:
            json.dump(
                {"frames": self.frames},
                f,
                indent=2,
            )
