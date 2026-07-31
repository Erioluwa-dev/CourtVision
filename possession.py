"""
PossessionTracker (Phase 4 - Possession Engine).

Determines which player / team controls the basketball.

Adds to the original nearest-player heuristic:

  * Velocity threshold - a ball moving faster than
    POSSESSION_MAX_BALL_SPEED is in flight (pass/shot) and cannot
    be possessed.
  * Time threshold (hysteresis) - a new candidate must hold the
    ball for POSSESSION_HYSTERESIS_FRAMES before the possessor
    switches, preventing flicker.
  * Possession timeline - ordered (player, team, start, end) list.
  * Team possession - which team controls the ball and possession
    percentages.
"""

import config


class PossessionTracker:
    """
    Determines which player
    currently possesses the basketball.
    """

    # Kept as a class attribute for backward compatibility.
    POSSESSION_DISTANCE = config.POSSESSION_DISTANCE_PX

    def __init__(self):

        self.current_player = None

        self.last_player = None

        self.history = []

        # --------------------------------------------------
        # Phase 4 additions
        # --------------------------------------------------

        # Frame counter for the current possession.
        self._frame = 0

        # Hysteresis bookkeeping.
        self._candidate = None
        self._candidate_frames = 0

        # Consecutive frames with no ball / fast ball.
        self._loose_frames = 0

        # Ordered possession timeline.
        # Entries: {"player", "team", "start_frame", "end_frame"}
        self.timeline = []

        # Ordered team possession timeline.
        # Entries: {"team", "start_frame", "end_frame"}
        self.team_possessions = []

        # Total frames each team has possessed the ball.
        self.team_possession_frames = {}

        # Total frames the ball was possessed by anyone.
        self.total_possession_frames = 0

        # player_id -> team name (fed from the TeamClassifier).
        self.team_lookup = {}

    # --------------------------------------------------------
    # Timeline helpers
    # --------------------------------------------------------

    def _open_timeline_entry(self, player, team=None):
        """
        Start a new possession entry for a player.
        """

        entry = {
            "player": player,
            "team": team,
            "start_frame": self._frame,
            "end_frame": None,
        }

        self.timeline.append(entry)

        return entry

    def _close_timeline_entry(self):
        """
        Close the current possession entry (no-op when closed).
        """

        if self.timeline and self.timeline[-1]["end_frame"] is None:
            self.timeline[-1]["end_frame"] = self._frame

    def _open_team_entry(self, team):
        """
        Start a new team possession entry.
        """

        entry = {
            "team": team,
            "start_frame": self._frame,
            "end_frame": None,
        }

        self.team_possessions.append(entry)

        return entry

    def _close_team_entry(self):
        """
        Close the current team possession entry (no-op when closed).
        """

        if self.team_possessions and self.team_possessions[-1]["end_frame"] is None:
            self.team_possessions[-1]["end_frame"] = self._frame

    def _end_possession(self):
        """
        Release the ball (loose ball / in flight / end of segment).
        """

        if self.current_player is None:
            return

        self._close_timeline_entry()
        self._close_team_entry()

        self.last_player = self.current_player
        self.current_player = None

        self._candidate = None
        self._candidate_frames = 0

    def _start_possession(self, player):
        """
        Assign possession to a player.
        """

        self.last_player = self.current_player
        self.current_player = player

        team = self.team_lookup.get(player, "Unknown")

        # Close the previous segment if it is still open.
        self._close_timeline_entry()
        self._close_team_entry()

        self._open_timeline_entry(player, team)
        self._open_team_entry(team)

        self._candidate = None
        self._candidate_frames = 0

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def update(
        self,
        tracked_players,
        tracked_ball,
        ball_speed=None,
        frame_number=None,
        team_lookup=None,
    ):
        """
        Determine which player possesses the basketball.

        Args:
            tracked_players: CourtVision player dicts.
            tracked_ball: CourtVision ball dict or None.
            ball_speed: current ball speed in px/frame. When given,
                speeds above the threshold mean the ball is in
                flight and cannot be possessed.
            frame_number: pipeline frame counter (for the timeline).
            team_lookup: player_id -> team name mapping.
        """

        if frame_number is not None:
            self._frame = frame_number
        else:
            self._frame += 1

        if team_lookup:
            self.team_lookup.update(team_lookup)

        # ----------------------------
        # No basketball detected
        # ----------------------------

        if tracked_ball is None:

            self._loose_frames += 1

            if (
                self._loose_frames >= config.POSSESSION_LOOSE_AFTER_FRAMES
                and self.current_player is not None
            ):
                self._end_possession()

            self.history.append(self.current_player)

            return

        self._loose_frames = 0

        # ----------------------------
        # Ball in flight (too fast to hold)
        # ----------------------------

        if (
            ball_speed is not None
            and ball_speed > config.POSSESSION_MAX_BALL_SPEED_PX
        ):

            if self.current_player is not None:
                self._end_possession()

            self.history.append(None)

            return

        # ----------------------------
        # Find closest player
        # ----------------------------

        closest_player = None

        closest_distance = float("inf")

        ball_position = tracked_ball["position"]

        for player in tracked_players:

            distance = (
                (player["position"][0] - ball_position[0]) ** 2
                + (player["position"][1] - ball_position[1]) ** 2
            ) ** 0.5

            if distance < closest_distance:

                closest_distance = distance

                closest_player = player["id"]

        target = (
            closest_player
            if closest_distance <= config.POSSESSION_DISTANCE_PX
            else None
        )

        # ----------------------------
        # Hysteresis: only switch after
        # the candidate holds for N frames
        # ----------------------------

        if target == self.current_player:

            # Stable possession.
            self._candidate = None
            self._candidate_frames = 0

        elif target is not None:

            # New candidate building toward a switch.
            if self._candidate != target:
                self._candidate = target
                self._candidate_frames = 0

            self._candidate_frames += 1

            if self._candidate_frames >= config.POSSESSION_HYSTERESIS_FRAMES:
                self._start_possession(target)

        else:

            # No player near the ball: possible loose ball.
            if self._candidate is not None:
                self._candidate = None
                self._candidate_frames = 0

            if (
                self.current_player is not None
                and self._candidate is None
            ):
                self._candidate_frames += 1

                if self._candidate_frames >= config.POSSESSION_HYSTERESIS_FRAMES:
                    self._end_possession()
                    self._candidate_frames = 0

        # ----------------------------
        # Store possession history
        # ----------------------------

        self.history.append(self.current_player)

        # ----------------------------
        # Team possession frames
        # ----------------------------

        if self.current_player is not None:

            team = self.team_lookup.get(
                self.current_player,
                "Unknown",
            )

            self.team_possession_frames[team] = (
                self.team_possession_frames.get(team, 0) + 1
            )

            self.total_possession_frames += 1

    # --------------------------------------------------------
    # Getters
    # --------------------------------------------------------

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

    def get_timeline(
        self,
    ):
        """
        Return the ordered possession timeline:
        [{"player", "team", "start_frame", "end_frame"}, ...]
        """

        return self.timeline

    def get_team_possessions(
        self,
    ):
        """
        Return the ordered team possession timeline:
        [{"team", "start_frame", "end_frame"}, ...]
        """

        return self.team_possessions

    def possession_percentage(
        self,
    ):
        """
        Return the possession share per team, e.g.
        {"Team A": 62.0, "Team B": 38.0}
        """

        if self.total_possession_frames == 0:
            return {}

        return {
            team: round(
                frames / self.total_possession_frames * 100.0,
                1,
            )
            for team, frames in self.team_possession_frames.items()
        }
