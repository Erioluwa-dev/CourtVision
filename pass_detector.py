"""
PassDetector (Phase 5 - Passing Engine).

Detects passes by observing possession changes, and extends the
original detector with:

  * Success / failure classification - a pass is successful when
    the receiving player is on the same team; a possession change
    to the opposing team (or after a long loose-ball gap) is a
    failed pass.
  * Pass map - (from, to) player pair counts.
  * Passing network - per-team (from, to) adjacency.

The original event shape is preserved: every event keeps "from"
and "to" keys, plus richer metadata.
"""

import config


class PassDetector:
    """
    Detects completed passes
    between players.
    """

    def __init__(self):

        self.last_possessor = None

        self.current_possessor = None

        self.passes = []

        # --------------------------------------------------
        # Phase 5 additions
        # --------------------------------------------------

        # The last player to hold the ball (kept across the
        # loose/in-flight gap so passes are not lost).
        self._last_valid_possessor = None

        # Frame the last valid possession was established.
        self._last_possession_frame = None

        # Consecutive loose-ball frames since possession was lost.
        self._loose_frames = 0

        # player_id -> team name.
        self.team_lookup = {}

        # Pass map: (from_player, to_player) -> count.
        self.pass_map = {}

        # Passing network: team -> {(from, to): count}.
        self.networks = {}

    def _record_in_map(self, pass_event):
        """
        Update the pass map and passing network.
        """

        pair = (pass_event["from"], pass_event["to"])

        self.pass_map[pair] = self.pass_map.get(pair, 0) + 1

        team = pass_event.get("from_team", "Unknown")

        if team not in self.networks:
            self.networks[team] = {}

        self.networks[team][pair] = (
            self.networks[team].get(pair, 0) + 1
        )

    def update(
        self,
        current_possessor,
        frame_number=None,
        team_lookup=None,
    ):
        """
        Detect completed passes.

        Args:
            current_possessor: the player currently possessing the
                ball, or None when the ball is loose / in flight.
            frame_number: pipeline frame counter.
            team_lookup: player_id -> team name mapping.
        """

        if team_lookup:
            self.team_lookup.update(team_lookup)

        # Public attributes kept for introspection.
        self.last_possessor = self._last_valid_possessor
        self.current_possessor = current_possessor

        # ----------------------------
        # Ball is loose / in flight
        # ----------------------------

        if current_possessor is None:

            self._loose_frames += 1

            return

        # ----------------------------
        # First possession
        # ----------------------------

        if self._last_valid_possessor is None:

            self._last_valid_possessor = current_possessor

            self._last_possession_frame = frame_number

            self._loose_frames = 0

            return

        # ----------------------------
        # Same player still has ball
        # ----------------------------

        if self._last_valid_possessor == current_possessor:

            self._last_possession_frame = frame_number

            self._loose_frames = 0

            return

        # ----------------------------
        # A possession change occurred.
        # Classify it as a pass.
        # ----------------------------

        self.record_pass(
            self._last_valid_possessor,
            current_possessor,
            frame_number=frame_number,
        )

        self._last_valid_possessor = current_possessor

        self._last_possession_frame = frame_number

        self._loose_frames = 0

    def record_pass(
        self,
        from_player,
        to_player,
        frame_number=None,
        loose_frames=None,
    ):
        """
        Store a completed pass, classifying success/failure.
        """

        from_team = self.team_lookup.get(
            from_player,
            "Unknown",
        )

        to_team = self.team_lookup.get(
            to_player,
            "Unknown",
        )

        if loose_frames is None:
            loose_frames = self._loose_frames

        # --------------------------------------------------
        # Success heuristic
        # --------------------------------------------------
        # Same team -> a pass was completed.
        # Different team -> the ball changed sides (failed pass).
        # Unknown teams -> fall back to the loose-ball gap: a
        # possession regained after a long gap is treated as
        # failed, a quick catch as successful.
        # --------------------------------------------------

        if from_team != "Unknown" and to_team != "Unknown":
            success = from_team == to_team
        else:
            success = loose_frames <= config.PASS_FAIL_LOOSE_FRAMES

        pass_event = {
            "from": from_player,
            "to": to_player,
            "success": success,
            "from_team": from_team,
            "to_team": to_team,
            "loose_frames": loose_frames,
            "frame": frame_number,
        }

        self.passes.append(pass_event)

        self._record_in_map(pass_event)

    # --------------------------------------------------------
    # Getters
    # --------------------------------------------------------

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
        Return the number of detected pass events.
        """

        return len(self.passes)

    def successful_passes(
        self,
    ):
        """
        Return the number of successful passes.
        """

        return sum(
            1 for p in self.passes if p["success"]
        )

    def failed_passes(
        self,
    ):
        """
        Return the number of failed passes.
        """

        return sum(
            1 for p in self.passes if not p["success"]
        )

    def get_pass_map(
        self,
    ):
        """
        Return the pass map:
        {(from_player, to_player): count}
        """

        return self.pass_map

    def get_passing_network(
        self,
    ):
        """
        Return the passing network per team:
        {team: {(from_player, to_player): count}}
        """

        return self.networks
