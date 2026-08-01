"""
ShotDetector (Phase 6 - Shooting Engine).

Detects shot attempts and extends the original detector with:

  * Rim interaction - the ball must come close to the rim for an
    attempt to be made; passing below the rim plane marks a make.
  * Shot type - 3PT vs 2PT from the release distance in court
    space. (FT% needs foul detection - future work.)
  * Field goal / three point percentages.
  * Shot chart - release locations with outcomes.
  * Expected Shot Value (ESV) - a distance-based make-probability
    model (see config.ESV_MAKE_*).

Original attributes and methods (start_attempt, finish_attempt,
get_all_shots, total_shots, latest_shot) are preserved.
"""

import math

from . import config
from .zones import classify_position


class ShotDetector:
    """
    Detects shot attempts.
    """

    def __init__(self):
        """
        Initialize the shot detector.
        """

        self.shots = []

        self.current_attempt = None

        # Distance (in pixels) the ball must
        # travel away from the shooter before
        # we consider it a shot attempt.
        self.shot_start_distance = config.SHOT_START_DISTANCE_PX

        # --------------------------------------------------
        # Phase 6 additions
        # --------------------------------------------------

        # Rim centre (x, y) in image space, fed by the rim
        # detector each frame.
        self.rim_position = None

        # Frame the rim position was last refreshed from a detection.
        self.rim_last_frame = None

        # Maximum frames an attempt may stay open before it is
        # force-finished (ball out of bounds, never regained).
        self.max_attempt_frames = 150

        # Shooting tallies.
        self.fg_attempts = 0
        self.fg_made = 0
        self.threepoint_attempts = 0
        self.threepoint_made = 0
        # FT tallies exist but stay 0 until foul detection
        # (referee/whistle events) is implemented.
        self.ft_attempts = 0
        self.ft_made = 0

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def set_rim_position(self, rim_position, frame=None):
        """
        Feed the current rim centre (x, y) from the rim detector.
        """

        self.rim_position = rim_position

        if frame is not None:
            self.rim_last_frame = frame

    @staticmethod
    def _ball_descending(ball_points, bx, by):
        """
        True when the ball was moving downward (image y increasing)
        as it crossed the rim plane, so a rising ball passing under
        the rim (floater, lob, pass) is not counted as a make.
        """

        if len(ball_points) < 2:
            return True

        index = len(ball_points)

        for i, (px, py) in enumerate(ball_points):
            if px == bx and py == by:
                index = i
                break

        before = ball_points[index - 1] if index > 0 else None
        after = ball_points[index + 1] if index < len(ball_points) - 1 else None

        if after is not None:
            return after[1] > by

        if before is not None:
            return by > before[1]

        return True

    @staticmethod
    def _expected_shot_value(distance_m, points):
        """
        Expected Shot Value = P(make | distance) * points.
        """

        make_probability = (
            config.ESV_MAKE_BASE
            * math.exp(-config.ESV_MAKE_DECAY * distance_m)
        )

        return round(make_probability * points, 3)

    @staticmethod
    def _shot_type(distance_m):
        """
        Classify a shot as 3PT or 2PT by release distance.
        """

        if distance_m >= config.THREE_POINT_DISTANCE_M:
            return "3PT"

        return "2PT"

    # --------------------------------------------------------
    # Attempt lifecycle
    # --------------------------------------------------------

    def start_attempt(
        self,
        player_id,
        frame_number,
        court_position=None,
    ):
        """
        Begin tracking a shot attempt.
        """

        if self.current_attempt is not None:
            return

        self.current_attempt = {
            "player": player_id,
            "start_frame": frame_number,
            "end_frame": None,
            "made": None,
            "rim_reached": False,
            "possession_lost": False,
            "ball_positions": [],
            "court_position": court_position,
            "distance_m": None,
            "shot_type": None,
            "points": None,
            "esv": None,
            "zone": None,
        }

    def finish_attempt(
        self,
        frame_number,
        made=None,
    ):
        """
        Finish the current shot attempt, resolving the outcome.
        """

        if self.current_attempt is None:
            return

        attempt = self.current_attempt

        attempt["end_frame"] = frame_number

        # --------------------------------------------------
        # Resolve made / missed using the rim position.
        # --------------------------------------------------

        if made is None:

            rim_stale = (
                self.rim_position is None
                or (
                    self.rim_last_frame is not None
                    and frame_number is not None
                    and (
                        frame_number - self.rim_last_frame
                    ) > config.SHOT_RIM_MAX_STALE_FRAMES
                )
            )

            if rim_stale:

                # No usable rim information (never seen or stale):
                # treat as a miss and mark the outcome as unknown.
                attempt["made"] = False
                attempt["outcome_known"] = False

            else:

                rim_x, rim_y = self.rim_position

                ball_points = attempt["ball_positions"]

                reached = False
                went_through = False

                for bx, by in ball_points:

                    distance_to_rim = math.hypot(
                        bx - rim_x,
                        by - rim_y,
                    )

                    if distance_to_rim <= config.SHOT_RIM_PROXIMITY_PX:
                        reached = True

                    # The ball must descend through the rim plane
                    # within the rim's horizontal window. A rising
                    # ball below the rim is a floater/lob, not a
                    # made shot.
                    if (
                        abs(bx - rim_x) <= config.SHOT_RIM_PROXIMITY_PX
                        and by > rim_y + config.SHOT_RIM_CLEARANCE_PX
                        and self._ball_descending(
                            ball_points,
                            bx,
                            by,
                        )
                    ):
                        went_through = True

                attempt["rim_reached"] = reached

                attempt["made"] = reached and went_through
                attempt["outcome_known"] = True

        else:

            attempt["made"] = made
            attempt["outcome_known"] = True

        # --------------------------------------------------
        # Shot type, points, expected value.
        # --------------------------------------------------

        court_position = attempt.get("court_position")

        if court_position is not None:

            attack_hoop = (
                (0.0, config.COURT_WIDTH_M / 2.0)
                if config.HALF_COURT
                else (config.COURT_LENGTH_M, config.COURT_WIDTH_M / 2.0)
            )

            distance_m = math.hypot(
                court_position[0] - attack_hoop[0],
                court_position[1] - attack_hoop[1],
            )

            shot_type = self._shot_type(distance_m)

            points = 3 if shot_type == "3PT" else 2

            attempt["distance_m"] = round(distance_m, 2)
            attempt["shot_type"] = shot_type
            attempt["points"] = points
            attempt["esv"] = self._expected_shot_value(distance_m, points)
            attempt["zone"] = classify_position(
                court_position[0],
                court_position[1],
            )

        # --------------------------------------------------
        # Tallies.
        # --------------------------------------------------

        self.fg_attempts += 1

        if attempt["made"]:
            self.fg_made += 1

        if attempt["shot_type"] == "3PT":

            self.threepoint_attempts += 1

            if attempt["made"]:
                self.threepoint_made += 1

        self.shots.append(attempt)

        self.current_attempt = None

    # --------------------------------------------------------
    # Getters
    # --------------------------------------------------------

    def get_all_shots(
        self,
    ):
        """
        Return every detected shot.
        """

        return self.shots

    def total_shots(
        self,
    ):
        """
        Return the total number
        of shot attempts.
        """

        return len(self.shots)

    def latest_shot(
        self,
    ):
        """
        Return the newest shot attempt.
        """

        if len(self.shots) == 0:
            return None

        return self.shots[-1]

    def field_goal_percentage(
        self,
    ):
        """
        Return overall FG% (0-100), or None with no attempts.
        """

        if self.fg_attempts == 0:
            return None

        return round(
            self.fg_made / self.fg_attempts * 100.0,
            1,
        )

    def three_point_percentage(
        self,
    ):
        """
        Return 3PT% (0-100), or None with no attempts.
        """

        if self.threepoint_attempts == 0:
            return None

        return round(
            self.threepoint_made / self.threepoint_attempts * 100.0,
            1,
        )

    def free_throw_percentage(
        self,
    ):
        """
        Return FT% (0-100), or None with no attempts.

        Note: free throws are not detected yet (requires foul
        detection); this stays None until then.
        """

        if self.ft_attempts == 0:
            return None

        return round(
            self.ft_made / self.ft_attempts * 100.0,
            1,
        )

    def get_shot_chart(
        self,
    ):
        """
        Return shot chart data: every attempt with release
        location, outcome, type, points and ESV.
        """

        return [
            {
                "player": s["player"],
                "frame": s["start_frame"],
                "court_position": s.get("court_position"),
                "made": s.get("made"),
                "outcome_known": s.get("outcome_known"),
                "shot_type": s.get("shot_type"),
                "points": s.get("points"),
                "esv": s.get("esv"),
                "zone": s.get("zone"),
            }
            for s in self.shots
        ]

    # --------------------------------------------------------
    # Per-frame update
    # --------------------------------------------------------

    def update(
        self,
        possession_tracker,
        tracked_players,
        tracked_ball,
        frame_number,
        mapped_ball=None,
    ):
        """
        Detect shot attempts.

        Args:
            possession_tracker: current PossessionTracker.
            tracked_players: CourtVision player dicts.
            tracked_ball: CourtVision ball dict or None.
            frame_number: pipeline frame counter.
            mapped_ball: ball dict with court_position (release
                location), when available.
        """

        shooter = (
            possession_tracker.get_current_player()
        )

        # The ball is in flight right after the release, so
        # possession is momentarily None. Fall back to the last
        # possessor - the player who shot.
        if shooter is None:
            shooter = possession_tracker.get_last_player()

        # Nobody has possession.
        if shooter is None:
            return

        # Ball is not visible.
        if tracked_ball is None:

            # The attempt must still be closed out while the ball is
            # hidden - otherwise a ball lost to occlusion leaves the
            # attempt open forever and swallows the next shot.
            if self.current_attempt is not None:

                elapsed = frame_number - self.current_attempt["start_frame"]

                if elapsed > self.max_attempt_frames:
                    self.finish_attempt(frame_number)

            return

        # ----------------------------
        # Find the shooter
        # ----------------------------

        shooter_data = None

        for player in tracked_players:

            if player["id"] == shooter:

                shooter_data = player

                break

        if shooter_data is None:
            return

        # ----------------------------
        # Measure ball distance
        # ----------------------------

        distance = math.hypot(
            shooter_data["position"][0] - tracked_ball["position"][0],
            shooter_data["position"][1] - tracked_ball["position"][1],
        )

        # ----------------------------
        # Start shot attempt
        # ----------------------------

        if (
            distance > self.shot_start_distance
            and self.current_attempt is None
        ):

            court_position = None

            if mapped_ball is not None:
                court_position = mapped_ball.get("court_position")

            self.start_attempt(
                shooter,
                frame_number,
                court_position=court_position,
            )

        # ----------------------------
        # Track the ball during the attempt
        # ----------------------------

        if self.current_attempt is not None:

            self.current_attempt["ball_positions"].append(
                tracked_ball["position"]
            )

        # ----------------------------
        # Finish shot attempt
        # ----------------------------
        # The attempt ends when another player gains possession
        # (catch or rebound), when the shooter regains it after
        # the ball was in flight (offensive rebound), or when the
        # attempt has run too long (ball out of bounds).
        # ----------------------------

        current_possessor = (
            possession_tracker.get_current_player()
        )

        attempt = self.current_attempt

        if attempt is not None:

            attempt_player = attempt["player"]

            # The ball left the shooter's hands whenever possession
            # is lost while an attempt is active.
            if current_possessor is None:
                attempt["possession_lost"] = True

            elapsed = frame_number - attempt["start_frame"]

            if (
                current_possessor is not None
                and current_possessor != attempt_player
            ):

                # Another player gained possession (catch or rebound).
                self.finish_attempt(frame_number)

            elif (
                current_possessor == attempt_player
                and attempt["possession_lost"]
            ):

                # The shooter regained the ball: offensive rebound.
                self.finish_attempt(frame_number, made=False)

            elif elapsed > self.max_attempt_frames:

                # Ball never regained: out of bounds. Resolve from
                # rim data when available; otherwise the outcome is
                # unknown rather than a forced miss.
                self.finish_attempt(frame_number)
