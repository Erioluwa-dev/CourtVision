"""
CourtVision synthetic logic tests.

Validates the Phase 1-6 analytics logic with synthetic data - no
video files or ML models required.

Run on Colab (or anywhere with the repo deps installed):

    !python tests/test_logic.py

Exits with code 0 when every check passes.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from zones import classify_position  # noqa: E402
from ball import BallTracker  # noqa: E402
from possession import PossessionTracker  # noqa: E402
from pass_detector import PassDetector  # noqa: E402
from shot_detector import ShotDetector  # noqa: E402

PASSED = 0
FAILED = 0


def check(name, condition):
    global PASSED, FAILED

    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}")


def player(pid, x, y):
    return {
        "id": pid,
        "position": (float(x), float(y)),
        "bounding_box": (int(x) - 10, int(y) - 10, 20, 40),
        "confidence": 0.9,
    }


def ball_at(x, y):
    return {
        "position": (float(x), float(y)),
        "bounding_box": (int(x) - 5, int(y) - 5, 10, 10),
        "confidence": 0.9,
    }


# ============================================================
# Phase 2 - Court zones
# ============================================================

def test_zones():
    print("\nPhase 2 - Court zones")

    check(
        "Paint: centre of the key",
        classify_position(3.0, 7.5) == "Paint",
    )

    check(
        "Left Corner: baseline + sideline",
        classify_position(1.0, 1.0) == "Left Corner",
    )

    check(
        "Right Corner: baseline + sideline",
        classify_position(1.0, 14.0) == "Right Corner",
    )

    check(
        "Top of Key: free-throw circle",
        classify_position(5.8, 7.5) == "Top of Key",
    )

    check(
        "Left Wing: beside the key",
        classify_position(6.5, 2.5) == "Left Wing",
    )

    check(
        "Right Wing: beside the key",
        classify_position(6.5, 12.5) == "Right Wing",
    )

    check(
        "Midrange: above the break",
        classify_position(9.0, 7.5) == "Midrange",
    )

    check(
        "Half Court: near midcourt",
        classify_position(12.5, 7.5) == "Half Court",
    )

    check(
        "Backcourt: full-court far half",
        classify_position(5.0, 7.5, half_court=False) == "Backcourt",
    )

    check(
        "Attack half zone in full court",
        classify_position(23.0, 3.0, half_court=False) == "Left Corner",
    )

    check(
        "Outside width is Unknown",
        classify_position(5.0, 20.0) == "Unknown",
    )


# ============================================================
# Phase 1 + 3 - Ball tracker
# ============================================================

def test_ball():
    print("\nPhase 1 + 3 - Ball tracker")

    # Deterministic path: use EMA with alpha=1.0 (no smoothing).
    config.BALL_SMOOTHING = "ema"
    config.BALL_EMA_ALPHA = 1.0

    tracker = BallTracker()

    # Constant velocity: 2 px/frame rightwards.
    for frame in range(10):
        tracker.update(ball_at(100 + frame * 2, 200))

    check(
        "Velocity estimated ~ (2, 0)",
        abs(tracker.velocity_vector[0] - 2.0) < 0.5
        and abs(tracker.velocity_vector[1]) < 0.5,
    )

    check(
        "Current speed ~ 2 px/frame",
        abs(tracker.current_speed - 2.0) < 0.5,
    )

    predicted = tracker.predict_position(5)

    check(
        "Prediction lands 5 frames ahead",
        predicted is not None
        and abs(predicted[0] - (100 + 9 * 2 + 5 * 2)) < 2.0,
    )

    check(
        "Trajectory has 10 detected points",
        len(tracker.trajectory) == 10,
    )

    # --------------------------------------------------
    # Interpolation: hide the ball for 4 frames.
    # --------------------------------------------------

    before = tracker.current_position

    for _ in range(4):
        tracker.update(None)

    check(
        "Lost ball is interpolated (not dropped)",
        tracker.current_position is not None
        and tracker.interpolated,
    )

    check(
        "Interpolated position moved with velocity",
        tracker.current_position[0] > before[0],
    )

    # Long loss (beyond the max) drops the trail.
    for _ in range(config.BALL_MAX_INTERPOLATION_FRAMES + 2):
        tracker.update(None)

    check(
        "Prolonged loss clears the position",
        tracker.current_position is None,
    )

    # --------------------------------------------------
    # Reset.
    # --------------------------------------------------

    tracker.reset()

    check(
        "Reset clears state",
        tracker.current_position is None
        and tracker.frames_seen == 0
        and tracker.distance_pixels == 0.0,
    )

    # --------------------------------------------------
    # Kalman smoothing smoke test.
    # --------------------------------------------------

    config.BALL_SMOOTHING = "kalman"

    kalman_tracker = BallTracker()

    for frame in range(20):
        kalman_tracker.update(ball_at(100 + frame * 2, 200))

    check(
        "Kalman path produces a position",
        kalman_tracker.current_position is not None,
    )

    check(
        "Kalman path estimates velocity",
        kalman_tracker.velocity_vector[0] > 0.0,
    )

    check(
        "Kalman path keeps speed near 2 px/frame",
        abs(kalman_tracker.current_speed - 2.0) < 1.0,
    )

    config.BALL_SMOOTHING = "kalman"


# ============================================================
# Phase 4 - Possession engine
# ============================================================

def test_possession():
    print("\nPhase 4 - Possession engine")

    tracker = PossessionTracker()

    team_lookup = {1: "Team A", 2: "Team A", 3: "Team B"}

    # Player 1 close to a slow ball -> possession after hysteresis.
    for frame in range(config.POSSESSION_HYSTERESIS_FRAMES + 2):
        tracker.update(
            [player(1, 500, 500), player(3, 700, 500)],
            ball_at(515, 500),
            ball_speed=2.0,
            frame_number=frame,
            team_lookup=team_lookup,
        )

    check(
        "Slow ball + near player -> possession",
        tracker.get_current_player() == 1,
    )

    check(
        "Timeline entry opened",
        len(tracker.get_timeline()) == 1
        and tracker.get_timeline()[0]["player"] == 1,
    )

    # --------------------------------------------------
    # Fast ball cannot be possessed.
    # --------------------------------------------------

    for frame in range(10, 15):
        tracker.update(
            [player(1, 500, 500)],
            ball_at(515, 500),
            ball_speed=60.0,
            frame_number=frame,
            team_lookup=team_lookup,
        )

    check(
        "Fast (in-flight) ball -> no possession",
        tracker.get_current_player() is None,
    )

    # --------------------------------------------------
    # Team possession percentages.
    # --------------------------------------------------

    pct = tracker.possession_percentage()

    check(
        "Possession % reported for Team A",
        pct.get("Team A", 0.0) > 0.0,
    )

    # Player 3 (Team B) catches next.
    for frame in range(20, 20 + config.POSSESSION_HYSTERESIS_FRAMES + 2):
        tracker.update(
            [player(3, 700, 500)],
            ball_at(710, 500),
            ball_speed=2.0,
            frame_number=frame,
            team_lookup=team_lookup,
        )

    check(
        "Possession switched to Player 3",
        tracker.get_current_player() == 3,
    )

    check(
        "Team possession timeline tracks Team B",
        any(
            entry["team"] == "Team B"
            for entry in tracker.get_team_possessions()
        ),
    )


# ============================================================
# Phase 5 - Passing engine
# ============================================================

def test_passes():
    print("\nPhase 5 - Passing engine")

    detector = PassDetector()

    team_lookup = {1: "Team A", 2: "Team A", 3: "Team B"}

    # Player 1 -> Player 2 (same team): successful pass.
    detector.update(1, frame_number=0, team_lookup=team_lookup)
    detector.update(None, frame_number=1, team_lookup=team_lookup)
    detector.update(None, frame_number=2, team_lookup=team_lookup)
    detector.update(2, frame_number=3, team_lookup=team_lookup)

    check(
        "Same-team possession change = successful pass",
        len(detector.get_all_passes()) == 1
        and detector.get_all_passes()[0]["success"],
    )

    # Player 2 -> Player 3 (other team): failed pass.
    detector.update(2, frame_number=4, team_lookup=team_lookup)
    detector.update(None, frame_number=5, team_lookup=team_lookup)
    detector.update(3, frame_number=6, team_lookup=team_lookup)

    check(
        "Cross-team possession change = failed pass",
        detector.failed_passes() == 1,
    )

    check(
        "Totals add up",
        detector.total_passes() == 2
        and detector.successful_passes() == 1,
    )

    pass_map = detector.get_pass_map()

    check(
        "Pass map records both pairs",
        pass_map.get((1, 2)) == 1
        and pass_map.get((2, 3)) == 1,
    )

    network = detector.get_passing_network()

    check(
        "Passing network is per-team",
        (1, 2) in network.get("Team A", {})
        and (2, 3) in network.get("Team B", {}),
    )


# ============================================================
# Phase 6 - Shooting engine
# ============================================================

def test_shots():
    print("\nPhase 6 - Shooting engine")

    team_lookup = {1: "Team A", 2: "Team A"}

    rim = (600, 300)

    def run_scenario(ball_path, release_court_pos, catch_ball):
        """
        Run a full shot scenario:

          * frames 0-8: player 1 holds (possession establishes)
          * frames 9-16: ball released, travels along ball_path
          * frames 17-22: player 2 catches

        Returns the finished ShotDetector.
        """

        detector = ShotDetector()

        possession = PossessionTracker()

        detector.set_rim_position(rim)

        # Hold phase.
        for frame in range(9):
            possession.update(
                [player(1, 500, 500), player(2, 700, 500)],
                ball_at(505, 500),
                ball_speed=2.0,
                frame_number=frame,
                team_lookup=team_lookup,
            )

            detector.update(
                possession,
                [player(1, 500, 500), player(2, 700, 500)],
                ball_at(505, 500),
                frame,
                mapped_ball={"court_position": (6.0, 7.5)},
            )

        # Release phase (fast ball -> in flight).
        for frame, bx, by in enumerate(ball_path, start=9):
            possession.update(
                [player(1, 500, 500), player(2, 700, 500)],
                ball_at(bx, by),
                ball_speed=60.0,
                frame_number=frame,
                team_lookup=team_lookup,
            )

            detector.update(
                possession,
                [player(1, 500, 500), player(2, 700, 500)],
                ball_at(bx, by),
                frame,
                mapped_ball={"court_position": release_court_pos},
            )

        # Catch phase.
        for frame in range(17, 23):
            possession.update(
                [player(1, 500, 500), player(2, 700, 500)],
                catch_ball,
                ball_speed=5.0,
                frame_number=frame,
                team_lookup=team_lookup,
            )

            detector.update(
                possession,
                [player(1, 500, 500), player(2, 700, 500)],
                catch_ball,
                frame,
                mapped_ball={"court_position": release_court_pos},
            )

        return detector

    # --------------------------------------------------
    # Scenario 1: ball reaches the rim and drops through.
    # --------------------------------------------------

    made_path = [
        (520, 490),
        (535, 475),
        (550, 455),
        (565, 435),
        (580, 415),
        (592, 395),
        (600, 365),
        (598, 340),
    ]

    detector = run_scenario(
        made_path,
        (6.0, 7.5),
        ball_at(710, 500),
    )

    shot = detector.latest_shot()

    check(
        "Shot attempt recorded",
        shot is not None,
    )

    check(
        "Ball through the rim = made",
        shot["made"] is True
        and shot["rim_reached"] is True,
    )

    check(
        "Near release classified as 2PT",
        shot["shot_type"] == "2PT",
    )

    check(
        "ESV is a positive value",
        shot["esv"] is not None and shot["esv"] > 0,
    )

    check(
        "FG% computed after one make",
        detector.field_goal_percentage() == 100.0,
    )

    # --------------------------------------------------
    # Scenario 2: ball never reaches the rim (missed 3PT).
    # --------------------------------------------------

    missed_path = [
        (520, 490),
        (540, 480),
        (560, 470),
        (580, 460),
        (600, 450),
        (620, 440),
        (640, 430),
        (660, 420),
    ]

    detector2 = run_scenario(
        missed_path,
        (24.0, 7.5),
        ball_at(710, 500),
    )

    shot2 = detector2.latest_shot()

    check(
        "Far release classified as 3PT",
        shot2 is not None and shot2["shot_type"] == "3PT",
    )

    check(
        "Ball away from rim = miss",
        shot2["made"] is False,
    )

    check(
        "3PT% is 0.0 after a miss",
        detector2.three_point_percentage() == 0.0,
    )


# ============================================================
# Runner
# ============================================================

def main():
    test_zones()
    test_ball()
    test_possession()
    test_passes()
    test_shots()

    print()
    print(f"Results: {PASSED} passed, {FAILED} failed")

    if FAILED:
        sys.exit(1)

    print("All CourtVision logic checks passed.")


if __name__ == "__main__":
    main()
