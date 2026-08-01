import sys
import os
import cv2

# Fallback for cv2_imshow when not running in Google Colab
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    import numpy as np

    def cv2_imshow(frame):
        cv2.imshow("CourtVision", frame)
        cv2.waitKey(1)

import config

from team import TeamClassifier
from stats import PlayerStats
from data import MatchData
from trajectory import TrajectoryTracker
from ball import BallTracker
from possession import PossessionTracker
from pass_detector import PassDetector
from commentary import CommentaryEngine
from shot_detector import ShotDetector
from court import CourtMapper
from hoop import HoopTracker
from zones import classify_position
from heatmap import HeatmapGenerator
from court_detection import (
    detect_court_lines,
    setup_court_mapper,
    draw_court_lines,
)
from rim import rim_center

from vision import (
    load_video,
    read_frame,
    release_video,
)

from tracker import (
    run_tracker,
    track_players,
    annotate_tracked_frame,
    debug_detections,
)

from detect import (
    load_models,
    detect_ball,
    detect_rim,
    box_to_coords,
)


"""
CourtVision

Main application entry point - Phase 1-6 pipeline:
detection (players / ball / rim / court) -> tracking -> homography
-> zones -> heatmaps -> team classification -> stats -> possession
-> passes -> shots -> commentary.
"""

# Ensure the latest version of vision.py is loaded
if "vision" in sys.modules:
    del sys.modules["vision"]


def build_tracked_ball(ball_box):
    """
    Convert a raw YOLO ball detection box into the same
    dict shape the rest of the pipeline (BallTracker,
    CourtMapper, possession/pass/shot detectors) expects.
    """
    if ball_box is None:
        return None

    x, y, w, h, conf = box_to_coords(ball_box)
    center_x = x + w / 2
    center_y = y + h / 2

    return {
        "position": (center_x, center_y),
        "bounding_box": (x, y, w, h),
        "confidence": conf,
    }


def rim_box_center(rim_box):
    """
    Return the centre (x, y) of a rim box whether it comes from
    the YOLO model or the classical-CV detector.
    """
    if rim_box is None:
        return None

    if hasattr(rim_box, "xyxy"):
        x, y, w, h, _ = box_to_coords(rim_box)
        return (x + w // 2, y + h // 2)

    return rim_center(rim_box)


def run(video_path):
    """
    Run the complete CourtVision pipeline.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print("🏀 Starting CourtVision...")

    player_model, ball_model, rim_model = load_models()

    trajectory_tracker = TrajectoryTracker()
    player_stats = PlayerStats()
    match = MatchData()

    team_classifier = TeamClassifier()

    ball_tracker = BallTracker()
    possession_tracker = PossessionTracker()
    pass_detector = PassDetector()
    commentary = CommentaryEngine()
    shot_detector = ShotDetector()
    court_mapper = CourtMapper()
    hoop_tracker = HoopTracker()
    heatmaps = HeatmapGenerator()

    print("✅ Player model loaded.")
    print("✅ Custom ball model loaded.")
    print("✅ Trajectory tracker initialized.")
    print("✅ Player stats initialized.")
    print("✅ Team classifier initialized.")
    print("✅ Ball tracker initialized.")
    print("✅ Possession tracker initialized.")
    print("✅ Pass detector initialized.")
    print("✅ Commentary engine initialized.")
    print("✅ Shot detector initialized.")
    print("✅ Hoop tracker initialized.")
    print("✅ Heatmap generator initialized.")
    print("✅ Match data initialized.")

    print("📹 Loading video...")

    video = load_video(video_path)

    print("✅ Video loaded successfully.")

    # ---------------------------------------------------------
    # Court mapping: automatic line detection with manual fallback
    # ---------------------------------------------------------

    success, first_frame = read_frame(video)

    if not success:
        release_video(video)
        raise RuntimeError("Could not read the first frame.")

    manual_fallback = [
        (100, 100),
        (500, 100),
        (500, 400),
        (100, 400),
    ]

    court_info = setup_court_mapper(
        first_frame,
        court_mapper,
        manual_image_points=manual_fallback,
    )

    if court_info["auto"]:
        print(
            f"✅ Court detected automatically "
            f"(fit score {court_info['fit_score']:.2f})."
        )
    else:
        print(
            "⚠️ Automatic court detection failed — "
            "using manual reference points. "
            "Tune config.COURT_* or provide real points."
        )

    court_segments = (
        detect_court_lines(first_frame)
        if court_info["auto"]
        else []
    )

    frame_count = 1

    while True:

        success, frame = read_frame(video)

        if not success:
            break

        # ---------------------------------
        # Run YOLO + ByteTrack for players
        # ---------------------------------

        result = run_tracker(
            player_model,
            frame,
            conf=config.PLAYER_CONFIDENCE,
            classes=[0],  # COCO class 0 = person
        )

        if frame_count % 100 == 0:
            debug_detections(
                result,
            )

        # ---------------------------------
        # Run custom model for ball
        # ---------------------------------

        ball_box = detect_ball(
            ball_model,
            frame,
        )

        tracked_ball = build_tracked_ball(ball_box)

        # ---------------------------------
        # Rim detection (model or classical CV)
        # ---------------------------------

        rim_box = detect_rim(
            rim_model,
            frame,
        )

        hoop_tracker.update(
            rim_box_center(rim_box),
        )

        rim_position = hoop_tracker.get_smoothed_position()

        shot_detector.set_rim_position(rim_position)

        # ---------------------------------
        # Extract tracked objects
        # ---------------------------------

        tracked_players = track_players(
            result,
        )

        mapped_players = court_mapper.map_players(
            tracked_players,
        )

        mapped_ball = court_mapper.map_ball(
            tracked_ball,
        )

        # ---------------------------------
        # Update analytics
        # ---------------------------------

        trajectory_tracker.update(
            tracked_players,
        )

        player_stats.update(
            tracked_players,
        )

        ball_tracker.update(
            tracked_ball,
        )

        team_lookup = team_classifier.get_all_teams()

        possession_tracker.update(
            tracked_players,
            tracked_ball,
            ball_speed=ball_tracker.current_speed,
            frame_number=frame_count,
            team_lookup=team_lookup,
        )

        pass_detector.update(
            possession_tracker.get_current_player(),
            frame_number=frame_count,
            team_lookup=team_lookup,
        )

        shot_detector.update(
            possession_tracker,
            tracked_players,
            tracked_ball,
            frame_count,
            mapped_ball=mapped_ball,
        )

        commentary.update_possession(
            possession_tracker,
        )

        commentary.update_passes(
            pass_detector,
        )

        commentary.update_shots(
            shot_detector,
        )

        heatmaps.update(
            mapped_players,
            mapped_ball,
            possession_player=(
                possession_tracker.get_current_player()
            ),
        )

        match.add_frame(
            frame_count,
            tracked_players,
            tracked_ball=tracked_ball,
            rim_position=rim_position,
            court_lines=(
                court_segments if court_info["auto"] else None
            ),
            possession_player=(
                possession_tracker.get_current_player()
            ),
        )

        # ---------------------------------
        # Team classification
        # ---------------------------------

        for player in mapped_players:

            jersey = team_classifier.extract_jersey(
                frame,
                player["bounding_box"],
            )

            color = team_classifier.average_jersey_color(
                jersey,
            )

            team_classifier.store_color(
                player["id"],
                color,
            )

        if (
            frame_count > 0
            and frame_count % 30 == 0
        ):
            team_classifier.classify_teams()

        # ---------------------------------
        # Draw overlays
        # ---------------------------------

        frame = annotate_tracked_frame(
            frame,
            result,
        )

        frame = trajectory_tracker.draw_trajectories(
            frame,
        )

        if ball_box is not None:
            x, y, w, h, conf = box_to_coords(ball_box)
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"ball {conf:.2f}",
                (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
            )

        if rim_box is not None:

            if hasattr(rim_box, "xyxy"):
                x, y, w, h, _ = box_to_coords(rim_box)
            else:
                x1, y1, x2, y2, _ = rim_box
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "RIM",
                (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        if court_info["auto"]:
            frame = draw_court_lines(
                frame,
                court_segments,
            )

        # ---------------------------------
        # Logging every 100 frames
        # ---------------------------------

        if frame_count % 100 == 0:

            ids = [
                player["id"]
                for player in tracked_players
            ]

            print()
            print(f"Frame {frame_count}")
            print(f"Players tracked: {len(tracked_players)}")
            print(f"IDs: {ids}")

            latest = commentary.latest_event()

            print()
            print("Latest Commentary:")

            if latest is not None:
                print(latest)
            else:
                print("No commentary yet.")

            print()

            if tracked_ball is not None:

                print(
                    f"Ball Position: {tracked_ball['position']}"
                )

                if ball_tracker.interpolated:
                    print("Ball: INTERPOLATED (temporarily lost)")

                print(
                    f"Ball Speed: "
                    f"{ball_tracker.current_speed:.2f}px/frame"
                )

                print(
                    f"Ball Predicted (5 frames): "
                    f"{ball_tracker.predict_position(5)}"
                )

            else:

                print("Ball: Not detected")

            print(
                f"Rim: "
                f"{rim_position if rim_position is not None else 'not detected'}"
            )

            print(
                f"Current Possession: "
                f"{possession_tracker.get_current_player()}"
            )

            print(
                f"Possession %: "
                f"{possession_tracker.possession_percentage()}"
            )

            print(
                f"Total Passes: "
                f"{pass_detector.total_passes()} "
                f"(successful: {pass_detector.successful_passes()}, "
                f"failed: {pass_detector.failed_passes()})"
            )

            print()

            for player in mapped_players:

                stats = player_stats.players[
                    player["id"]
                ]

                team = team_classifier.get_team(
                    player["id"]
                )

                color = (
                    team_classifier
                    .get_all_colors()
                    .get(player["id"])
                )

                zone = classify_position(
                    player["court_position"][0],
                    player["court_position"][1],
                )

                print(
                    f"Player {player['id']} | "
                    f"Team: {team} | Zone: {zone}"
                )

                print(
                    f"Frames: {stats['frames_seen']} | "
                    f"Distance: {stats['distance_pixels']:.2f}px | "
                    f"Speed: {stats['current_speed']:.2f}px/frame"
                )

                print(
                    f"Jersey Color: {color}"
                )
                print(
                    f"Court Position: "
                    f"{player['court_position']}"
                )

                print()

            if pass_detector.total_passes() > 0:

                print("Detected Passes:")

                for index, pass_event in enumerate(
                    pass_detector.get_all_passes(),
                    start=1,
                ):

                    status = (
                        "✅"
                        if pass_event["success"]
                        else "❌"
                    )

                    print(
                        f"{index}. {status} "
                        f"{pass_event['from']} "
                        f"→ "
                        f"{pass_event['to']}"
                    )

                print()

            cv2_imshow(frame)

        frame_count += 1

    # ---------------------------------
    # Cleanup
    # ---------------------------------

    release_video(video)

    cv2.destroyAllWindows()

    print()
    print(
        f"Stored {len(match.get_all_frames())} frames."
    )

    print()

    print("First stored frame:")

    print(
        match.get_frame(1)
    )

    print()

    print("Ball Summary")

    print(
        f"Frames Seen: {ball_tracker.frames_seen}"
    )

    print(
        f"Maximum Speed: "
        f"{ball_tracker.max_speed:.2f}px/frame"
    )

    print(
        f"Trajectory Points: "
        f"{len(ball_tracker.trajectory)}"
    )

    print(
        f"Velocity: "
        f"{ball_tracker.get_velocity()}"
    )

    print()

    print("Possession Summary")

    print(
        f"Current Player: "
        f"{possession_tracker.get_current_player()}"
    )

    print(
        f"Possession Records: "
        f"{len(possession_tracker.get_history())}"
    )

    print(
        f"Timeline Entries: "
        f"{len(possession_tracker.get_timeline())}"
    )

    print(
        f"Team Possession %: "
        f"{possession_tracker.possession_percentage()}"
    )

    print()

    print("Pass Summary")

    print(
        f"Total Passes: "
        f"{pass_detector.total_passes()}"
    )

    print(
        f"Successful: "
        f"{pass_detector.successful_passes()}"
    )

    print(
        f"Failed: "
        f"{pass_detector.failed_passes()}"
    )

    print("Pass Map:")

    for (from_player, to_player), count in sorted(
        pass_detector.get_pass_map().items()
    ):
        print(
            f"  Player {from_player} → Player {to_player}: {count}"
        )

    print("Passing Network (per team):")

    for team, network in (
        pass_detector.get_passing_network().items()
    ):
        print(f"  {team}: {network}")

    for index, pass_event in enumerate(
        pass_detector.get_all_passes(),
        start=1,
    ):
        print(
            f"Pass {index}: "
            f"{pass_event['from']} "
            f"→ "
            f"{pass_event['to']} "
            f"({'success' if pass_event['success'] else 'FAILED'})"
        )

    print()

    print("Shot Summary")

    print(
        f"Total Attempts: "
        f"{shot_detector.total_shots()}"
    )

    print(
        f"FG%: "
        f"{shot_detector.field_goal_percentage()}"
    )

    print(
        f"3PT%: "
        f"{shot_detector.three_point_percentage()}"
    )

    chart = shot_detector.get_shot_chart()

    print(
        f"Shot Chart Entries: {len(chart)}"
    )

    for shot in chart:
        print(
            f"  Player {shot['player']} "
            f"{shot['shot_type']} from {shot['court_position']} "
            f"zone={shot['zone']} "
            f"{'MADE' if shot['made'] else 'miss'} "
            f"ESV={shot['esv']}"
        )

    print()

    print("✅ Video processing complete.")

    print(
        f"Total frames processed: {frame_count}"
    )


if __name__ == "__main__":
    run(config.VIDEO_PATH)
