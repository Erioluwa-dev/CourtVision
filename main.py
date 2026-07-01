import sys
import cv2

from google.colab.patches import cv2_imshow

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

from vision import (
    load_video,
    read_frame,
    release_video,
)

from tracker import (
    run_tracker,
    track_players,
    track_ball,
    annotate_tracked_frame,
    debug_detections,
)

from detect import load_model


"""
CourtVision

Main application entry point.
"""

# Ensure the latest version of vision.py is loaded
if "vision" in sys.modules:
    del sys.modules["vision"]


def run(video_path):
    """
    Run the complete CourtVision pipeline.
    """

    print("🏀 Starting CourtVision...")

    model = load_model()

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


    image_points = [

        (100, 100),
        (500, 100),
        (500, 400),
        (100, 400),

    ]

    court_points = [

        (0, 0),
        (28, 0),
        (28, 15),
        (0, 15),

    ]

    court_mapper.set_reference_points(

        image_points,
        court_points,

    )

    court_mapper.compute_homography()

    print("✅ AI model loaded.")
    print("✅ Trajectory tracker initialized.")
    print("✅ Player stats initialized.")
    print("✅ Team classifier initialized.")
    print("✅ Ball tracker initialized.")
    print("✅ Possession tracker initialized.")
    print("✅ Pass detector initialized.")
    print("✅ Commentary engine initialized.")
    print("✅ Shot detector initialized.")
    print("✅ Match data initialized.")

    print("📹 Loading video...")

    video = load_video(video_path)

    print("✅ Video loaded successfully.")

    frame_count = 0

    while True:

        success, frame = read_frame(video)

        if not success:
            break

        # ---------------------------------
        # Run YOLO + ByteTrack
        # ---------------------------------

        result = run_tracker(
            model,
            frame,
        )
        if frame_count % 100 == 0:

            debug_detections(
            result,
        )

        # ---------------------------------
        # Extract tracked objects
        # ---------------------------------

        tracked_players = track_players(
            result,
        )

        tracked_ball = track_ball(
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

        possession_tracker.update(
            tracked_players,
            tracked_ball,
        )

        pass_detector.update(
            possession_tracker.get_current_player(),
        )

        shot_detector.update(
            possession_tracker,
            tracked_players,
            tracked_ball,
            frame_count,
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

        match.add_frame(
            frame_count,
            tracked_players,
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
                if mapped_ball is not None:

                    print(
                        f"Ball Court Position: "
                        f"{mapped_ball['court_position']}"
                    )

                print(
                    f"Ball Speed: "
                    f"{ball_tracker.current_speed:.2f}px/frame"
                )

            else:

                print("Ball: Not detected")

            print(
                f"Current Possession: "
                f"{possession_tracker.get_current_player()}"
            )

            print(
                f"Total Passes: "
                f"{pass_detector.total_passes()}"
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

                print(
                    f"Player {player['id']} | "
                    f"Team: {team}"
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

                    print(
                        f"{index}. "
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
        match.get_frame(0)
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
        f"{len(ball_tracker.positions)}"
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

    print()

    print("Pass Summary")

    print(
        f"Total Passes: "
        f"{pass_detector.total_passes()}"
    )

    for index, pass_event in enumerate(
        pass_detector.get_all_passes(),
        start=1,
    ):

        print(
            f"Pass {index}: "
            f"{pass_event['from']} "
            f"→ "
            f"{pass_event['to']}"
        )

    print()

    print("✅ Video processing complete.")

    print(
        f"Total frames processed: {frame_count}"
    )


if __name__ == "__main__":
    run("/content/CourtVision/footage.mp4")
