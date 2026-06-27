import sys

"""
CourtVision

Main application entry point.
"""

# Ensure the latest version of vision.py is loaded
if 'vision' in sys.modules:
    del sys.modules['vision']

from trajectory import TrajectoryTracker
from vision import (
    load_video,
    read_frame,
    release_video,
)
from detect import (
    load_model,
    detect_players,
)


def run(video_path):
    """
    Run the complete CourtVision pipeline.
    """

    print("🏀 Starting CourtVision...")

    tracker = TrajectoryTracker()
    model = load_model()

    print("✅ Tracker initialized.")

    print("📹 Loading video...")

    video = load_video(video_path)

    print("✅ Video loaded successfully.")

    frame_count = 0
    boxes = [] # Initialize boxes to handle case where loop might not run

    while True:

        success, frame = read_frame(video)

        if not success:
            break

        boxes = detect_players(
            model,
            frame,
        )

        frame_count += 1

        print(
            f"Frame {frame_count}: "
            f"{len(boxes)} players detected"
        )

        if frame_count % 100 == 0:
            print(
                f"Processed {frame_count} frames..."
            )

    

    print("✅ Video processing complete.")
    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    run("footage.mp4")
