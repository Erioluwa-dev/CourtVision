"""
Diagnostic: run the ball model on a single frame where the ball should
be visible, at low confidence, and show what it actually detects.

Why: confirms (a) whether the ball model works at all on your footage,
(b) what it fires on instead of the ball (scoreboard, crowd, ads), and
(c) the real confidence values.

Usage (Colab):
    !python diagnostics/test_ball_frame.py --frame 500
"""

import argparse
import os

import cv2
from ultralytics import YOLO

from google.colab.patches import cv2_imshow  # noqa: F401  (Colab only)

MODEL_DIR = os.environ.get(
    "COURTVISION_MODEL_DIR",
    "/content/CourtVision/CourtVision/models",
)
VIDEO_PATH = os.environ.get(
    "COURTVISION_VIDEO_PATH",
    "/content/CourtVision/CourtVision/footage.mp4",
)
BALL_MODEL_PATH = os.path.join(MODEL_DIR, "basketball_ball_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=500)
    args = parser.parse_args()

    ball_model = YOLO(BALL_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"!! Could not read frame {args.frame}")
        return

    print(f"Frame {args.frame}, size {frame.shape[1]}x{frame.shape[0]}")

    results = ball_model(frame, conf=0.1)
    boxes = results[0].boxes
    print(f"Raw detections at conf>=0.1: {len(boxes)}")
    for box in boxes:
        cls = int(box.cls[0])
        print(f"  class={results[0].names[cls]} conf={float(box.conf[0]):.2f}")

    annotated = results[0].plot()
    cv2_imshow(annotated)


if __name__ == "__main__":
    main()
