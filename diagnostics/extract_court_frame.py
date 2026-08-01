"""
Diagnostic: extract a representative frame with the full court visible
and save it so the user can identify the 4 court-corner pixel
coordinates.

Why: the current manual fallback points in main.py are a placeholder
rectangle that produces a garbage homography (court positions up to
60 m on a 28 m court -> every zone "Unknown"). Real corners unblock
zones, heatmaps, and shot charts.

Usage (Colab):
    !python diagnostics/extract_court_frame.py --frame 500 --output /content/court_frame.jpg
"""

import argparse
import os

import cv2

VIDEO_PATH = os.environ.get(
    "COURTVISION_VIDEO_PATH",
    "/content/CourtVision/CourtVision/footage.mp4",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=500)
    parser.add_argument("--output", default="/content/court_frame.jpg")
    args = parser.parse_args()

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"!! Could not read frame {args.frame}")
        return

    cv2.imwrite(args.output, frame)
    print(f"Saved {args.output}")
    print(f"Frame size: {frame.shape[1]} x {frame.shape[0]}")
    print("Open the image and note the 4 pixel coords where the court's")
    print("sidelines meet the visible baseline (usually a trapezoid).")


if __name__ == "__main__":
    main()
