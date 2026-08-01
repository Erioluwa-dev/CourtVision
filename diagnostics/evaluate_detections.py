"""
Diagnostic: evaluate detection precision / recall on labeled frames.

Runs the player, ball (and rim, when labeled) detectors over a set
of labeled frames and reports TP / FP / FN plus precision, recall
and F1 per class, so detection regressions are visible.

Labels file format (JSON):

    {
      "100": {"ball": [x1, y1, x2, y2], "players": [[x1, y1, x2, y2], ...]},
      "250": {"players": [[x1, y1, x2, y2], ...]}
    }

Frame numbers are keys; box coords are pixels in (x1, y1, x2, y2).

Usage (Colab):
    !python diagnostics/evaluate_detections.py \
        --video /content/footage.mp4 \
        --labels /content/labels.json \
        --iou 0.5
"""

import argparse
import json
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

import cv2
from ultralytics import YOLO

from courtvision import config
from courtvision.tracker import iou
from courtvision.detect import (
    detect_ball,
    detect_rim,
    box_to_coords,
)

MODEL_DIR = os.environ.get(
    "COURTVISION_MODEL_DIR",
    os.path.dirname(config.PLAYER_MODEL_PATH),
)
VIDEO_PATH = os.environ.get(
    "COURTVISION_VIDEO_PATH",
    "/content/CourtVision/CourtVision/footage.mp4",
)


def _box_xyxy(bounding_box):
    """
    Convert a CourtVision (x, y, w, h) bounding box to (x1, y1, x2, y2).
    """
    x, y, w, h = bounding_box
    return (x, y, x + w, y + h)


def _greedy_match(predicted, ground_truth, iou_threshold):
    """
    Match predicted boxes to ground truth by IoU (greedy, best-first).

    Returns (true_positives, false_positives, false_negatives).
    """

    matched_gt = set()
    true_positives = 0

    for pred in predicted:

        best_gt = None
        best_iou = iou_threshold

        for index, gt in enumerate(ground_truth):

            if index in matched_gt:
                continue

            score = iou(pred, gt)

            if score > best_iou:
                best_iou = score
                best_gt = index

        if best_gt is not None:
            matched_gt.add(best_gt)
            true_positives += 1

    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - len(matched_gt)

    return true_positives, false_positives, false_negatives


def _metrics(true_positives, false_positives, false_negatives):
    """
    Precision / recall / F1 from the confusion counts.
    """

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection precision/recall on labeled frames."
    )

    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Path to the JSON labels file.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for a match (default: 0.5).",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )

    parser.add_argument(
        "--make-template",
        default=None,
        help="Write an empty labels template to this path and exit.",
    )

    args = parser.parse_args()

    if args.make_template:

        with open(args.make_template, "w") as f:
            json.dump(
                {
                    "100": {
                        "ball": [0, 0, 0, 0],
                        "players": [[0, 0, 0, 0]],
                    }
                },
                f,
                indent=2,
            )

        print(f"Template written to {args.make_template}")
        return

    if not os.path.exists(args.video):
        raise FileNotFoundError(
            f"Video not found: {args.video}"
        )

    with open(args.labels) as f:
        labels = json.load(f)

    player_model = YOLO(config.PLAYER_MODEL_PATH)
    ball_model = YOLO(config.BALL_MODEL_PATH)
    rim_model = (
        YOLO(config.RIM_MODEL_PATH)
        if config.RIM_MODEL_PATH
        else None
    )

    cap = cv2.VideoCapture(args.video)

    counts = {
        "players": [0, 0, 0],
        "ball": [0, 0, 0],
        "rim": [0, 0, 0],
    }

    evaluated = 0

    for frame_number, ground_truth in sorted(
        labels.items(),
        key=lambda item: int(item[0]),
    ):

        frame_number = int(frame_number)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()

        if not ok:
            print(
                f"!! Skipping frame {frame_number}: could not read."
            )
            continue

        # ---------------- players ----------------

        result = player_model.track(
            frame,
            persist=True,
            conf=args.conf,
            classes=[0],
            verbose=False,
        )[0]

        predicted_players = []

        for box in result.boxes:

            if box.id is None:
                continue

            x, y, w, h, _ = box_to_coords(box)
            predicted_players.append((x, y, x + w, y + h))

        gt_players = [
            tuple(coords)
            for coords in ground_truth.get("players", [])
        ]

        counts["players"] = [
            a + b
            for a, b in zip(
                counts["players"],
                _greedy_match(
                    predicted_players,
                    gt_players,
                    args.iou,
                ),
            )
        ]

        # ---------------- ball ----------------

        ball_box = detect_ball(
            ball_model,
            frame,
            confidence=args.conf,
        )

        predicted_ball = []

        if ball_box is not None:
            x, y, w, h, _ = box_to_coords(ball_box)
            predicted_ball.append((x, y, x + w, y + h))

        gt_ball = (
            [tuple(ground_truth["ball"])]
            if ground_truth.get("ball")
            else []
        )

        counts["ball"] = [
            a + b
            for a, b in zip(
                counts["ball"],
                _greedy_match(
                    predicted_ball,
                    gt_ball,
                    args.iou,
                ),
            )
        ]

        # ---------------- rim ----------------

        rim_box = detect_rim(
            rim_model,
            frame,
            confidence=args.conf,
        )

        predicted_rim = []

        if rim_box is not None:

            if hasattr(rim_box, "xyxy"):
                x, y, w, h, _ = box_to_coords(rim_box)
                predicted_rim.append((x, y, x + w, y + h))
            else:
                x1, y1, x2, y2, _ = rim_box
                predicted_rim.append((x1, y1, x2, y2))

        gt_rim = (
            [tuple(ground_truth["rim"])]
            if ground_truth.get("rim")
            else []
        )

        counts["rim"] = [
            a + b
            for a, b in zip(
                counts["rim"],
                _greedy_match(
                    predicted_rim,
                    gt_rim,
                    args.iou,
                ),
            )
        ]

        evaluated += 1

    cap.release()

    print()
    print(f"Evaluated {evaluated} labeled frames at IoU >= {args.iou}.")
    print()

    print(f"{'class':<10}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'precision':>12}{'recall':>10}{'F1':>8}")

    for class_name in ("players", "ball", "rim"):

        tp, fp, fn = counts[class_name]
        precision, recall, f1 = _metrics(tp, fp, fn)

        print(
            f"{class_name:<10}{tp:>5}{fp:>5}{fn:>5}"
            f"{precision:>12.3f}{recall:>10.3f}{f1:>8.3f}"
        )

    print()
    print(
        "SUMMARY "
        f"frames={evaluated} "
        f"players_tp={counts['players'][0]} "
        f"players_fp={counts['players'][1]} "
        f"players_fn={counts['players'][2]} "
        f"ball_tp={counts['ball'][0]} "
        f"ball_fp={counts['ball'][1]} "
        f"ball_fn={counts['ball'][2]}"
    )


if __name__ == "__main__":
    main()
