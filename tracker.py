
import cv2
import math

from vision import draw_box
from detect import box_to_coords


# ============================================================
# Geometry Utilities
# ============================================================

def iou(box_a, box_b):
    """
    Calculate Intersection over Union.
    """

    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (
        (x_right - x_left)
        * (y_bottom - y_top)
    )

    area_a = (
        (box_a[2] - box_a[0])
        * (box_a[3] - box_a[1])
    )

    area_b = (
        (box_b[2] - box_b[0])
        * (box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union


def get_centroid(
    x,
    y,
    w,
    h,
):
    """
    Return center of a bounding box.
    """

    return (
        x + w // 2,
        y + h // 2,
    )


def centroid_distance(
    point_a,
    point_b,
):
    """
    Distance between two points.
    """

    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2
    )


# ============================================================
# YOLO + ByteTrack
# ============================================================

def run_tracker(
    model,
    frame,
):
    """
    Run YOLO tracking.
    """

    results = model.track(
        frame,
        persist=True,
    )

    return results[0]


# ============================================================
# Debugging
# ============================================================

def debug_detections(
    result,
):
    """
    Print every detection made by YOLO.
    """

    print("\nYOLO Detections")

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = result.names[class_id]

        confidence = float(box.conf[0])

        print(
            f"{class_name:<15}"
            f"conf={confidence:.2f}"
        )


# ============================================================
# Raw YOLO Objects
# ============================================================

def get_tracked_persons(
    result,
):
    """
    Return tracked people.
    """

    persons = []

    for box in result.boxes:

        class_name = result.names[
            int(box.cls[0])
        ]

        if (
            class_name == "person"
            and box.id is not None
        ):

            persons.append(box)

    return persons


def get_tracked_ball(
    result,
):
    """
    Return tracked basketball.
    """

    for box in result.boxes:

        class_name = result.names[
            int(box.cls[0])
        ]

        if class_name == "sports ball":

            return box

    return None


# ============================================================
# CourtVision Objects
# ============================================================

def build_tracked_players(
    boxes,
):
    """
    Convert YOLO people into CourtVision players.
    """

    tracked_players = []

    for box in boxes:

        x, y, w, h, confidence = box_to_coords(
            box,
        )

        tracked_players.append(

            {
                "id": int(box.id[0]),
                "position": get_centroid(
                    x,
                    y,
                    w,
                    h,
                ),
                "bounding_box": (
                    x,
                    y,
                    w,
                    h,
                ),
                "confidence": confidence,
            }

        )

    return tracked_players


def build_tracked_ball(
    box,
):
    """
    Convert YOLO basketball into CourtVision ball.
    """

    if box is None:
        return None

    x, y, w, h, confidence = box_to_coords(
        box,
    )

    return {

        "position": get_centroid(
            x,
            y,
            w,
            h,
        ),

        "bounding_box": (
            x,
            y,
            w,
            h,
        ),

        "confidence": confidence,

    }


# ============================================================
# Public API
# ============================================================

def track_players(
    result,
):
    """
    Return CourtVision players.
    """

    return build_tracked_players(

        get_tracked_persons(
            result,
        )

    )


def track_ball(
    result,
):
    """
    Return CourtVision basketball.
    """

    return build_tracked_ball(

        get_tracked_ball(
            result,
        )

    )


# ============================================================
# Drawing
# ============================================================

def annotate_tracked_frame(
    frame,
    result,
):
    """
    Draw every YOLO detection.
    Useful for debugging ball detection.
    """

    for box in result.boxes:

        x, y, w, h, confidence = box_to_coords(
            box,
        )

        class_name = result.names[
            int(box.cls[0])
        ]

        label = class_name

        if box.id is not None:

            label += f" {int(box.id[0])}"

        frame = draw_box(

            frame,

            x,
            y,
            w,
            h,

            label,

        )

    return frame


def draw_ball(
    frame,
    tracked_ball,
):
    """
    Draw basketball.
    """

    if tracked_ball is None:
        return frame

    x, y, w, h = tracked_ball[
        "bounding_box"
    ]

    cv2.rectangle(

        frame,

        (x, y),

        (x + w, y + h),

        (0, 165, 255),

        2,

    )

    cv2.putText(

        frame,

        "BALL",

        (x, y - 10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (0, 165, 255),

        2,

    )

    return frame
