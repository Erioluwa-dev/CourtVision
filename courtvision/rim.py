"""
Classical computer-vision rim detection (Phase 1).

Used automatically when no trained rim model is available.
Two-stage approach:

  1. Orange mask - rims are painted orange in most venues. The
     largest orange contour in the upper part of the frame is the
     rim.
  2. Backboard fallback - when no orange is found, a large white
     rectangle (backboard) is located and the rim is estimated
     just below its centre.

Returns boxes as (x1, y1, x2, y2, confidence) tuples compatible
with detect.box_to_coords output.
"""

import cv2
import numpy as np
from . import config


def _mask_contours(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return contours


def _orange_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [
            config.RIM_ORANGE_HUE_LOW,
            config.RIM_ORANGE_MIN_SATURATION,
            config.RIM_ORANGE_MIN_VALUE,
        ]
    )

    upper = np.array(
        [
            config.RIM_ORANGE_HUE_HIGH,
            255,
            255,
        ]
    )

    return cv2.inRange(hsv, lower, upper)


def _backboard_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [
            0,
            0,
            config.BACKBOARD_MIN_VALUE,
        ]
    )

    upper = np.array(
        [
            179,
            config.BACKBOARD_MAX_SATURATION,
            255,
        ]
    )

    return cv2.inRange(hsv, lower, upper)


def _clean(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )


def _best_rim_contour(contours, frame_height):
    """
    Pick the largest contour whose bounding box is mostly in the
    upper portion of the frame and plausible for a rim.
    """

    search_limit = int(
        frame_height * config.RIM_SEARCH_TOP_FRACTION
    )

    best = None
    best_area = 0

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if y > search_limit:
            continue

        if w * h < config.RIM_MIN_AREA:
            continue

        if w > frame_height * 0.5:
            continue

        area = w * h

        if area > best_area:
            best = (x, y, w, h)
            best_area = area

    return best


def detect_rim_box(frame):
    """
    Detect the rim and return (x1, y1, x2, y2, confidence) or None.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------
    # Pass 1: orange rim
    # --------------------------------------------------

    orange = _clean(_orange_mask(frame))

    rim = _best_rim_contour(
        _mask_contours(orange),
        height,
    )

    if rim is not None:

        x, y, w, h = rim

        return (x, y, x + w, y + h, 0.8)

    # --------------------------------------------------
    # Pass 2: backboard fallback
    # --------------------------------------------------

    backboard = _clean(_backboard_mask(frame))

    best_board = _best_rim_contour(
        _mask_contours(backboard),
        height,
    )

    if best_board is None:
        return None

    x, y, w, h = best_board

    # The rim hangs below the backboard's centre.
    rim_x = x + w // 2
    rim_y = y + h

    rim_w = max(w // 3, 10)
    rim_h = max(h // 6, 5)

    return (
        rim_x - rim_w // 2,
        rim_y - rim_h // 2,
        rim_x + rim_w // 2,
        rim_y + rim_h // 2,
        0.6,
    )


def rim_center(box):
    """
    Return the centre (x, y) of a rim box.
    """

    if box is None:
        return None

    x1, y1, x2, y2, _ = box

    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2,
    )
