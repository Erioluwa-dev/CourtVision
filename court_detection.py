"""
Classical computer-vision court detection (Phase 1).

Detects court boundary lines with Canny + Hough line detection and
derives a quad (court boundaries) from the extreme lines. The quad
feeds a homography into the CourtMapper.

When the automatic fit scores below config.COURT_MIN_FIT_SCORE the
pipeline falls back to manual reference points (see main.py).

Note: the automatic detector assumes a roughly straight-on or
slightly angled camera view. Wide broadcast angles and tight
zooms may fail the fit gate and require manual points.
"""

import cv2
import numpy as np

import config


def _to_horizontal(line):
    x1, y1, x2, y2 = line
    return abs(y2 - y1) <= abs(x2 - x1)


def _midpoint(line):
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detect_court_lines(frame):
    """
    Return long court line segments in original frame coordinates:
    [(x1, y1, x2, y2), ...]
    """

    height, width = frame.shape[:2]

    scale = config.COURT_DETECT_MAX_WIDTH / width

    small = cv2.resize(
        frame,
        (config.COURT_DETECT_MAX_WIDTH, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        gray,
        config.COURT_CANNY_LOW,
        config.COURT_CANNY_HIGH,
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=config.COURT_HOUGH_THRESHOLD,
        minLineLength=config.COURT_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.COURT_HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        return []

    inv = 1.0 / scale

    segments = []

    # HoughLinesP shape differs by OpenCV version: (N, 1, 4) or (N, 4).
    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        x1 = float(x1) * inv
        y1 = float(y1) * inv
        x2 = float(x2) * inv
        y2 = float(y2) * inv

        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        if length >= config.COURT_HOUGH_MIN_LINE_LENGTH * inv:
            segments.append((x1, y1, x2, y2))

    return segments


def detect_court_quad(frame):
    """
    Fit a quad (court boundaries) from the detected lines.

    Returns corners ordered as
    [top-left, top-right, bottom-right, bottom-left]
    in original frame coordinates, or None when the fit is poor.
    """

    segments = detect_court_lines(frame)

    horizontals = [s for s in segments if _to_horizontal(s)]
    verticals = [s for s in segments if not _to_horizontal(s)]

    if not horizontals or len(verticals) < 2:
        return None

    baseline_y = max(
        _midpoint(s)[1] for s in horizontals
    )

    far_y = min(
        _midpoint(s)[1] for s in horizontals
    )

    left_x = min(
        min(s[0], s[2]) for s in verticals
    )

    right_x = max(
        max(s[0], s[2]) for s in verticals
    )

    if right_x - left_x < config.COURT_DETECT_MAX_WIDTH * 0.3:
        return None

    quad = np.array(
        [
            [left_x, far_y],
            [right_x, far_y],
            [right_x, baseline_y],
            [left_x, baseline_y],
        ],
        dtype=np.float32,
    )

    if _fit_score(segments, quad) < config.COURT_MIN_FIT_SCORE:
        return None

    return quad


def _fit_score(segments, quad):
    """
    Fraction of detected lines whose midpoints lie near a quad edge.
    """

    if not segments:
        return 0.0

    xs = quad[:, 0]
    ys = quad[:, 1]

    x_edges = [
        (quad[0], quad[1]),  # top
        (quad[2], quad[3]),  # bottom
    ]

    y_edges = [
        (quad[0], quad[3]),  # left
        (quad[1], quad[2]),  # right
    ]

    def distance_to_edges(point, edges):
        best = float("inf")

        for (ax, ay), (bx, by) in edges:

            dx = bx - ax
            dy = by - ay
            length_sq = dx * dx + dy * dy

            if length_sq == 0:
                continue

            t = max(
                0.0,
                min(
                    1.0,
                    ((point[0] - ax) * dx + (point[1] - ay) * dy)
                    / length_sq,
                ),
            )

            nearest_x = ax + t * dx
            nearest_y = ay + t * dy

            distance = (
                (point[0] - nearest_x) ** 2
                + (point[1] - nearest_y) ** 2
            ) ** 0.5

            best = min(best, distance)

        return best

    aligned = 0

    for segment in segments:

        midpoint = _midpoint(segment)

        if _to_horizontal(segment):

            if distance_to_edges(midpoint, x_edges) <= config.COURT_LINE_TOLERANCE_PX:
                aligned += 1

        else:

            if distance_to_edges(midpoint, y_edges) <= config.COURT_LINE_TOLERANCE_PX:
                aligned += 1

    return aligned / len(segments)


def court_reference_points(half_court=None, baseline_at_top=None):
    """
    Build the court-space reference rectangle matching the
    detected quad's corner order.

    Corners are ordered [top-left, top-right, bottom-right,
    bottom-left]; x is along the court length (attack hoop at the
    baseline edge), y across the width.
    """

    if half_court is None:
        half_court = config.HALF_COURT

    if baseline_at_top is None:
        baseline_at_top = config.COURT_BASELINE_AT_TOP

    length = config.COURT_LENGTH_M
    width = config.COURT_WIDTH_M
    half = length / 2.0

    if half_court:

        if baseline_at_top:
            return [(0, 0), (half, 0), (half, width), (0, width)]

        return [(half, 0), (half, width), (0, width), (0, 0)]

    if baseline_at_top:
        return [(length, 0), (length, width), (0, width), (0, 0)]

    return [(0, 0), (0, width), (length, width), (length, 0)]


def setup_court_mapper(frame, court_mapper, manual_image_points=None):
    """
    Configure a CourtMapper with either automatically detected
    court lines or the provided manual reference points.

    Returns a dict describing what was configured:
        {"auto": bool, "quad": quad or None, "fit_score": float}
    """

    quad = detect_court_quad(frame)

    if quad is not None:

        court_mapper.set_reference_points(
            quad.tolist(),
            court_reference_points(),
        )

        court_mapper.compute_homography()

        return {
            "auto": True,
            "quad": quad,
            "fit_score": _fit_score(
                detect_court_lines(frame),
                quad,
            ),
        }

    if manual_image_points is not None:

        court_mapper.set_reference_points(
            manual_image_points,
            court_reference_points(),
        )

        court_mapper.compute_homography()

        return {
            "auto": False,
            "quad": None,
            "fit_score": 0.0,
        }

    return {
        "auto": False,
        "quad": None,
        "fit_score": 0.0,
    }


def draw_court_lines(frame, segments, color=(0, 255, 255), thickness=1):
    """
    Draw detected court lines on a frame (debugging aid).
    """

    for x1, y1, x2, y2 in segments:

        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness,
        )

    return frame
