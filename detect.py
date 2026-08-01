from ultralytics import YOLO

import config
from rim import detect_rim_box


# ============================================================
# MODELS
# ============================================================

PLAYER_MODEL_PATH = config.PLAYER_MODEL_PATH
BALL_MODEL_PATH = config.BALL_MODEL_PATH
RIM_MODEL_PATH = config.RIM_MODEL_PATH


def load_models():
    player_model = YOLO(PLAYER_MODEL_PATH)
    ball_model = YOLO(BALL_MODEL_PATH)
    rim_model = YOLO(RIM_MODEL_PATH) if RIM_MODEL_PATH else None

    print("✅ Player model loaded.")
    print("✅ Custom ball model loaded.")
    if rim_model:
        print("✅ Rim model loaded.")
    else:
        print("ℹ️ No rim model set — using classical-CV rim detection.")

    return player_model, ball_model, rim_model


# ============================================================
# DETECTION
# ============================================================

def run_detection(model, frame, conf=0.25, verbose=False):
    result = model(frame, verbose=False, conf=conf)[0]

    if verbose:
        print("\nYOLO Detections")
        for box in result.boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            print(f"{result.names[cls]:15} conf={conf_val:.2f}")

    return result


# ============================================================
# CLASS HELPERS
# ============================================================

def _get_boxes(result, class_names):
    return [
        box
        for box in result.boxes
        if result.names[int(box.cls[0])] in class_names
    ]


def filter_boxes(boxes, confidence=0.5):
    return [
        box
        for box in boxes
        if float(box.conf[0]) >= confidence
    ]


# ============================================================
# PUBLIC DETECTORS
# ============================================================

def detect_ball(ball_model, frame, confidence=0.25):
    result = run_detection(ball_model, frame, conf=confidence)
    boxes = _get_boxes(result, config.BALL_CLASS_NAMES)
    balls = filter_boxes(boxes, confidence)
    return balls[0] if balls else None


def detect_rim(rim_model, frame, confidence=0.4):
    """
    Detect the rim.

    Uses the trained rim model when available; otherwise falls
    back to the classical-CV detector (rim.py).
    """

    if rim_model is not None:

        result = run_detection(rim_model, frame, conf=confidence)
        boxes = _get_boxes(result, ["rim", "hoop"])
        rims = filter_boxes(boxes, confidence)
        return rims[0] if rims else None

    return detect_rim_box(frame)


# ============================================================
# BOX UTILITIES
# ============================================================

def box_to_coords(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)
    confidence = float(box.conf[0])
    return x, y, w, h, confidence
