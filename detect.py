import os
import cv2
from ultralytics import YOLO

from vision import draw_box, extract_frame


# ============================================================
# MODELS
# ============================================================

PLAYER_MODEL_PATH = "models/yolov8n.pt"              # existing COCO-based player model
BALL_MODEL_PATH = "models/basketball_ball_best.pt"   # your new custom ball-only model
RIM_MODEL_PATH = None  # set this once you train rim/backboard model — leave None for now


def load_models():
    player_model = YOLO(PLAYER_MODEL_PATH)
    ball_model = YOLO(BALL_MODEL_PATH)
    rim_model = YOLO(RIM_MODEL_PATH) if RIM_MODEL_PATH else None

    print("✅ Player model loaded.")
    print("✅ Custom ball model loaded.")
    if rim_model:
        print("✅ Rim model loaded.")
    else:
        print("⚠️ No rim model set yet — rim detection disabled.")

    return player_model, ball_model, rim_model


# ============================================================
# DETECTION
# ============================================================

def run_detection(model, frame, conf=0.25):
    result = model(frame, verbose=False, conf=conf)[0]

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

def detect_players(player_model, frame, confidence=0.5):
    result = run_detection(player_model, frame, conf=confidence)
    boxes = _get_boxes(result, ["person"])  # COCO's actual class name
    return filter_boxes(boxes, confidence)


def detect_ball(ball_model, frame, confidence=0.25):
    result = run_detection(ball_model, frame, conf=confidence)
    boxes = _get_boxes(result, ["ball"])  # matches your custom model's class
    balls = filter_boxes(boxes, confidence)
    return balls[0] if balls else None


def detect_rim(rim_model, frame, confidence=0.4):
    if rim_model is None:
        return None  # not trained yet

    result = run_detection(rim_model, frame, conf=confidence)
    boxes = _get_boxes(result, ["rim", "hoop"])
    rims = filter_boxes(boxes, confidence)
    return rims[0] if rims else None


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


# ============================================================
# DRAWING
# ============================================================

def annotate_frame(frame, boxes):
    for box in boxes:
        x, y, w, h, conf = box_to_coords(box)
        label = f"{conf:.2f}"
        frame = draw_box(frame, x, y, w, h, label)
    return frame


# ============================================================
# VIDEO
# ============================================================

def process_video(
    player_model,
    ball_model,
    rim_model,
    video_path,
    output_dir,
    every_n_frames=30,
    player_confidence=0.5,
    ball_confidence=0.25,
    rim_confidence=0.4,
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % every_n_frames == 0:
            players = detect_players(player_model, frame, player_confidence)
            ball = detect_ball(ball_model, frame, ball_confidence)
            rim = detect_rim(rim_model, frame, rim_confidence)

            all_boxes = list(players)
            if ball:
                all_boxes.append(ball)
            if rim:
                all_boxes.append(rim)

            annotated = annotate_frame(frame, all_boxes)

            cv2.imwrite(
                os.path.join(output_dir, f"frame_{frame_index}.jpg"),
                annotated
            )

            print(
                f"Frame {frame_index}: "
                f"{len(players)} players, "
                f"{'1' if ball else '0'} ball, "
                f"{'1' if rim else '0'} rim"
            )
            saved += 1

        frame_index += 1

    cap.release()
    print(f"\nProcessed {saved} frames.")
    return saved


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    player_model, ball_model, rim_model = load_models()

    frame = extract_frame("footage.mp4", 10)

    players = detect_players(player_model, frame)
    ball = detect_ball(ball_model, frame)
    rim = detect_rim(rim_model, frame)

    print(f"Players : {len(players)}")
    print(f"Ball    : {'found' if ball else 'not found'}")
    print(f"Rim     : {'found' if rim else 'not found'}")

    boxes_to_draw = list(players)
    if ball:
        boxes_to_draw.append(ball)
    if rim:
        boxes_to_draw.append(rim)

    annotated = annotate_frame(frame, boxes_to_draw)
    cv2.imwrite("detected_frame.jpg", annotated)
    print("Saved detected_frame.jpg")
