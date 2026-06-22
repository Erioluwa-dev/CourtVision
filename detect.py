import os
import cv2
from ultralytics import YOLO

from vision import draw_box, extract_frame


MODEL_PATH = "yolov8n.pt"


# ---------------------------
# Model
# ---------------------------

def load_model(model_path=MODEL_PATH):
    return YOLO(model_path)


# ---------------------------
# Detection
# ---------------------------

def run_detection(model, frame):
    return model(frame)[0]


def get_person_boxes(result):
    return [
        box
        for box in result.boxes
        if result.names[int(box.cls[0])] == "person"
    ]


def filter_boxes(boxes, confidence=0.5):
    return [
        box
        for box in boxes
        if float(box.conf[0]) >= confidence
    ]


def detect_players(model, frame, confidence=0.5):
    result = run_detection(model, frame)
    persons = get_person_boxes(result)
    return filter_boxes(persons, confidence)


def count_players(result, confidence=0.5):
    persons = get_person_boxes(result)
    return len(filter_boxes(persons, confidence))


# ---------------------------
# Box Utilities
# ---------------------------

def box_to_coords(box):
    x1, y1, x2, y2 = box.xyxy[0]

    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)

    confidence = float(box.conf[0])

    return x, y, w, h, confidence


# ---------------------------
# Visualization
# ---------------------------

def annotate_frame(frame, boxes):
    for box in boxes:
        x, y, w, h, conf = box_to_coords(box)

        frame = draw_box(
            frame,
            x,
            y,
            w,
            h,
            f"Player {conf:.2f}"
        )

    return frame


# ---------------------------
# Video Processing
# ---------------------------

def process_video(
    model,
    video_path,
    output_dir,
    every_n_frames=30,
    confidence=0.5,
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % every_n_frames == 0:
            boxes = detect_players(model, frame, confidence)

            annotated = annotate_frame(frame, boxes)

            output_path = os.path.join(
                output_dir,
                f"detected_{frame_index}.jpg"
            )

            cv2.imwrite(output_path, annotated)

            print(
                f"Frame {frame_index}: "
                f"{len(boxes)} players detected"
            )

            saved_count += 1

        frame_index += 1

    cap.release()

    print(f"Processed {saved_count} frames → {output_dir}")

    return saved_count


# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    model = load_model()

    frame = extract_frame("footage.mp4", 10)

    boxes = detect_players(model, frame)

    print(f"Detected {len(boxes)} players")

    annotated = annotate_frame(frame, boxes)

    cv2.imwrite("detected_frame.jpg", annotated)

    print("Saved detected_frame.jpg")
