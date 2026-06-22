import cv2
import os
import math

from vision import draw_box
from detect import box_to_coords


# ============================================================
# IoU & Centroid Utilities
# ============================================================

def iou(box_a, box_b):
    x_left   = max(box_a[0], box_b[0])
    y_top    = max(box_a[1], box_b[1])
    x_right  = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - intersection

    return intersection / union


def get_centroid(x, y, w, h):
    return (x + w // 2, y + h // 2)


def centroid_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


# ============================================================
# ByteTrack — Detection
# ============================================================

def run_tracker(model, frame):
    results = model.track(frame, persist=True)
    return results[0]


def get_track_ids(result):
    return [
        int(box.id[0]) if box.id is not None else -1
        for box in result.boxes
    ]


def get_tracked_persons(result):
    return [
        box for box in result.boxes
        if result.names[int(box.cls[0])] == "person"
        and box.id is not None
    ]


def track_players_in_frame(model, frame):
    result = run_tracker(model, frame)
    return get_tracked_persons(result), result


# ============================================================
# Visualization
# ============================================================

def annotate_tracked_frame(frame, result):
    for box in get_tracked_persons(result):
        x, y, w, h, conf = box_to_coords(box)
        track_id = int(box.id[0])
        frame = draw_box(frame, x, y, w, h, f"ID:{track_id}")
    return frame


# ============================================================
# Video Pipeline
# ============================================================

def process_video_with_tracking(model, video_path, output_dir, every_n=30):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % every_n == 0:
            boxes, result = track_players_in_frame(model, frame)
            annotated    = annotate_tracked_frame(frame.copy(), result)

            out_path = os.path.join(output_dir, f"tracked_{frame_index}.jpg")
            cv2.imwrite(out_path, annotated)

            ids = [int(b.id[0]) for b in boxes]
            print(f"Frame {frame_index}: {len(boxes)} players → IDs {ids}")
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Processed {saved_count} frames → {output_dir}/")
    return saved_count


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    from detect import load_model
    from vision import extract_frame

    VIDEO_PATH = "/content/CourtVision/footage.mp4"

    # --- math tests (no model needed) ---
    box_a = (0, 0, 100, 100)
    box_b = (50, 50, 150, 150)
    print(f"IoU: {iou(box_a, box_b):.2f}")

    c1 = get_centroid(0, 0, 100, 100)
    c2 = get_centroid(50, 50, 100, 100)
    print(f"Centroid 1: {c1}, Centroid 2: {c2}")
    print(f"Distance: {centroid_distance(c1, c2):.2f}")

    # --- tracking tests ---
    model = load_model()
    frame = extract_frame(VIDEO_PATH, 10)
    print("Frame loaded:", frame is not None)

    boxes, result = track_players_in_frame(model, frame)
    ids = [int(b.id[0]) for b in boxes]
    print(f"Frame 10: {len(boxes)} players tracked → IDs {ids}")

    annotated = annotate_tracked_frame(frame.copy(), result)
    cv2.imwrite("tracked_frame.jpg", annotated)
    print("Saved tracked_frame.jpg")

    # --- bonus pipeline ---
    process_video_with_tracking(model, VIDEO_PATH, "tracked", every_n=30)
