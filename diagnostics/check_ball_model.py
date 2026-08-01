"""
Diagnostic: print the class names of the player and ball YOLO models.

Why: detect.detect_ball() filters detections by the literal class name
"ball". If the custom ball model was trained with a different class
name (e.g. "basketball"), detect_ball silently returns None on every
frame - which would explain "Ball: Not detected" for ~99% of a run.

Usage (Colab):
    !python diagnostics/check_ball_model.py
"""

import os

from ultralytics import YOLO

MODEL_DIR = os.environ.get(
    "COURTVISION_MODEL_DIR",
    "/content/CourtVision/CourtVision/models",
)
PLAYER_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
BALL_MODEL_PATH = os.path.join(MODEL_DIR, "basketball_ball_best.pt")

print(f"Player model: {PLAYER_MODEL_PATH}")
player_model = YOLO(PLAYER_MODEL_PATH)
print("Player model classes:", player_model.names)
print()

print(f"Ball model: {BALL_MODEL_PATH}")
ball_model = YOLO(BALL_MODEL_PATH)
print("Ball model classes:", ball_model.names)
print()

if "ball" not in ball_model.names.values():
    print("!! WARNING: ball model has no class named 'ball'.")
    print("!! detect.detect_ball() filters for the literal name 'ball'")
    print("!! and will return None on every frame. Fix the filter in")
    print("!! detect.py or retrain/rename the model class.")
else:
    print(
        "OK: ball model has a class named 'ball' - "
        "the filter in detect.py matches."
    )
