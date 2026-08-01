"""
CourtVision configuration.

Central home for every tunable in the Phase 1-6 pipeline so you
can adjust behaviour on Colab without hunting through modules.

All values use the same units the modules document (pixels/frame
for image-space, metres for court space).
"""

import os


# ============================================================
# Models & inputs
# ============================================================
# Update these to your Colab paths.
MODEL_DIR = os.environ.get("COURTVISION_MODEL_DIR", "models")
PLAYER_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
BALL_MODEL_PATH = os.path.join(MODEL_DIR, "basketball_ball_best.pt")
# Set to a path once you train a rim/backboard model; when None the
# classical-CV rim detector (rim.py) is used automatically.
RIM_MODEL_PATH = os.environ.get("COURTVISION_RIM_MODEL_PATH", None)

VIDEO_PATH = os.environ.get("COURTVISION_VIDEO_PATH", "footage.mp4")

# Detection confidence thresholds.
PLAYER_CONFIDENCE = 0.5
BALL_CONFIDENCE = 0.25
RIM_CONFIDENCE = 0.4

# Ball model class label(s). Discover with diagnostics/check_ball_model.py.
BALL_CLASS_NAMES = ["ball"]

# How often (frames) the tracker debug dump and progress log runs.
LOG_EVERY_N_FRAMES = 100


# ============================================================
# Phase 1 — Ball handling
# ============================================================
# Frames the ball can be missing before the interpolated position
# is dropped (the ball is assumed to be genuinely lost).
BALL_MAX_INTERPOLATION_FRAMES = 10

# Ball smoothing method: "kalman" or "ema".
BALL_SMOOTHING = "kalman"
# Exponential moving average alpha when BALL_SMOOTHING == "ema".
BALL_EMA_ALPHA = 0.4

# Ball velocity is measured over this many frames (smooths jitter).
BALL_VELOCITY_WINDOW = 5


# ============================================================
# Phase 1 — Classical rim detector (rim.py)
# ============================================================
RIM_ORANGE_HUE_LOW = 3
RIM_ORANGE_HUE_HIGH = 22
RIM_ORANGE_MIN_SATURATION = 90
RIM_ORANGE_MIN_VALUE = 90
# Only look for the rim in the upper portion of the frame.
RIM_SEARCH_TOP_FRACTION = 0.65
# Minimum contour area (px^2) to accept as a rim candidate.
RIM_MIN_AREA = 80

# Backboard fallback thresholds (white rectangle detection).
BACKBOARD_MIN_VALUE = 200
BACKBOARD_MAX_SATURATION = 45


# ============================================================
# Phase 1 — Classical court detector (court_detection.py)
# ============================================================
# FIBA full court is 28m x 15m. When only one half is visible
# (typical camera angle) HALF_COURT=True maps the visible quad
# to 14m x 15m with the attack basket on the baseline edge.
COURT_LENGTH_M = 28.0
COURT_WIDTH_M = 15.0
HALF_COURT = True

# Preprocessing resolution for line detection (fast downscale).
COURT_DETECT_MAX_WIDTH = 960
COURT_CANNY_LOW = 50
COURT_CANNY_HIGH = 150
COURT_HOUGH_THRESHOLD = 90
COURT_HOUGH_MIN_LINE_LENGTH = 120
COURT_HOUGH_MAX_LINE_GAP = 20
# Minimum confidence for the automatic court fit (0-1). When the
# detected quad scores below this the mapper falls back to the
# manual reference points.
COURT_MIN_FIT_SCORE = 0.4

# True when the far edge of the frame (top) is the attack baseline
# (camera between midcourt and the hoop). False when the camera
# sits behind the hoop (near edge = baseline).
COURT_BASELINE_AT_TOP = True

# Max distance (px, original frame scale) from a detected line's
# midpoint to a quad edge for it to count as aligning with it.
COURT_LINE_TOLERANCE_PX = 12


# ============================================================
# Phase 2 — Court zones (zones.py)
# ============================================================
# FIBA key/paint dimensions in metres.
PAINT_DEPTH_M = 5.8
PAINT_WIDTH_M = 4.9
FREE_THROW_CIRCLE_RADIUS_M = 1.8
# FIBA 3-point line distance from the hoop (arc) in metres.
THREE_POINT_DISTANCE_M = 6.75
# Corner three: the 3pt line is 0.9m from the sideline.
CORNER_SIDELINE_DISTANCE_M = 0.9
# "Half court" band width around midcourt for half-court views.
HALF_COURT_BAND_M = 3.0


# ============================================================
# Phase 4 — Possession engine (possession.py)
# ============================================================
# Maximum image-space distance between ball and player for
# possession to be considered.
POSSESSION_DISTANCE_PX = 60
# Ball speeds above this (px/frame) are treated as a loose ball
# (pass in flight / shot) — no possession.
POSSESSION_MAX_BALL_SPEED_PX = 40
# Consecutive frames a new candidate must hold before the
# possessor switches (prevents flicker).
POSSESSION_HYSTERESIS_FRAMES = 5
# Frames with no valid possessor before the ball is declared loose.
POSSESSION_LOOSE_AFTER_FRAMES = 3


# ============================================================
# Phase 5 — Passing engine (pass_detector.py)
# ============================================================
# A possession change through a loose-ball gap longer than this
# counts the preceding possession as a failed pass / turnover.
PASS_FAIL_LOOSE_FRAMES = 8

# Minimum consecutive loose-ball frames between two possessions for
# a pass to be recorded. A possession that switches with no real
# in-flight gap is almost always ByteTrack ID flicker or a loose-ball
# recovery, not a pass - rebounds and steals are filtered out by
# requiring the ball to actually be in flight.
PASS_MIN_LOOSE_FRAMES = 2


# ============================================================
# Phase 6 — Shooting engine (shot_detector.py)
# ============================================================
# Image-space distance the ball must travel away from the shooter
# before a shot attempt is registered.
SHOT_START_DISTANCE_PX = 120
# How close (px) the ball must come to the rim centre to count as
# "reaching the rim" (separates airballs from real attempts).
SHOT_RIM_PROXIMITY_PX = 80
# Ball must be this many px below the rim centre (image space)
# to be considered as having gone through.
SHOT_RIM_CLEARANCE_PX = 25
# Rim positions older than this many frames are stale - shot
# outcomes are not resolved from a stale rim (camera cut, wrong
# hoop detected, etc.).
SHOT_RIM_MAX_STALE_FRAMES = 15
# Expected Shot Value make-probability model.
#   p_make(distance_m) = ESV_MAKE_BASE * exp(-ESV_MAKE_DECAY * distance_m)
# Fitted roughly to league-average shooting (0.65 at the rim,
# ~0.36 at the 3PT line). Tunable; this is a stopgap model.
ESV_MAKE_BASE = 0.65
ESV_MAKE_DECAY = 0.08


# ============================================================
# Phase 2 — Heatmaps (heatmap.py)
# ============================================================
# Bin size in metres for the court-space heatmap grid.
HEATMAP_BIN_SIZE_M = 0.5


# ============================================================
# Match data storage (data.py)
# ============================================================
# Keep every Nth frame in memory. A full game at 30fps is ~86k
# frames; sampling bounds RAM while the JSONL spool keeps the full
# record on disk when the pipeline is given an output directory.
MATCH_FRAME_SAMPLING = 5
