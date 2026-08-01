# CourtVision

AI-powered basketball video analytics pipeline: detects players, the ball, and the rim in game footage, tracks them frame-to-frame, and derives teams, possession, passes, shots, player stats, and commentary.

**Status: in development (MVP).** The detection/tracking pipeline runs end-to-end on video, and the analytical modules (possession, passes, shots, stats, commentary, heatmaps, export) are implemented and unit-tested. See [Limitations](#limitations) for what is not yet reliable.

## Quickstart

### 1. Install

Runtime dependencies (requires a CUDA/Colab environment for reasonable speed):

```bash
pip install -r requirements.txt
```

### 2. Download the models

```bash
python diagnostics/fetch_models.py --ball-url <roboflow-download-url>
```

The player model (official YOLOv8n) downloads automatically. The ball model is a custom Roboflow-trained detector, so you must pass its download URL; without it the pipeline runs but ball detection is disabled. Pass `--rim-url` if you have a custom rim model. Models land in `models/` by default:

| Model | Default path | Purpose |
|---|---|---|
| Player | `models/yolov8n.pt` | YOLOv8 player detection |
| Ball | `models/basketball_ball_best.pt` | Fine-tuned basketball detector (Roboflow) |
| Rim | `models/...` (optional) | If unset, a classical-CV rim detector is used |

Set `COURTVISION_MODEL_DIR` to override the model directory, or edit [courtvision/config.py](courtvision/config.py). Existing files are skipped, so re-running the downloader is safe.

### 3. Run the pipeline

```bash
py main.py footage.mp4
```

Export frames, a game summary, and heatmaps to a directory:

```bash
py main.py footage.mp4 --output out/
```

This writes:

- `out/frames.json` — sampled per-frame record (every Nth frame, see `MATCH_FRAME_SAMPLING`)
- `out/summary.json` — game-level totals (possession, passes, shots, stats)
- `out/heatmap_*.jpg` — court-space heatmaps
- `out/spool.jsonl` — full per-frame JSONL spool

### 4. Run the tests

The logic test suite needs only the dev requirements (no ML models):

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

or standalone: `python tests/test_logic.py`

### 5. Evaluate detection quality

Ground-truth labels against pipeline detections to measure precision/recall/F1:

```bash
py diagnostics/evaluate_detections.py --video footage.mp4 --labels labels.json
```

See the docstring in [diagnostics/evaluate_detections.py](diagnostics/evaluate_detections.py) for the label format and options.

## Repository layout

```
main.py                        CLI entry point
courtvision/
  pipeline.py                  main run() loop
  detect.py                    YOLO model loading + detection
  tracker.py                   ByteTrack integration, ball interpolation
  team.py                      jersey-color team classification (KMeans)
  possession.py                ball possession engine
  pass_detector.py             pass/success/fail detection
  shot_detector.py             shot attempts + made/missed
  rim.py                       classical-CV rim detector
  court_detection.py           court quad detection
  zones.py                     court-space zone classification
  heatmap.py                   court-space heatmap grid
  data.py                      match data storage (sampled + JSONL spool)
  export.py                    JSON summary + heatmap export
  commentary.py                human-readable commentary
  stats.py, trajectory.py      player stats and movement trajectories
  config.py                    every tunable parameter
tests/test_logic.py            11 unit tests, 67 checks
diagnostics/
  evaluate_detections.py       precision/recall/F1 eval harness
  check_ball_model.py          ball-model class-label discovery
```

## Limitations

Honest list of what the MVP does not yet do reliably:

- **Shot detection** is conservative by design. A shot attempt is only registered when the ball moves away from the shooter AND approaches the rim; outcomes (made/missed) resolve only from a *recent* rim observation. Airballs and rim-distant shots are reported as unresolved rather than guessed.
- **Ball detection** depends on the fine-tuned ball model; the stock YOLOv8 COCO model does not reliably detect a small fast ball. Expect dropped detections on motion blur and occlusions (interpolation covers short gaps, not long ones).
- **Possession and passes** are heuristic (nearest-player within a distance threshold, speed gating, hysteresis). They can mislabel loose-ball scrambles, and passes through heavy traffic are under-detected.
- **Team classification** clusters jersey colors with KMeans from detected player crops; jerseys with similar colors, or weak detections, can flip assignments mid-game.
- **Court detection** assumes a fairly standard half-court broadcast angle and maps it to a fixed FIBA half-court model (`HALF_COURT=True`). Unusual camera angles degrade zone and heatmap accuracy.
- **Player identity** is inherited from ByteTrack; ID switches during occlusions/group play are possible and propagate to per-player stats.
- **ESV (Expected Shot Value)** uses a simple distance-decay curve fit to league averages, not a trained model.
