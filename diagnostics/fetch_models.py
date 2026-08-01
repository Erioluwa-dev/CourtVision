"""
Utility: download the CourtVision detection models into models/.

The player model is the official Ultralytics YOLOv8n weights and is
downloaded automatically. The ball (and optional rim) models are
custom Roboflow-trained weights and need a direct download URL, which
you pass with --ball-url (and --rim-url when you have one).

Usage:
    # Player model only (official YOLOv8n):
    python diagnostics/fetch_models.py

    # Player + custom ball model:
    python diagnostics/fetch_models.py --ball-url <roboflow-download-url>

    # Everything:
    python diagnostics/fetch_models.py \
        --ball-url <roboflow-download-url> --rim-url <roboflow-download-url>

Files already present are skipped, so re-running is safe.
"""

import argparse
import os
import sys
import urllib.request

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from courtvision import config

# Official Ultralytics YOLOv8n weights (COCO). ~6.5 MB.
DEFAULT_PLAYER_URL = (
    "https://github.com/ultralytics/assets/releases/download/"
    "v8.3.0/yolov8n.pt"
)

# A download counts as present when it is at least this big; guards
# against truncated/empty files left behind by a failed run.
MIN_VALID_BYTES = 1024


def _download(url, dest_path):
    """Download url to dest_path, skipping when the file is valid."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) >= MIN_VALID_BYTES:
        print(f"⏭️  Already present: {dest_path}")
        return

    print(f"⬇️  Downloading {url}")
    print(f"   -> {dest_path}")

    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as exc:
        # Remove a partial file so a retry does not trust it.
        if os.path.exists(dest_path):
            os.remove(dest_path)
        print(f"❌ Failed to download {url}: {exc}")
        raise

    size_kb = os.path.getsize(dest_path) / 1024
    print(f"✅ Downloaded {size_kb:.0f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Download CourtVision detection models."
    )

    parser.add_argument(
        "--model-dir",
        default=config.MODEL_DIR,
        help="Directory to download models into (default: %(default)s).",
    )

    parser.add_argument(
        "--player-url",
        default=DEFAULT_PLAYER_URL,
        help="Player model weights URL (default: official YOLOv8n).",
    )

    parser.add_argument(
        "--ball-url",
        default=None,
        help="Custom ball model weights URL (from Roboflow). Required for "
        "ball detection.",
    )

    parser.add_argument(
        "--rim-url",
        default=None,
        help="Optional custom rim model weights URL.",
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    _download(
        args.player_url,
        os.path.join(args.model_dir, os.path.basename(config.PLAYER_MODEL_PATH)),
    )

    if args.ball_url:
        _download(
            args.ball_url,
            os.path.join(args.model_dir, os.path.basename(config.BALL_MODEL_PATH)),
        )
    else:
        print(
            "ℹ️  No --ball-url given. The custom ball model is NOT downloaded; "
            "ball detection will not work until you provide one."
        )

    if args.rim_url:
        _download(
            args.rim_url,
            os.path.join(args.model_dir, os.path.basename(config.RIM_MODEL_PATH))
            if config.RIM_MODEL_PATH
            else os.path.join(args.model_dir, "rim.pt"),
        )


if __name__ == "__main__":
    main()
