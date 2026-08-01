import argparse

from courtvision import config
from courtvision.pipeline import run


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CourtVision basketball analytics pipeline."
    )

    parser.add_argument(
        "video_path",
        nargs="?",
        default=config.VIDEO_PATH,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--output",
        default=config.OUTPUT_DIR,
        help="Directory for exported frames, summary and heatmaps.",
    )

    args = parser.parse_args()

    run(
        args.video_path,
        output_dir=args.output,
    )
