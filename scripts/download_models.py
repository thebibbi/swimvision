#!/usr/bin/env python3
"""Download YOLO11 pose estimation models."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def download_model(model_name: str, output_dir: Path) -> None:
    """Download a YOLO model.

    Args:
        model_name: Name of the YOLO model (e.g., 'yolo11n-pose.pt').
        output_dir: Directory to save the model.
    """
    print(f"Downloading {model_name}...")

    try:
        # Initialize YOLO model (this will download if not present)
        model = YOLO(model_name)

        # Get the model file path
        model_path = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None

        if model_path and model_path.exists():
            print(f"✅ {model_name} downloaded successfully")
            print(f"   Location: {model_path}")
        else:
            print(f"✅ {model_name} loaded (may already be cached)")

    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download YOLO11 pose models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n-pose.pt", "yolo11s-pose.pt"],
        help="List of models to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/pose_models"),
        help="Output directory for models",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SwimVision - YOLO11 Model Downloader")
    print("=" * 60)
    print()

    # Download each model
    for model_name in args.models:
        download_model(model_name, args.output_dir)
        print()

    print("=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
