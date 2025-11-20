"""
Phase 1 Pipeline Demo
Demonstrates the complete Phase 1 pipeline:
- RTMPose pose estimation
- ByteTrack multi-swimmer tracking
- Format conversion (COCO -> SMPL -> OpenSim)
- Real-time visualization

Usage:
    python demos/demo_phase1_pipeline.py --video path/to/video.mp4
    python demos/demo_phase1_pipeline.py --webcam  # Use webcam

Author: SwimVision Pro Team
Date: 2025-01-20
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.orchestrator import (
    SwimVisionPipeline,
    PipelineConfig,
    ProcessingMode
)


def create_test_video(output_path: str, duration_sec: int = 5):
    """
    Create a synthetic test video with moving stick figures.

    Args:
        output_path: Where to save the video
        duration_sec: Duration in seconds
    """
    width, height = 1280, 720
    fps = 30
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_sec}s, Resolution: {width}x{height}, FPS: {fps}")

    for frame_idx in range(total_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw swimming pool lane lines
        for y in range(0, height, 100):
            cv2.line(frame, (0, y), (width, y), (100, 100, 100), 1)

        # Swimmer 1: Moving left to right
        x1 = int(100 + (width - 200) * (frame_idx / total_frames))
        y1 = height // 3
        draw_swimmer(frame, x1, y1, scale=1.0)

        # Swimmer 2: Moving right to left
        x2 = int(width - 100 - (width - 200) * (frame_idx / total_frames))
        y2 = 2 * height // 3
        draw_swimmer(frame, x2, y2, scale=0.8)

        # Add frame number
        cv2.putText(
            frame, f"Frame: {frame_idx}/{total_frames}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        writer.write(frame)

    writer.release()
    print(f"✅ Test video created: {output_path}")


def draw_swimmer(frame, x, y, scale=1.0):
    """Draw a simple swimmer stick figure."""
    s = int(30 * scale)  # Scale factor

    # Head
    cv2.circle(frame, (x, y), int(15 * scale), (255, 200, 100), -1)

    # Body
    cv2.line(frame, (x, y + int(15 * scale)), (x, y + int(50 * scale)),
             (255, 200, 100), int(8 * scale))

    # Arms (swimming position)
    arm_y = y + int(25 * scale)
    cv2.line(frame, (x, arm_y), (x - s, arm_y - s // 2),
             (255, 200, 100), int(6 * scale))
    cv2.line(frame, (x, arm_y), (x + s, arm_y - s // 2),
             (255, 200, 100), int(6 * scale))

    # Legs (kicking)
    leg_y = y + int(50 * scale)
    cv2.line(frame, (x, leg_y), (x - s // 2, leg_y + s),
             (255, 200, 100), int(6 * scale))
    cv2.line(frame, (x, leg_y), (x + s // 2, leg_y + s),
             (255, 200, 100), int(6 * scale))


def demo_webcam(pipeline: SwimVisionPipeline):
    """Run pipeline on webcam feed."""
    print("\n" + "="*60)
    print("WEBCAM DEMO")
    print("="*60)
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("="*60 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = pipeline.process_frame(frame, frame_id=frame_count)

            # Show result
            if result.visualized_frame is not None:
                cv2.imshow("SwimVision - Webcam Demo", result.visualized_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, result.visualized_frame)
                print(f"Screenshot saved: {filename}")

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Print statistics
    print_statistics(pipeline)


def demo_video(pipeline: SwimVisionPipeline, video_path: str, output_path: str = None):
    """Run pipeline on video file."""
    print("\n" + "="*60)
    print("VIDEO DEMO")
    print("="*60)
    print(f"Input: {video_path}")
    if output_path:
        print(f"Output: {output_path}")
    print("Press 'q' to quit early")
    print("="*60 + "\n")

    results = pipeline.process_video(
        video_path,
        output_path=output_path,
        show_preview=True
    )

    print(f"\n✅ Processed {len(results)} frames")

    # Print statistics
    print_statistics(pipeline)

    # Analyze tracking performance
    analyze_tracking(results)


def print_statistics(pipeline: SwimVisionPipeline):
    """Print pipeline statistics."""
    stats = pipeline.get_statistics()

    print("\n" + "="*60)
    print("PIPELINE STATISTICS")
    print("="*60)
    print(f"Total frames:          {stats.get('total_frames', 0)}")
    print(f"Average FPS:           {stats.get('avg_fps', 0):.1f}")
    print(f"Min FPS:               {stats.get('min_fps', 0):.1f}")
    print(f"Max FPS:               {stats.get('max_fps', 0):.1f}")
    print(f"Avg processing time:   {stats.get('avg_processing_time', 0)*1000:.1f} ms")
    print(f"Avg swimmers/frame:    {stats.get('avg_swimmers_per_frame', 0):.1f}")
    print(f"Max swimmers detected: {stats.get('max_swimmers', 0)}")
    print(f"Total runtime:         {stats.get('total_runtime', 0):.1f} s")
    print("="*60 + "\n")


def analyze_tracking(results):
    """Analyze tracking performance across frames."""
    print("\n" + "="*60)
    print("TRACKING ANALYSIS")
    print("="*60)

    # Collect all unique track IDs
    all_track_ids = set()
    track_appearances = {}

    for result in results:
        for swimmer in result.tracked_swimmers:
            track_id = swimmer.get('track_id')
            if track_id is not None:
                all_track_ids.add(track_id)
                track_appearances[track_id] = track_appearances.get(track_id, 0) + 1

    print(f"Unique tracks detected: {len(all_track_ids)}")
    print(f"\nTrack persistence:")
    for track_id, count in sorted(track_appearances.items()):
        percentage = (count / len(results)) * 100
        print(f"  Track {track_id}: {count}/{len(results)} frames ({percentage:.1f}%)")

    # Analyze track consistency
    if len(all_track_ids) > 0:
        avg_persistence = np.mean(list(track_appearances.values()))
        print(f"\nAverage track persistence: {avg_persistence:.1f} frames")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SwimVision Phase 1 Pipeline Demo"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam instead of video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output video"
    )
    parser.add_argument(
        "--create-test-video",
        action="store_true",
        help="Create a synthetic test video"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rtmpose-m",
        choices=["rtmpose-t", "rtmpose-s", "rtmpose-m", "rtmpose-l"],
        help="RTMPose model variant"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="realtime",
        choices=["realtime", "balanced", "accuracy"],
        help="Processing mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable tracking"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["coco17", "smpl24"],
        choices=["coco17", "smpl24", "opensim"],
        help="Output formats for keypoints"
    )

    args = parser.parse_args()

    # Create test video if requested
    if args.create_test_video:
        test_video_path = "data/videos/test_swimmers.mp4"
        Path(test_video_path).parent.mkdir(parents=True, exist_ok=True)
        create_test_video(test_video_path, duration_sec=10)
        print(f"\n✅ Test video created. Run demo with:")
        print(f"   python {__file__} --video {test_video_path}")
        return

    # Validate input
    if not args.webcam and not args.video:
        print("❌ Error: Provide either --video or --webcam")
        print(f"   Or use --create-test-video to generate a test video")
        parser.print_help()
        return

    # Configure pipeline
    mode_map = {
        "realtime": ProcessingMode.REALTIME,
        "balanced": ProcessingMode.BALANCED,
        "accuracy": ProcessingMode.ACCURACY
    }

    config = PipelineConfig(
        pose_models=[args.model],
        enable_tracking=not args.no_tracking,
        mode=mode_map[args.mode],
        output_formats=args.formats,
        visualize=True,
        show_tracking_ids=True,
        show_keypoints=True,
        show_skeleton=True,
        device=args.device
    )

    print("\n" + "="*60)
    print("SWIMVISION PHASE 1 PIPELINE DEMO")
    print("="*60)
    print(f"Pose model:    {args.model}")
    print(f"Tracking:      {'Enabled' if not args.no_tracking else 'Disabled'}")
    print(f"Mode:          {args.mode}")
    print(f"Device:        {args.device}")
    print(f"Output formats: {', '.join(args.formats)}")
    print("="*60)

    try:
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = SwimVisionPipeline(config)
        print("✅ Pipeline initialized\n")

        # Run demo
        if args.webcam:
            demo_webcam(pipeline)
        else:
            demo_video(pipeline, args.video, args.output)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Make sure all dependencies are installed:")
        print("   bash scripts/setup_advanced_features.sh")
        return


if __name__ == "__main__":
    main()
