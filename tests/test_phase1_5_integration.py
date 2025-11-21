"""
Phase 1.5 Integration Test
===========================

End-to-end test of the 3D reconstruction pipeline:
1. 2D pose estimation (RTMPose)
2. Temporal 2D→3D lifting (MotionAGFormer or PoseFormerV2)
3. 3D visualization

Usage:
    python tests/test_phase1_5_integration.py --video path/to/video.mp4
    python tests/test_phase1_5_integration.py --model motionagformer  # or poseformerv2
    python tests/test_phase1_5_integration.py --quick  # Use sample video
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_motionagformer_integration(video_path: str, variant: str = 'xs'):
    """Test MotionAGFormer 2D→3D lifting."""
    logger.info("=" * 60)
    logger.info("Testing MotionAGFormer Integration")
    logger.info("=" * 60)

    # Import components
    from src.pose.rtmpose_estimator import RTMPoseEstimator
    from src.reconstruction.motionagformer_estimator import MotionAGFormerEstimator
    import cv2

    # Initialize 2D pose estimator
    logger.info("Initializing RTMPose...")
    pose_2d = RTMPoseEstimator(
        model_size='m',
        det_model='yolox-s',
        device='auto'
    )

    # Initialize 3D lifter
    logger.info(f"Initializing MotionAGFormer-{variant.upper()}...")
    sequence_length = 27 if variant == 'xs' else 81 if variant == 's' else 243
    lifter_3d = MotionAGFormerEstimator(
        model_variant=variant,
        sequence_length=sequence_length,
        device='auto'
    )

    # Open video
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS")

    # Process frames
    results_2d = []
    results_3d = []
    frame_times_2d = []
    frame_times_3d = []

    logger.info("Processing frames...")
    frame_idx = 0
    max_frames = min(100, total_frames)  # Process first 100 frames for test

    start_time = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 2D pose estimation
        t0 = time.time()
        pose_data, _ = pose_2d.estimate_pose(frame, return_image=False)
        t1 = time.time()
        frame_times_2d.append(t1 - t0)

        if pose_data is None:
            keypoints_2d = None
        else:
            keypoints_2d = pose_data['keypoints']

        results_2d.append(keypoints_2d)

        # 3D lifting (streaming mode)
        t0 = time.time()
        pose_3d = lifter_3d.add_frame_2d(keypoints_2d)
        t1 = time.time()
        frame_times_3d.append(t1 - t0)

        results_3d.append(pose_3d)

        if frame_idx % 10 == 0:
            logger.info(f"  Frame {frame_idx}/{max_frames}")

        frame_idx += 1

    cap.release()

    total_time = time.time() - start_time

    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Processed {frame_idx} frames in {total_time:.2f}s")
    logger.info(f"Overall FPS: {frame_idx / total_time:.2f}")
    logger.info(f"\n2D Pose Estimation:")
    logger.info(f"  Average time: {np.mean(frame_times_2d) * 1000:.2f}ms")
    logger.info(f"  FPS: {1.0 / np.mean(frame_times_2d):.2f}")
    logger.info(f"\n3D Pose Lifting:")
    logger.info(f"  Average time: {np.mean(frame_times_3d) * 1000:.2f}ms")
    logger.info(f"  FPS: {1.0 / np.mean(frame_times_3d):.2f}")

    # Count successful 3D reconstructions
    num_3d = sum(1 for p in results_3d if p is not None)
    logger.info(f"\n3D Poses Reconstructed: {num_3d}/{frame_idx}")

    if num_3d > 0:
        logger.info("✅ Integration test PASSED")
    else:
        logger.warning("⚠️  No 3D poses reconstructed (weights may not be loaded)")

    return results_2d, results_3d


def test_poseformerv2_integration(video_path: str, variant: str = '3-27-47.9'):
    """Test PoseFormerV2 2D→3D lifting."""
    logger.info("=" * 60)
    logger.info("Testing PoseFormerV2 Integration")
    logger.info("=" * 60)

    # Import components
    from src.pose.rtmpose_estimator import RTMPoseEstimator
    from src.reconstruction.poseformerv2_estimator import PoseFormerV2Estimator
    import cv2

    # Initialize 2D pose estimator
    logger.info("Initializing RTMPose...")
    pose_2d = RTMPoseEstimator(
        model_size='m',
        det_model='yolox-s',
        device='auto'
    )

    # Initialize 3D lifter
    logger.info(f"Initializing PoseFormerV2 ({variant})...")
    sequence_length = int(variant.split('-')[1])
    lifter_3d = PoseFormerV2Estimator(
        variant=variant,
        sequence_length=sequence_length,
        device='auto'
    )

    # Open video
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS")

    # Process frames
    results_2d = []
    results_3d = []
    frame_times_2d = []
    frame_times_3d = []

    logger.info("Processing frames...")
    frame_idx = 0
    max_frames = min(100, total_frames)

    start_time = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 2D pose estimation
        t0 = time.time()
        pose_data, _ = pose_2d.estimate_pose(frame, return_image=False)
        t1 = time.time()
        frame_times_2d.append(t1 - t0)

        if pose_data is None:
            keypoints_2d = None
        else:
            keypoints_2d = pose_data['keypoints']

        results_2d.append(keypoints_2d)

        # 3D lifting
        t0 = time.time()
        pose_3d = lifter_3d.add_frame_2d(keypoints_2d)
        t1 = time.time()
        frame_times_3d.append(t1 - t0)

        results_3d.append(pose_3d)

        if frame_idx % 10 == 0:
            logger.info(f"  Frame {frame_idx}/{max_frames}")

        frame_idx += 1

    cap.release()

    total_time = time.time() - start_time

    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    logger.info(f"Processed {frame_idx} frames in {total_time:.2f}s")
    logger.info(f"Overall FPS: {frame_idx / total_time:.2f}")
    logger.info(f"\n2D Pose Estimation:")
    logger.info(f"  Average time: {np.mean(frame_times_2d) * 1000:.2f}ms")
    logger.info(f"  FPS: {1.0 / np.mean(frame_times_2d):.2f}")
    logger.info(f"\n3D Pose Lifting:")
    logger.info(f"  Average time: {np.mean(frame_times_3d) * 1000:.2f}ms")
    logger.info(f"  FPS: {1.0 / np.mean(frame_times_3d):.2f}")

    # Count successful 3D reconstructions
    num_3d = sum(1 for p in results_3d if p is not None)
    logger.info(f"\n3D Poses Reconstructed: {num_3d}/{frame_idx}")

    if num_3d > 0:
        logger.info("✅ Integration test PASSED")
    else:
        logger.warning("⚠️  No 3D poses reconstructed (weights may not be loaded)")

    return results_2d, results_3d


def test_pipeline3d(video_path: str):
    """Test unified Pipeline3D orchestrator."""
    logger.info("=" * 60)
    logger.info("Testing Pipeline3D Orchestrator")
    logger.info("=" * 60)

    from src.reconstruction.pipeline_3d import create_pipeline, PRESET_REALTIME

    logger.info("Creating pipeline with REALTIME preset...")
    pipeline = create_pipeline(PRESET_REALTIME)

    logger.info(f"Processing video: {video_path}")

    try:
        results = pipeline.process_video(video_path=video_path, show_progress=True)
        logger.info(f"✅ Pipeline processed {len(results)} frames")
        return results
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Test Phase 1.5 Integration')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument(
        '--model',
        choices=['motionagformer', 'poseformerv2', 'pipeline'],
        default='motionagformer',
        help='Model to test'
    )
    parser.add_argument(
        '--variant',
        type=str,
        help='Model variant (xs/s/b/l for MAG, 3-27-47.9 for PFV2)'
    )
    parser.add_argument('--quick', action='store_true', help='Use sample video')

    args = parser.parse_args()

    # Determine video path
    if args.quick or args.video is None:
        # Use sample video from MotionAGFormer
        video_path = str(PROJECT_ROOT / "models" / "motionagformer" / "demo" / "video" / "sample_video.mp4")
        if not Path(video_path).exists():
            # Try PoseFormerV2 sample
            video_path = str(PROJECT_ROOT / "models" / "poseformerv2" / "demo" / "video" / "sample_video.mp4")
        logger.info(f"Using sample video: {video_path}")
    else:
        video_path = args.video

    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Run test
    try:
        if args.model == 'motionagformer':
            variant = args.variant or 'xs'
            test_motionagformer_integration(video_path, variant)

        elif args.model == 'poseformerv2':
            variant = args.variant or '3-27-47.9'
            test_poseformerv2_integration(video_path, variant)

        elif args.model == 'pipeline':
            test_pipeline3d(video_path)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
