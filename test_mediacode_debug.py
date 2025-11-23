#!/usr/bin/env python3
"""Debug MediaPipe processing issue."""

import logging
from pathlib import Path

import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_mediacode_processing():
    """Test MediaPipe with the actual video that's failing."""
    print("=" * 60)
    print("Testing MediaPipe Video Processing")
    print("=" * 60)

    try:
        from src.pose.mediapipe_estimator import MediaPipeEstimator
        from src.pose.swimming_keypoints import SwimmingKeypoints
    except ImportError as e:
        print(f"❌ Cannot import MediaPipe: {e}")
        return False

    # Create estimators
    estimator = MediaPipeEstimator(confidence=0.5, model_complexity=1)
    analyzer = SwimmingKeypoints()

    print("✅ MediaPipe Estimator created")
    print("✅ Swimming Analyzer created")

    # Test with the actual video
    video_path = Path(
        "/Users/ahmedayoub/Documents/VibeClauding/swimvision/data/videos/TestSwim.mp4"
    )

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False

    print(f"✅ Found video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("❌ Cannot open video")
        return False

    frame_count = 0
    success_count = 0
    error_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            print(f"\n🔍 Processing frame {frame_count}...")

            try:
                # Pose estimation
                pose_result, _ = estimator.estimate_pose(frame, return_image=False)

                if pose_result is not None:
                    # Normalize to list format
                    if isinstance(pose_result, list):
                        pose_data_list = pose_result
                    else:
                        pose_data_list = [pose_result]

                    if len(pose_data_list) > 0:
                        pose_data = pose_data_list[0]

                        print(f"✅ Pose detected - type: {type(pose_data)}")
                        print(f"✅ Keys: {list(pose_data.keys())}")

                        # Check keypoints format
                        keypoints = pose_data["keypoints"]
                        print(f"✅ Keypoints type: {type(keypoints)}")

                        if isinstance(keypoints, dict):
                            print(f"✅ Keypoints dict format: {len(keypoints)} keypoints")
                            sample_key = list(keypoints.keys())[0]
                            sample_val = keypoints[sample_key]
                            print(f"✅ Sample keypoint: {sample_key} -> {sample_val}")
                        else:
                            print(f"❌ Unexpected keypoints format: {type(keypoints)}")

                        # Test swimming analyzer
                        try:
                            angles = analyzer.get_body_angles(pose_data)
                            print(f"✅ Angles calculated: {len(angles)} angles")

                            # Test individual keypoint extraction
                            left_wrist = analyzer._get_keypoint(pose_data, "left_wrist")
                            right_wrist = analyzer._get_keypoint(pose_data, "right_wrist")
                            print(f"✅ Left wrist: {left_wrist}")
                            print(f"✅ Right wrist: {right_wrist}")

                            success_count += 1

                        except Exception as e:
                            print(f"❌ Swimming analyzer failed: {e}")
                            import traceback

                            print(f"Full traceback:\n{traceback.format_exc()}")
                            error_count += 1
                            break
                    else:
                        print("⚠️  Empty pose list")
                else:
                    print("⚠️  No pose detected")

            except Exception as e:
                print(f"❌ Frame processing failed: {e}")
                import traceback

                print(f"Full traceback:\n{traceback.format_exc()}")
                error_count += 1
                break

            # Stop after 20 frames to avoid infinite loop
            if frame_count >= 20:
                print("🔍 Stopping after 20 frames for testing")
                break

    finally:
        cap.release()

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Successful frames: {success_count}")
    print(f"Failed frames: {error_count}")

    if error_count == 0:
        print("✅ MediaPipe video processing test PASSED")
        return True
    else:
        print("❌ MediaPipe video processing test FAILED")
        return False


if __name__ == "__main__":
    test_mediacode_processing()
