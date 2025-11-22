#!/usr/bin/env python3
"""Test trajectory extraction with YOLO output."""

import logging

import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from src.pose import YOLOPoseEstimator


def test_trajectory_extraction():
    """Test trajectory extraction with YOLO output."""
    print("=" * 60)
    print("Testing Trajectory Extraction")
    print("=" * 60)

    # Create estimator
    estimator = YOLOPoseEstimator()
    print(f"✅ Estimator created: {estimator.model_name}")

    # Create test pose data (simulating YOLO output)
    keypoints = np.random.rand(17, 3) * 100  # Random [x, y, conf]
    pose_data = {
        "keypoints": keypoints,
        "keypoint_names": estimator.KEYPOINT_NAMES,
        "bbox": [0, 0, 100, 100],
        "person_id": 0,
        "format": estimator.get_keypoint_format(),
        "confidence": 0.8,
        "metadata": {"model": estimator.model_name},
    }

    print(f"✅ Created test pose_data with {len(keypoints)} keypoints")
    print(f"✅ Keypoints type: {type(keypoints)}")
    print(f"✅ Keypoint names: {len(pose_data['keypoint_names'])}")

    # Test get_keypoint method
    try:
        print("\n🔍 Testing get_keypoint for left_wrist...")
        left_wrist = estimator.get_keypoint(pose_data, "left_wrist")
        print(f"✅ Left wrist: {left_wrist}")

        print("\n🔍 Testing get_keypoint for right_wrist...")
        right_wrist = estimator.get_keypoint(pose_data, "right_wrist")
        print(f"✅ Right wrist: {right_wrist}")

        # Test trajectory extraction logic
        print("\n🔍 Testing trajectory extraction logic...")
        left_hand_path = []
        right_hand_path = []

        if left_wrist:
            left_hand_path.append((left_wrist[0], left_wrist[1]))
            print(f"✅ Added left wrist to path: {left_hand_path}")

        if right_wrist:
            right_hand_path.append((right_wrist[0], right_wrist[1]))
            print(f"✅ Added right wrist to path: {right_hand_path}")

        print("\n✅ Trajectory extraction test PASSED")

    except AttributeError as e:
        print(f"\n❌ AttributeError: {e}")
        import traceback

        print(f"Full traceback:\n{traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"\n❌ Other error: {e}")
        import traceback

        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

    print("\n" + "=" * 60)
    print("Test Complete - SUCCESS")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_trajectory_extraction()
    if not success:
        exit(1)
