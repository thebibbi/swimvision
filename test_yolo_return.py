#!/usr/bin/env python3
"""Test what YOLO estimator actually returns."""

import logging

import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from src.pose import YOLOPoseEstimator


def test_yolo_return():
    """Test YOLO estimator return format."""
    print("=" * 60)
    print("Testing YOLO Estimator Return Format")
    print("=" * 60)

    # Create estimator
    estimator = YOLOPoseEstimator()
    print(f"✅ Estimator created: {estimator.model_name}")
    print(f"✅ Device: {estimator.device}")
    print(f"✅ Supports multi-person: {estimator.supports_multi_person()}")

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"✅ Test image created: {test_image.shape}")

    # Run inference
    print("\n🔍 Running inference...")
    pose_result, annotated = estimator.estimate_pose(test_image, return_image=False)

    print(f"\n📊 Return type: {type(pose_result)}")

    if pose_result is None:
        print("❌ No pose detected")
        return

    # Check if it's a list (multi-person) or dict (single-person)
    if isinstance(pose_result, list):
        print(f"✅ Returns list with {len(pose_result)} detections")
        if len(pose_result) > 0:
            first_detection = pose_result[0]
            print(f"First detection type: {type(first_detection)}")
            if isinstance(first_detection, dict):
                print(f"✅ First detection keys: {list(first_detection.keys())}")
                if "keypoints" in first_detection:
                    kpts = first_detection["keypoints"]
                    print(f"✅ Keypoints shape: {kpts.shape}")
                    print(f"✅ Keypoints type: {type(kpts)}")
            else:
                print(f"❌ First detection is not a dict: {type(first_detection)}")
                if isinstance(first_detection, np.ndarray):
                    print(f"❌ It's a numpy array with shape: {first_detection.shape}")
    elif isinstance(pose_result, dict):
        print("✅ Returns single dict")
        print(f"✅ Dict keys: {list(pose_result.keys())}")
        if "keypoints" in pose_result:
            kpts = pose_result["keypoints"]
            print(f"✅ Keypoints shape: {kpts.shape}")
            print(f"✅ Keypoints type: {type(kpts)}")
    else:
        print(f"❌ Unexpected return type: {type(pose_result)}")
        if isinstance(pose_result, np.ndarray):
            print(f"❌ It's a numpy array with shape: {pose_result.shape}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_yolo_return()
