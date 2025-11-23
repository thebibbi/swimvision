#!/usr/bin/env python3
"""Audit all pose estimators to check their return formats."""

import logging

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def audit_estimator_formats():
    """Check what format each estimator returns."""
    print("=" * 60)
    print("Auditing Pose Estimator Return Formats")
    print("=" * 60)

    # Test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    estimators_to_check = []

    # Check YOLO (already fixed)
    try:
        from src.pose import YOLOPoseEstimator

        estimators_to_check.append(("YOLO", YOLOPoseEstimator()))
    except Exception as e:
        print(f"❌ Could not import YOLO: {e}")

    # Check MediaPipe
    try:
        from src.pose import MediaPipeEstimator

        estimators_to_check.append(("MediaPipe", MediaPipeEstimator()))
    except Exception as e:
        print(f"❌ Could not import MediaPipe: {e}")

    # Check RTMPose
    try:
        from src.pose import RTMPoseEstimator

        estimators_to_check.append(("RTMPose", RTMPoseEstimator()))
    except Exception as e:
        print(f"❌ Could not import RTMPose: {e}")

    # Check ViTPose
    try:
        from src.pose import ViTPoseEstimator

        estimators_to_check.append(("ViTPose", ViTPoseEstimator()))
    except Exception as e:
        print(f"❌ Could not import ViTPose: {e}")

    # Test each estimator
    results = {}

    for name, estimator in estimators_to_check:
        print(f"\n🔍 Testing {name}...")
        try:
            pose_result, _ = estimator.estimate_pose(test_image, return_image=False)

            if pose_result is None:
                print(f"⚠️  {name}: No pose detected (expected for random image)")
                results[name] = {"status": "no_pose", "format": None}
                continue

            # Normalize to list format
            if isinstance(pose_result, list):
                pose_data_list = pose_result
            else:
                pose_data_list = [pose_result]

            if len(pose_data_list) == 0:
                print(f"⚠️  {name}: Empty pose list")
                results[name] = {"status": "empty_list", "format": None}
                continue

            # Check first detection
            pose_data = pose_data_list[0]

            if not isinstance(pose_data, dict):
                print(f"❌ {name}: pose_data is not a dict! It's {type(pose_data)}")
                results[name] = {"status": "wrong_type", "format": type(pose_data)}
                continue

            # Check keypoints format
            if "keypoints" not in pose_data:
                print(f"❌ {name}: No 'keypoints' key in pose_data")
                results[name] = {"status": "missing_keypoints", "format": None}
                continue

            keypoints = pose_data["keypoints"]

            if isinstance(keypoints, dict):
                print(f"✅ {name}: Keypoints are in dict format (correct)")
                if keypoints:
                    sample_key = list(keypoints.keys())[0]
                    sample_val = keypoints[sample_key]
                    print(f"   Sample: {sample_key} -> {sample_val}")
                results[name] = {"status": "correct", "format": "dict"}
            elif isinstance(keypoints, np.ndarray):
                print(f"❌ {name}: Keypoints are numpy array (needs fix)")
                print(f"   Shape: {keypoints.shape}")
                results[name] = {"status": "needs_fix", "format": "numpy_array"}
            else:
                print(f"❌ {name}: Unknown keypoints format: {type(keypoints)}")
                results[name] = {"status": "unknown", "format": type(keypoints)}

        except Exception as e:
            print(f"❌ {name}: Error during inference: {e}")
            results[name] = {"status": "error", "format": None}

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in results.items():
        status = result["status"]
        format_type = result["format"]

        if status == "correct":
            print(f"✅ {name}: {format_type} format (OK)")
        elif status == "needs_fix":
            print(f"🔧 {name}: {format_type} (NEEDS FIX)")
        elif status == "no_pose":
            print(f"⚠️  {name}: No pose detected (can't test format)")
        else:
            print(f"❌ {name}: {status}")

    return results


if __name__ == "__main__":
    audit_estimator_formats()
