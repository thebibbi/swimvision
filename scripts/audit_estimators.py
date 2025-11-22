#!/usr/bin/env python3
"""
Audit all pose estimators for missing abstract methods.

This script checks that all estimators properly implement the BasePoseEstimator interface.
"""

import importlib
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose.base_estimator import BasePoseEstimator

# Required abstract methods
REQUIRED_METHODS = [
    "load_model",
    "estimate_pose",
    "get_keypoint_format",
    "supports_3d",
    "supports_multi_person",
]

# Estimator modules to check
ESTIMATORS = [
    ("src.pose.yolo_estimator", "YOLOPoseEstimator"),
    ("src.pose.mediapipe_estimator", "MediaPipeEstimator"),
    ("src.pose.rtmpose_estimator", "RTMPoseEstimator"),
    ("src.pose.vitpose_estimator", "ViTPoseEstimator"),
    ("src.pose.alphapose_estimator", "AlphaPoseEstimator"),
    ("src.pose.openpose_estimator", "OpenPoseEstimator"),
    ("src.pose.smpl_estimator", "SMPLEstimator"),
]


def check_estimator(module_name: str, class_name: str) -> dict:
    """Check if an estimator implements all required methods."""
    result = {
        "name": class_name,
        "module": module_name,
        "missing_methods": [],
        "can_instantiate": False,
        "error": None,
    }

    try:
        # Import module
        module = importlib.import_module(module_name)
        estimator_class = getattr(module, class_name)

        # Check if it's a subclass of BasePoseEstimator
        if not issubclass(estimator_class, BasePoseEstimator):
            result["error"] = f"{class_name} does not inherit from BasePoseEstimator"
            return result

        # Check for missing methods
        for method_name in REQUIRED_METHODS:
            if not hasattr(estimator_class, method_name):
                result["missing_methods"].append(method_name)
            else:
                method = getattr(estimator_class, method_name)
                # Check if it's still abstract
                if hasattr(method, "__isabstractmethod__") and method.__isabstractmethod__:
                    result["missing_methods"].append(f"{method_name} (still abstract)")

        # Try to check if class can be instantiated (without actually instantiating)
        if not result["missing_methods"]:
            result["can_instantiate"] = True

    except ImportError as e:
        result["error"] = f"Import error: {e}"
    except AttributeError as e:
        result["error"] = f"Attribute error: {e}"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"

    return result


def main():
    """Run audit on all estimators."""
    print("=" * 80)
    print("POSE ESTIMATOR AUDIT")
    print("=" * 80)
    print()

    all_pass = True
    results = []

    for module_name, class_name in ESTIMATORS:
        result = check_estimator(module_name, class_name)
        results.append(result)

        status = "✅ PASS" if result["can_instantiate"] and not result["error"] else "❌ FAIL"
        print(f"{status} {result['name']}")

        if result["error"]:
            print(f"   Error: {result['error']}")
            all_pass = False

        if result["missing_methods"]:
            print(f"   Missing methods: {', '.join(result['missing_methods'])}")
            all_pass = False

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["can_instantiate"] and not r["error"])
    failed = len(results) - passed

    print(f"Total estimators: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if all_pass:
        print("✅ All estimators implement the required interface!")
        return 0
    else:
        print("❌ Some estimators have issues. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
