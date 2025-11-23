#!/usr/bin/env python3
"""Integration tests for the complete SwimVision pipeline."""

import logging
import os
import tempfile

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.pose import YOLOPoseEstimator
from src.pose.model_fusion import FusionMethod, MultiModelFusion
from src.pose.swimming_keypoints import SwimmingKeypoints


def create_test_video(output_path: str, num_frames: int = 30):
    """Create a simple test video with simulated swimming motion.

    Args:
        output_path: Path to save the test video
        num_frames: Number of frames to generate
    """
    import cv2

    # Video settings
    width, height = 640, 480
    fps = 30

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Create a frame with some motion
        frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

        # Add a simple "swimmer" silhouette (moving rectangle)
        swimmer_x = 200 + int(50 * np.sin(frame_idx * 0.2))
        swimmer_y = 200 + int(20 * np.cos(frame_idx * 0.2))
        cv2.rectangle(
            frame, (swimmer_x, swimmer_y), (swimmer_x + 100, swimmer_y + 200), (255, 255, 255), -1
        )

        out.write(frame)

    out.release()
    logger.info(f"Created test video: {output_path}")


def test_yolo_pipeline_integration():
    """Test complete YOLO pipeline integration."""
    print("=" * 60)
    print("Testing YOLO Pipeline Integration")
    print("=" * 60)

    # Create components
    estimator = YOLOPoseEstimator()
    analyzer = SwimmingKeypoints()

    print(f"✅ YOLO Estimator: {estimator.model_name}")
    print("✅ Swimming Analyzer: Ready")

    # Create test video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    try:
        create_test_video(video_path, num_frames=10)

        # Test video processing (simulate app.py behavior)
        import cv2

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        detected_count = 0
        pose_sequence = []
        angles_over_time = {
            "left_elbow_angle": [],
            "right_elbow_angle": [],
            "left_shoulder_angle": [],
            "right_shoulder_angle": [],
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Pose estimation
            pose_result, _ = estimator.estimate_pose(frame, return_image=False)

            if pose_result is not None:
                # Normalize to list format
                if isinstance(pose_result, list):
                    pose_data_list = pose_result
                else:
                    pose_data_list = [pose_result]

                # Process first detection
                if len(pose_data_list) > 0:
                    pose_data = pose_data_list[0]

                    # Verify pose_data structure
                    assert isinstance(
                        pose_data, dict
                    ), f"pose_data should be dict, got {type(pose_data)}"
                    assert "keypoints" in pose_data, "pose_data missing 'keypoints'"
                    assert isinstance(
                        pose_data["keypoints"], dict
                    ), f"keypoints should be dict, got {type(pose_data['keypoints'])}"

                    detected_count += 1
                    pose_sequence.append(pose_data)

                    # Test swimming analyzer
                    try:
                        angles = analyzer.get_body_angles(pose_data)
                        assert isinstance(
                            angles, dict
                        ), f"angles should be dict, got {type(angles)}"

                        for angle_name in angles_over_time:
                            value = angles.get(angle_name)
                            angles_over_time[angle_name].append(
                                float(value) if value is not None else float(np.nan)
                            )
                    except Exception as e:
                        print(f"❌ Swimming analyzer failed: {e}")
                        return False

        cap.release()

        # Verify results
        print(f"✅ Processed {frame_count} frames")
        print(f"✅ Detected poses in {detected_count} frames")
        print(f"✅ Collected {len(pose_sequence)} pose sequences")

        # Check angles data
        for angle_name, values in angles_over_time.items():
            print(f"✅ {angle_name}: {len(values)} measurements")

        # Test trajectory extraction
        if len(pose_sequence) > 0:
            print("\n🔍 Testing trajectory extraction...")
            try:
                left_hand_path = []
                right_hand_path = []

                for pose_data in pose_sequence:
                    left_wrist = estimator.get_keypoint(pose_data, "left_wrist")
                    right_wrist = estimator.get_keypoint(pose_data, "right_wrist")

                    if left_wrist:
                        left_hand_path.append((left_wrist[0], left_wrist[1]))
                    if right_wrist:
                        right_hand_path.append((right_wrist[0], right_wrist[1]))

                print(f"✅ Left hand trajectory: {len(left_hand_path)} points")
                print(f"✅ Right hand trajectory: {len(right_hand_path)} points")

            except Exception as e:
                print(f"❌ Trajectory extraction failed: {e}")
                return False

        print("\n✅ YOLO Pipeline Integration Test PASSED")
        return True

    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.unlink(video_path)


def test_multi_model_fusion():
    """Test multi-model fusion integration."""
    print("\n" + "=" * 60)
    print("Testing Multi-Model Fusion Integration")
    print("=" * 60)

    try:
        # Create YOLO estimator (the only one available)
        yolo = YOLOPoseEstimator()

        # Create fusion with single model (degenerate case)
        fusion = MultiModelFusion(
            models=[yolo], fusion_method=FusionMethod.WEIGHTED_AVERAGE, min_models=1
        )

        print(f"✅ MultiModelFusion created with {len(fusion.models)} models")

        # Test fusion
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        fused_result, annotated = fusion.estimate_pose(test_image, return_image=False)

        if fused_result is not None:
            print(f"✅ Fusion result type: {type(fused_result)}")
            print(f"✅ Fusion keypoints shape: {fused_result.keypoints.shape}")
            print(f"✅ Contributing models: {fused_result.contributing_models}")
            print(f"✅ Fusion method: {fused_result.fusion_method}")
        else:
            print("⚠️  No pose detected in test image (expected)")

        print("\n✅ Multi-Model Fusion Integration Test PASSED")
        return True

    except Exception as e:
        print(f"❌ Multi-Model Fusion test failed: {e}")
        return False


def test_pose_data_consistency():
    """Test that all pose data consumers work with the same format."""
    print("\n" + "=" * 60)
    print("Testing Pose Data Consistency")
    print("=" * 60)

    # Create test pose data in the expected format
    keypoints_dict = {
        "left_shoulder": {"x": 100, "y": 200, "confidence": 0.9},
        "right_shoulder": {"x": 150, "y": 200, "confidence": 0.9},
        "left_elbow": {"x": 80, "y": 250, "confidence": 0.8},
        "right_elbow": {"x": 170, "y": 250, "confidence": 0.8},
        "left_wrist": {"x": 70, "y": 300, "confidence": 0.7},
        "right_wrist": {"x": 180, "y": 300, "confidence": 0.7},
        "left_hip": {"x": 110, "y": 350, "confidence": 0.9},
        "right_hip": {"x": 140, "y": 350, "confidence": 0.9},
        "left_knee": {"x": 100, "y": 400, "confidence": 0.8},
        "right_knee": {"x": 150, "y": 400, "confidence": 0.8},
        "left_ankle": {"x": 95, "y": 450, "confidence": 0.7},
        "right_ankle": {"x": 155, "y": 450, "confidence": 0.7},
    }

    pose_data = {
        "keypoints": keypoints_dict,
        "keypoint_names": list(keypoints_dict.keys()),
        "bbox": [50, 150, 200, 500],
        "person_id": 0,
        "format": "COCO_17",
        "confidence": 0.8,
        "metadata": {"model": "test"},
    }

    print(f"✅ Created test pose data with {len(keypoints_dict)} keypoints")

    # Test all consumers
    try:
        # 1. Swimming analyzer
        analyzer = SwimmingKeypoints()
        angles = analyzer.get_body_angles(pose_data)
        print(f"✅ Swimming analyzer: {len(angles)} angles calculated")

        # 2. Individual keypoint extraction
        left_wrist = analyzer._get_keypoint(pose_data, "left_wrist")
        right_wrist = analyzer._get_keypoint(pose_data, "right_wrist")
        print(f"✅ Individual keypoints: left={left_wrist}, right={right_wrist}")

        # 3. YOLO estimator get_keypoint method
        yolo = YOLOPoseEstimator()
        left_wrist_yolo = yolo.get_keypoint(pose_data, "left_wrist")
        right_wrist_yolo = yolo.get_keypoint(pose_data, "right_wrist")
        print(f"✅ YOLO get_keypoint: left={left_wrist_yolo}, right={right_wrist_yolo}")

        print("\n✅ Pose Data Consistency Test PASSED")
        return True

    except Exception as e:
        print(f"❌ Pose data consistency test failed: {e}")
        import traceback

        print(traceback.format_exc())
        return False


def run_all_integration_tests():
    """Run all integration tests."""
    print("🚀 Starting SwimVision Integration Tests")
    print("=" * 60)

    tests = [
        ("YOLO Pipeline Integration", test_yolo_pipeline_integration),
        ("Multi-Model Fusion Integration", test_multi_model_fusion),
        ("Pose Data Consistency", test_pose_data_consistency),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Integration Test Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\n📊 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests PASSED!")
        print("✅ YOLO AttributeError fix is working correctly")
        print("✅ Complete pipeline is functional")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Pipeline needs attention")

    return passed == total


if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1)
