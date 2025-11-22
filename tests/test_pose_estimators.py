"""Unit tests for pose estimators."""

from pathlib import Path

import numpy as np
import pytest

from src.pose import (
    KeypointFormat,
    MediaPipeEstimator,
    RTMPoseEstimator,
    ViTPoseEstimator,
    YOLOPoseEstimator,
)


@pytest.fixture
def test_image():
    """Create a test image (dummy data)."""
    # Create a simple 640x480 RGB image
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_image_with_person():
    """Load a real test image with a person (if available)."""
    test_image_path = Path("tests/data/test_swimmer.jpg")
    if test_image_path.exists():
        import cv2

        return cv2.imread(str(test_image_path))
    else:
        # Fallback to dummy image
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestYOLOPoseEstimator:
    """Test YOLO11 pose estimator."""

    def test_instantiation(self):
        """Test that YOLO estimator can be instantiated."""
        estimator = YOLOPoseEstimator()
        assert estimator is not None
        assert estimator.model is not None
        assert estimator.model_name == "yolo11n-pose.pt"

    def test_abstract_methods(self):
        """Test that all abstract methods are implemented."""
        estimator = YOLOPoseEstimator()
        assert hasattr(estimator, "estimate_pose")
        assert hasattr(estimator, "load_model")
        assert hasattr(estimator, "get_keypoint_format")
        assert hasattr(estimator, "supports_3d")
        assert hasattr(estimator, "supports_multi_person")

    def test_supports_multi_person(self):
        """Test that YOLO supports multi-person detection."""
        estimator = YOLOPoseEstimator()
        assert estimator.supports_multi_person() is True

    def test_supports_3d(self):
        """Test that YOLO does not support 3D."""
        estimator = YOLOPoseEstimator()
        assert estimator.supports_3d() is False

    def test_keypoint_format(self):
        """Test that YOLO returns COCO-17 format."""
        estimator = YOLOPoseEstimator()
        assert estimator.get_keypoint_format() == KeypointFormat.COCO_17

    def test_estimate_pose_returns_list(self, test_image):
        """Test that estimate_pose returns list[dict] or None."""
        estimator = YOLOPoseEstimator()
        pose_data, annotated = estimator.estimate_pose(test_image, return_image=False)

        # Should return either None or list[dict]
        assert pose_data is None or isinstance(pose_data, list)

        if pose_data is not None:
            assert len(pose_data) > 0
            # Check first detection has required fields
            first_detection = pose_data[0]
            assert "keypoints" in first_detection
            assert "keypoint_names" in first_detection
            assert "confidence" in first_detection
            assert "format" in first_detection
            assert isinstance(first_detection["keypoints"], np.ndarray)
            assert first_detection["keypoints"].shape == (17, 3)

    def test_device_selection(self):
        """Test device selection."""
        estimator_cpu = YOLOPoseEstimator(device="cpu")
        assert "cpu" in estimator_cpu.device.lower()

        # Test auto device selection
        estimator_auto = YOLOPoseEstimator(device="auto")
        assert estimator_auto.device in ["cpu", "cuda", "cuda:0", "mps"]


class TestMediaPipeEstimator:
    """Test MediaPipe pose estimator."""

    def test_instantiation(self):
        """Test that MediaPipe estimator can be instantiated."""
        if MediaPipeEstimator is None:
            pytest.skip("MediaPipe not installed")

        estimator = MediaPipeEstimator()
        assert estimator is not None

    def test_supports_multi_person(self):
        """Test that MediaPipe is single-person only."""
        if MediaPipeEstimator is None:
            pytest.skip("MediaPipe not installed")

        estimator = MediaPipeEstimator()
        assert estimator.supports_multi_person() is False

    def test_supports_3d(self):
        """Test that MediaPipe supports 3D landmarks."""
        if MediaPipeEstimator is None:
            pytest.skip("MediaPipe not installed")

        estimator = MediaPipeEstimator()
        assert estimator.supports_3d() is True

    def test_keypoint_format(self):
        """Test that MediaPipe returns MediaPipe-33 format."""
        if MediaPipeEstimator is None:
            pytest.skip("MediaPipe not installed")

        estimator = MediaPipeEstimator()
        assert estimator.get_keypoint_format() == KeypointFormat.MEDIAPIPE_33


class TestRTMPoseEstimator:
    """Test RTMPose estimator."""

    def test_instantiation(self):
        """Test that RTMPose estimator can be instantiated."""
        if RTMPoseEstimator is None:
            pytest.skip("RTMPose (MMPose) not installed")

        estimator = RTMPoseEstimator(model_variant="rtmpose-m")
        assert estimator is not None

    def test_supports_multi_person(self):
        """Test that RTMPose supports multi-person detection."""
        if RTMPoseEstimator is None:
            pytest.skip("RTMPose (MMPose) not installed")

        estimator = RTMPoseEstimator()
        assert estimator.supports_multi_person() is True

    def test_supports_3d(self):
        """Test that RTMPose is 2D only."""
        if RTMPoseEstimator is None:
            pytest.skip("RTMPose (MMPose) not installed")

        estimator = RTMPoseEstimator()
        assert estimator.supports_3d() is False

    def test_keypoint_format(self):
        """Test that RTMPose returns COCO-17 format."""
        if RTMPoseEstimator is None:
            pytest.skip("RTMPose (MMPose) not installed")

        estimator = RTMPoseEstimator()
        assert estimator.get_keypoint_format() == KeypointFormat.COCO_17

    def test_estimate_pose_returns_list(self, test_image):
        """Test that estimate_pose returns list[dict] or None."""
        if RTMPoseEstimator is None:
            pytest.skip("RTMPose (MMPose) not installed")

        estimator = RTMPoseEstimator()
        pose_data, annotated = estimator.estimate_pose(test_image, return_image=False)

        # Should return either None or list[dict]
        assert pose_data is None or isinstance(pose_data, list)


class TestViTPoseEstimator:
    """Test ViTPose estimator."""

    def test_instantiation(self):
        """Test that ViTPose estimator can be instantiated."""
        if ViTPoseEstimator is None:
            pytest.skip("ViTPose (MMPose) not installed")

        estimator = ViTPoseEstimator(model_variant="vitpose-b")
        assert estimator is not None

    def test_supports_multi_person(self):
        """Test that ViTPose supports multi-person detection."""
        if ViTPoseEstimator is None:
            pytest.skip("ViTPose (MMPose) not installed")

        estimator = ViTPoseEstimator()
        assert estimator.supports_multi_person() is True

    def test_supports_3d(self):
        """Test that ViTPose is 2D only."""
        if ViTPoseEstimator is None:
            pytest.skip("ViTPose (MMPose) not installed")

        estimator = ViTPoseEstimator()
        assert estimator.supports_3d() is False

    def test_keypoint_format(self):
        """Test that ViTPose returns COCO-17 format."""
        if ViTPoseEstimator is None:
            pytest.skip("ViTPose (MMPose) not installed")

        estimator = ViTPoseEstimator()
        assert estimator.get_keypoint_format() == KeypointFormat.COCO_17


class TestEstimatorCompatibility:
    """Test compatibility between estimators."""

    def test_all_estimators_return_consistent_format(self, test_image):
        """Test that all estimators return data in consistent format."""
        estimators = []

        # Add available estimators
        estimators.append(YOLOPoseEstimator())

        if MediaPipeEstimator is not None:
            estimators.append(MediaPipeEstimator())

        if RTMPoseEstimator is not None:
            estimators.append(RTMPoseEstimator())

        if ViTPoseEstimator is not None:
            estimators.append(ViTPoseEstimator())

        for estimator in estimators:
            pose_data, _ = estimator.estimate_pose(test_image, return_image=False)

            if pose_data is not None:
                # Normalize to list
                if not isinstance(pose_data, list):
                    pose_data = [pose_data]

                # Check each detection has required fields
                for detection in pose_data:
                    assert "keypoints" in detection
                    assert "keypoint_names" in detection
                    assert "confidence" in detection
                    assert "format" in detection
                    assert isinstance(detection["keypoints"], np.ndarray)
                    assert detection["keypoints"].ndim == 2
                    assert detection["keypoints"].shape[1] == 3  # (x, y, conf)

    def test_model_info(self):
        """Test that get_model_info returns expected fields."""
        estimator = YOLOPoseEstimator()
        info = estimator.get_model_info()

        assert "name" in info
        assert "device" in info
        assert "confidence_threshold" in info
        assert "format" in info
        assert "supports_3d" in info
        assert "supports_multi_person" in info

        assert info["supports_multi_person"] is True
        assert info["supports_3d"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
