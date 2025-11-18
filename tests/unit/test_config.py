"""Unit tests for configuration management."""

import pytest

from src.utils.config import Config, get_config


class TestConfig:
    """Test configuration loading and access."""

    def test_load_pose_config(self):
        """Test loading pose configuration."""
        config = Config()
        pose_config = config.load("pose_config")

        assert pose_config is not None
        assert "yolo" in pose_config
        assert "coco_keypoints" in pose_config

    def test_load_camera_config(self):
        """Test loading camera configuration."""
        config = Config()
        camera_config = config.load("camera_config")

        assert camera_config is not None
        assert "webcam" in camera_config
        assert "video_file" in camera_config

    def test_load_analysis_config(self):
        """Test loading analysis configuration."""
        config = Config()
        analysis_config = config.load("analysis_config")

        assert analysis_config is not None
        assert "dtw" in analysis_config
        assert "stroke_phases" in analysis_config

    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        config = Config()

        # Test nested access
        model = config.get("pose_config", "yolo.model")
        assert model is not None
        assert isinstance(model, str)

        confidence = config.get("pose_config", "yolo.confidence")
        assert confidence is not None
        assert isinstance(confidence, (int, float))

    def test_get_with_default(self):
        """Test getting value with default."""
        config = Config()

        # Non-existent key should return default
        value = config.get("pose_config", "nonexistent.key", default="default_value")
        assert value == "default_value"

    def test_singleton_behavior(self):
        """Test that get_config() returns the same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_config_reload(self):
        """Test configuration reload."""
        config = Config()

        # Load once
        pose_config1 = config.load("pose_config")

        # Reload
        pose_config2 = config.reload("pose_config")

        # Should be equal (but potentially different object)
        assert pose_config1 == pose_config2


class TestPoseConfig:
    """Test specific pose configuration values."""

    def test_yolo_model_name(self):
        """Test YOLO model name is set."""
        config = Config()
        model = config.get("pose_config", "yolo.model")

        assert model.endswith("-pose.pt")

    def test_confidence_threshold_range(self):
        """Test confidence threshold is in valid range."""
        config = Config()
        confidence = config.get("pose_config", "yolo.confidence")

        assert 0.0 <= confidence <= 1.0

    def test_keypoint_names_complete(self):
        """Test all COCO keypoints are defined."""
        config = Config()
        pose_config = config.load("pose_config")

        keypoints = pose_config.get("coco_keypoints", {})

        # Should have 17 COCO keypoints
        expected_keypoints = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        for kpt in expected_keypoints:
            assert kpt in keypoints


class TestCameraConfig:
    """Test camera configuration values."""

    def test_webcam_default_resolution(self):
        """Test webcam has default resolution."""
        config = Config()

        width = config.get("camera_config", "webcam.width")
        height = config.get("camera_config", "webcam.height")

        assert width > 0
        assert height > 0

    def test_webcam_fps(self):
        """Test webcam FPS is reasonable."""
        config = Config()

        fps = config.get("camera_config", "webcam.fps")

        assert fps >= 15  # Minimum reasonable FPS
        assert fps <= 120  # Maximum reasonable FPS


class TestAnalysisConfig:
    """Test analysis configuration values."""

    def test_dtw_window_size(self):
        """Test DTW window size is positive."""
        config = Config()

        window_size = config.get("analysis_config", "dtw.window_size")

        assert window_size > 0

    def test_similarity_weights_sum_to_one(self):
        """Test similarity weights sum to approximately 1.0."""
        config = Config()
        analysis_config = config.load("analysis_config")

        weights = analysis_config.get("similarity_weights", {})
        weight_sum = sum(weights.values())

        # Should sum to 1.0 (within floating point tolerance)
        assert abs(weight_sum - 1.0) < 0.01
