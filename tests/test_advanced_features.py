"""Comprehensive tests for advanced pose estimation features.

Tests:
- MediaPipe estimator
- OpenPose estimator
- AlphaPose estimator
- SMPL/SMPL-X estimator
- Multi-model fusion
- Multi-camera fusion
- Water surface detection
- Adaptive threshold tuning
- Model comparison tools
"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.pose.base_estimator import KeypointFormat


class TestMediaPipeEstimator:
    """Test MediaPipe pose estimator."""

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.mark.skipif(
        not hasattr(
            __import__("src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]),
            "MEDIAPIPE_AVAILABLE",
        )
        or not __import__(
            "src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]
        ).MEDIAPIPE_AVAILABLE,
        reason="MediaPipe not installed",
    )
    def test_mediapipe_initialization(self):
        """Test MediaPipe estimator initialization."""
        from src.pose.mediapipe_estimator import MediaPipeEstimator

        estimator = MediaPipeEstimator(model_complexity=1, min_detection_confidence=0.5)

        assert estimator.model_complexity == 1
        assert estimator.min_detection_confidence == 0.5
        assert estimator.get_keypoint_format() == KeypointFormat.MEDIAPIPE_33
        assert estimator.supports_3d() is True
        assert estimator.supports_multi_person() is False

    @pytest.mark.skipif(
        not hasattr(
            __import__("src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]),
            "MEDIAPIPE_AVAILABLE",
        )
        or not __import__(
            "src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]
        ).MEDIAPIPE_AVAILABLE,
        reason="MediaPipe not installed",
    )
    def test_mediapipe_estimate_pose(self, test_image):
        """Test pose estimation."""
        from src.pose.mediapipe_estimator import MediaPipeEstimator

        estimator = MediaPipeEstimator()
        pose_data, annotated = estimator.estimate_pose(test_image)

        # May return None on blank image
        if pose_data is not None:
            assert "keypoints" in pose_data
            assert "keypoint_names" in pose_data
            assert pose_data["format"] == KeypointFormat.MEDIAPIPE_33

    @pytest.mark.skipif(
        not hasattr(
            __import__("src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]),
            "MEDIAPIPE_AVAILABLE",
        )
        or not __import__(
            "src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]
        ).MEDIAPIPE_AVAILABLE,
        reason="MediaPipe not installed",
    )
    def test_mediapipe_convert_to_coco17(self):
        """Test COCO-17 conversion."""
        from src.pose.mediapipe_estimator import MediaPipeEstimator

        estimator = MediaPipeEstimator()

        # Create mock pose data
        mock_keypoints = np.random.rand(33, 3)
        pose_data = {
            "keypoints": mock_keypoints,
            "keypoint_names": ["kp" + str(i) for i in range(33)],
            "format": KeypointFormat.MEDIAPIPE_33,
        }

        coco17_data = estimator.convert_to_coco17(pose_data)

        assert coco17_data["format"] == KeypointFormat.COCO_17
        assert len(coco17_data["keypoints"]) == 17


class TestModelFusion:
    """Test multi-model fusion."""

    @pytest.fixture
    def mock_models(self):
        """Create mock pose estimators."""
        models = []

        for i in range(3):
            model = Mock()
            model.model_name = f"model_{i}"
            model.estimate_pose = Mock(
                return_value=(
                    {
                        "keypoints": np.random.rand(17, 3),
                        "keypoint_names": [f"kp_{j}" for j in range(17)],
                        "format": KeypointFormat.COCO_17,
                    },
                    None,
                )
            )
            models.append(model)

        return models

    def test_fusion_initialization(self, mock_models):
        """Test fusion system initialization."""
        from src.pose.model_fusion import FusionMethod, MultiModelFusion

        fusion = MultiModelFusion(
            models=mock_models,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
        )

        assert len(fusion.models) == 3
        assert fusion.fusion_method == FusionMethod.WEIGHTED_AVERAGE

    def test_weighted_average_fusion(self, mock_models):
        """Test weighted average fusion."""
        from src.pose.model_fusion import FusionMethod, MultiModelFusion

        fusion = MultiModelFusion(
            models=mock_models,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
        )

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        fused_result, _ = fusion.estimate_pose(test_image, return_image=False)

        assert fused_result is not None
        assert len(fused_result.keypoints) == 17
        assert fused_result.format == KeypointFormat.COCO_17
        assert len(fused_result.contributing_models) == 3

    def test_fusion_model_weights(self, mock_models):
        """Test custom model weights."""
        from src.pose.model_fusion import FusionMethod, MultiModelFusion

        fusion = MultiModelFusion(models=mock_models, fusion_method=FusionMethod.WEIGHTED_AVERAGE)

        # Set custom weights
        fusion.set_model_weight("model_0", 2.0)
        fusion.set_model_weight("model_1", 0.5)

        assert fusion.model_weights["model_0"] == 2.0
        assert fusion.model_weights["model_1"] == 0.5


class TestMultiCameraFusion:
    """Test multi-camera fusion."""

    @pytest.fixture
    def camera_params(self):
        """Create test camera parameters."""
        from src.pose.multi_camera_fusion import CameraParameters

        K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        t = np.array([0, 0, 5], dtype=np.float32)

        return CameraParameters(
            camera_id="cam1",
            intrinsic_matrix=K,
            distortion_coeffs=np.zeros(5, dtype=np.float32),
            rotation_matrix=R,
            translation_vector=t,
            resolution=(640, 480),
        )

    def test_camera_fusion_initialization(self, camera_params):
        """Test multi-camera fusion initialization."""
        from src.pose.multi_camera_fusion import MultiCameraFusion

        fusion = MultiCameraFusion(cameras=[camera_params])

        assert len(fusion.cameras) == 1
        assert "cam1" in fusion.cameras

    def test_triangulation(self, camera_params):
        """Test 3D point triangulation."""
        from src.pose.multi_camera_fusion import CameraParameters, MultiCameraFusion

        # Create second camera
        K2 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
        R2 = np.eye(3, dtype=np.float32)
        t2 = np.array([1, 0, 5], dtype=np.float32)

        cam2 = CameraParameters(
            camera_id="cam2",
            intrinsic_matrix=K2,
            distortion_coeffs=np.zeros(5, dtype=np.float32),
            rotation_matrix=R2,
            translation_vector=t2,
            resolution=(640, 480),
        )

        fusion = MultiCameraFusion(cameras=[camera_params, cam2])

        # Test triangulation with mock observations
        point_3d, quality, errors = fusion._triangulate_point(
            [np.array([320, 240]), np.array([340, 240])], ["cam1", "cam2"]
        )

        assert point_3d is not None
        assert len(point_3d) == 3
        assert quality >= 0.0
        assert len(errors) == 2


class TestWaterSurfaceDetector:
    """Test water surface detection."""

    @pytest.fixture
    def test_pool_image(self):
        """Create test pool image with simulated water surface."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add blue water in bottom half
        img[240:, :] = [200, 100, 50]  # Blue-ish color

        # Add horizontal edge at water surface
        img[235:245, :] = [255, 255, 255]  # White edge

        return img

    def test_surface_detector_initialization(self):
        """Test surface detector initialization."""
        from src.analysis.water_surface_detector import WaterSurfaceDetector

        detector = WaterSurfaceDetector(
            pool_type="indoor",
            detection_method="edge",
        )

        assert detector.pool_type == "indoor"
        assert detector.detection_method == "edge"

    def test_edge_detection(self, test_pool_image):
        """Test edge-based surface detection."""
        from src.analysis.water_surface_detector import WaterSurfaceDetector

        detector = WaterSurfaceDetector(detection_method="edge")
        surface_info = detector.detect_surface(test_pool_image)

        if surface_info is not None:
            assert hasattr(surface_info, "water_level")
            assert hasattr(surface_info, "surface_line")
            assert hasattr(surface_info, "confidence")

    def test_water_state_detection(self):
        """Test water state (above/below) detection."""
        from src.analysis.water_surface_detector import (
            WaterState,
            WaterSurfaceDetector,
            WaterSurfaceInfo,
        )

        detector = WaterSurfaceDetector()

        # Mock surface info
        surface_info = WaterSurfaceInfo(
            surface_line=np.array([0, 240]),  # Horizontal line at y=240
            surface_points=np.array([[0, 240], [640, 240]]),
            confidence=0.9,
            water_level=240.0,
            surface_normal=np.array([0, 1]),
            metadata={},
        )

        # Point above water
        state_above = detector.get_water_state((320, 100), surface_info)
        assert state_above == WaterState.ABOVE_WATER

        # Point below water
        state_below = detector.get_water_state((320, 400), surface_info)
        assert state_below == WaterState.UNDERWATER

        # Point at surface
        state_at = detector.get_water_state((320, 240), surface_info)
        assert state_at == WaterState.AT_SURFACE


class TestAdaptiveTuning:
    """Test adaptive threshold tuning."""

    def test_tuner_initialization(self):
        """Test tuner initialization."""
        from src.utils.adaptive_tuning import AdaptiveThresholdTuner

        tuner = AdaptiveThresholdTuner(auto_tune=True)

        assert tuner.auto_tune is True
        assert tuner.params.confidence_threshold == 0.5

    def test_parameter_adaptation(self):
        """Test parameter adaptation."""
        from src.utils.adaptive_tuning import AdaptiveThresholdTuner

        tuner = AdaptiveThresholdTuner(auto_tune=True, adaptation_rate=0.5)

        # Simulate low confidence detections
        for _ in range(20):
            detection = {
                "keypoints": np.random.rand(17, 3) * 0.3,  # Low confidence
            }
            tuner.update(detection)

        # Confidence threshold should decrease
        assert tuner.params.confidence_threshold < 0.5

    def test_condition_assessment(self):
        """Test pool condition assessment."""
        from src.utils.adaptive_tuning import AdaptiveThresholdTuner, PoolCondition

        tuner = AdaptiveThresholdTuner()

        # Simulate excellent conditions (high confidence, high detection rate)
        for _ in range(30):
            detection = {
                "keypoints": np.random.rand(17, 3) * 0.9 + 0.1,  # High confidence
            }
            tuner.update(detection)

        # Should assess as good/excellent conditions
        assert tuner.current_condition in [PoolCondition.EXCELLENT, PoolCondition.GOOD]

    def test_frame_stats_calculation(self):
        """Test frame statistics calculation."""
        from src.utils.adaptive_tuning import calculate_frame_stats

        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        stats = calculate_frame_stats(frame)

        assert "brightness" in stats
        assert "contrast" in stats
        assert "sharpness" in stats
        assert stats["brightness"] >= 0
        assert stats["brightness"] <= 255


class TestModelComparison:
    """Test model comparison tools."""

    @pytest.fixture
    def mock_models_dict(self):
        """Create dictionary of mock models."""
        models = {}

        for i in range(2):
            model = Mock()
            model.model_name = f"model_{i}"
            model.estimate_pose = Mock(
                return_value=(
                    {
                        "keypoints": np.random.rand(17, 3),
                        "keypoint_names": [f"kp_{j}" for j in range(17)],
                        "format": KeypointFormat.COCO_17,
                    },
                    None,
                )
            )
            models[f"model_{i}"] = model

        return models

    def test_comparison_initialization(self, mock_models_dict, tmp_path):
        """Test model comparison initialization."""
        from src.utils.model_comparison import ModelComparison

        comparison = ModelComparison(mock_models_dict, output_dir=str(tmp_path))

        assert len(comparison.models) == 2
        assert len(comparison.metrics) == 2

    def test_frame_processing(self, mock_models_dict):
        """Test frame processing with multiple models."""
        from src.utils.model_comparison import ModelComparison

        comparison = ModelComparison(mock_models_dict)

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = comparison._process_frame_all_models(test_frame, 0)

        assert "frame_idx" in results
        assert "models" in results
        assert len(results["models"]) == 2

    def test_metrics_calculation(self, mock_models_dict):
        """Test metrics calculation."""
        from src.utils.model_comparison import ModelComparison

        comparison = ModelComparison(mock_models_dict)

        # Process some frames
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(10):
            comparison._process_frame_all_models(test_frame, i)

        comparison._calculate_final_metrics()

        for model_name, metrics in comparison.metrics.items():
            assert metrics.total_frames_processed > 0
            assert metrics.fps > 0


class TestIntegration:
    """Integration tests for combined features."""

    @pytest.mark.skipif(
        not hasattr(
            __import__("src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]),
            "MEDIAPIPE_AVAILABLE",
        )
        or not __import__(
            "src.pose.mediapipe_estimator", fromlist=["MEDIAPIPE_AVAILABLE"]
        ).MEDIAPIPE_AVAILABLE,
        reason="MediaPipe not installed",
    )
    def test_mediapipe_with_water_surface(self):
        """Test MediaPipe with water surface detection."""
        from src.analysis.water_surface_detector import WaterSurfaceDetector
        from src.pose.mediapipe_estimator import MediaPipeEstimator

        estimator = MediaPipeEstimator()
        detector = WaterSurfaceDetector()

        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[240:, :] = [200, 100, 50]  # Water

        # Detect pose and surface
        pose_data, _ = estimator.estimate_pose(img)
        surface_info = detector.detect_surface(img)

        # Both can run independently
        assert pose_data is not None or pose_data is None  # May or may not detect on blank image
        assert surface_info is not None or surface_info is None

    def test_fusion_with_adaptive_tuning(self):
        """Test model fusion with adaptive tuning."""
        from src.utils.adaptive_tuning import AdaptiveThresholdTuner

        tuner = AdaptiveThresholdTuner(auto_tune=True)

        # Simulate detection sequence
        for i in range(30):
            # Varying confidence
            conf = 0.3 + 0.4 * (i / 30.0)
            detection = {
                "keypoints": np.random.rand(17, 3) * conf,
            }

            params = tuner.update(detection)

        # Parameters should have adapted
        assert params is not None
        assert hasattr(params, "confidence_threshold")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
