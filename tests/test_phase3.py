"""Tests for Phase 3: Feature Extraction and Symmetry Analysis."""

import numpy as np
import pytest

from src.analysis.features_extractor import FeaturesExtractor
from src.analysis.symmetry_analyzer import SymmetryAnalyzer
from src.utils.smoothing import (
    KalmanFilter1D,
    KalmanFilter2D,
    calculate_acceleration,
    calculate_speed,
    calculate_velocity,
    detect_outliers_iqr,
    detect_outliers_zscore,
    moving_average,
    smooth_trajectory_kalman,
    smooth_trajectory_ma,
    smooth_trajectory_savgol,
)


class TestKalmanFilter1D:
    """Test 1D Kalman filter."""

    def test_initialization(self):
        """Test filter initialization."""
        kf = KalmanFilter1D()
        assert kf.process_variance == 1e-5
        assert kf.measurement_variance == 1e-1
        assert kf.state == 0.0
        assert kf.covariance == 1.0

    def test_update(self):
        """Test filter update."""
        kf = KalmanFilter1D()
        measurements = [1.0, 2.0, 3.0, 4.0, 5.0]

        for measurement in measurements:
            state = kf.update(measurement)
            assert isinstance(state, float)

    def test_reset(self):
        """Test filter reset."""
        kf = KalmanFilter1D()
        kf.update(5.0)
        kf.reset(0.0)
        assert kf.state == 0.0


class TestKalmanFilter2D:
    """Test 2D Kalman filter."""

    def test_initialization(self):
        """Test filter initialization."""
        kf = KalmanFilter2D()
        assert kf.state.shape == (4,)
        assert kf.F.shape == (4, 4)
        assert kf.H.shape == (2, 4)

    def test_update(self):
        """Test filter update."""
        kf = KalmanFilter2D()
        measurements = np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ]
        )

        for measurement in measurements:
            position = kf.update(measurement)
            assert position.shape == (2,)

    def test_velocity_estimation(self):
        """Test velocity estimation."""
        kf = KalmanFilter2D(dt=1.0)
        measurements = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )

        for measurement in measurements:
            kf.update(measurement)

        velocity = kf.get_velocity()
        assert velocity.shape == (2,)
        # Velocity should be approximately [1, 0] after these measurements
        assert abs(velocity[0] - 1.0) < 0.5


class TestTrajectorySmoothing:
    """Test trajectory smoothing functions."""

    def test_smooth_trajectory_kalman(self):
        """Test Kalman trajectory smoothing."""
        trajectory = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )

        smoothed, velocities = smooth_trajectory_kalman(trajectory)

        assert smoothed.shape == trajectory.shape
        assert velocities.shape == trajectory.shape

    def test_smooth_trajectory_savgol(self):
        """Test Savitzky-Golay smoothing."""
        trajectory = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )

        smoothed = smooth_trajectory_savgol(trajectory, window_length=3, polyorder=2)

        assert smoothed.shape == trajectory.shape

    def test_smooth_trajectory_ma(self):
        """Test moving average smoothing."""
        trajectory = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )

        smoothed = smooth_trajectory_ma(trajectory, window_size=3)

        assert smoothed.shape == trajectory.shape


class TestKinematicCalculations:
    """Test kinematic calculations."""

    def test_calculate_velocity(self):
        """Test velocity calculation."""
        trajectory = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        )

        velocities = calculate_velocity(trajectory, dt=1.0, smooth=False)

        assert velocities.shape == (3, 2)
        # Velocity should be [1, 0] for all steps
        np.testing.assert_array_almost_equal(velocities[:, 0], np.ones(3), decimal=1)

    def test_calculate_acceleration(self):
        """Test acceleration calculation."""
        trajectory = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [6.0, 0.0],
            ]
        )

        accelerations = calculate_acceleration(trajectory, dt=1.0, smooth=False)

        assert accelerations.shape == (2, 2)
        # Acceleration should be positive in x-direction
        assert np.all(accelerations[:, 0] > 0)

    def test_calculate_speed(self):
        """Test speed calculation."""
        velocities = np.array(
            [
                [3.0, 4.0],
                [5.0, 12.0],
            ]
        )

        speeds = calculate_speed(velocities)

        assert speeds.shape == (2,)
        # Speed should be magnitude of velocity
        np.testing.assert_array_almost_equal(speeds, [5.0, 13.0], decimal=1)


class TestOutlierDetection:
    """Test outlier detection."""

    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection."""
        signal = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0])

        outliers = detect_outliers_zscore(signal, threshold=3.0)

        assert outliers.shape == signal.shape
        assert outliers[3] == True  # 100.0 should be detected as outlier

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        signal = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0])

        outliers = detect_outliers_iqr(signal, factor=1.5)

        assert outliers.shape == signal.shape
        assert outliers[3] == True  # 100.0 should be detected as outlier


class TestFeaturesExtractor:
    """Test biomechanical features extractor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample swimming data."""
        num_frames = 50
        t = np.linspace(0, 2 * np.pi, num_frames)

        left_hand_path = np.column_stack([100 + 50 * np.sin(t), 200 + 30 * np.cos(t)])

        right_hand_path = np.column_stack([300 + 50 * np.sin(t), 200 + 30 * np.cos(t)])

        angles_over_time = {
            "left_elbow": 90 + 40 * np.sin(t),
            "right_elbow": 90 + 40 * np.sin(t),
            "left_shoulder": 120 + 30 * np.cos(t),
            "right_shoulder": 120 + 30 * np.cos(t),
        }

        pose_sequence = [{"frame": i} for i in range(num_frames)]

        return {
            "pose_sequence": pose_sequence,
            "left_hand_path": left_hand_path,
            "right_hand_path": right_hand_path,
            "angles_over_time": angles_over_time,
        }

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = FeaturesExtractor(fps=30.0)
        assert extractor.fps == 30.0
        assert extractor.dt == 1.0 / 30.0

    def test_extract_stroke_features(self, sample_data):
        """Test stroke features extraction."""
        extractor = FeaturesExtractor(fps=30.0)

        features = extractor.extract_stroke_features(
            sample_data["pose_sequence"],
            sample_data["left_hand_path"],
            sample_data["right_hand_path"],
            sample_data["angles_over_time"],
        )

        assert isinstance(features, dict)
        assert len(features) > 0

        # Check for key features
        assert "stroke_rate" in features or "num_strokes" in features
        assert any("velocity" in k for k in features)
        assert any("elbow" in k for k in features)

    def test_extract_injury_risk_features(self, sample_data):
        """Test injury risk features extraction."""
        extractor = FeaturesExtractor(fps=30.0)

        features = extractor.extract_injury_risk_features(
            sample_data["pose_sequence"],
            sample_data["angles_over_time"],
            sample_data["left_hand_path"],
            sample_data["right_hand_path"],
        )

        assert isinstance(features, dict)

    def test_extract_all_features(self, sample_data):
        """Test extraction of all features."""
        extractor = FeaturesExtractor(fps=30.0)

        features = extractor.extract_all_features(
            sample_data["pose_sequence"],
            sample_data["left_hand_path"],
            sample_data["right_hand_path"],
            sample_data["angles_over_time"],
        )

        assert isinstance(features, dict)
        assert len(features) > 10  # Should have many features


class TestSymmetryAnalyzer:
    """Test symmetry analyzer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample swimming data."""
        num_frames = 50
        t = np.linspace(0, 2 * np.pi, num_frames)

        left_hand_path = np.column_stack([100 + 50 * np.sin(t), 200 + 30 * np.cos(t)])

        # Right hand with slight asymmetry
        right_hand_path = np.column_stack([300 + 45 * np.sin(t), 200 + 28 * np.cos(t)])

        angles_over_time = {
            "left_elbow": 90 + 40 * np.sin(t),
            "right_elbow": 92 + 38 * np.sin(t),  # Slight asymmetry
            "left_shoulder": 120 + 30 * np.cos(t),
            "right_shoulder": 118 + 32 * np.cos(t),  # Slight asymmetry
        }

        return {
            "left_hand_path": left_hand_path,
            "right_hand_path": right_hand_path,
            "angles_over_time": angles_over_time,
        }

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SymmetryAnalyzer(fps=30.0)
        assert analyzer.fps == 30.0
        assert analyzer.dt == 1.0 / 30.0

    def test_analyze_arm_symmetry(self, sample_data):
        """Test arm symmetry analysis."""
        analyzer = SymmetryAnalyzer(fps=30.0)

        metrics = analyzer.analyze_arm_symmetry(
            sample_data["left_hand_path"],
            sample_data["right_hand_path"],
            sample_data["angles_over_time"]["left_elbow"],
            sample_data["angles_over_time"]["right_elbow"],
        )

        assert isinstance(metrics, dict)
        assert "path_length_asymmetry_pct" in metrics
        assert "left_path_length" in metrics
        assert "right_path_length" in metrics

    def test_analyze_temporal_symmetry(self, sample_data):
        """Test temporal symmetry analysis."""
        analyzer = SymmetryAnalyzer(fps=30.0)

        metrics = analyzer.analyze_temporal_symmetry(
            sample_data["left_hand_path"],
            sample_data["right_hand_path"],
        )

        assert isinstance(metrics, dict)

    def test_estimate_force_imbalance(self, sample_data):
        """Test force imbalance estimation."""
        analyzer = SymmetryAnalyzer(fps=30.0)

        metrics = analyzer.estimate_force_imbalance(
            sample_data["left_hand_path"],
            sample_data["right_hand_path"],
        )

        assert isinstance(metrics, dict)

    def test_comprehensive_symmetry_analysis(self, sample_data):
        """Test comprehensive symmetry analysis."""
        analyzer = SymmetryAnalyzer(fps=30.0)

        results = analyzer.comprehensive_symmetry_analysis(
            sample_data["left_hand_path"],
            sample_data["right_hand_path"],
            sample_data["angles_over_time"],
        )

        assert isinstance(results, dict)
        assert "overall_symmetry_score" in results
        assert "interpretation" in results
        assert "recommendations" in results
        assert "arm_symmetry" in results
        assert "temporal_symmetry" in results
        assert "force_imbalance" in results

        # Score should be between 0 and 100
        assert 0 <= results["overall_symmetry_score"] <= 100


class TestMovingAverage:
    """Test moving average function."""

    def test_moving_average_valid(self):
        """Test moving average with valid mode."""
        signal = np.array([1, 2, 3, 4, 5])
        result = moving_average(signal, window_size=3, mode="valid")

        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_moving_average_same(self):
        """Test moving average with same mode."""
        signal = np.array([1, 2, 3, 4, 5])
        result = moving_average(signal, window_size=3, mode="same")

        assert len(result) == len(signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
