"""Tests for occlusion detection and hand tracking."""

import numpy as np
import pytest

from src.analysis.stroke_phases import StrokePhase
from src.tracking.hand_tracker import HandTracker, TrackingMethod, TrackingResult
from src.tracking.occlusion_detector import OcclusionDetector, OcclusionState


class TestOcclusionDetector:
    """Test occlusion detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = OcclusionDetector()
        assert detector.current_state == OcclusionState.VISIBLE
        assert detector.total_occlusion_events == 0

    def test_detect_from_high_confidence(self):
        """Test detection with high confidence (visible)."""
        detector = OcclusionDetector(confidence_threshold_high=0.5)
        state = detector.detect(confidence=0.8)
        assert state == OcclusionState.VISIBLE

    def test_detect_from_low_confidence(self):
        """Test detection with low confidence (occluded)."""
        detector = OcclusionDetector(
            confidence_threshold_low=0.3,
            min_occlusion_frames=1,
        )
        state = detector.detect(confidence=0.1)
        assert state == OcclusionState.FULLY_OCCLUDED

    def test_detect_from_underwater_phase(self):
        """Test detection from underwater stroke phase."""
        detector = OcclusionDetector(
            use_phase_detection=True,
            min_occlusion_frames=1,
        )

        # Pull phase should be underwater
        state = detector.detect(confidence=0.5, stroke_phase="pull")
        assert state == OcclusionState.FULLY_OCCLUDED

    def test_detect_from_recovery_phase(self):
        """Test detection from recovery phase (visible)."""
        detector = OcclusionDetector(
            use_phase_detection=True,
            min_occlusion_frames=1,
        )

        # Recovery phase should be visible
        state = detector.detect(confidence=0.5, stroke_phase="recovery")
        assert state == OcclusionState.VISIBLE

    def test_temporal_smoothing(self):
        """Test temporal smoothing prevents rapid state changes."""
        detector = OcclusionDetector(min_occlusion_frames=3)

        # First frame with low confidence
        state1 = detector.detect(confidence=0.1)
        # Should not immediately change state
        assert state1 == OcclusionState.VISIBLE

        # Second frame with low confidence
        state2 = detector.detect(confidence=0.1)
        assert state2 == OcclusionState.VISIBLE

        # Third frame with low confidence
        state3 = detector.detect(confidence=0.1)
        # Now should change to occluded
        assert state3 == OcclusionState.FULLY_OCCLUDED

    def test_is_occluded(self):
        """Test is_occluded helper method."""
        detector = OcclusionDetector(min_occlusion_frames=1)

        detector.detect(confidence=0.9)
        assert not detector.is_occluded()

        detector.detect(confidence=0.1)
        assert detector.is_occluded()

    def test_statistics(self):
        """Test statistics gathering."""
        detector = OcclusionDetector(min_occlusion_frames=1)

        # Visible frames
        for _ in range(5):
            detector.detect(confidence=0.9)

        # Occluded frames
        for _ in range(3):
            detector.detect(confidence=0.1)

        stats = detector.get_statistics()
        assert stats["total_frames"] == 8
        assert stats["total_occluded_frames"] == 3
        assert abs(stats["occlusion_percentage"] - 37.5) < 0.1

    def test_reset(self):
        """Test reset functionality."""
        detector = OcclusionDetector(min_occlusion_frames=1)

        detector.detect(confidence=0.1)
        assert len(detector.occlusion_history) > 0

        detector.reset()
        assert detector.current_state == OcclusionState.VISIBLE
        assert len(detector.occlusion_history) == 0


class TestHandTracker:
    """Test hand tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_ONLY, fps=30.0)
        assert tracker.method == TrackingMethod.KALMAN_ONLY
        assert tracker.fps == 30.0

    def test_track_kalman_only_visible(self):
        """Test Kalman-only tracking with visible observation."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_ONLY, fps=30.0)

        observation = np.array([100.0, 200.0])
        result = tracker.update(observation, confidence=0.9)

        assert isinstance(result, TrackingResult)
        assert result.occlusion_state == OcclusionState.VISIBLE
        assert not result.is_predicted
        assert np.allclose(result.position, observation, atol=10)

    def test_track_kalman_only_occluded(self):
        """Test Kalman-only tracking during occlusion."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_ONLY, fps=30.0)

        # First visible
        tracker.update(np.array([100.0, 200.0]), confidence=0.9)

        # Then occluded (no observation)
        result = tracker.update(None, confidence=0.1)

        assert result.is_predicted
        # Position should still be estimated

    def test_track_kalman_predict(self):
        """Test Kalman prediction method."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_PREDICT, fps=30.0)

        # Visible sequence
        for i in range(5):
            tracker.update(np.array([100.0 + i * 10, 200.0]), confidence=0.9)

        # Occluded sequence
        for i in range(3):
            result = tracker.update(None, confidence=0.1, stroke_phase=StrokePhase.PULL)
            assert result.is_predicted
            assert result.occlusion_state == OcclusionState.FULLY_OCCLUDED

    def test_track_interpolation(self):
        """Test interpolation method."""
        tracker = HandTracker(method=TrackingMethod.INTERPOLATION, fps=30.0)

        # Start visible
        tracker.update(np.array([100.0, 200.0]), confidence=0.9)

        # Occluded segment
        for _ in range(5):
            tracker.update(None, confidence=0.1)

        # Reappear
        result = tracker.update(np.array([150.0, 250.0]), confidence=0.9)

        # Check that interpolation was performed
        assert len(tracker.position_history) > 6

    def test_track_hybrid(self):
        """Test hybrid tracking method."""
        tracker = HandTracker(method=TrackingMethod.HYBRID, fps=30.0)

        # Visible
        result1 = tracker.update(np.array([100.0, 200.0]), confidence=0.9)
        assert not result1.is_predicted

        # Occluded
        result2 = tracker.update(None, confidence=0.1, stroke_phase=StrokePhase.CATCH)
        assert result2.is_predicted
        assert result2.method_used == TrackingMethod.HYBRID

    def test_get_trajectory(self):
        """Test trajectory retrieval."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_ONLY, fps=30.0)

        positions = [
            np.array([100.0, 200.0]),
            np.array([110.0, 210.0]),
            np.array([120.0, 220.0]),
        ]

        for pos in positions:
            tracker.update(pos, confidence=0.9)

        trajectory = tracker.get_trajectory()
        assert len(trajectory) == 3
        assert trajectory.shape == (3, 2)

    def test_statistics(self):
        """Test tracking statistics."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_PREDICT, fps=30.0)

        # Visible frames
        for i in range(5):
            tracker.update(np.array([100.0 + i, 200.0]), confidence=0.9)

        # Occluded frames
        for _ in range(3):
            tracker.update(None, confidence=0.1)

        stats = tracker.get_statistics()
        assert stats["total_tracked_frames"] == 8
        assert stats["predicted_frames"] == 3
        assert stats["tracking_method"] == "kalman_predict"

    def test_reset(self):
        """Test tracker reset."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_ONLY, fps=30.0)

        tracker.update(np.array([100.0, 200.0]), confidence=0.9)
        assert len(tracker.position_history) > 0

        tracker.reset()
        assert len(tracker.position_history) == 0
        assert len(tracker.confidence_history) == 0

    def test_velocity_estimation(self):
        """Test velocity estimation during tracking."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_PREDICT, fps=30.0)

        # Moving sequence
        for i in range(5):
            result = tracker.update(np.array([100.0 + i * 10, 200.0]), confidence=0.9)

        # Velocity should be approximately [10, 0] pixels/frame
        assert result.velocity is not None
        assert abs(result.velocity[0] - 10) < 5  # Allow some error

    def test_confidence_decay_during_occlusion(self):
        """Test that confidence decays during occlusion."""
        tracker = HandTracker(method=TrackingMethod.KALMAN_PREDICT, fps=30.0)

        # Start visible
        result1 = tracker.update(np.array([100.0, 200.0]), confidence=0.9)
        initial_conf = result1.confidence

        # Occlude
        for _ in range(5):
            result = tracker.update(None, confidence=0.1)

        # Confidence should have decreased
        assert result.confidence < initial_conf

    def test_multiple_tracking_methods(self):
        """Test all tracking methods produce valid results."""
        methods = [
            TrackingMethod.KALMAN_ONLY,
            TrackingMethod.KALMAN_PREDICT,
            TrackingMethod.PHASE_AWARE,
            TrackingMethod.HYBRID,
        ]

        for method in methods:
            tracker = HandTracker(method=method, fps=30.0)

            # Visible
            result1 = tracker.update(np.array([100.0, 200.0]), confidence=0.9)
            assert isinstance(result1, TrackingResult)

            # Occluded
            result2 = tracker.update(None, confidence=0.1, stroke_phase=StrokePhase.PULL)
            assert isinstance(result2, TrackingResult)
            assert result2.is_predicted


class TestTrackingIntegration:
    """Integration tests for occlusion tracking."""

    def test_full_stroke_cycle(self):
        """Test tracking through a full stroke cycle."""
        tracker = HandTracker(method=TrackingMethod.HYBRID, fps=30.0)

        # Simulate a stroke cycle
        # Entry (visible)
        for i in range(5):
            pos = np.array([100.0 + i * 5, 200.0 + i * 10])
            tracker.update(pos, confidence=0.9, stroke_phase=StrokePhase.ENTRY)

        # Catch (transitioning to underwater)
        for i in range(3):
            pos = np.array([125.0, 250.0 + i * 5])
            confidence = 0.9 - i * 0.2
            tracker.update(pos, confidence=confidence, stroke_phase=StrokePhase.CATCH)

        # Pull (underwater - occluded)
        for _ in range(10):
            tracker.update(None, confidence=0.1, stroke_phase=StrokePhase.PULL)

        # Push (still underwater)
        for _ in range(5):
            tracker.update(None, confidence=0.1, stroke_phase=StrokePhase.PUSH)

        # Recovery (reappearing)
        for i in range(5):
            pos = np.array([100.0 - i * 5, 200.0 - i * 10])
            tracker.update(pos, confidence=0.9, stroke_phase=StrokePhase.RECOVERY)

        # Check trajectory is complete
        trajectory = tracker.get_trajectory()
        assert len(trajectory) > 20

        # Check statistics
        stats = tracker.get_statistics()
        assert stats["total_occluded_frames"] > 10
        assert stats["occlusion_percentage"] > 30

    def test_comparison_of_methods(self):
        """Compare different tracking methods on same data."""
        # Generate synthetic trajectory
        true_positions = []
        for i in range(30):
            x = 100 + 50 * np.sin(i * 0.2)
            y = 200 + 30 * np.cos(i * 0.2)
            true_positions.append(np.array([x, y]))

        # Simulate occlusion (frames 10-20)
        observations = []
        confidences = []
        for i, pos in enumerate(true_positions):
            if 10 <= i < 20:
                observations.append(None)
                confidences.append(0.1)
            else:
                observations.append(pos + np.random.normal(0, 2, 2))  # Add noise
                confidences.append(0.9)

        # Test each method
        methods = [
            TrackingMethod.KALMAN_ONLY,
            TrackingMethod.KALMAN_PREDICT,
            TrackingMethod.HYBRID,
        ]

        results = {}
        for method in methods:
            tracker = HandTracker(method=method, fps=30.0)

            for obs, conf in zip(observations, confidences, strict=False):
                tracker.update(obs, conf)

            trajectory = tracker.get_trajectory()
            results[method.value] = trajectory

        # All methods should produce trajectories
        for method_name, traj in results.items():
            assert len(traj) == 30, f"{method_name} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
