"""Unified hand tracker with multiple occlusion handling methods.

Combines different tracking approaches:
- Enhanced Kalman filtering with prediction
- Stroke phase-aware prediction
- Interpolation-based trajectory completion
- Hybrid approaches
"""

from enum import Enum
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from src.tracking.occlusion_detector import OcclusionDetector, OcclusionState
from src.utils.smoothing import KalmanFilter2D
from src.analysis.stroke_phases import StrokePhase


class TrackingMethod(Enum):
    """Available tracking methods for occlusion handling."""
    KALMAN_ONLY = "kalman"                    # Pure Kalman filtering
    KALMAN_PREDICT = "kalman_predict"        # Kalman with prediction during occlusion
    PHASE_AWARE = "phase_aware"              # Stroke phase constraints
    INTERPOLATION = "interpolation"          # Post-process interpolation
    HYBRID = "hybrid"                        # Combine multiple methods


@dataclass
class TrackingResult:
    """Result from hand tracking."""
    position: np.ndarray                     # Estimated position [x, y]
    confidence: float                        # Tracking confidence (0-1)
    occlusion_state: OcclusionState         # Current occlusion state
    method_used: TrackingMethod             # Method that produced this result
    is_predicted: bool                      # True if position is predicted (not observed)
    velocity: Optional[np.ndarray] = None   # Estimated velocity [vx, vy]


class HandTracker:
    """Unified hand tracker with multiple occlusion handling methods."""

    def __init__(
        self,
        method: TrackingMethod = TrackingMethod.HYBRID,
        fps: float = 30.0,
        kalman_process_variance: float = 1e-4,
        kalman_measurement_variance: float = 1e-1,
        confidence_threshold: float = 0.3,
        use_phase_detection: bool = True,
    ):
        """Initialize hand tracker.

        Args:
            method: Tracking method to use.
            fps: Video frame rate.
            kalman_process_variance: Process noise for Kalman filter.
            kalman_measurement_variance: Measurement noise for Kalman filter.
            confidence_threshold: Confidence threshold for occlusion detection.
            use_phase_detection: Use stroke phase for better prediction.
        """
        self.method = method
        self.fps = fps
        self.dt = 1.0 / fps

        # Initialize Kalman filter
        self.kalman = KalmanFilter2D(
            process_variance=kalman_process_variance,
            measurement_variance=kalman_measurement_variance,
            dt=self.dt,
        )

        # Initialize occlusion detector
        self.occlusion_detector = OcclusionDetector(
            confidence_threshold_high=0.5,
            confidence_threshold_low=confidence_threshold,
            min_occlusion_frames=3,
            use_phase_detection=use_phase_detection,
        )

        # Tracking history
        self.position_history: List[np.ndarray] = []
        self.confidence_history: List[float] = []
        self.occlusion_history: List[OcclusionState] = []
        self.prediction_segments: List[Dict] = []  # Track prediction segments

        # Phase-aware tracking
        self.current_phase: Optional[StrokePhase] = None
        self.phase_templates: Dict[str, np.ndarray] = {}  # Ideal phase trajectories

        # For interpolation method
        self.pending_interpolation: Optional[Dict] = None

    def update(
        self,
        observation: Optional[np.ndarray],
        confidence: float,
        stroke_phase: Optional[StrokePhase] = None,
    ) -> TrackingResult:
        """Update tracker with new observation.

        Args:
            observation: Observed position [x, y] or None if not detected.
            confidence: Detection confidence (0-1).
            stroke_phase: Current stroke phase.

        Returns:
            Tracking result with estimated position.
        """
        # Update phase
        if stroke_phase is not None:
            self.current_phase = stroke_phase

        # Detect occlusion
        occlusion_state = self.occlusion_detector.detect(
            confidence,
            stroke_phase.value if stroke_phase else None,
            observation,
        )

        # Route to appropriate tracking method
        if self.method == TrackingMethod.KALMAN_ONLY:
            result = self._track_kalman_only(observation, confidence, occlusion_state)

        elif self.method == TrackingMethod.KALMAN_PREDICT:
            result = self._track_kalman_predict(observation, confidence, occlusion_state)

        elif self.method == TrackingMethod.PHASE_AWARE:
            result = self._track_phase_aware(observation, confidence, occlusion_state)

        elif self.method == TrackingMethod.INTERPOLATION:
            result = self._track_interpolation(observation, confidence, occlusion_state)

        elif self.method == TrackingMethod.HYBRID:
            result = self._track_hybrid(observation, confidence, occlusion_state)

        else:
            raise ValueError(f"Unknown tracking method: {self.method}")

        # Update history
        self.position_history.append(result.position)
        self.confidence_history.append(result.confidence)
        self.occlusion_history.append(occlusion_state)

        return result

    def _track_kalman_only(
        self,
        observation: Optional[np.ndarray],
        confidence: float,
        occlusion_state: OcclusionState,
    ) -> TrackingResult:
        """Track using only Kalman filter (no special occlusion handling).

        Args:
            observation: Observed position.
            confidence: Detection confidence.
            occlusion_state: Current occlusion state.

        Returns:
            Tracking result.
        """
        if observation is not None and confidence > 0.1:
            # Update with observation
            position = self.kalman.update(observation)
            is_predicted = False
        else:
            # No observation, use prediction
            position = self.kalman.state[0:2]
            is_predicted = True

        velocity = self.kalman.get_velocity()

        return TrackingResult(
            position=position,
            confidence=confidence,
            occlusion_state=occlusion_state,
            method_used=TrackingMethod.KALMAN_ONLY,
            is_predicted=is_predicted,
            velocity=velocity,
        )

    def _track_kalman_predict(
        self,
        observation: Optional[np.ndarray],
        confidence: float,
        occlusion_state: OcclusionState,
    ) -> TrackingResult:
        """Track with Kalman prediction during occlusion.

        Args:
            observation: Observed position.
            confidence: Detection confidence.
            occlusion_state: Current occlusion state.

        Returns:
            Tracking result.
        """
        is_occluded = occlusion_state in [
            OcclusionState.FULLY_OCCLUDED,
            OcclusionState.PARTIALLY_OCCLUDED,
        ]

        if not is_occluded and observation is not None:
            # Visible - update with measurement
            position = self.kalman.update(observation)
            is_predicted = False
            tracking_confidence = confidence
        else:
            # Occluded - predict only (no measurement update)
            # Prediction step
            self.kalman.state = self.kalman.F @ self.kalman.state
            self.kalman.P = self.kalman.F @ self.kalman.P @ self.kalman.F.T + self.kalman.Q

            position = self.kalman.state[0:2]
            is_predicted = True

            # Confidence degrades during occlusion
            tracking_confidence = max(0.1, confidence * 0.9)

        velocity = self.kalman.get_velocity()

        return TrackingResult(
            position=position,
            confidence=tracking_confidence,
            occlusion_state=occlusion_state,
            method_used=TrackingMethod.KALMAN_PREDICT,
            is_predicted=is_predicted,
            velocity=velocity,
        )

    def _track_phase_aware(
        self,
        observation: Optional[np.ndarray],
        confidence: float,
        occlusion_state: OcclusionState,
    ) -> TrackingResult:
        """Track with stroke phase constraints.

        Args:
            observation: Observed position.
            confidence: Detection confidence.
            occlusion_state: Current occlusion state.

        Returns:
            Tracking result.
        """
        is_occluded = occlusion_state in [
            OcclusionState.FULLY_OCCLUDED,
            OcclusionState.PARTIALLY_OCCLUDED,
        ]

        if not is_occluded and observation is not None:
            # Visible - normal Kalman update
            position = self.kalman.update(observation)
            is_predicted = False
            tracking_confidence = confidence

        else:
            # Occluded - use phase-aware prediction
            if self.current_phase is not None and self.current_phase.value in self.phase_templates:
                # Use template for this phase
                template = self.phase_templates[self.current_phase.value]
                # TODO: Interpolate along template based on time in phase
                # For now, use Kalman prediction with phase constraints
                position = self._predict_with_phase_constraint()
            else:
                # No template, fall back to Kalman prediction
                self.kalman.state = self.kalman.F @ self.kalman.state
                self.kalman.P = self.kalman.F @ self.kalman.P @ self.kalman.F.T + self.kalman.Q
                position = self.kalman.state[0:2]

            is_predicted = True
            tracking_confidence = max(0.2, confidence * 0.8)

        velocity = self.kalman.get_velocity()

        return TrackingResult(
            position=position,
            confidence=tracking_confidence,
            occlusion_state=occlusion_state,
            method_used=TrackingMethod.PHASE_AWARE,
            is_predicted=is_predicted,
            velocity=velocity,
        )

    def _predict_with_phase_constraint(self) -> np.ndarray:
        """Predict position with stroke phase constraints.

        Returns:
            Predicted position.
        """
        # Get Kalman prediction
        predicted_state = self.kalman.F @ self.kalman.state
        predicted_pos = predicted_state[0:2]

        # TODO: Apply phase-specific constraints
        # For example, during pull phase, hand should move backward and down
        # This could be implemented by:
        # 1. Defining velocity bounds for each phase
        # 2. Constraining prediction to stay within expected region

        return predicted_pos

    def _track_interpolation(
        self,
        observation: Optional[np.ndarray],
        confidence: float,
        occlusion_state: OcclusionState,
    ) -> TrackingResult:
        """Track with post-process interpolation.

        This method waits for hand to reappear, then retroactively fills trajectory.

        Args:
            observation: Observed position.
            confidence: Detection confidence.
            occlusion_state: Current occlusion state.

        Returns:
            Tracking result.
        """
        is_occluded = occlusion_state in [
            OcclusionState.FULLY_OCCLUDED,
            OcclusionState.PARTIALLY_OCCLUDED,
        ]

        if not is_occluded and observation is not None:
            # Visible
            if self.pending_interpolation is not None:
                # Hand just reappeared - perform interpolation
                self._complete_interpolation(observation)
                self.pending_interpolation = None

            position = observation
            is_predicted = False
            tracking_confidence = confidence

        else:
            # Occluded
            if self.pending_interpolation is None:
                # Start new occlusion segment
                last_visible = self.position_history[-1] if self.position_history else np.zeros(2)
                self.pending_interpolation = {
                    'start_pos': last_visible,
                    'start_frame': len(self.position_history),
                }

            # Use last known position (will be updated retroactively)
            position = self.pending_interpolation['start_pos']
            is_predicted = True
            tracking_confidence = 0.1

        velocity = None  # Interpolation doesn't provide real-time velocity

        return TrackingResult(
            position=position,
            confidence=tracking_confidence,
            occlusion_state=occlusion_state,
            method_used=TrackingMethod.INTERPOLATION,
            is_predicted=is_predicted,
            velocity=velocity,
        )

    def _complete_interpolation(self, end_position: np.ndarray):
        """Complete interpolation for occluded segment.

        Args:
            end_position: Position where hand reappeared.
        """
        if self.pending_interpolation is None:
            return

        start_pos = self.pending_interpolation['start_pos']
        start_frame = self.pending_interpolation['start_frame']
        end_frame = len(self.position_history)
        num_frames = end_frame - start_frame

        if num_frames < 2:
            return

        # Interpolate using cubic spline
        t = np.array([0, num_frames])
        positions = np.array([start_pos, end_position])

        # Create interpolator
        try:
            interpolator_x = interp1d(t, positions[:, 0], kind='cubic', fill_value='extrapolate')
            interpolator_y = interp1d(t, positions[:, 1], kind='cubic', fill_value='extrapolate')

            # Fill in interpolated positions
            t_interp = np.arange(1, num_frames)
            for i, t_val in enumerate(t_interp):
                frame_idx = start_frame + i
                if frame_idx < len(self.position_history):
                    self.position_history[frame_idx] = np.array([
                        interpolator_x(t_val),
                        interpolator_y(t_val),
                    ])
        except:
            # Fallback to linear interpolation
            for i in range(1, num_frames):
                alpha = i / num_frames
                interp_pos = (1 - alpha) * start_pos + alpha * end_position
                frame_idx = start_frame + i
                if frame_idx < len(self.position_history):
                    self.position_history[frame_idx] = interp_pos

    def _track_hybrid(
        self,
        observation: Optional[np.ndarray],
        confidence: float,
        occlusion_state: OcclusionState,
    ) -> TrackingResult:
        """Hybrid tracking: Combine Kalman + Phase awareness.

        Args:
            observation: Observed position.
            confidence: Detection confidence.
            occlusion_state: Current occlusion state.

        Returns:
            Tracking result.
        """
        is_occluded = occlusion_state in [
            OcclusionState.FULLY_OCCLUDED,
            OcclusionState.PARTIALLY_OCCLUDED,
        ]

        if not is_occluded and observation is not None:
            # Visible - normal Kalman update
            position = self.kalman.update(observation)
            is_predicted = False
            tracking_confidence = confidence

        else:
            # Occluded - use both Kalman prediction and phase constraints
            # Get Kalman prediction
            self.kalman.state = self.kalman.F @ self.kalman.state
            self.kalman.P = self.kalman.F @ self.kalman.P @ self.kalman.F.T + self.kalman.Q
            kalman_pos = self.kalman.state[0:2]

            # Apply phase constraints if available
            if self.current_phase is not None:
                position = self._apply_phase_constraints(kalman_pos, self.current_phase)
            else:
                position = kalman_pos

            is_predicted = True
            tracking_confidence = max(0.15, confidence * 0.85)

        velocity = self.kalman.get_velocity()

        return TrackingResult(
            position=position,
            confidence=tracking_confidence,
            occlusion_state=occlusion_state,
            method_used=TrackingMethod.HYBRID,
            is_predicted=is_predicted,
            velocity=velocity,
        )

    def _apply_phase_constraints(
        self,
        predicted_pos: np.ndarray,
        phase: StrokePhase,
    ) -> np.ndarray:
        """Apply stroke phase constraints to predicted position.

        Args:
            predicted_pos: Kalman predicted position.
            phase: Current stroke phase.

        Returns:
            Constrained position.
        """
        # Define expected velocity directions for each phase
        # (in image coordinates: +x = right, +y = down)
        phase_velocity_hints = {
            StrokePhase.ENTRY: (0, 1),      # Moving down (entering water)
            StrokePhase.CATCH: (-0.5, 1),   # Down and slightly back
            StrokePhase.PULL: (-1, 0.5),    # Backward and down
            StrokePhase.PUSH: (-1, -0.5),   # Backward and up
            StrokePhase.RECOVERY: (1, -1),  # Forward and up
        }

        # For now, just return predicted position
        # TODO: Implement actual constraint logic
        # This could involve:
        # 1. Checking if velocity aligns with expected direction
        # 2. Bounding position to expected region for phase
        # 3. Adjusting prediction to be more biomechanically plausible

        return predicted_pos

    def set_phase_template(self, phase: StrokePhase, template: np.ndarray):
        """Set ideal trajectory template for a stroke phase.

        Args:
            phase: Stroke phase.
            template: Ideal trajectory for this phase (Nx2 array).
        """
        self.phase_templates[phase.value] = template

    def get_trajectory(self) -> np.ndarray:
        """Get full tracked trajectory.

        Returns:
            Trajectory array (Nx2).
        """
        if not self.position_history:
            return np.array([])
        return np.array(self.position_history)

    def get_statistics(self) -> Dict[str, any]:
        """Get tracking statistics.

        Returns:
            Dictionary with statistics.
        """
        stats = self.occlusion_detector.get_statistics()

        # Add tracking-specific stats
        stats['tracking_method'] = self.method.value
        stats['total_tracked_frames'] = len(self.position_history)

        # Calculate percentage of predicted frames
        predicted_frames = sum(
            1 for state in self.occlusion_history
            if state in [OcclusionState.FULLY_OCCLUDED, OcclusionState.PARTIALLY_OCCLUDED]
        )
        stats['predicted_frames'] = predicted_frames
        stats['prediction_percentage'] = (
            predicted_frames / len(self.position_history) * 100
            if self.position_history else 0.0
        )

        return stats

    def reset(self):
        """Reset tracker state."""
        self.kalman.reset()
        self.occlusion_detector.reset()
        self.position_history = []
        self.confidence_history = []
        self.occlusion_history = []
        self.prediction_segments = []
        self.pending_interpolation = None
