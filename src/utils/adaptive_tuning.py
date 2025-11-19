"""Adaptive threshold tuning system for swimming analysis.

Automatically adjusts detection thresholds and parameters based on:
- Pool conditions (lighting, water clarity)
- Camera settings
- Detection performance metrics
- Environmental factors
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque


class PoolCondition(Enum):
    """Pool environment conditions."""
    EXCELLENT = "excellent"        # Perfect lighting, clear water
    GOOD = "good"                 # Good conditions
    FAIR = "fair"                 # Acceptable conditions
    POOR = "poor"                 # Challenging conditions
    VERY_POOR = "very_poor"       # Very difficult conditions


@dataclass
class AdaptiveParameters:
    """Adaptive parameters for detection."""
    confidence_threshold: float = 0.5
    occlusion_threshold: float = 0.3
    tracking_confidence: float = 0.4
    min_detection_confidence: float = 0.3
    max_reprojection_error: float = 10.0
    kalman_process_noise: float = 1e-5
    kalman_measurement_noise: float = 1e-1
    iou_threshold: float = 0.5
    nms_threshold: float = 0.4

    # Water surface detection
    water_edge_threshold: int = 50
    water_color_tolerance: int = 30

    # Temporal smoothing
    smoothing_window: int = 5

    metadata: Dict = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Detection performance metrics for adaptive tuning."""
    avg_confidence: float = 0.0
    detection_rate: float = 0.0        # % of frames with detections
    tracking_stability: float = 0.0    # Stability score
    false_positive_rate: float = 0.0
    avg_keypoint_visibility: float = 0.0
    temporal_consistency: float = 0.0


class AdaptiveThresholdTuner:
    """Automatically tune detection thresholds based on conditions."""

    def __init__(
        self,
        initial_params: Optional[AdaptiveParameters] = None,
        adaptation_rate: float = 0.1,
        history_size: int = 30,
        auto_tune: bool = True,
    ):
        """Initialize adaptive tuner.

        Args:
            initial_params: Initial parameters (uses defaults if None).
            adaptation_rate: Rate of parameter adaptation (0-1).
            history_size: Number of frames to track for metrics.
            auto_tune: Enable automatic tuning.
        """
        self.params = initial_params or AdaptiveParameters()
        self.adaptation_rate = adaptation_rate
        self.history_size = history_size
        self.auto_tune = auto_tune

        # Metrics history
        self.confidence_history = deque(maxlen=history_size)
        self.detection_history = deque(maxlen=history_size)
        self.stability_history = deque(maxlen=history_size)
        self.visibility_history = deque(maxlen=history_size)

        # Previous detection for tracking
        self.previous_detections = []

        # Condition assessment
        self.current_condition = PoolCondition.GOOD

    def update(
        self,
        detection_result: Optional[Dict],
        frame_stats: Optional[Dict] = None,
    ) -> AdaptiveParameters:
        """Update parameters based on detection results.

        Args:
            detection_result: Detection results (pose_data or None).
            frame_stats: Frame statistics (brightness, contrast, etc.).

        Returns:
            Updated parameters.
        """
        # Collect metrics
        metrics = self._calculate_metrics(detection_result, frame_stats)

        # Update history
        self._update_history(metrics, detection_result is not None)

        # Assess current conditions
        self._assess_conditions(metrics, frame_stats)

        # Auto-tune if enabled
        if self.auto_tune:
            self._auto_tune(metrics)

        # Store previous detection for next frame
        if detection_result is not None:
            self.previous_detections.append(detection_result)
            if len(self.previous_detections) > self.history_size:
                self.previous_detections.pop(0)

        return self.params

    def _calculate_metrics(
        self,
        detection_result: Optional[Dict],
        frame_stats: Optional[Dict],
    ) -> PerformanceMetrics:
        """Calculate performance metrics.

        Args:
            detection_result: Detection results.
            frame_stats: Frame statistics.

        Returns:
            Performance metrics.
        """
        metrics = PerformanceMetrics()

        if detection_result is not None:
            # Handle both single and multi-person results
            if isinstance(detection_result, list):
                if len(detection_result) > 0:
                    detection_result = detection_result[0]
                else:
                    return metrics

            keypoints = detection_result.get('keypoints', np.array([]))

            if len(keypoints) > 0:
                # Average confidence
                confidences = keypoints[:, 2]
                metrics.avg_confidence = float(np.mean(confidences))

                # Keypoint visibility (% with confidence > threshold)
                visible = np.sum(confidences > self.params.confidence_threshold)
                metrics.avg_keypoint_visibility = visible / len(keypoints)

                # Tracking stability (compare with previous frame)
                if len(self.previous_detections) > 0:
                    prev_det = self.previous_detections[-1]
                    prev_kps = prev_det.get('keypoints', np.array([]))

                    if len(prev_kps) == len(keypoints):
                        # Calculate position differences
                        pos_diff = np.linalg.norm(keypoints[:, :2] - prev_kps[:, :2], axis=1)
                        # Stability is inverse of average movement (normalized)
                        avg_movement = np.mean(pos_diff)
                        metrics.tracking_stability = 1.0 / (1.0 + avg_movement / 10.0)

                # Temporal consistency (confidence variation)
                if len(self.confidence_history) > 5:
                    conf_std = np.std(list(self.confidence_history)[-5:])
                    metrics.temporal_consistency = 1.0 / (1.0 + conf_std)

        return metrics

    def _update_history(self, metrics: PerformanceMetrics, detected: bool):
        """Update metrics history.

        Args:
            metrics: Current metrics.
            detected: Whether detection succeeded.
        """
        self.confidence_history.append(metrics.avg_confidence)
        self.detection_history.append(1.0 if detected else 0.0)
        self.stability_history.append(metrics.tracking_stability)
        self.visibility_history.append(metrics.avg_keypoint_visibility)

    def _assess_conditions(
        self,
        metrics: PerformanceMetrics,
        frame_stats: Optional[Dict],
    ):
        """Assess current pool conditions.

        Args:
            metrics: Performance metrics.
            frame_stats: Frame statistics.
        """
        # Calculate condition score (0-1, higher is better)
        score = 0.0
        factors = 0

        # Factor 1: Detection rate
        if len(self.detection_history) > 0:
            detection_rate = np.mean(self.detection_history)
            score += detection_rate
            factors += 1

        # Factor 2: Average confidence
        if len(self.confidence_history) > 0:
            avg_conf = np.mean(self.confidence_history)
            score += avg_conf
            factors += 1

        # Factor 3: Visibility
        if len(self.visibility_history) > 0:
            avg_vis = np.mean(self.visibility_history)
            score += avg_vis
            factors += 1

        # Factor 4: Frame brightness (if available)
        if frame_stats and 'brightness' in frame_stats:
            brightness = frame_stats['brightness']
            # Normalize brightness (assuming 0-255 range)
            # Optimal brightness around 100-150
            if 80 <= brightness <= 170:
                brightness_score = 1.0
            elif brightness < 80:
                brightness_score = brightness / 80.0
            else:
                brightness_score = max(0, 1.0 - (brightness - 170) / 85.0)

            score += brightness_score
            factors += 1

        # Average score
        if factors > 0:
            score = score / factors

        # Map score to condition
        if score >= 0.8:
            self.current_condition = PoolCondition.EXCELLENT
        elif score >= 0.65:
            self.current_condition = PoolCondition.GOOD
        elif score >= 0.5:
            self.current_condition = PoolCondition.FAIR
        elif score >= 0.3:
            self.current_condition = PoolCondition.POOR
        else:
            self.current_condition = PoolCondition.VERY_POOR

    def _auto_tune(self, metrics: PerformanceMetrics):
        """Automatically tune parameters based on metrics.

        Args:
            metrics: Current performance metrics.
        """
        # Tune confidence threshold
        if len(self.confidence_history) >= 10:
            avg_conf = np.mean(self.confidence_history)

            # If average confidence is low, lower threshold
            if avg_conf < 0.4:
                target_threshold = max(0.2, avg_conf * 0.7)
                self.params.confidence_threshold = self._adapt_param(
                    self.params.confidence_threshold,
                    target_threshold,
                )

            # If average confidence is high and detection rate is good, raise threshold
            elif avg_conf > 0.7 and np.mean(self.detection_history) > 0.8:
                target_threshold = min(0.7, avg_conf * 0.8)
                self.params.confidence_threshold = self._adapt_param(
                    self.params.confidence_threshold,
                    target_threshold,
                )

        # Tune occlusion threshold (lower if visibility is poor)
        if len(self.visibility_history) >= 10:
            avg_visibility = np.mean(self.visibility_history)

            if avg_visibility < 0.5:
                # Lower occlusion threshold to be more aggressive
                target_occlusion = max(0.15, avg_visibility * 0.5)
                self.params.occlusion_threshold = self._adapt_param(
                    self.params.occlusion_threshold,
                    target_occlusion,
                )

        # Tune tracking confidence based on stability
        if len(self.stability_history) >= 10:
            avg_stability = np.mean(self.stability_history)

            if avg_stability < 0.5:
                # Increase tracking confidence requirement if tracking is unstable
                target_tracking = min(0.6, 0.3 + (1.0 - avg_stability) * 0.3)
                self.params.tracking_confidence = self._adapt_param(
                    self.params.tracking_confidence,
                    target_tracking,
                )
            else:
                # Can be more lenient if tracking is stable
                target_tracking = max(0.3, 0.5 * avg_stability)
                self.params.tracking_confidence = self._adapt_param(
                    self.params.tracking_confidence,
                    target_tracking,
                )

        # Tune Kalman noise based on tracking stability
        if len(self.stability_history) >= 10:
            avg_stability = np.mean(self.stability_history)

            if avg_stability < 0.6:
                # Increase process noise if movement is erratic
                target_process_noise = min(1e-3, 1e-5 / max(avg_stability, 0.1))
                self.params.kalman_process_noise = self._adapt_param(
                    self.params.kalman_process_noise,
                    target_process_noise,
                )

        # Condition-based tuning
        self._tune_for_condition()

    def _tune_for_condition(self):
        """Tune parameters based on assessed conditions."""
        if self.current_condition == PoolCondition.EXCELLENT:
            # Can use stricter thresholds
            targets = {
                'confidence_threshold': 0.6,
                'occlusion_threshold': 0.35,
                'min_detection_confidence': 0.4,
            }
        elif self.current_condition == PoolCondition.GOOD:
            # Standard thresholds
            targets = {
                'confidence_threshold': 0.5,
                'occlusion_threshold': 0.3,
                'min_detection_confidence': 0.3,
            }
        elif self.current_condition == PoolCondition.FAIR:
            # Slightly more lenient
            targets = {
                'confidence_threshold': 0.4,
                'occlusion_threshold': 0.25,
                'min_detection_confidence': 0.25,
            }
        elif self.current_condition == PoolCondition.POOR:
            # More lenient thresholds
            targets = {
                'confidence_threshold': 0.35,
                'occlusion_threshold': 0.2,
                'min_detection_confidence': 0.2,
            }
        else:  # VERY_POOR
            # Very lenient
            targets = {
                'confidence_threshold': 0.25,
                'occlusion_threshold': 0.15,
                'min_detection_confidence': 0.15,
            }

        # Gradually adapt to target values
        for param_name, target_value in targets.items():
            current_value = getattr(self.params, param_name)
            new_value = self._adapt_param(current_value, target_value)
            setattr(self.params, param_name, new_value)

    def _adapt_param(self, current: float, target: float) -> float:
        """Gradually adapt parameter to target value.

        Args:
            current: Current parameter value.
            target: Target parameter value.

        Returns:
            Adapted parameter value.
        """
        # Exponential moving average
        return current + self.adaptation_rate * (target - current)

    def set_condition(self, condition: PoolCondition):
        """Manually set pool condition.

        Args:
            condition: Pool condition to set.
        """
        self.current_condition = condition
        self._tune_for_condition()

    def get_current_metrics(self) -> Dict:
        """Get current performance metrics.

        Returns:
            Dictionary with current metrics.
        """
        return {
            'avg_confidence': float(np.mean(self.confidence_history)) if len(self.confidence_history) > 0 else 0.0,
            'detection_rate': float(np.mean(self.detection_history)) if len(self.detection_history) > 0 else 0.0,
            'avg_stability': float(np.mean(self.stability_history)) if len(self.stability_history) > 0 else 0.0,
            'avg_visibility': float(np.mean(self.visibility_history)) if len(self.visibility_history) > 0 else 0.0,
            'current_condition': self.current_condition.value,
        }

    def get_parameters(self) -> AdaptiveParameters:
        """Get current adaptive parameters.

        Returns:
            Current parameters.
        """
        return self.params

    def reset(self):
        """Reset tuner state."""
        self.confidence_history.clear()
        self.detection_history.clear()
        self.stability_history.clear()
        self.visibility_history.clear()
        self.previous_detections.clear()
        self.current_condition = PoolCondition.GOOD


def calculate_frame_stats(frame: np.ndarray) -> Dict:
    """Calculate frame statistics for adaptive tuning.

    Args:
        frame: Input frame (BGR).

    Returns:
        Dictionary with frame statistics.
    """
    import cv2

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate statistics
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Calculate sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Calculate color distribution
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    avg_hue = float(np.mean(h))
    avg_saturation = float(np.mean(s))

    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'avg_hue': avg_hue,
        'avg_saturation': avg_saturation,
    }
