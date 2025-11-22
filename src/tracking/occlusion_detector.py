"""Occlusion detection for swimming hand tracking.

Detects when swimmer's hands are underwater (occluded) using:
- Pose estimation confidence scores
- Stroke phase information
- Temporal consistency checks
"""

from enum import Enum
from typing import Any

import numpy as np


class OcclusionState(Enum):
    """State of hand visibility."""

    VISIBLE = "visible"  # Hand clearly visible
    PARTIALLY_OCCLUDED = "partial"  # Hand partially visible (low confidence)
    FULLY_OCCLUDED = "occluded"  # Hand underwater/not visible
    TRANSITIONING = "transitioning"  # Entering or exiting water


class OcclusionDetector:
    """Detect when hands are occluded (underwater) during swimming."""

    def __init__(
        self,
        confidence_threshold_high: float = 0.5,
        confidence_threshold_low: float = 0.3,
        min_occlusion_frames: int = 3,
        use_phase_detection: bool = True,
    ):
        """Initialize occlusion detector.

        Args:
            confidence_threshold_high: Confidence above this = visible.
            confidence_threshold_low: Confidence below this = occluded.
            min_occlusion_frames: Minimum frames to confirm occlusion.
            use_phase_detection: Use stroke phase to help detect occlusion.
        """
        self.confidence_threshold_high = confidence_threshold_high
        self.confidence_threshold_low = confidence_threshold_low
        self.min_occlusion_frames = min_occlusion_frames
        self.use_phase_detection = use_phase_detection

        # State tracking
        self.current_state = OcclusionState.VISIBLE
        self.frames_in_current_state = 0
        self.occlusion_history: list[OcclusionState] = []

        # Statistics
        self.total_occlusion_events = 0
        self.total_occluded_frames = 0

    def detect(
        self,
        confidence: float,
        stroke_phase: str | None = None,
        position: np.ndarray | None = None,
    ) -> OcclusionState:
        """Detect current occlusion state.

        Args:
            confidence: Pose keypoint confidence (0-1).
            stroke_phase: Current stroke phase (e.g., 'catch', 'pull', 'push').
            position: Current hand position [x, y] if available.

        Returns:
            Current occlusion state.
        """
        # Method 1: Confidence-based detection
        state_from_confidence = self._detect_from_confidence(confidence)

        # Method 2: Phase-based detection (if enabled)
        if self.use_phase_detection and stroke_phase is not None:
            state_from_phase = self._detect_from_phase(stroke_phase)
            # Combine both signals
            state = self._combine_detections(state_from_confidence, state_from_phase)
        else:
            state = state_from_confidence

        # Apply temporal smoothing (require min frames to change state)
        state = self._apply_temporal_smoothing(state)

        # Update history
        self.occlusion_history.append(state)

        # Update statistics
        if state == OcclusionState.FULLY_OCCLUDED:
            self.total_occluded_frames += 1

        return state

    def _detect_from_confidence(self, confidence: float) -> OcclusionState:
        """Detect occlusion based on confidence score.

        Args:
            confidence: Pose keypoint confidence.

        Returns:
            Occlusion state based on confidence.
        """
        if confidence >= self.confidence_threshold_high:
            return OcclusionState.VISIBLE
        elif confidence <= self.confidence_threshold_low:
            return OcclusionState.FULLY_OCCLUDED
        else:
            return OcclusionState.PARTIALLY_OCCLUDED

    def _detect_from_phase(self, stroke_phase: str) -> OcclusionState:
        """Detect occlusion based on stroke phase.

        During catch, pull, and push phases, hands are typically underwater.

        Args:
            stroke_phase: Current stroke phase.

        Returns:
            Occlusion state based on phase.
        """
        # Underwater phases
        underwater_phases = ["catch", "pull", "push"]
        # Above water phases
        visible_phases = ["entry", "recovery"]
        # Transitional phases
        transitional_phases = ["release"]

        phase_lower = stroke_phase.lower()

        if phase_lower in underwater_phases:
            return OcclusionState.FULLY_OCCLUDED
        elif phase_lower in visible_phases:
            return OcclusionState.VISIBLE
        elif phase_lower in transitional_phases:
            return OcclusionState.TRANSITIONING
        else:
            # Unknown phase, default to partially occluded
            return OcclusionState.PARTIALLY_OCCLUDED

    def _combine_detections(
        self,
        confidence_state: OcclusionState,
        phase_state: OcclusionState,
    ) -> OcclusionState:
        """Combine confidence and phase-based detections.

        Args:
            confidence_state: State from confidence detection.
            phase_state: State from phase detection.

        Returns:
            Combined state (conservative approach).
        """
        # If either method says fully occluded, trust it
        if (
            confidence_state == OcclusionState.FULLY_OCCLUDED
            or phase_state == OcclusionState.FULLY_OCCLUDED
        ):
            return OcclusionState.FULLY_OCCLUDED

        # If both say visible, it's visible
        if confidence_state == OcclusionState.VISIBLE and phase_state == OcclusionState.VISIBLE:
            return OcclusionState.VISIBLE

        # If transitioning according to phase
        if phase_state == OcclusionState.TRANSITIONING:
            return OcclusionState.TRANSITIONING

        # Otherwise, partially occluded
        return OcclusionState.PARTIALLY_OCCLUDED

    def _apply_temporal_smoothing(self, new_state: OcclusionState) -> OcclusionState:
        """Apply temporal smoothing to avoid rapid state changes.

        Args:
            new_state: Newly detected state.

        Returns:
            Smoothed state.
        """
        if new_state == self.current_state:
            # Same state, increment counter
            self.frames_in_current_state += 1
            return self.current_state
        else:
            # Different state
            self.frames_in_current_state = 1

            # Only change state if we've seen enough consistent frames
            # Exception: allow immediate transition to visible
            if (
                self.frames_in_current_state >= self.min_occlusion_frames
                or new_state == OcclusionState.VISIBLE
            ):
                # Track occlusion events
                if (
                    self.current_state != OcclusionState.FULLY_OCCLUDED
                    and new_state == OcclusionState.FULLY_OCCLUDED
                ):
                    self.total_occlusion_events += 1

                self.current_state = new_state
                return new_state
            else:
                # Not enough frames, keep current state
                return self.current_state

    def is_occluded(self) -> bool:
        """Check if currently occluded.

        Returns:
            True if fully or partially occluded.
        """
        return self.current_state in [
            OcclusionState.FULLY_OCCLUDED,
            OcclusionState.PARTIALLY_OCCLUDED,
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get occlusion statistics.

        Returns:
            Dictionary with statistics.
        """
        total_frames = len(self.occlusion_history)

        return {
            "total_frames": total_frames,
            "total_occlusion_events": self.total_occlusion_events,
            "total_occluded_frames": self.total_occluded_frames,
            "occlusion_percentage": (
                self.total_occluded_frames / total_frames * 100 if total_frames > 0 else 0.0
            ),
            "current_state": self.current_state.value,
        }

    def reset(self):
        """Reset detector state."""
        self.current_state = OcclusionState.VISIBLE
        self.frames_in_current_state = 0
        self.occlusion_history = []
        self.total_occlusion_events = 0
        self.total_occluded_frames = 0
