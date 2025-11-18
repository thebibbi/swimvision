"""Stroke phase detection for swimming analysis."""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from src.utils.config import load_analysis_config


class StrokePhase(Enum):
    """Enumeration of freestyle stroke phases."""

    ENTRY = "entry"  # Hand enters water
    CATCH = "catch"  # Hand catches water
    PULL = "pull"  # Hand pulls through water
    PUSH = "push"  # Hand pushes to completion
    RECOVERY = "recovery"  # Hand recovers above water


class StrokePhaseDetector:
    """Detect stroke phases from pose sequences."""

    def __init__(
        self,
        velocity_threshold: Optional[float] = None,
        min_phase_duration: Optional[float] = None,
    ):
        """Initialize stroke phase detector.

        Args:
            velocity_threshold: Velocity threshold for phase transitions.
            min_phase_duration: Minimum phase duration in seconds.
        """
        # Load configuration
        config = load_analysis_config()
        phase_config = config.get("stroke_phases", {})

        self.velocity_threshold = (
            velocity_threshold
            if velocity_threshold is not None
            else phase_config.get("velocity_threshold", 0.1)
        )
        self.min_phase_duration = (
            min_phase_duration
            if min_phase_duration is not None
            else phase_config.get("min_phase_duration", 0.2)
        )

    def detect_phases_freestyle(
        self,
        hand_positions: np.ndarray,
        fps: float,
        side: str = "right",
    ) -> List[Dict]:
        """Detect freestyle stroke phases from hand trajectory.

        Args:
            hand_positions: Hand positions over time (n_frames, 2 or 3).
            fps: Frames per second.
            side: Which hand ('left' or 'right').

        Returns:
            List of phase dictionaries with start_frame, end_frame, phase.
        """
        if len(hand_positions) < 10:
            return []

        # Calculate velocities
        velocities = self._calculate_velocities(hand_positions, fps)

        # Calculate vertical movement (y-axis)
        y_positions = hand_positions[:, 1]

        # Detect phase transitions
        phases = self._detect_phase_transitions(
            hand_positions, velocities, y_positions, fps
        )

        return phases

    def _calculate_velocities(
        self, positions: np.ndarray, fps: float
    ) -> np.ndarray:
        """Calculate velocity magnitude over time.

        Args:
            positions: Position array (n_frames, n_dims).
            fps: Frames per second.

        Returns:
            Velocity array (n_frames,).
        """
        velocities = np.zeros(len(positions))

        dt = 1.0 / fps

        for i in range(1, len(positions)):
            displacement = positions[i] - positions[i - 1]
            velocity = np.linalg.norm(displacement) / dt
            velocities[i] = velocity

        # Smooth velocities
        window_size = max(3, int(fps * 0.1))  # 0.1 second window
        velocities = self._smooth_signal(velocities, window_size)

        return velocities

    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth signal using moving average.

        Args:
            signal: Input signal.
            window_size: Window size for smoothing.

        Returns:
            Smoothed signal.
        """
        if window_size < 2:
            return signal

        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1

        half_window = window_size // 2
        smoothed = np.zeros_like(signal)

        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window + 1)
            smoothed[i] = np.mean(signal[start:end])

        return smoothed

    def _detect_phase_transitions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        y_positions: np.ndarray,
        fps: float,
    ) -> List[Dict]:
        """Detect phase transitions based on kinematics.

        Args:
            positions: Hand positions.
            velocities: Velocity magnitudes.
            y_positions: Vertical positions.
            fps: Frames per second.

        Returns:
            List of detected phases.
        """
        phases = []

        # Find local minima and maxima in velocity
        min_distance = int(fps * self.min_phase_duration)

        # Find velocity peaks (high movement)
        vel_peaks, _ = find_peaks(velocities, distance=min_distance)

        # Find velocity valleys (low movement)
        vel_valleys, _ = find_peaks(-velocities, distance=min_distance)

        # Find vertical position peaks (hand highest/lowest)
        y_peaks, _ = find_peaks(y_positions, distance=min_distance)
        y_valleys, _ = find_peaks(-y_positions, distance=min_distance)

        # Combine all key points
        key_frames = sorted(
            set(list(vel_peaks) + list(vel_valleys) + list(y_peaks) + list(y_valleys))
        )

        if len(key_frames) < 2:
            return []

        # Classify phases between key frames
        for i in range(len(key_frames) - 1):
            start_frame = key_frames[i]
            end_frame = key_frames[i + 1]

            if end_frame - start_frame < min_distance:
                continue

            # Analyze this segment
            segment_velocities = velocities[start_frame:end_frame]
            segment_y = y_positions[start_frame:end_frame]

            phase = self._classify_phase_segment(
                segment_velocities,
                segment_y,
                start_frame,
                end_frame,
            )

            phases.append(
                {
                    "phase": phase,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "duration": (end_frame - start_frame) / fps,
                }
            )

        return phases

    def _classify_phase_segment(
        self,
        velocities: np.ndarray,
        y_positions: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> StrokePhase:
        """Classify a phase segment based on kinematics.

        Args:
            velocities: Velocity values in segment.
            y_positions: Vertical positions in segment.
            start_frame: Start frame index.
            end_frame: End frame index.

        Returns:
            Detected stroke phase.
        """
        avg_velocity = np.mean(velocities)
        y_trend = y_positions[-1] - y_positions[0]  # Positive = moving down

        # Classification rules (heuristic)
        if avg_velocity < self.velocity_threshold:
            # Low velocity = entry or catch
            if y_trend > 0:
                return StrokePhase.ENTRY  # Moving down slowly
            else:
                return StrokePhase.CATCH  # Pausing at depth

        elif avg_velocity > self.velocity_threshold * 3:
            # High velocity = pull or push
            if abs(y_trend) < np.std(y_positions):
                return StrokePhase.PULL  # Moving horizontally
            else:
                return StrokePhase.PUSH  # Moving up/back

        else:
            # Medium velocity = recovery
            if y_trend < 0:
                return StrokePhase.RECOVERY  # Moving up/forward
            else:
                return StrokePhase.PULL  # Default to pull

    def detect_cycle_boundaries(
        self,
        hand_positions: np.ndarray,
        fps: float,
    ) -> List[Tuple[int, int]]:
        """Detect stroke cycle boundaries (start and end of each stroke).

        Args:
            hand_positions: Hand trajectory.
            fps: Frames per second.

        Returns:
            List of (start_frame, end_frame) tuples for each cycle.
        """
        if len(hand_positions) < 10:
            return []

        # Use vertical position peaks as cycle markers
        y_positions = hand_positions[:, 1]

        # Find peaks (hand at highest point = end of recovery)
        min_distance = int(fps * 0.5)  # Minimum 0.5s between strokes
        peaks, _ = find_peaks(y_positions, distance=min_distance)

        # Create cycles from consecutive peaks
        cycles = []
        for i in range(len(peaks) - 1):
            cycles.append((int(peaks[i]), int(peaks[i + 1])))

        return cycles

    def validate_phase_sequence(self, phases: List[Dict]) -> bool:
        """Validate that phase sequence is logical.

        Args:
            phases: List of detected phases.

        Returns:
            True if sequence is valid.
        """
        if len(phases) < 2:
            return True

        # Expected phase order (can repeat)
        expected_order = [
            StrokePhase.ENTRY,
            StrokePhase.CATCH,
            StrokePhase.PULL,
            StrokePhase.PUSH,
            StrokePhase.RECOVERY,
        ]

        # Check that phases generally follow expected order
        for i in range(len(phases) - 1):
            current_phase = phases[i]["phase"]
            next_phase = phases[i + 1]["phase"]

            # Get indices in expected order
            try:
                curr_idx = expected_order.index(current_phase)
                next_idx = expected_order.index(next_phase)

                # Allow wrapping around
                if next_idx < curr_idx and next_idx != 0:
                    return False  # Invalid transition

            except ValueError:
                return False  # Unknown phase

        return True

    def get_phase_durations(self, phases: List[Dict]) -> Dict[str, float]:
        """Calculate average duration for each phase.

        Args:
            phases: List of detected phases.

        Returns:
            Dictionary mapping phase names to average durations.
        """
        durations = {}

        for phase_type in StrokePhase:
            matching_phases = [
                p for p in phases if p["phase"] == phase_type
            ]

            if matching_phases:
                avg_duration = np.mean([p["duration"] for p in matching_phases])
                durations[phase_type.value] = float(avg_duration)

        return durations

    def analyze_phase_timing(
        self,
        phases_left: List[Dict],
        phases_right: List[Dict],
    ) -> Dict:
        """Analyze timing coordination between left and right arms.

        Args:
            phases_left: Phases for left arm.
            phases_right: Phases for right arm.

        Returns:
            Dictionary with timing analysis.
        """
        results = {}

        # Calculate phase overlap
        if phases_left and phases_right:
            # Find pull phases
            left_pulls = [
                p for p in phases_left if p["phase"] == StrokePhase.PULL
            ]
            right_pulls = [
                p for p in phases_right if p["phase"] == StrokePhase.PULL
            ]

            if left_pulls and right_pulls:
                # Check if pulls overlap (should be alternating)
                overlap_count = 0
                for left_pull in left_pulls:
                    for right_pull in right_pulls:
                        # Check overlap
                        if not (
                            left_pull["end_frame"] < right_pull["start_frame"]
                            or left_pull["start_frame"] > right_pull["end_frame"]
                        ):
                            overlap_count += 1

                results["pull_overlap_count"] = overlap_count
                results["pulls_alternate"] = overlap_count == 0

        return results
