"""Symmetry analysis for swimming biomechanics.

This module analyzes left/right symmetry in swimming technique:
- Arm symmetry (trajectory, timing, angles)
- Leg symmetry (kick timing, power)
- Temporal symmetry (stroke timing consistency)
- Force imbalance estimation
- Rotation imbalance
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

from src.analysis.dtw_analyzer import DTWAnalyzer
from src.utils.smoothing import smooth_trajectory_kalman, calculate_velocity, calculate_speed


class SymmetryAnalyzer:
    """Analyze symmetry in swimming technique."""

    def __init__(self, fps: float = 30.0):
        """Initialize symmetry analyzer.

        Args:
            fps: Video frame rate (frames per second).
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.dtw_analyzer = DTWAnalyzer()

    def analyze_arm_symmetry(
        self,
        left_hand_path: np.ndarray,
        right_hand_path: np.ndarray,
        left_elbow_angles: Optional[np.ndarray] = None,
        right_elbow_angles: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Analyze arm symmetry (trajectory and angles).

        Args:
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.
            left_elbow_angles: Array of left elbow angles over time.
            right_elbow_angles: Array of right elbow angles over time.

        Returns:
            Dictionary of symmetry metrics.
        """
        metrics = {}

        # Trajectory symmetry
        if len(left_hand_path) > 0 and len(right_hand_path) > 0:
            # Path length comparison
            left_length = self._calculate_path_length(left_hand_path)
            right_length = self._calculate_path_length(right_hand_path)

            metrics["left_path_length"] = left_length
            metrics["right_path_length"] = right_length

            # Path length asymmetry (%)
            if left_length + right_length > 0:
                asymmetry = abs(left_length - right_length) / (left_length + right_length) * 100
                metrics["path_length_asymmetry_pct"] = asymmetry
            else:
                metrics["path_length_asymmetry_pct"] = 0.0

            # DTW distance between trajectories (shape similarity)
            # Mirror right hand path for comparison
            right_mirrored = right_hand_path.copy()
            right_mirrored[:, 0] = -right_mirrored[:, 0]  # Mirror x-coordinate

            dtw_dist = self.dtw_analyzer.compute_dtw(left_hand_path, right_mirrored)
            metrics["trajectory_dtw_distance"] = dtw_dist

            # Velocity symmetry
            left_velocities = calculate_velocity(left_hand_path, self.dt, smooth=True)
            right_velocities = calculate_velocity(right_hand_path, self.dt, smooth=True)

            if len(left_velocities) > 0 and len(right_velocities) > 0:
                left_speeds = calculate_speed(left_velocities)
                right_speeds = calculate_speed(right_velocities)

                # Mean speed comparison
                left_mean_speed = np.mean(left_speeds)
                right_mean_speed = np.mean(right_speeds)

                metrics["left_mean_speed"] = left_mean_speed
                metrics["right_mean_speed"] = right_mean_speed

                # Speed asymmetry (%)
                if left_mean_speed + right_mean_speed > 0:
                    speed_asymmetry = abs(left_mean_speed - right_mean_speed) / (
                        left_mean_speed + right_mean_speed
                    ) * 100
                    metrics["speed_asymmetry_pct"] = speed_asymmetry
                else:
                    metrics["speed_asymmetry_pct"] = 0.0

                # Max speed comparison
                metrics["left_max_speed"] = np.max(left_speeds)
                metrics["right_max_speed"] = np.max(right_speeds)

        # Angular symmetry
        if left_elbow_angles is not None and right_elbow_angles is not None:
            # Remove NaN values
            valid_mask = ~(np.isnan(left_elbow_angles) | np.isnan(right_elbow_angles))
            left_valid = left_elbow_angles[valid_mask]
            right_valid = right_elbow_angles[valid_mask]

            if len(left_valid) > 0:
                # Mean angle comparison
                left_mean_angle = np.mean(left_valid)
                right_mean_angle = np.mean(right_valid)

                metrics["left_mean_elbow_angle"] = left_mean_angle
                metrics["right_mean_elbow_angle"] = right_mean_angle

                # Angle asymmetry
                angle_diff = np.abs(left_valid - right_valid)
                metrics["elbow_angle_asymmetry_mean"] = np.mean(angle_diff)
                metrics["elbow_angle_asymmetry_max"] = np.max(angle_diff)

                # Correlation between left and right angles
                if len(left_valid) > 1:
                    corr, _ = pearsonr(left_valid, right_valid)
                    metrics["elbow_angle_correlation"] = corr

        return metrics

    def analyze_leg_symmetry(
        self,
        left_ankle_path: Optional[np.ndarray] = None,
        right_ankle_path: Optional[np.ndarray] = None,
        left_knee_angles: Optional[np.ndarray] = None,
        right_knee_angles: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Analyze leg symmetry (kick timing and power).

        Args:
            left_ankle_path: Nx2 array of left ankle positions.
            right_ankle_path: Nx2 array of right ankle positions.
            left_knee_angles: Array of left knee angles over time.
            right_knee_angles: Array of right knee angles over time.

        Returns:
            Dictionary of leg symmetry metrics.
        """
        metrics = {}

        # Kick trajectory symmetry
        if left_ankle_path is not None and right_ankle_path is not None:
            if len(left_ankle_path) > 0 and len(right_ankle_path) > 0:
                # Path length comparison
                left_length = self._calculate_path_length(left_ankle_path)
                right_length = self._calculate_path_length(right_ankle_path)

                metrics["left_kick_length"] = left_length
                metrics["right_kick_length"] = right_length

                # Kick asymmetry (%)
                if left_length + right_length > 0:
                    asymmetry = abs(left_length - right_length) / (left_length + right_length) * 100
                    metrics["kick_length_asymmetry_pct"] = asymmetry
                else:
                    metrics["kick_length_asymmetry_pct"] = 0.0

        # Knee angle symmetry
        if left_knee_angles is not None and right_knee_angles is not None:
            # Remove NaN values
            valid_mask = ~(np.isnan(left_knee_angles) | np.isnan(right_knee_angles))
            left_valid = left_knee_angles[valid_mask]
            right_valid = right_knee_angles[valid_mask]

            if len(left_valid) > 0:
                # Mean angle comparison
                angle_diff = np.abs(left_valid - right_valid)
                metrics["knee_angle_asymmetry_mean"] = np.mean(angle_diff)

        return metrics

    def analyze_temporal_symmetry(
        self,
        left_hand_path: np.ndarray,
        right_hand_path: np.ndarray,
    ) -> Dict[str, float]:
        """Analyze temporal symmetry (stroke timing consistency).

        Args:
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.

        Returns:
            Dictionary of temporal symmetry metrics.
        """
        metrics = {}

        if len(left_hand_path) < 10 or len(right_hand_path) < 10:
            return metrics

        # Detect stroke cycles for each arm
        from scipy.signal import find_peaks

        # Use y-coordinate for peak detection
        left_y = left_hand_path[:, 1]
        right_y = right_hand_path[:, 1]

        # Find peaks (hand at highest point)
        left_peaks, _ = find_peaks(left_y, distance=int(0.5 * self.fps))
        right_peaks, _ = find_peaks(right_y, distance=int(0.5 * self.fps))

        metrics["left_num_strokes"] = len(left_peaks)
        metrics["right_num_strokes"] = len(right_peaks)

        # Stroke count asymmetry
        if len(left_peaks) + len(right_peaks) > 0:
            count_asymmetry = abs(len(left_peaks) - len(right_peaks)) / (
                len(left_peaks) + len(right_peaks)
            ) * 100
            metrics["stroke_count_asymmetry_pct"] = count_asymmetry
        else:
            metrics["stroke_count_asymmetry_pct"] = 0.0

        # Stroke timing consistency
        if len(left_peaks) >= 2:
            left_intervals = np.diff(left_peaks) / self.fps
            metrics["left_stroke_interval_mean"] = np.mean(left_intervals)
            metrics["left_stroke_interval_std"] = np.std(left_intervals)
            metrics["left_stroke_consistency"] = 1.0 - min(
                np.std(left_intervals) / np.mean(left_intervals), 1.0
            )

        if len(right_peaks) >= 2:
            right_intervals = np.diff(right_peaks) / self.fps
            metrics["right_stroke_interval_mean"] = np.mean(right_intervals)
            metrics["right_stroke_interval_std"] = np.std(right_intervals)
            metrics["right_stroke_consistency"] = 1.0 - min(
                np.std(right_intervals) / np.mean(right_intervals), 1.0
            )

        # Bilateral timing (phase difference between arms)
        if len(left_peaks) >= 1 and len(right_peaks) >= 1:
            # Find closest right peak for each left peak
            phase_diffs = []
            for left_peak in left_peaks:
                closest_right = right_peaks[np.argmin(np.abs(right_peaks - left_peak))]
                phase_diff = abs(left_peak - closest_right) / self.fps
                phase_diffs.append(phase_diff)

            metrics["bilateral_phase_difference_mean"] = np.mean(phase_diffs)
            metrics["bilateral_phase_difference_std"] = np.std(phase_diffs)

        return metrics

    def estimate_force_imbalance(
        self,
        left_hand_path: np.ndarray,
        right_hand_path: np.ndarray,
    ) -> Dict[str, float]:
        """Estimate force imbalance from hand trajectories.

        Note: This is an indirect estimate based on velocity and acceleration.
        True force measurement requires pressure sensors.

        Args:
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.

        Returns:
            Dictionary of force imbalance estimates.
        """
        metrics = {}

        if len(left_hand_path) < 3 or len(right_hand_path) < 3:
            return metrics

        # Smooth trajectories
        left_smooth, _ = smooth_trajectory_kalman(left_hand_path, dt=self.dt)
        right_smooth, _ = smooth_trajectory_kalman(right_hand_path, dt=self.dt)

        # Calculate velocities
        from src.utils.smoothing import calculate_acceleration

        left_accel = calculate_acceleration(left_smooth, dt=self.dt, smooth=True)
        right_accel = calculate_acceleration(right_smooth, dt=self.dt, smooth=True)

        if len(left_accel) > 0 and len(right_accel) > 0:
            # Acceleration magnitude (proxy for force)
            left_accel_mag = calculate_speed(left_accel)
            right_accel_mag = calculate_speed(right_accel)

            # Mean acceleration comparison
            left_mean_accel = np.mean(left_accel_mag)
            right_mean_accel = np.mean(right_accel_mag)

            metrics["left_mean_acceleration"] = left_mean_accel
            metrics["right_mean_acceleration"] = right_mean_accel

            # Force imbalance estimate (%)
            if left_mean_accel + right_mean_accel > 0:
                force_imbalance = abs(left_mean_accel - right_mean_accel) / (
                    left_mean_accel + right_mean_accel
                ) * 100
                metrics["force_imbalance_pct"] = force_imbalance
            else:
                metrics["force_imbalance_pct"] = 0.0

            # Peak force comparison
            metrics["left_peak_acceleration"] = np.max(left_accel_mag)
            metrics["right_peak_acceleration"] = np.max(right_accel_mag)

        return metrics

    def estimate_rotation_imbalance(
        self,
        shoulder_angles: Optional[Dict[str, np.ndarray]] = None,
        hip_angles: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Estimate body rotation imbalance.

        Args:
            shoulder_angles: Dictionary with 'left' and 'right' shoulder angles.
            hip_angles: Dictionary with 'left' and 'right' hip angles.

        Returns:
            Dictionary of rotation imbalance metrics.
        """
        metrics = {}

        # Shoulder rotation imbalance
        if shoulder_angles is not None:
            if "left" in shoulder_angles and "right" in shoulder_angles:
                left = shoulder_angles["left"]
                right = shoulder_angles["right"]

                # Remove NaN values
                valid_mask = ~(np.isnan(left) | np.isnan(right))
                left_valid = left[valid_mask]
                right_valid = right[valid_mask]

                if len(left_valid) > 0:
                    # Rotation asymmetry (difference between sides)
                    rotation_diff = left_valid - right_valid
                    metrics["shoulder_rotation_bias"] = np.mean(rotation_diff)
                    metrics["shoulder_rotation_variability"] = np.std(rotation_diff)

        # Hip rotation imbalance
        if hip_angles is not None:
            if "left" in hip_angles and "right" in hip_angles:
                left = hip_angles["left"]
                right = hip_angles["right"]

                # Remove NaN values
                valid_mask = ~(np.isnan(left) | np.isnan(right))
                left_valid = left[valid_mask]
                right_valid = right[valid_mask]

                if len(left_valid) > 0:
                    # Rotation asymmetry
                    rotation_diff = left_valid - right_valid
                    metrics["hip_rotation_bias"] = np.mean(rotation_diff)
                    metrics["hip_rotation_variability"] = np.std(rotation_diff)

        return metrics

    def comprehensive_symmetry_analysis(
        self,
        left_hand_path: np.ndarray,
        right_hand_path: np.ndarray,
        angles_over_time: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, any]:
        """Perform comprehensive symmetry analysis.

        Args:
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.
            angles_over_time: Dictionary of joint angles over time.

        Returns:
            Dictionary with all symmetry metrics and interpretation.
        """
        results = {}

        # Arm symmetry
        left_elbow = angles_over_time.get("left_elbow") if angles_over_time else None
        right_elbow = angles_over_time.get("right_elbow") if angles_over_time else None

        arm_symmetry = self.analyze_arm_symmetry(
            left_hand_path, right_hand_path, left_elbow, right_elbow
        )
        results["arm_symmetry"] = arm_symmetry

        # Temporal symmetry
        temporal_symmetry = self.analyze_temporal_symmetry(
            left_hand_path, right_hand_path
        )
        results["temporal_symmetry"] = temporal_symmetry

        # Force imbalance
        force_imbalance = self.estimate_force_imbalance(
            left_hand_path, right_hand_path
        )
        results["force_imbalance"] = force_imbalance

        # Overall symmetry score (0-100, 100 = perfect symmetry)
        overall_score = self._calculate_overall_symmetry_score(
            arm_symmetry, temporal_symmetry, force_imbalance
        )
        results["overall_symmetry_score"] = overall_score

        # Interpretation
        results["interpretation"] = self._interpret_symmetry_score(overall_score)

        # Recommendations
        results["recommendations"] = self._generate_symmetry_recommendations(
            arm_symmetry, temporal_symmetry, force_imbalance
        )

        return results

    def _calculate_path_length(self, path: np.ndarray) -> float:
        """Calculate total path length.

        Args:
            path: Nx2 array of positions.

        Returns:
            Total path length.
        """
        if len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            length += euclidean(path[i], path[i + 1])
        return length

    def _calculate_overall_symmetry_score(
        self,
        arm_symmetry: Dict[str, float],
        temporal_symmetry: Dict[str, float],
        force_imbalance: Dict[str, float],
    ) -> float:
        """Calculate overall symmetry score (0-100).

        Args:
            arm_symmetry: Arm symmetry metrics.
            temporal_symmetry: Temporal symmetry metrics.
            force_imbalance: Force imbalance metrics.

        Returns:
            Overall symmetry score (100 = perfect).
        """
        scores = []

        # Path length symmetry (weight: 30%)
        if "path_length_asymmetry_pct" in arm_symmetry:
            # Convert asymmetry % to score (0% = 100, 20% = 0)
            path_score = max(0, 100 - arm_symmetry["path_length_asymmetry_pct"] * 5)
            scores.append((path_score, 0.3))

        # Speed symmetry (weight: 20%)
        if "speed_asymmetry_pct" in arm_symmetry:
            speed_score = max(0, 100 - arm_symmetry["speed_asymmetry_pct"] * 5)
            scores.append((speed_score, 0.2))

        # Temporal symmetry (weight: 25%)
        if "stroke_count_asymmetry_pct" in temporal_symmetry:
            temporal_score = max(0, 100 - temporal_symmetry["stroke_count_asymmetry_pct"] * 5)
            scores.append((temporal_score, 0.25))

        # Force balance (weight: 25%)
        if "force_imbalance_pct" in force_imbalance:
            force_score = max(0, 100 - force_imbalance["force_imbalance_pct"] * 5)
            scores.append((force_score, 0.25))

        # Calculate weighted average
        if scores:
            total_weight = sum(weight for _, weight in scores)
            overall = sum(score * weight for score, weight in scores) / total_weight
            return overall
        else:
            return 50.0  # Default if no metrics available

    def _interpret_symmetry_score(self, score: float) -> str:
        """Interpret symmetry score.

        Args:
            score: Symmetry score (0-100).

        Returns:
            Interpretation string.
        """
        if score >= 90:
            return "Excellent symmetry - very balanced technique"
        elif score >= 75:
            return "Good symmetry - minor imbalances present"
        elif score >= 60:
            return "Moderate symmetry - noticeable imbalances that should be addressed"
        elif score >= 40:
            return "Poor symmetry - significant imbalances requiring correction"
        else:
            return "Very poor symmetry - major imbalances present, high injury risk"

    def _generate_symmetry_recommendations(
        self,
        arm_symmetry: Dict[str, float],
        temporal_symmetry: Dict[str, float],
        force_imbalance: Dict[str, float],
    ) -> List[str]:
        """Generate recommendations based on symmetry analysis.

        Args:
            arm_symmetry: Arm symmetry metrics.
            temporal_symmetry: Temporal symmetry metrics.
            force_imbalance: Force imbalance metrics.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Path length asymmetry
        if arm_symmetry.get("path_length_asymmetry_pct", 0) > 15:
            recommendations.append(
                "⚠️ Significant path length asymmetry detected. "
                "Focus on making both arms travel equal distances during the stroke."
            )

        # Speed asymmetry
        if arm_symmetry.get("speed_asymmetry_pct", 0) > 15:
            recommendations.append(
                "⚠️ Speed imbalance between arms. "
                "Work on equalizing stroke speed for both arms."
            )

        # Temporal asymmetry
        if temporal_symmetry.get("stroke_count_asymmetry_pct", 0) > 10:
            recommendations.append(
                "⚠️ Stroke count imbalance. "
                "Ensure both arms are performing equal numbers of strokes."
            )

        # Force imbalance
        if force_imbalance.get("force_imbalance_pct", 0) > 20:
            recommendations.append(
                "⚠️ Force imbalance detected. "
                "Focus on generating equal power with both arms. Consider strength training for the weaker side."
            )

        # Angular asymmetry
        if arm_symmetry.get("elbow_angle_asymmetry_mean", 0) > 20:
            recommendations.append(
                "⚠️ Elbow angle asymmetry. "
                "Work on maintaining similar elbow angles throughout the stroke on both sides."
            )

        if not recommendations:
            recommendations.append("✅ Good symmetry overall! Continue maintaining balanced technique.")

        return recommendations
