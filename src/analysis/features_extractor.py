"""Biomechanical feature extraction for swimming analysis.

This module extracts comprehensive biomechanical features from swimming pose sequences:
- Temporal features (stroke rate, cycle time, tempo)
- Kinematic features (velocities, accelerations, speed)
- Angular features (joint angles, angular velocities)
- Symmetry features (left/right balance)
- Spatial features (hand paths, body alignment)
- Injury risk features (shoulder angles, asymmetry, workload)
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean

from src.utils.smoothing import (
    calculate_acceleration,
    calculate_speed,
    calculate_velocity,
    smooth_signal_savgol,
    smooth_trajectory_kalman,
)


class FeaturesExtractor:
    """Extract biomechanical features from swimming pose sequences."""

    def __init__(self, fps: float = 30.0):
        """Initialize features extractor.

        Args:
            fps: Video frame rate (frames per second).
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time step between frames

    def extract_stroke_features(
        self,
        pose_sequence: list[dict],
        left_hand_path: np.ndarray | None = None,
        right_hand_path: np.ndarray | None = None,
        angles_over_time: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Extract comprehensive stroke features.

        Args:
            pose_sequence: List of pose dictionaries.
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.
            angles_over_time: Dictionary of joint angles over time.

        Returns:
            Dictionary of features and their values.
        """
        features = {}

        # Temporal features
        temporal = self._extract_temporal_features(left_hand_path, right_hand_path)
        features.update(temporal)

        # Kinematic features
        if left_hand_path is not None and len(left_hand_path) > 0:
            kinematic_left = self._extract_kinematic_features(left_hand_path, prefix="left_hand")
            features.update(kinematic_left)

        if right_hand_path is not None and len(right_hand_path) > 0:
            kinematic_right = self._extract_kinematic_features(right_hand_path, prefix="right_hand")
            features.update(kinematic_right)

        # Angular features
        if angles_over_time is not None:
            angular = self._extract_angular_features(angles_over_time)
            features.update(angular)

        # Spatial features
        if left_hand_path is not None and right_hand_path is not None:
            spatial = self._extract_spatial_features(left_hand_path, right_hand_path)
            features.update(spatial)

        # Symmetry features
        if left_hand_path is not None and right_hand_path is not None:
            symmetry = self._extract_symmetry_features(
                left_hand_path, right_hand_path, angles_over_time
            )
            features.update(symmetry)

        return features

    def _extract_temporal_features(
        self,
        left_hand_path: np.ndarray | None,
        right_hand_path: np.ndarray | None,
    ) -> dict[str, float]:
        """Extract temporal features (stroke rate, cycle time).

        Args:
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.

        Returns:
            Dictionary of temporal features.
        """
        features = {}

        # Use left hand by default
        hand_path = left_hand_path if left_hand_path is not None else right_hand_path

        if hand_path is None or len(hand_path) < 10:
            return {
                "stroke_rate": 0.0,
                "stroke_cycle_time": 0.0,
                "tempo": 0.0,
                "num_strokes": 0.0,
            }

        # Detect stroke cycles by finding peaks in y-coordinate (vertical motion)
        y_coords = hand_path[:, 1]

        # Smooth y-coords for better peak detection
        if len(y_coords) > 5:
            y_smooth = smooth_signal_savgol(y_coords, window_length=5, polyorder=2)
        else:
            y_smooth = y_coords

        # Find peaks (hand at highest point in stroke)
        peaks, _ = find_peaks(y_smooth, distance=int(0.5 * self.fps))  # Min 0.5s between strokes

        num_strokes = len(peaks)
        features["num_strokes"] = float(num_strokes)

        if num_strokes >= 2:
            # Calculate stroke cycle time (average time between strokes)
            cycle_times = np.diff(peaks) / self.fps
            avg_cycle_time = np.mean(cycle_times)
            features["stroke_cycle_time"] = avg_cycle_time

            # Stroke rate (strokes per minute)
            features["stroke_rate"] = 60.0 / avg_cycle_time if avg_cycle_time > 0 else 0.0

            # Tempo (ratio of fastest to slowest stroke)
            if len(cycle_times) > 1:
                tempo_consistency = (
                    np.std(cycle_times) / np.mean(cycle_times) if np.mean(cycle_times) > 0 else 0.0
                )
                features["tempo"] = 1.0 - min(tempo_consistency, 1.0)  # 1.0 = perfect consistency
            else:
                features["tempo"] = 1.0

            # Cycle time variability
            features["cycle_time_std"] = np.std(cycle_times)
        else:
            features["stroke_cycle_time"] = 0.0
            features["stroke_rate"] = 0.0
            features["tempo"] = 0.0
            features["cycle_time_std"] = 0.0

        return features

    def _extract_kinematic_features(
        self,
        trajectory: np.ndarray,
        prefix: str = "hand",
    ) -> dict[str, float]:
        """Extract kinematic features (velocity, acceleration, speed).

        Args:
            trajectory: Nx2 array of positions.
            prefix: Feature name prefix.

        Returns:
            Dictionary of kinematic features.
        """
        features = {}

        if len(trajectory) < 2:
            return features

        # Smooth trajectory first
        smoothed, _ = smooth_trajectory_kalman(trajectory, dt=self.dt)

        # Calculate velocities
        velocities = calculate_velocity(smoothed, dt=self.dt, smooth=True)
        if len(velocities) > 0:
            speeds = calculate_speed(velocities)

            features[f"{prefix}_mean_velocity"] = np.mean(speeds)
            features[f"{prefix}_max_velocity"] = np.max(speeds)
            features[f"{prefix}_min_velocity"] = np.min(speeds)
            features[f"{prefix}_velocity_std"] = np.std(speeds)

        # Calculate accelerations
        accelerations = calculate_acceleration(smoothed, dt=self.dt, smooth=True)
        if len(accelerations) > 0:
            accel_magnitudes = calculate_speed(accelerations)

            features[f"{prefix}_mean_acceleration"] = np.mean(accel_magnitudes)
            features[f"{prefix}_max_acceleration"] = np.max(accel_magnitudes)
            features[f"{prefix}_acceleration_std"] = np.std(accel_magnitudes)

        # Path length (total distance traveled)
        path_length = 0.0
        for i in range(len(smoothed) - 1):
            path_length += euclidean(smoothed[i], smoothed[i + 1])
        features[f"{prefix}_path_length"] = path_length

        # Displacement (straight-line distance from start to end)
        displacement = euclidean(smoothed[0], smoothed[-1])
        features[f"{prefix}_displacement"] = displacement

        # Path efficiency (displacement / path_length)
        if path_length > 0:
            features[f"{prefix}_path_efficiency"] = displacement / path_length
        else:
            features[f"{prefix}_path_efficiency"] = 0.0

        return features

    def _extract_angular_features(
        self,
        angles_over_time: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Extract angular features from joint angles.

        Args:
            angles_over_time: Dictionary of joint angles over time.

        Returns:
            Dictionary of angular features.
        """
        features = {}

        for joint_name, angles in angles_over_time.items():
            # Remove NaN values
            valid_angles = angles[~np.isnan(angles)]

            if len(valid_angles) == 0:
                continue

            # Basic statistics
            features[f"{joint_name}_mean"] = np.mean(valid_angles)
            features[f"{joint_name}_max"] = np.max(valid_angles)
            features[f"{joint_name}_min"] = np.min(valid_angles)
            features[f"{joint_name}_std"] = np.std(valid_angles)
            features[f"{joint_name}_range"] = np.max(valid_angles) - np.min(valid_angles)

            # Angular velocity (rate of change)
            if len(valid_angles) > 1:
                angular_velocity = np.abs(np.diff(valid_angles)) / self.dt
                features[f"{joint_name}_angular_velocity_mean"] = np.mean(angular_velocity)
                features[f"{joint_name}_angular_velocity_max"] = np.max(angular_velocity)

        return features

    def _extract_spatial_features(
        self,
        left_hand_path: np.ndarray,
        right_hand_path: np.ndarray,
    ) -> dict[str, float]:
        """Extract spatial features (hand positions, body alignment).

        Args:
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.

        Returns:
            Dictionary of spatial features.
        """
        features = {}

        # Hand separation (average distance between hands)
        if len(left_hand_path) == len(right_hand_path):
            separations = [
                euclidean(left_hand_path[i], right_hand_path[i]) for i in range(len(left_hand_path))
            ]
            features["hand_separation_mean"] = np.mean(separations)
            features["hand_separation_std"] = np.std(separations)
            features["hand_separation_max"] = np.max(separations)
            features["hand_separation_min"] = np.min(separations)

        # Hand path width (lateral spread)
        if len(left_hand_path) > 0:
            features["left_hand_width"] = np.max(left_hand_path[:, 0]) - np.min(
                left_hand_path[:, 0]
            )
            features["left_hand_depth"] = np.max(left_hand_path[:, 1]) - np.min(
                left_hand_path[:, 1]
            )

        if len(right_hand_path) > 0:
            features["right_hand_width"] = np.max(right_hand_path[:, 0]) - np.min(
                right_hand_path[:, 0]
            )
            features["right_hand_depth"] = np.max(right_hand_path[:, 1]) - np.min(
                right_hand_path[:, 1]
            )

        return features

    def _extract_symmetry_features(
        self,
        left_hand_path: np.ndarray,
        right_hand_path: np.ndarray,
        angles_over_time: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Extract symmetry features (left/right balance).

        Args:
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.
            angles_over_time: Dictionary of joint angles over time.

        Returns:
            Dictionary of symmetry features.
        """
        features = {}

        # Hand path symmetry
        if len(left_hand_path) > 0 and len(right_hand_path) > 0:
            left_length = 0.0
            for i in range(len(left_hand_path) - 1):
                left_length += euclidean(left_hand_path[i], left_hand_path[i + 1])

            right_length = 0.0
            for i in range(len(right_hand_path) - 1):
                right_length += euclidean(right_hand_path[i], right_hand_path[i + 1])

            # Path length asymmetry (%)
            if left_length + right_length > 0:
                asymmetry = abs(left_length - right_length) / (left_length + right_length) * 100
                features["path_length_asymmetry"] = asymmetry

        # Angular symmetry
        if angles_over_time is not None:
            # Compare left and right elbow angles
            if "left_elbow" in angles_over_time and "right_elbow" in angles_over_time:
                left_elbow = angles_over_time["left_elbow"]
                right_elbow = angles_over_time["right_elbow"]

                # Remove NaN values
                valid_mask = ~(np.isnan(left_elbow) | np.isnan(right_elbow))
                left_valid = left_elbow[valid_mask]
                right_valid = right_elbow[valid_mask]

                if len(left_valid) > 0:
                    elbow_diff = np.abs(left_valid - right_valid)
                    features["elbow_angle_asymmetry"] = np.mean(elbow_diff)

            # Compare left and right shoulder angles
            if "left_shoulder" in angles_over_time and "right_shoulder" in angles_over_time:
                left_shoulder = angles_over_time["left_shoulder"]
                right_shoulder = angles_over_time["right_shoulder"]

                # Remove NaN values
                valid_mask = ~(np.isnan(left_shoulder) | np.isnan(right_shoulder))
                left_valid = left_shoulder[valid_mask]
                right_valid = right_shoulder[valid_mask]

                if len(left_valid) > 0:
                    shoulder_diff = np.abs(left_valid - right_valid)
                    features["shoulder_angle_asymmetry"] = np.mean(shoulder_diff)

        return features

    def extract_injury_risk_features(
        self,
        pose_sequence: list[dict],
        angles_over_time: dict[str, np.ndarray],
        left_hand_path: np.ndarray | None = None,
        right_hand_path: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Extract features specific to injury risk prediction.

        Args:
            pose_sequence: List of pose dictionaries.
            angles_over_time: Dictionary of joint angles over time.
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.

        Returns:
            Dictionary of injury risk features.
        """
        features = {}

        # Shoulder hyperextension risk
        if "left_shoulder" in angles_over_time:
            left_shoulder = angles_over_time["left_shoulder"]
            valid_angles = left_shoulder[~np.isnan(left_shoulder)]

            if len(valid_angles) > 0:
                # Count frames with extreme shoulder angles (>170° or <30°)
                extreme_angles = np.sum((valid_angles > 170) | (valid_angles < 30))
                features["left_shoulder_extreme_angle_pct"] = (
                    extreme_angles / len(valid_angles)
                ) * 100

        if "right_shoulder" in angles_over_time:
            right_shoulder = angles_over_time["right_shoulder"]
            valid_angles = right_shoulder[~np.isnan(right_shoulder)]

            if len(valid_angles) > 0:
                extreme_angles = np.sum((valid_angles > 170) | (valid_angles < 30))
                features["right_shoulder_extreme_angle_pct"] = (
                    extreme_angles / len(valid_angles)
                ) * 100

        # Elbow drop (elbow angle too small during pull)
        if "left_elbow" in angles_over_time:
            left_elbow = angles_over_time["left_elbow"]
            valid_angles = left_elbow[~np.isnan(left_elbow)]

            if len(valid_angles) > 0:
                # Count frames with elbow angle < 90° (elbow drop)
                dropped_elbow = np.sum(valid_angles < 90)
                features["left_elbow_drop_pct"] = (dropped_elbow / len(valid_angles)) * 100

        if "right_elbow" in angles_over_time:
            right_elbow = angles_over_time["right_elbow"]
            valid_angles = right_elbow[~np.isnan(right_elbow)]

            if len(valid_angles) > 0:
                dropped_elbow = np.sum(valid_angles < 90)
                features["right_elbow_drop_pct"] = (dropped_elbow / len(valid_angles)) * 100

        # Asymmetry (risk factor)
        symmetry_features = self._extract_symmetry_features(
            left_hand_path, right_hand_path, angles_over_time
        )

        # High asymmetry is a risk factor
        if "path_length_asymmetry" in symmetry_features:
            features["asymmetry_risk_score"] = min(
                symmetry_features["path_length_asymmetry"] / 10.0, 10.0
            )

        # Workload (total path length - proxy for training volume)
        if left_hand_path is not None and len(left_hand_path) > 1:
            left_length = 0.0
            for i in range(len(left_hand_path) - 1):
                left_length += euclidean(left_hand_path[i], left_hand_path[i + 1])
            features["left_hand_workload"] = left_length

        if right_hand_path is not None and len(right_hand_path) > 1:
            right_length = 0.0
            for i in range(len(right_hand_path) - 1):
                right_length += euclidean(right_hand_path[i], right_hand_path[i + 1])
            features["right_hand_workload"] = right_length

        return features

    def extract_all_features(
        self,
        pose_sequence: list[dict],
        left_hand_path: np.ndarray | None = None,
        right_hand_path: np.ndarray | None = None,
        angles_over_time: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Extract all features (stroke + injury risk).

        Args:
            pose_sequence: List of pose dictionaries.
            left_hand_path: Nx2 array of left hand positions.
            right_hand_path: Nx2 array of right hand positions.
            angles_over_time: Dictionary of joint angles over time.

        Returns:
            Dictionary of all features.
        """
        stroke_features = self.extract_stroke_features(
            pose_sequence, left_hand_path, right_hand_path, angles_over_time
        )

        injury_features = self.extract_injury_risk_features(
            pose_sequence, angles_over_time, left_hand_path, right_hand_path
        )

        # Combine all features
        all_features = {**stroke_features, **injury_features}

        return all_features


def format_features_table(features: dict[str, float]) -> str:
    """Format features as a readable table.

    Args:
        features: Dictionary of features.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Feature':<50} {'Value':>15}")
    lines.append("=" * 80)

    # Group features by category
    categories = {
        "Temporal": ["stroke_rate", "stroke_cycle_time", "tempo", "num_strokes", "cycle_time_std"],
        "Kinematic (Left Hand)": [
            k
            for k in features
            if k.startswith("left_hand") and "velocity" in k or "acceleration" in k or "path" in k
        ],
        "Kinematic (Right Hand)": [
            k
            for k in features
            if k.startswith("right_hand") and "velocity" in k or "acceleration" in k or "path" in k
        ],
        "Angular (Elbows)": [k for k in features if "elbow" in k and "drop" not in k],
        "Angular (Shoulders)": [k for k in features if "shoulder" in k and "extreme" not in k],
        "Spatial": [k for k in features if "hand_separation" in k or "width" in k or "depth" in k],
        "Symmetry": [k for k in features if "asymmetry" in k],
        "Injury Risk": [
            k for k in features if "extreme" in k or "drop" in k or "risk" in k or "workload" in k
        ],
    }

    for category, feature_keys in categories.items():
        if not feature_keys:
            continue

        lines.append(f"\n{category}:")
        lines.append("-" * 80)

        for key in feature_keys:
            if key in features:
                value = features[key]
                lines.append(f"{key:<50} {value:>15.2f}")

    lines.append("=" * 80)

    return "\n".join(lines)
