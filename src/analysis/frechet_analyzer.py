"""Fréchet distance analysis for trajectory comparison."""

from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import euclidean
from similaritymeasures import frechet_dist


class FrechetAnalyzer:
    """Fréchet distance analyzer for hand path and trajectory comparison."""

    def __init__(self):
        """Initialize Fréchet analyzer."""
        pass

    def compute_distance(
        self,
        trajectory1: np.ndarray,
        trajectory2: np.ndarray,
    ) -> float:
        """Compute discrete Fréchet distance between two trajectories.

        Args:
            trajectory1: First trajectory (n_points, n_dims).
            trajectory2: Second trajectory (m_points, n_dims).

        Returns:
            Fréchet distance.
        """
        # Ensure 2D arrays
        if trajectory1.ndim == 1:
            trajectory1 = trajectory1.reshape(-1, 1)
        if trajectory2.ndim == 1:
            trajectory2 = trajectory2.reshape(-1, 1)

        # Compute Fréchet distance using library
        distance = frechet_dist(trajectory1, trajectory2)

        return float(distance)

    def compare_hand_paths(
        self,
        left_path1: List[Tuple[float, float]],
        right_path1: List[Tuple[float, float]],
        left_path2: List[Tuple[float, float]],
        right_path2: List[Tuple[float, float]],
    ) -> dict:
        """Compare hand paths between two swimmers.

        Args:
            left_path1: Left hand path of first swimmer.
            right_path1: Right hand path of first swimmer.
            left_path2: Left hand path of second swimmer.
            right_path2: Right hand path of second swimmer.

        Returns:
            Dictionary with Fréchet distances for each hand and overall.
        """
        results = {}

        # Convert to numpy arrays
        left1 = np.array(left_path1)
        right1 = np.array(right_path1)
        left2 = np.array(left_path2)
        right2 = np.array(right_path2)

        # Compare left hands
        if len(left1) > 0 and len(left2) > 0:
            results["left_hand_distance"] = self.compute_distance(left1, left2)

        # Compare right hands
        if len(right1) > 0 and len(right2) > 0:
            results["right_hand_distance"] = self.compute_distance(right1, right2)

        # Compute overall distance (average)
        if "left_hand_distance" in results and "right_hand_distance" in results:
            results["overall_distance"] = (
                results["left_hand_distance"] + results["right_hand_distance"]
            ) / 2.0

        return results

    def analyze_trajectory_shape(
        self,
        trajectory: np.ndarray,
    ) -> dict:
        """Analyze shape characteristics of a trajectory.

        Args:
            trajectory: Trajectory array (n_points, 2 or 3).

        Returns:
            Dictionary with shape metrics.
        """
        if len(trajectory) < 2:
            return {}

        metrics = {}

        # Total path length
        path_length = 0.0
        for i in range(len(trajectory) - 1):
            path_length += euclidean(trajectory[i], trajectory[i + 1])
        metrics["path_length"] = path_length

        # Straight-line distance (start to end)
        straight_distance = euclidean(trajectory[0], trajectory[-1])
        metrics["straight_distance"] = straight_distance

        # Path efficiency (straightness)
        if path_length > 0:
            metrics["path_efficiency"] = straight_distance / path_length
        else:
            metrics["path_efficiency"] = 0.0

        # Bounding box dimensions
        min_coords = np.min(trajectory, axis=0)
        max_coords = np.max(trajectory, axis=0)
        metrics["bbox_width"] = float(max_coords[0] - min_coords[0])
        metrics["bbox_height"] = float(max_coords[1] - min_coords[1])

        # Centroid
        centroid = np.mean(trajectory, axis=0)
        metrics["centroid"] = centroid.tolist()

        # Average distance from centroid (spread)
        distances_from_centroid = [euclidean(pt, centroid) for pt in trajectory]
        metrics["avg_spread"] = float(np.mean(distances_from_centroid))

        return metrics

    def classify_pull_pattern(
        self,
        hand_path: np.ndarray,
    ) -> str:
        """Classify hand pull pattern (S-pull, I-pull, straight).

        Args:
            hand_path: Hand trajectory (n_points, 2).

        Returns:
            Pull pattern classification.
        """
        if len(hand_path) < 3:
            return "unknown"

        # Analyze shape metrics
        metrics = self.analyze_trajectory_shape(hand_path)
        efficiency = metrics.get("path_efficiency", 0.0)

        # Calculate curvature indicator
        # Check if path curves significantly
        curvature = self._estimate_curvature(hand_path)

        # Classification thresholds
        if efficiency > 0.85:
            return "straight"  # Very direct path
        elif curvature > 0.3:
            return "S-pull"  # High curvature (S-shaped)
        elif efficiency > 0.65:
            return "I-pull"  # Moderately straight
        else:
            return "irregular"  # Complex or inefficient pattern

    def _estimate_curvature(self, trajectory: np.ndarray) -> float:
        """Estimate trajectory curvature.

        Args:
            trajectory: Path array (n_points, 2).

        Returns:
            Curvature metric (0 = straight, higher = more curved).
        """
        if len(trajectory) < 3:
            return 0.0

        # Calculate changes in direction
        curvatures = []

        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i - 1]
            v2 = trajectory[i + 1] - trajectory[i]

            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm

                # Calculate angle change
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)

        if len(curvatures) > 0:
            # Return average curvature
            return float(np.mean(curvatures))
        else:
            return 0.0

    def compare_stroke_trajectories(
        self,
        trajectory1: np.ndarray,
        trajectory2: np.ndarray,
    ) -> dict:
        """Comprehensive trajectory comparison.

        Args:
            trajectory1: First trajectory.
            trajectory2: Second trajectory.

        Returns:
            Dictionary with comparison metrics.
        """
        results = {}

        # Fréchet distance
        results["frechet_distance"] = self.compute_distance(trajectory1, trajectory2)

        # Shape analysis for both
        shape1 = self.analyze_trajectory_shape(trajectory1)
        shape2 = self.analyze_trajectory_shape(trajectory2)

        # Compare shape metrics
        if shape1 and shape2:
            results["path_length_diff"] = abs(
                shape1["path_length"] - shape2["path_length"]
            )
            results["efficiency_diff"] = abs(
                shape1["path_efficiency"] - shape2["path_efficiency"]
            )

        # Pull pattern classification
        pattern1 = self.classify_pull_pattern(trajectory1)
        pattern2 = self.classify_pull_pattern(trajectory2)

        results["pattern1"] = pattern1
        results["pattern2"] = pattern2
        results["patterns_match"] = pattern1 == pattern2

        return results

    def compute_similarity_score(
        self,
        distance: float,
        max_distance: float = 100.0,
    ) -> float:
        """Convert Fréchet distance to similarity score (0-100).

        Args:
            distance: Fréchet distance.
            max_distance: Maximum expected distance.

        Returns:
            Similarity score (0-100).
        """
        distance = min(distance, max_distance)
        similarity = (1.0 - distance / max_distance) * 100.0
        return max(0.0, min(100.0, similarity))
