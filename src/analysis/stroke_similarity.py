"""Comprehensive stroke similarity analysis ensemble."""

from typing import Dict, List, Optional

import numpy as np

from src.analysis.dtw_analyzer import DTWAnalyzer
from src.analysis.frechet_analyzer import FrechetAnalyzer
from src.analysis.similarity_measures import (
    CosineSimilarityAnalyzer,
    CrossCorrelationAnalyzer,
    EuclideanDistanceAnalyzer,
    LCSS,
    SoftDTW,
)
from src.analysis.stroke_phases import StrokePhaseDetector
from src.utils.config import load_analysis_config


class StrokeSimilarityEnsemble:
    """Ensemble of similarity measures for comprehensive stroke comparison."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize stroke similarity ensemble.

        Args:
            weights: Custom weights for each metric (default from config).
        """
        # Load configuration
        config = load_analysis_config()

        if weights is None:
            weights = config.get("similarity_weights", {})

        self.weights = {
            "dtw": weights.get("dtw_score", 0.4),
            "frechet": weights.get("frechet_score", 0.3),
            "phase_timing": weights.get("phase_timing_score", 0.2),
            "angle": weights.get("angle_similarity", 0.1),
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Initialize analyzers
        self.dtw_analyzer = DTWAnalyzer()
        self.frechet_analyzer = FrechetAnalyzer()
        self.phase_detector = StrokePhaseDetector()
        self.soft_dtw = SoftDTW(gamma=1.0)
        self.lcss = LCSS(epsilon=0.5, delta=5)

    def comprehensive_comparison(
        self,
        stroke1: Dict,
        stroke2: Dict,
        fps: float = 30.0,
    ) -> Dict:
        """Perform comprehensive comparison between two strokes.

        Args:
            stroke1: First stroke data with trajectories and angles.
            stroke2: Second stroke data with trajectories and angles.
            fps: Frames per second.

        Returns:
            Dictionary with all comparison metrics and overall score.
        """
        results = {
            "individual_scores": {},
            "distances": {},
            "recommendations": [],
        }

        # 1. DTW comparison on hand trajectories
        if "left_hand_path" in stroke1 and "left_hand_path" in stroke2:
            dtw_left = self.dtw_analyzer.compute_distance(
                np.array(stroke1["left_hand_path"]),
                np.array(stroke2["left_hand_path"]),
                normalize=True,
            )
            results["distances"]["dtw_left_hand"] = dtw_left
            results["individual_scores"]["dtw_left"] = self._distance_to_score(
                dtw_left, max_dist=10.0
            )

        if "right_hand_path" in stroke1 and "right_hand_path" in stroke2:
            dtw_right = self.dtw_analyzer.compute_distance(
                np.array(stroke1["right_hand_path"]),
                np.array(stroke2["right_hand_path"]),
                normalize=True,
            )
            results["distances"]["dtw_right_hand"] = dtw_right
            results["individual_scores"]["dtw_right"] = self._distance_to_score(
                dtw_right, max_dist=10.0
            )

        # Average DTW score
        if "dtw_left" in results["individual_scores"] and "dtw_right" in results["individual_scores"]:
            results["individual_scores"]["dtw_overall"] = (
                results["individual_scores"]["dtw_left"]
                + results["individual_scores"]["dtw_right"]
            ) / 2.0

        # 2. Fréchet distance comparison
        if all(k in stroke1 and k in stroke2 for k in ["left_hand_path", "right_hand_path"]):
            frechet_results = self.frechet_analyzer.compare_hand_paths(
                stroke1["left_hand_path"],
                stroke1["right_hand_path"],
                stroke2["left_hand_path"],
                stroke2["right_hand_path"],
            )

            if "overall_distance" in frechet_results:
                results["distances"]["frechet"] = frechet_results["overall_distance"]
                results["individual_scores"]["frechet"] = self._distance_to_score(
                    frechet_results["overall_distance"], max_dist=100.0
                )

        # 3. Phase timing comparison
        phase_score = self._compare_phase_timing(stroke1, stroke2, fps)
        if phase_score is not None:
            results["individual_scores"]["phase_timing"] = phase_score

        # 4. Joint angle similarity (cosine)
        if "angles" in stroke1 and "angles" in stroke2:
            angle_similarity = self._compare_angles(stroke1["angles"], stroke2["angles"])
            results["individual_scores"]["angle_similarity"] = angle_similarity * 100

        # 5. Additional measures
        results["additional_measures"] = self._compute_additional_measures(
            stroke1, stroke2
        )

        # Calculate weighted overall score
        results["overall_score"] = self._calculate_weighted_score(
            results["individual_scores"]
        )

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _compare_phase_timing(
        self,
        stroke1: Dict,
        stroke2: Dict,
        fps: float,
    ) -> Optional[float]:
        """Compare phase timing between strokes.

        Args:
            stroke1: First stroke.
            stroke2: Second stroke.
            fps: Frames per second.

        Returns:
            Phase timing similarity score (0-100).
        """
        # Detect phases for both strokes
        if "left_hand_path" not in stroke1 or "left_hand_path" not in stroke2:
            return None

        phases1 = self.phase_detector.detect_phases_freestyle(
            np.array(stroke1["left_hand_path"]), fps, "left"
        )
        phases2 = self.phase_detector.detect_phases_freestyle(
            np.array(stroke2["left_hand_path"]), fps, "left"
        )

        if not phases1 or not phases2:
            return None

        # Compare phase durations
        durations1 = self.phase_detector.get_phase_durations(phases1)
        durations2 = self.phase_detector.get_phase_durations(phases2)

        # Calculate similarity
        common_phases = set(durations1.keys()) & set(durations2.keys())

        if not common_phases:
            return None

        duration_diffs = []
        for phase in common_phases:
            diff = abs(durations1[phase] - durations2[phase])
            max_duration = max(durations1[phase], durations2[phase])
            if max_duration > 0:
                normalized_diff = diff / max_duration
                duration_diffs.append(normalized_diff)

        if duration_diffs:
            avg_similarity = 1.0 - np.mean(duration_diffs)
            return max(0.0, min(100.0, avg_similarity * 100))

        return None

    def _compare_angles(
        self,
        angles1: Dict[str, np.ndarray],
        angles2: Dict[str, np.ndarray],
    ) -> float:
        """Compare joint angle sequences using cosine similarity.

        Args:
            angles1: First stroke angles.
            angles2: Second stroke angles.

        Returns:
            Average cosine similarity (0-1).
        """
        similarities = []

        for joint_name in angles1.keys():
            if joint_name in angles2:
                seq1 = angles1[joint_name]
                seq2 = angles2[joint_name]

                # Remove NaN values
                seq1 = seq1[~np.isnan(seq1)]
                seq2 = seq2[~np.isnan(seq2)]

                if len(seq1) > 0 and len(seq2) > 0:
                    similarity = CosineSimilarityAnalyzer.compute_similarity(
                        seq1, seq2
                    )
                    similarities.append(similarity)

        if similarities:
            return float(np.mean(similarities))
        else:
            return 0.0

    def _compute_additional_measures(
        self,
        stroke1: Dict,
        stroke2: Dict,
    ) -> Dict:
        """Compute additional similarity measures.

        Args:
            stroke1: First stroke.
            stroke2: Second stroke.

        Returns:
            Dictionary with additional metrics.
        """
        additional = {}

        # Soft-DTW
        if "left_hand_path" in stroke1 and "left_hand_path" in stroke2:
            soft_dtw_dist = self.soft_dtw.compute_distance(
                np.array(stroke1["left_hand_path"]),
                np.array(stroke2["left_hand_path"]),
            )
            additional["soft_dtw"] = soft_dtw_dist

        # LCSS
        if "left_hand_path" in stroke1 and "left_hand_path" in stroke2:
            lcss_sim = self.lcss.compute_similarity(
                np.array(stroke1["left_hand_path"]),
                np.array(stroke2["left_hand_path"]),
                normalize=True,
            )
            additional["lcss_similarity"] = lcss_sim * 100

        # Cross-correlation
        if "left_hand_path" in stroke1 and "left_hand_path" in stroke2:
            path1 = np.array(stroke1["left_hand_path"])
            path2 = np.array(stroke2["left_hand_path"])

            # Use x-coordinate for correlation
            corr = CrossCorrelationAnalyzer.compute_max_correlation(
                path1[:, 0], path2[:, 0]
            )
            additional["cross_correlation"] = corr * 100

        return additional

    def _distance_to_score(self, distance: float, max_dist: float) -> float:
        """Convert distance to similarity score (0-100).

        Args:
            distance: Distance value.
            max_dist: Maximum expected distance.

        Returns:
            Similarity score.
        """
        distance = min(distance, max_dist)
        score = (1.0 - distance / max_dist) * 100.0
        return max(0.0, min(100.0, score))

    def _calculate_weighted_score(self, individual_scores: Dict) -> float:
        """Calculate weighted overall score.

        Args:
            individual_scores: Dictionary of individual metric scores.

        Returns:
            Weighted overall score (0-100).
        """
        weighted_sum = 0.0
        total_weight = 0.0

        score_mapping = {
            "dtw_overall": "dtw",
            "frechet": "frechet",
            "phase_timing": "phase_timing",
            "angle_similarity": "angle",
        }

        for score_key, weight_key in score_mapping.items():
            if score_key in individual_scores:
                weighted_sum += individual_scores[score_key] * self.weights[weight_key]
                total_weight += self.weights[weight_key]

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate technique recommendations based on comparison.

        Args:
            results: Comparison results.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        scores = results["individual_scores"]

        # DTW recommendations
        if "dtw_left" in scores and scores["dtw_left"] < 70:
            recommendations.append(
                "Left hand path differs significantly from reference. "
                "Focus on maintaining consistent hand trajectory."
            )

        if "dtw_right" in scores and scores["dtw_right"] < 70:
            recommendations.append(
                "Right hand path differs significantly from reference. "
                "Focus on maintaining consistent hand trajectory."
            )

        # Phase timing recommendations
        if "phase_timing" in scores and scores["phase_timing"] < 70:
            recommendations.append(
                "Stroke phase timing differs from reference. "
                "Work on stroke rhythm and timing consistency."
            )

        # Angle recommendations
        if "angle_similarity" in scores and scores["angle_similarity"] < 70:
            recommendations.append(
                "Joint angles differ from reference technique. "
                "Focus on proper arm positioning throughout the stroke."
            )

        # Overall recommendation
        overall = results.get("overall_score", 0)
        if overall >= 90:
            recommendations.insert(
                0, "Excellent technique! Very close to reference stroke."
            )
        elif overall >= 75:
            recommendations.insert(0, "Good technique with minor areas for improvement.")
        elif overall >= 60:
            recommendations.insert(
                0, "Moderate similarity. Several aspects need attention."
            )
        else:
            recommendations.insert(
                0, "Significant differences from reference. Focus on fundamental technique."
            )

        return recommendations

    def progressive_analysis(
        self,
        stroke_sequence: List[Dict],
        fps: float = 30.0,
    ) -> Dict:
        """Analyze progression/fatigue over multiple strokes.

        Args:
            stroke_sequence: List of strokes in chronological order.
            fps: Frames per second.

        Returns:
            Dictionary with progression analysis.
        """
        if len(stroke_sequence) < 2:
            return {"error": "Need at least 2 strokes for progression analysis"}

        results = {
            "stroke_count": len(stroke_sequence),
            "consistency_scores": [],
            "trend": "stable",
            "fatigue_detected": False,
        }

        # Compare consecutive strokes
        for i in range(len(stroke_sequence) - 1):
            comparison = self.comprehensive_comparison(
                stroke_sequence[i],
                stroke_sequence[i + 1],
                fps,
            )
            results["consistency_scores"].append(comparison["overall_score"])

        # Analyze trend
        if len(results["consistency_scores"]) >= 3:
            scores = np.array(results["consistency_scores"])

            # Linear regression to detect trend
            x = np.arange(len(scores))
            slope, _ = np.polyfit(x, scores, 1)

            if slope < -2:
                results["trend"] = "degrading"
                results["fatigue_detected"] = True
            elif slope > 2:
                results["trend"] = "improving"
            else:
                results["trend"] = "stable"

            results["trend_slope"] = float(slope)

        return results
