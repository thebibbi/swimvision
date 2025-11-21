"""Multi-model fusion system for pose estimation.

Combines predictions from multiple pose estimation models to improve
accuracy and robustness, especially for challenging underwater scenarios.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.pose.base_estimator import (
    BasePoseEstimator,
    KeypointFormat,
    map_keypoints_to_coco17,
)


class FusionMethod(Enum):
    """Fusion methods for combining predictions."""

    WEIGHTED_AVERAGE = "weighted_average"  # Confidence-weighted averaging
    MEDIAN = "median"  # Median of predictions
    MAX_CONFIDENCE = "max_confidence"  # Take highest confidence
    KALMAN_FUSION = "kalman_fusion"  # Kalman filter fusion
    LEARNED_FUSION = "learned_fusion"  # Machine learning fusion


@dataclass
class FusedPrediction:
    """Result of multi-model fusion."""

    keypoints: np.ndarray  # Fused keypoints (Nx3)
    keypoint_names: list[str]  # Keypoint names
    bbox: list[float]  # Bounding box
    format: KeypointFormat  # Output format
    confidence_scores: dict[str, float]  # Per-model confidence
    contributing_models: list[str]  # Models that contributed
    fusion_method: FusionMethod  # Fusion method used
    metadata: dict  # Additional metadata


class MultiModelFusion:
    """Fuse predictions from multiple pose estimation models."""

    def __init__(
        self,
        models: list[BasePoseEstimator],
        fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE,
        target_format: KeypointFormat = KeypointFormat.COCO_17,
        min_models: int = 1,
        confidence_threshold: float = 0.3,
    ):
        """Initialize multi-model fusion.

        Args:
            models: List of pose estimators to fuse.
            fusion_method: Method for fusing predictions.
            target_format: Target keypoint format for fusion.
            min_models: Minimum number of models required for fusion.
            confidence_threshold: Minimum confidence for a prediction.
        """
        self.models = models
        self.fusion_method = fusion_method
        self.target_format = target_format
        self.min_models = min_models
        self.confidence_threshold = confidence_threshold

        # Model weights (can be learned or manually set)
        self.model_weights = {model.model_name: 1.0 for model in models}

    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
    ) -> tuple[FusedPrediction | None, np.ndarray | None]:
        """Estimate pose using model fusion.

        Args:
            image: Input image.
            return_image: Whether to return annotated image.

        Returns:
            Tuple of (fused_prediction, annotated_image).
        """
        # Collect predictions from all models
        predictions = []
        annotated_images = []

        for model in self.models:
            try:
                pose_data, annotated = model.estimate_pose(image, return_image)

                if pose_data is not None:
                    # Handle multi-person detection (take first person)
                    if isinstance(pose_data, list):
                        pose_data = pose_data[0] if len(pose_data) > 0 else None

                    if pose_data is not None:
                        # Convert to target format
                        if pose_data["format"] != self.target_format:
                            pose_data = self._convert_format(pose_data)

                        predictions.append(
                            {
                                "model_name": model.model_name,
                                "pose_data": pose_data,
                                "weight": self.model_weights.get(model.model_name, 1.0),
                            }
                        )

                        if annotated is not None:
                            annotated_images.append(annotated)

            except Exception as e:
                print(f"Warning: Model {model.model_name} failed: {e}")
                continue

        # Check if we have enough predictions
        if len(predictions) < self.min_models:
            return None, image if return_image else None

        # Fuse predictions
        fused = self._fuse_predictions(predictions)

        # Create annotated image
        annotated_image = None
        if return_image:
            if len(annotated_images) > 0:
                # Use the first annotated image as base
                annotated_image = annotated_images[0]
            else:
                annotated_image = image

            # Draw fused keypoints
            annotated_image = self._draw_fused_keypoints(annotated_image, fused)

        return fused, annotated_image

    def _fuse_predictions(self, predictions: list[dict]) -> FusedPrediction:
        """Fuse multiple predictions into one.

        Args:
            predictions: List of prediction dictionaries.

        Returns:
            Fused prediction.
        """
        if self.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(predictions)
        elif self.fusion_method == FusionMethod.MEDIAN:
            return self._median_fusion(predictions)
        elif self.fusion_method == FusionMethod.MAX_CONFIDENCE:
            return self._max_confidence_fusion(predictions)
        elif self.fusion_method == FusionMethod.KALMAN_FUSION:
            return self._kalman_fusion(predictions)
        else:
            # Default to weighted average
            return self._weighted_average_fusion(predictions)

    def _weighted_average_fusion(self, predictions: list[dict]) -> FusedPrediction:
        """Fuse using confidence-weighted averaging.

        Args:
            predictions: List of predictions.

        Returns:
            Fused prediction.
        """
        # Get number of keypoints from target format
        num_keypoints = self._get_num_keypoints(self.target_format)

        # Initialize fused keypoints
        fused_keypoints = np.zeros((num_keypoints, 3))
        weight_sum = np.zeros(num_keypoints)

        # Accumulate weighted predictions
        for pred in predictions:
            pose_data = pred["pose_data"]
            model_weight = pred["weight"]
            keypoints = pose_data["keypoints"]

            for i in range(min(num_keypoints, len(keypoints))):
                kp = keypoints[i]
                confidence = kp[2]

                if confidence > self.confidence_threshold:
                    # Weight by both model weight and keypoint confidence
                    weight = model_weight * confidence

                    fused_keypoints[i, :2] += kp[:2] * weight
                    fused_keypoints[i, 2] += weight
                    weight_sum[i] += weight

        # Normalize by total weight
        for i in range(num_keypoints):
            if weight_sum[i] > 0:
                fused_keypoints[i, :2] /= weight_sum[i]
                fused_keypoints[i, 2] /= weight_sum[i]

        # Calculate overall confidence scores
        confidence_scores = {}
        for pred in predictions:
            model_name = pred["model_name"]
            pose_data = pred["pose_data"]
            avg_conf = np.mean(pose_data["keypoints"][:, 2])
            confidence_scores[model_name] = float(avg_conf)

        # Calculate bounding box
        bbox = self._calculate_bbox(fused_keypoints)

        return FusedPrediction(
            keypoints=fused_keypoints,
            keypoint_names=self._get_keypoint_names(self.target_format),
            bbox=bbox,
            format=self.target_format,
            confidence_scores=confidence_scores,
            contributing_models=[p["model_name"] for p in predictions],
            fusion_method=self.fusion_method,
            metadata={
                "num_models": len(predictions),
                "fusion_weights": {p["model_name"]: p["weight"] for p in predictions},
            },
        )

    def _median_fusion(self, predictions: list[dict]) -> FusedPrediction:
        """Fuse using median of predictions.

        Args:
            predictions: List of predictions.

        Returns:
            Fused prediction.
        """
        num_keypoints = self._get_num_keypoints(self.target_format)
        fused_keypoints = np.zeros((num_keypoints, 3))

        # Collect all predictions for each keypoint
        for i in range(num_keypoints):
            x_values, y_values, conf_values = [], [], []

            for pred in predictions:
                keypoints = pred["pose_data"]["keypoints"]
                if i < len(keypoints):
                    kp = keypoints[i]
                    if kp[2] > self.confidence_threshold:
                        x_values.append(kp[0])
                        y_values.append(kp[1])
                        conf_values.append(kp[2])

            # Calculate median
            if len(x_values) > 0:
                fused_keypoints[i, 0] = np.median(x_values)
                fused_keypoints[i, 1] = np.median(y_values)
                fused_keypoints[i, 2] = np.mean(conf_values)  # Average confidence

        # Calculate confidence scores
        confidence_scores = {}
        for pred in predictions:
            model_name = pred["model_name"]
            pose_data = pred["pose_data"]
            avg_conf = np.mean(pose_data["keypoints"][:, 2])
            confidence_scores[model_name] = float(avg_conf)

        bbox = self._calculate_bbox(fused_keypoints)

        return FusedPrediction(
            keypoints=fused_keypoints,
            keypoint_names=self._get_keypoint_names(self.target_format),
            bbox=bbox,
            format=self.target_format,
            confidence_scores=confidence_scores,
            contributing_models=[p["model_name"] for p in predictions],
            fusion_method=self.fusion_method,
            metadata={"num_models": len(predictions)},
        )

    def _max_confidence_fusion(self, predictions: list[dict]) -> FusedPrediction:
        """Fuse by taking highest confidence prediction for each keypoint.

        Args:
            predictions: List of predictions.

        Returns:
            Fused prediction.
        """
        num_keypoints = self._get_num_keypoints(self.target_format)
        fused_keypoints = np.zeros((num_keypoints, 3))

        # For each keypoint, take the prediction with highest confidence
        for i in range(num_keypoints):
            best_conf = 0
            best_kp = None

            for pred in predictions:
                keypoints = pred["pose_data"]["keypoints"]
                if i < len(keypoints):
                    kp = keypoints[i]
                    if kp[2] > best_conf:
                        best_conf = kp[2]
                        best_kp = kp

            if best_kp is not None:
                fused_keypoints[i] = best_kp

        confidence_scores = {}
        for pred in predictions:
            model_name = pred["model_name"]
            pose_data = pred["pose_data"]
            avg_conf = np.mean(pose_data["keypoints"][:, 2])
            confidence_scores[model_name] = float(avg_conf)

        bbox = self._calculate_bbox(fused_keypoints)

        return FusedPrediction(
            keypoints=fused_keypoints,
            keypoint_names=self._get_keypoint_names(self.target_format),
            bbox=bbox,
            format=self.target_format,
            confidence_scores=confidence_scores,
            contributing_models=[p["model_name"] for p in predictions],
            fusion_method=self.fusion_method,
            metadata={"num_models": len(predictions)},
        )

    def _kalman_fusion(self, predictions: list[dict]) -> FusedPrediction:
        """Fuse using Kalman filtering approach.

        Args:
            predictions: List of predictions.

        Returns:
            Fused prediction.
        """
        # Treat each prediction as a measurement
        # Use measurement uncertainty based on inverse confidence
        # This is a simplified version - full implementation would maintain Kalman state

        num_keypoints = self._get_num_keypoints(self.target_format)
        fused_keypoints = np.zeros((num_keypoints, 3))

        for i in range(num_keypoints):
            measurements = []
            uncertainties = []

            for pred in predictions:
                keypoints = pred["pose_data"]["keypoints"]
                if i < len(keypoints):
                    kp = keypoints[i]
                    if kp[2] > self.confidence_threshold:
                        measurements.append(kp[:2])
                        # Uncertainty inversely proportional to confidence
                        uncertainty = 1.0 / (kp[2] + 1e-6)
                        uncertainties.append(uncertainty)

            if len(measurements) > 0:
                measurements = np.array(measurements)
                uncertainties = np.array(uncertainties)

                # Weighted average with inverse uncertainty weighting
                weights = 1.0 / uncertainties
                weights /= np.sum(weights)

                fused_pos = np.sum(measurements * weights[:, np.newaxis], axis=0)
                fused_conf = np.mean(
                    [
                        kp[2]
                        for pred in predictions
                        for kp in [pred["pose_data"]["keypoints"][i]]
                        if i < len(pred["pose_data"]["keypoints"])
                        and kp[2] > self.confidence_threshold
                    ]
                )

                fused_keypoints[i] = [fused_pos[0], fused_pos[1], fused_conf]

        confidence_scores = {}
        for pred in predictions:
            model_name = pred["model_name"]
            pose_data = pred["pose_data"]
            avg_conf = np.mean(pose_data["keypoints"][:, 2])
            confidence_scores[model_name] = float(avg_conf)

        bbox = self._calculate_bbox(fused_keypoints)

        return FusedPrediction(
            keypoints=fused_keypoints,
            keypoint_names=self._get_keypoint_names(self.target_format),
            bbox=bbox,
            format=self.target_format,
            confidence_scores=confidence_scores,
            contributing_models=[p["model_name"] for p in predictions],
            fusion_method=self.fusion_method,
            metadata={"num_models": len(predictions)},
        )

    def _convert_format(self, pose_data: dict) -> dict:
        """Convert pose data to target format.

        Args:
            pose_data: Pose data in source format.

        Returns:
            Pose data in target format.
        """
        source_format = pose_data["format"]

        if source_format == self.target_format:
            return pose_data

        # Convert to COCO-17 (our common format)
        if self.target_format == KeypointFormat.COCO_17:
            keypoints = map_keypoints_to_coco17(
                pose_data["keypoints"],
                source_format,
            )

            return {
                **pose_data,
                "keypoints": keypoints,
                "format": KeypointFormat.COCO_17,
            }

        # TODO: Add other format conversions as needed
        return pose_data

    def _calculate_bbox(self, keypoints: np.ndarray) -> list[float]:
        """Calculate bounding box from keypoints.

        Args:
            keypoints: Keypoints array (Nx3).

        Returns:
            Bounding box [x1, y1, x2, y2].
        """
        valid_keypoints = keypoints[keypoints[:, 2] > self.confidence_threshold]

        if len(valid_keypoints) == 0:
            return [0, 0, 0, 0]

        x_coords = valid_keypoints[:, 0]
        y_coords = valid_keypoints[:, 1]

        x1, y1 = np.min(x_coords), np.min(y_coords)
        x2, y2 = np.max(x_coords), np.max(y_coords)

        # Add padding
        width, height = x2 - x1, y2 - y1
        padding_x, padding_y = width * 0.1, height * 0.1

        return [
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            x2 + padding_x,
            y2 + padding_y,
        ]

    def _draw_fused_keypoints(
        self,
        image: np.ndarray,
        fused: FusedPrediction,
    ) -> np.ndarray:
        """Draw fused keypoints on image.

        Args:
            image: Input image.
            fused: Fused prediction.

        Returns:
            Annotated image.
        """
        import cv2

        annotated = image.copy()
        keypoints = fused.keypoints

        # COCO-17 skeleton
        skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head
            (5, 6),  # Shoulders
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),  # Arms
            (5, 11),
            (6, 12),
            (11, 12),  # Torso
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Legs
        ]

        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue

            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]

            if start_kp[2] > self.confidence_threshold and end_kp[2] > self.confidence_threshold:
                pt1 = (int(start_kp[0]), int(start_kp[1]))
                pt2 = (int(end_kp[0]), int(end_kp[1]))
                cv2.line(annotated, pt1, pt2, (255, 0, 255), 2)  # Magenta for fused

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > self.confidence_threshold:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(annotated, (x, y), 5, (255, 0, 255), -1)
                cv2.circle(annotated, (x, y), 5, (255, 255, 255), 1)

        # Draw model info
        y_offset = 30
        for model_name, conf in fused.confidence_scores.items():
            text = f"{model_name}: {conf:.2f}"
            cv2.putText(
                annotated, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            y_offset += 25

        return annotated

    def _get_num_keypoints(self, format: KeypointFormat) -> int:
        """Get number of keypoints for format."""
        mapping = {
            KeypointFormat.COCO_17: 17,
            KeypointFormat.COCO_133: 133,
            KeypointFormat.MEDIAPIPE_33: 33,
            KeypointFormat.SMPL_24: 24,
            KeypointFormat.SMPL_X_127: 127,
        }
        return mapping.get(format, 17)

    def _get_keypoint_names(self, format: KeypointFormat) -> list[str]:
        """Get keypoint names for format."""
        from src.pose.base_estimator import get_keypoint_names

        return get_keypoint_names(format)

    def set_model_weight(self, model_name: str, weight: float):
        """Set fusion weight for a specific model.

        Args:
            model_name: Name of the model.
            weight: Fusion weight (higher = more influence).
        """
        self.model_weights[model_name] = weight

    def get_model_statistics(self) -> dict:
        """Get statistics about contributing models.

        Returns:
            Dictionary with model statistics.
        """
        return {
            "num_models": len(self.models),
            "model_names": [m.model_name for m in self.models],
            "model_weights": self.model_weights.copy(),
            "fusion_method": self.fusion_method.value,
            "target_format": self.target_format.value,
        }
