"""YOLO11 pose estimation wrapper."""

import logging

import numpy as np
from ultralytics import YOLO

from src.pose.base_estimator import BasePoseEstimator, KeypointFormat
from src.utils.config import load_pose_config
from src.utils.device_utils import get_optimal_device

logger = logging.getLogger(__name__)


class YOLOPoseEstimator(BasePoseEstimator):
    """YOLO11 pose estimation wrapper for swimming analysis."""

    # COCO pose keypoint names (17 keypoints)
    KEYPOINT_NAMES = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        device: str = "auto",
        confidence: float = 0.5,
    ):
        """Initialize YOLO pose estimator.

        Args:
            model_name: YOLO model name (e.g., 'yolo11n-pose.pt').
            device: Device to run on ('cpu', 'cuda', 'mps', 'auto').
            confidence: Minimum confidence threshold (0-1).
        """
        # Auto-detect device if needed
        if device == "auto":
            device = get_optimal_device()
        else:
            device = get_optimal_device(preferred=device)

        # Initialize base class
        super().__init__(model_name, device, confidence)

        # Load configuration
        config = load_pose_config()
        yolo_config = config.get("yolo", {})

        self.iou = yolo_config.get("iou", 0.7)
        self.max_det = yolo_config.get("max_det", 1)
        self.imgsz = yolo_config.get("imgsz", 640)

        # Load model
        self.load_model()

    def load_model(self):
        """Load YOLO model."""
        try:
            self.model = YOLO(self.model_name)
            # Move to device
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def estimate_pose(
        self, image: np.ndarray, return_image: bool = True
    ) -> tuple[list[dict] | None, np.ndarray | None]:
        """Estimate pose from a single frame.

        Args:
            image: Input image (BGR format).
            return_image: Whether to return annotated image.

        Returns:
            Tuple of (pose_data_list, annotated_image).
            pose_data_list: List of pose dictionaries (one per person detected).
            annotated_image: Image with pose overlay (if return_image=True).
        """
        frame = image  # Alias for compatibility
        logger.debug(f"Input frame shape: {frame.shape}, dtype: {frame.dtype}")
        logger.debug(f"Model confidence threshold: {self.confidence}")

        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )

        # Log inference results
        logger.debug(f"Number of results: {len(results)}")
        if len(results) > 0:
            result = results[0]
            logger.debug(f"Has keypoints: {result.keypoints is not None}")
            if result.keypoints is not None:
                logger.debug(f"Number of keypoints detected: {len(result.keypoints.data)}")
            logger.debug(f"Has boxes: {result.boxes is not None}")
            if result.boxes is not None:
                logger.debug(f"Number of boxes: {len(result.boxes.data)}")

        # Extract pose data for all detections
        pose_data_list = []
        annotated_image = None

        if len(results) > 0 and results[0].keypoints is not None:
            result = results[0]

            # Get keypoints for ALL detections
            if len(result.keypoints.data) > 0:
                # Process each detection
                for idx, kpts in enumerate(result.keypoints.data):
                    kpts = kpts.cpu().numpy()  # Shape: (17, 3) [x, y, conf]
                    logger.debug(f"Person {idx} keypoints shape: {kpts.shape}")

                    # Check if keypoints are valid
                    if kpts.shape[0] > 0 and kpts.shape[1] >= 3:
                        # Get bounding box for this detection
                        bbox = None
                        if result.boxes is not None and idx < len(result.boxes.data):
                            box = (
                                result.boxes.data[idx].cpu().numpy()
                            )  # [x1, y1, x2, y2, conf, cls]
                            bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                            box_conf = float(box[4])
                        else:
                            box_conf = 0.0

                        # Build pose dictionary in standard format
                        pose_data = {
                            "keypoints": self._build_keypoints_dict(kpts),  # Convert to dict format
                            "keypoint_names": self.KEYPOINT_NAMES,
                            "bbox": bbox,
                            "person_id": idx,
                            "format": self.get_keypoint_format(),
                            "confidence": float(np.mean(kpts[:, 2])),
                            "metadata": {
                                "model": self.model_name,
                                "box_confidence": box_conf,
                                "image_shape": frame.shape,
                            },
                        }

                        # Only include if confidence is above threshold
                        if pose_data["confidence"] >= self.confidence:
                            pose_data_list.append(pose_data)
                            logger.debug(
                                f"Added person {idx} with confidence: {pose_data['confidence']:.3f}"
                            )
                    else:
                        logger.debug(f"Person {idx}: Invalid keypoints shape: {kpts.shape}")
            else:
                logger.debug("No keypoints data available")

            # Get annotated image if requested
            if return_image and len(pose_data_list) > 0:
                annotated_image = result.plot()
        else:
            logger.debug("No pose detected - returning None")

        # Return None if no valid poses found
        if len(pose_data_list) == 0:
            return None, annotated_image

        return pose_data_list, annotated_image

    def _build_keypoints_dict(self, kpts: np.ndarray) -> dict[str, dict[str, float]]:
        """Build keypoints dictionary from YOLO output.

        Args:
            kpts: Keypoints array (17, 3) [x, y, conf].

        Returns:
            Dictionary mapping keypoint names to {x, y, confidence}.
        """
        keypoints = {}
        for i, name in enumerate(self.KEYPOINT_NAMES):
            keypoints[name] = {
                "x": float(kpts[i, 0]),
                "y": float(kpts[i, 1]),
                "confidence": float(kpts[i, 2]),
            }
        return keypoints

    def estimate_poses_batch(self, frames: list[np.ndarray]) -> list[dict | None]:
        """Estimate poses from multiple frames (batch processing).

        Args:
            frames: List of input frames (BGR format).

        Returns:
            List of pose data dictionaries (one per frame).
        """
        # Run batch inference
        results = self.model.predict(
            frames,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )

        # Extract pose data for each frame
        poses = []
        for result in results:
            pose_data = None

            if result.keypoints is not None and len(result.keypoints.data) > 0:
                kpts = result.keypoints.data[0].cpu().numpy()

                bbox = None
                if result.boxes is not None and len(result.boxes.data) > 0:
                    box = result.boxes.data[0].cpu().numpy()
                    bbox = {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "confidence": float(box[4]),
                    }

                pose_data = {
                    "keypoints": self._build_keypoints_dict(kpts),
                    "bbox": bbox,
                    "confidence": float(result.boxes.data[0][4]) if bbox else 0.0,
                }

            poses.append(pose_data)

        return poses

    def get_keypoint(
        self,
        pose_data: dict,
        keypoint_name: str,
    ) -> tuple[float, float, float] | None:
        """Get a specific keypoint by name.

        Args:
            pose_data: Pose data dictionary.
            keypoint_name: Name of keypoint (e.g., 'left_wrist').

        Returns:
            Tuple of (x, y, confidence) or None if not found.
        """
        if pose_data is None or "keypoints" not in pose_data:
            return None

        keypoints = pose_data["keypoints"]

        # Handle numpy array format (YOLO's own output)
        if isinstance(keypoints, np.ndarray):
            # YOLO format: numpy array (17, 3) with [x, y, conf]
            # Need to map keypoint_name to index
            if "keypoint_names" not in pose_data:
                return None

            keypoint_names = pose_data["keypoint_names"]
            if keypoint_name not in keypoint_names:
                return None

            idx = keypoint_names.index(keypoint_name)
            if idx >= len(keypoints):
                return None

            kpt = keypoints[idx]  # [x, y, conf]
            return tuple(kpt)
        else:
            # Dictionary format (from other estimators)
            kpt = keypoints.get(keypoint_name)
            if kpt is None:
                return None

            return (kpt["x"], kpt["y"], kpt["confidence"])

    def is_keypoint_visible(
        self, pose_data: dict, keypoint_name: str, min_confidence: float = 0.3
    ) -> bool:
        """Check if a keypoint is visible (above confidence threshold).

        Args:
            pose_data: Pose data dictionary.
            keypoint_name: Name of keypoint.
            min_confidence: Minimum confidence threshold.

        Returns:
            True if keypoint is visible.
        """
        kpt = self.get_keypoint(pose_data, keypoint_name)
        if kpt is None:
            return False

        return kpt[2] >= min_confidence

    def get_keypoint_format(self) -> KeypointFormat:
        """Get the keypoint format this model outputs."""
        return KeypointFormat.COCO_17

    def supports_3d(self) -> bool:
        """Check if model supports 3D pose estimation."""
        return False

    def supports_multi_person(self) -> bool:
        """Check if model supports multi-person detection."""
        return True
