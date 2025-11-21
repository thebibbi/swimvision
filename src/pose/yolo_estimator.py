"""YOLO11 pose estimation wrapper."""

import numpy as np
from ultralytics import YOLO

from src.utils.config import load_pose_config


class YOLOPoseEstimator:
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
        model_name: str | None = None,
        device: str | None = None,
        confidence: float | None = None,
    ):
        """Initialize YOLO pose estimator.

        Args:
            model_name: YOLO model name (e.g., 'yolo11n-pose.pt').
            device: Device to run on ('cpu', 'cuda', 'mps').
            confidence: Minimum confidence threshold (0-1).
        """
        # Load configuration
        config = load_pose_config()
        yolo_config = config.get("yolo", {})

        self.model_name = model_name or yolo_config.get("model", "yolo11n-pose.pt")
        self.device = device or yolo_config.get("device", "cpu")
        self.confidence = confidence or yolo_config.get("confidence", 0.5)
        self.iou = yolo_config.get("iou", 0.7)
        self.max_det = yolo_config.get("max_det", 1)
        self.imgsz = yolo_config.get("imgsz", 640)

        # Initialize model
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Load YOLO model.

        Returns:
            YOLO model instance.
        """
        try:
            model = YOLO(self.model_name)
            # Move to device
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def estimate_pose(
        self, frame: np.ndarray, return_image: bool = False
    ) -> tuple[dict | None, np.ndarray | None]:
        """Estimate pose from a single frame.

        Args:
            frame: Input frame (BGR format).
            return_image: Whether to return annotated image.

        Returns:
            Tuple of (pose_data, annotated_image).
            pose_data contains keypoints, bbox, confidence.
            annotated_image is None if return_image=False.
        """
        # Debug: Log input frame info
        print(f"[DEBUG] Input frame shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"[DEBUG] Model confidence threshold: {self.confidence}")

        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )

        # Debug: Log inference results
        print(f"[DEBUG] Number of results: {len(results)}")
        if len(results) > 0:
            result = results[0]
            print(f"[DEBUG] Has keypoints: {result.keypoints is not None}")
            if result.keypoints is not None:
                print(f"[DEBUG] Number of keypoints detected: {len(result.keypoints.data)}")
            print(f"[DEBUG] Has boxes: {result.boxes is not None}")
            if result.boxes is not None:
                print(f"[DEBUG] Number of boxes: {len(result.boxes.data)}")

        # Extract pose data
        pose_data = None
        annotated_image = None

        if len(results) > 0 and results[0].keypoints is not None:
            result = results[0]

            # Get keypoints
            if len(result.keypoints.data) > 0:
                # Take first detection (single swimmer)
                kpts = result.keypoints.data[0].cpu().numpy()  # Shape: (17, 3) [x, y, conf]
                print(f"[DEBUG] Keypoints shape: {kpts.shape}")

                # Check if keypoints are not empty and have correct shape
                if kpts.shape[0] > 0 and kpts.shape[1] >= 3:  # Should be (17, 3)
                    # Get bounding box
                    bbox = None
                    if result.boxes is not None and len(result.boxes.data) > 0:
                        box = result.boxes.data[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
                        bbox = {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "confidence": float(box[4]),
                        }

                    # Build pose dictionary
                    pose_data = {
                        "keypoints": self._build_keypoints_dict(kpts),
                        "bbox": bbox,
                        "confidence": float(result.boxes.data[0][4]) if bbox else 0.0,
                    }
                    print(
                        f"[DEBUG] Successfully built pose_data with confidence: {pose_data['confidence']}"
                    )
                else:
                    print(f"[DEBUG] Keypoints tensor is empty or malformed: {kpts.shape}")
            else:
                print("[DEBUG] No keypoints data available")

            # Get annotated image if requested
            if return_image:
                annotated_image = result.plot()
        else:
            print("[DEBUG] No pose detected - returning None")

        return pose_data, annotated_image

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
        self, pose_data: dict, keypoint_name: str
    ) -> tuple[float, float, float] | None:
        """Get a specific keypoint from pose data.

        Args:
            pose_data: Pose data dictionary from estimate_pose().
            keypoint_name: Name of keypoint (e.g., 'left_shoulder').

        Returns:
            Tuple of (x, y, confidence) or None if not found.
        """
        if pose_data is None or "keypoints" not in pose_data:
            return None

        kpt = pose_data["keypoints"].get(keypoint_name)
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
