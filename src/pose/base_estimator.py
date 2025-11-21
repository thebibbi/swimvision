"""Unified pose estimation interface for multiple backends.

Supports:
- YOLO11-Pose (Ultralytics)
- MediaPipe Pose
- AlphaPose
- OpenPose
- SMPL/SMPL-X (3D body models)

Provides a consistent interface regardless of backend model.
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class PoseModel(Enum):
    """Available pose estimation models."""

    YOLO11_NANO = "yolo11n-pose"
    YOLO11_SMALL = "yolo11s-pose"
    YOLO11_MEDIUM = "yolo11m-pose"
    MEDIAPIPE = "mediapipe"
    ALPHAPOSE = "alphapose"
    OPENPOSE = "openpose"
    SMPL = "smpl"
    SMPL_X = "smplx"


class KeypointFormat(Enum):
    """Keypoint format standards."""

    COCO_17 = "coco17"  # 17 keypoints (YOLO, OpenPose)
    COCO_133 = "coco133"  # 133 keypoints (AlphaPose)
    MEDIAPIPE_33 = "mediapipe"  # 33 landmarks
    SMPL_24 = "smpl"  # 24 joints
    SMPL_X_127 = "smplx"  # 127 keypoints (body + face + hands)


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimators."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        confidence: float = 0.5,
    ):
        """Initialize pose estimator.

        Args:
            model_name: Name/path of the model.
            device: Device to run on (cpu, cuda, mps).
            confidence: Confidence threshold.
        """
        self.model_name = model_name
        self.device = device
        self.confidence = confidence
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the pose estimation model."""
        pass

    @abstractmethod
    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
    ) -> tuple[dict | None, np.ndarray | None]:
        """Estimate pose from image.

        Args:
            image: Input image (BGR format).
            return_image: Whether to return annotated image.

        Returns:
            Tuple of (pose_data, annotated_image).
            pose_data format:
            {
                'keypoints': np.ndarray,  # Nx3 array (x, y, confidence)
                'keypoint_names': List[str],
                'bbox': Optional[List[float]],  # [x1, y1, x2, y2]
                'person_id': int,
                'format': KeypointFormat,
                'metadata': Dict,  # Model-specific metadata
            }
        """
        pass

    @abstractmethod
    def get_keypoint_format(self) -> KeypointFormat:
        """Get the keypoint format this model outputs."""
        pass

    @abstractmethod
    def supports_3d(self) -> bool:
        """Check if model supports 3D pose estimation."""
        pass

    @abstractmethod
    def supports_multi_person(self) -> bool:
        """Check if model supports multi-person detection."""
        pass

    def normalize_keypoints(
        self,
        keypoints: np.ndarray,
        source_format: KeypointFormat,
        target_format: KeypointFormat = KeypointFormat.COCO_17,
    ) -> np.ndarray:
        """Normalize keypoints from source to target format.

        Args:
            keypoints: Source keypoints.
            source_format: Source format.
            target_format: Target format.

        Returns:
            Normalized keypoints.
        """
        # Implement format conversion if needed
        # For now, return as-is if formats match
        if source_format == target_format:
            return keypoints

        # TODO: Implement cross-format mapping
        # This would map keypoints from one standard to another
        return keypoints

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
        keypoint_names = pose_data.get("keypoint_names", [])
        keypoints = pose_data.get("keypoints", np.array([]))

        if keypoint_name not in keypoint_names:
            return None

        idx = keypoint_names.index(keypoint_name)
        if idx >= len(keypoints):
            return None

        return tuple(keypoints[idx])

    def get_model_info(self) -> dict:
        """Get model information.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence,
            "format": self.get_keypoint_format().value,
            "supports_3d": self.supports_3d(),
            "supports_multi_person": self.supports_multi_person(),
        }


# Keypoint name mappings for different formats
COCO_17_KEYPOINTS = [
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

MEDIAPIPE_33_LANDMARKS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

SMPL_24_JOINTS = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


def get_keypoint_names(format: KeypointFormat) -> list[str]:
    """Get keypoint names for a given format.

    Args:
        format: Keypoint format.

    Returns:
        List of keypoint names.
    """
    mappings = {
        KeypointFormat.COCO_17: COCO_17_KEYPOINTS,
        KeypointFormat.MEDIAPIPE_33: MEDIAPIPE_33_LANDMARKS,
        KeypointFormat.SMPL_24: SMPL_24_JOINTS,
    }
    return mappings.get(format, [])


def map_keypoints_to_coco17(
    keypoints: np.ndarray,
    source_format: KeypointFormat,
) -> np.ndarray:
    """Map keypoints from any format to COCO-17.

    Args:
        keypoints: Source keypoints (Nx3).
        source_format: Source format.

    Returns:
        COCO-17 keypoints (17x3).
    """
    if source_format == KeypointFormat.COCO_17:
        return keypoints[:17]  # Ensure 17 keypoints

    # Initialize COCO-17 output
    coco17 = np.zeros((17, 3))

    # Map from MediaPipe 33 to COCO-17
    if source_format == KeypointFormat.MEDIAPIPE_33:
        # Mapping indices
        mp_to_coco = {
            0: 0,  # nose
            2: 1,  # left_eye
            5: 2,  # right_eye
            7: 3,  # left_ear
            8: 4,  # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10,  # right_wrist
            23: 11,  # left_hip
            24: 12,  # right_hip
            25: 13,  # left_knee
            26: 14,  # right_knee
            27: 15,  # left_ankle
            28: 16,  # right_ankle
        }

        for mp_idx, coco_idx in mp_to_coco.items():
            if mp_idx < len(keypoints):
                coco17[coco_idx] = keypoints[mp_idx]

    # Map from SMPL 24 to COCO-17
    elif source_format == KeypointFormat.SMPL_24:
        smpl_to_coco = {
            15: 0,  # head -> nose (approximation)
            16: 5,  # left_shoulder
            17: 6,  # right_shoulder
            18: 7,  # left_elbow
            19: 8,  # right_elbow
            20: 9,  # left_wrist
            21: 10,  # right_wrist
            1: 11,  # left_hip
            2: 12,  # right_hip
            4: 13,  # left_knee
            5: 14,  # right_knee
            7: 15,  # left_ankle
            8: 16,  # right_ankle
        }

        for smpl_idx, coco_idx in smpl_to_coco.items():
            if smpl_idx < len(keypoints):
                coco17[coco_idx] = keypoints[smpl_idx]

    return coco17
