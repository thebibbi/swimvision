"""OpenPose wrapper for pose estimation.

OpenPose is a multi-person pose estimation system with high accuracy.
Supports COCO-18 body model, hand detection, and face detection.

Install: https://github.com/CMU-Perceptual-Computing-Lab/openpose
Python bindings: pip install openpose-python (or build from source)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

from src.pose.base_estimator import (
    BasePoseEstimator,
    KeypointFormat,
    PoseModel,
)

try:
    import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False


class OpenPoseEstimator(BasePoseEstimator):
    """OpenPose wrapper for multi-person pose estimation."""

    def __init__(
        self,
        model_folder: Optional[str] = None,
        net_resolution: str = "-1x368",  # Width x Height (-1 for automatic)
        face: bool = False,
        hand: bool = False,
        device: str = "cpu",  # OpenPose uses 'cpu' or GPU_ID
        confidence: float = 0.5,
        number_people_max: int = -1,  # -1 for no limit
    ):
        """Initialize OpenPose estimator.

        Args:
            model_folder: Path to OpenPose models folder.
            net_resolution: Network resolution (e.g., "-1x368", "656x368").
            face: Enable face keypoint detection.
            hand: Enable hand keypoint detection.
            device: Device to run on ('cpu' or GPU ID like '0').
            confidence: Confidence threshold.
            number_people_max: Maximum number of people to detect (-1 for unlimited).
        """
        super().__init__("openpose", device, confidence)

        if not OPENPOSE_AVAILABLE:
            raise ImportError(
                "OpenPose not installed. Install from: "
                "https://github.com/CMU-Perceptual-Computing-Lab/openpose"
            )

        self.model_folder = model_folder or self._get_default_model_folder()
        self.net_resolution = net_resolution
        self.face = face
        self.hand = hand
        self.number_people_max = number_people_max

        # OpenPose parameters
        self.params = {
            "model_folder": str(self.model_folder),
            "net_resolution": self.net_resolution,
            "face": face,
            "hand": hand,
            "number_people_max": number_people_max,
        }

        # Set device
        if device.lower() != "cpu":
            # GPU device ID
            try:
                gpu_id = int(device) if device.isdigit() else 0
                self.params["num_gpu"] = 1
                self.params["num_gpu_start"] = gpu_id
            except:
                pass  # Fall back to CPU

        self.load_model()

    def load_model(self):
        """Load OpenPose model."""
        # Create OpenPose wrapper
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

        # Datum for processing
        self.datum = op.Datum()

    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
    ) -> Tuple[Optional[List[Dict]], Optional[np.ndarray]]:
        """Estimate pose using OpenPose.

        Args:
            image: Input image (BGR format).
            return_image: Whether to return annotated image.

        Returns:
            Tuple of (pose_data_list, annotated_image).
            Note: Returns list of pose_data for multi-person support.
        """
        # Process image
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))

        # Check if any poses detected
        if self.datum.poseKeypoints is None or len(self.datum.poseKeypoints) == 0:
            return None, image if return_image else None

        # Extract poses (OpenPose returns multiple people)
        pose_data_list = []
        for person_id, person_keypoints in enumerate(self.datum.poseKeypoints):
            pose_data = self._format_output(person_keypoints, person_id, image.shape)

            # Filter by confidence
            if self._has_sufficient_confidence(pose_data):
                pose_data_list.append(pose_data)

        if len(pose_data_list) == 0:
            return None, image if return_image else None

        # Annotated image
        annotated_image = None
        if return_image:
            annotated_image = self.datum.cvOutputData

        # Return all detected people
        return pose_data_list, annotated_image

    def _format_output(
        self,
        keypoints: np.ndarray,
        person_id: int,
        image_shape: Tuple[int, int, int],
    ) -> Dict:
        """Format OpenPose output to standard format.

        Args:
            keypoints: OpenPose keypoints for one person (25x3 or 18x3).
            person_id: Person ID.
            image_shape: Image shape.

        Returns:
            Formatted pose data.
        """
        # OpenPose BODY_25 has 25 keypoints, COCO has 18
        # We'll use the first 18 to match COCO format
        num_keypoints = min(18, len(keypoints))
        keypoints = keypoints[:num_keypoints]

        # OpenPose keypoint names (COCO-18 format)
        keypoint_names = [
            'nose',
            'neck',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'mid_hip',
            'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle',
            'right_eye', 'left_eye',
            'right_ear', 'left_ear',
        ][:num_keypoints]

        # Calculate bounding box
        bbox = self._calculate_bbox(keypoints)

        # Extract hand and face keypoints if enabled
        metadata = {
            'model': 'openpose',
            'format': 'coco18',
        }

        if self.hand and hasattr(self.datum, 'handKeypoints') and self.datum.handKeypoints is not None:
            metadata['hand_keypoints'] = {
                'left': self.datum.handKeypoints[0][person_id] if len(self.datum.handKeypoints[0]) > person_id else None,
                'right': self.datum.handKeypoints[1][person_id] if len(self.datum.handKeypoints[1]) > person_id else None,
            }

        if self.face and hasattr(self.datum, 'faceKeypoints') and self.datum.faceKeypoints is not None:
            if len(self.datum.faceKeypoints) > person_id:
                metadata['face_keypoints'] = self.datum.faceKeypoints[person_id]

        return {
            'keypoints': keypoints,
            'keypoint_names': keypoint_names,
            'bbox': bbox,
            'person_id': person_id,
            'format': self.get_keypoint_format(),
            'metadata': metadata,
        }

    def _has_sufficient_confidence(self, pose_data: Dict) -> bool:
        """Check if pose has sufficient confidence.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            True if pose meets confidence threshold.
        """
        keypoints = pose_data['keypoints']
        confidences = keypoints[:, 2]

        # Require at least 50% of keypoints above threshold
        valid_keypoints = np.sum(confidences > self.confidence)
        return valid_keypoints >= len(keypoints) * 0.5

    def _calculate_bbox(self, keypoints: np.ndarray) -> List[float]:
        """Calculate bounding box from keypoints.

        Args:
            keypoints: Keypoints array (Nx3).

        Returns:
            Bounding box [x1, y1, x2, y2].
        """
        # Filter keypoints with sufficient confidence
        valid_keypoints = keypoints[keypoints[:, 2] > self.confidence]

        if len(valid_keypoints) == 0:
            return [0, 0, 0, 0]

        x_coords = valid_keypoints[:, 0]
        y_coords = valid_keypoints[:, 1]

        x1 = np.min(x_coords)
        y1 = np.min(y_coords)
        x2 = np.max(x_coords)
        y2 = np.max(y_coords)

        # Add padding (10%)
        width = x2 - x1
        height = y2 - y1
        padding_x = width * 0.1
        padding_y = height * 0.1

        return [
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            x2 + padding_x,
            y2 + padding_y,
        ]

    def get_keypoint_format(self) -> KeypointFormat:
        """Get keypoint format."""
        return KeypointFormat.COCO_17

    def supports_3d(self) -> bool:
        """OpenPose is 2D only (without additional depth sensors)."""
        return False

    def supports_multi_person(self) -> bool:
        """OpenPose supports multi-person detection."""
        return True

    def convert_to_coco17(self, pose_data: Dict) -> Dict:
        """Convert OpenPose COCO-18 to COCO-17 format.

        OpenPose COCO-18 includes 'neck' and 'mid_hip' which aren't in COCO-17.
        We need to map accordingly.

        Args:
            pose_data: OpenPose pose data.

        Returns:
            Pose data in COCO-17 format.
        """
        keypoints_18 = pose_data['keypoints']

        # Mapping from OpenPose COCO-18 to COCO-17
        # OpenPose: [nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist,
        #            mid_hip, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear]
        # COCO-17: [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow,
        #           l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]

        coco17_indices = [
            0,   # nose
            16,  # left_eye
            15,  # right_eye
            18,  # left_ear (if available, else 17)
            17,  # right_ear (if available, else 16)
            5,   # left_shoulder
            2,   # right_shoulder
            6,   # left_elbow
            3,   # right_elbow
            7,   # left_wrist
            4,   # right_wrist
            12,  # left_hip
            9,   # right_hip
            13,  # left_knee
            10,  # right_knee
            14,  # left_ankle
            11,  # right_ankle
        ]

        # Create COCO-17 keypoints
        coco17_keypoints = np.zeros((17, 3))
        for coco17_idx, openpose_idx in enumerate(coco17_indices):
            if openpose_idx < len(keypoints_18):
                coco17_keypoints[coco17_idx] = keypoints_18[openpose_idx]

        # COCO-17 keypoint names
        coco17_names = [
            'nose',
            'left_eye', 'right_eye',
            'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
        ]

        return {
            **pose_data,
            'keypoints': coco17_keypoints,
            'keypoint_names': coco17_names,
            'format': KeypointFormat.COCO_17,
        }

    def get_hand_keypoints(self, pose_data: Dict) -> Dict[str, Optional[np.ndarray]]:
        """Get hand keypoints if hand detection was enabled.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            Dictionary with 'left' and 'right' hand keypoints.
        """
        metadata = pose_data.get('metadata', {})
        return metadata.get('hand_keypoints', {'left': None, 'right': None})

    def get_face_keypoints(self, pose_data: Dict) -> Optional[np.ndarray]:
        """Get face keypoints if face detection was enabled.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            Face keypoints or None.
        """
        metadata = pose_data.get('metadata', {})
        return metadata.get('face_keypoints')

    def _get_default_model_folder(self) -> str:
        """Get default OpenPose model folder.

        Returns:
            Path to models folder.
        """
        # Try common installation locations
        possible_paths = [
            Path("/usr/local/share/openpose/models"),
            Path.home() / "openpose" / "models",
            Path("models") / "openpose",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Return default path (user may need to configure)
        return "models/openpose"

    def __del__(self):
        """Cleanup OpenPose resources."""
        if hasattr(self, 'opWrapper'):
            self.opWrapper.stop()
