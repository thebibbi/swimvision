"""MediaPipe Pose wrapper for pose estimation.

MediaPipe is a lightweight, cross-platform pose estimation solution.
Excellent for CPU-only environments and real-time performance.

Install: pip install mediapipe
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import logging

from src.pose.base_estimator import (
    BasePoseEstimator,
    KeypointFormat,
    PoseModel,
    map_keypoints_to_coco17,
)

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Install with: pip install mediapipe")


class MediaPipeEstimator(BasePoseEstimator):
    """MediaPipe Pose wrapper for lightweight pose estimation."""

    def __init__(
        self,
        model_complexity: int = 1,  # 0=lite, 1=full, 2=heavy
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True,
        device: str = "cpu",  # MediaPipe runs on CPU
    ):
        """Initialize MediaPipe estimator.

        Args:
            model_complexity: Model complexity (0-2).
            min_detection_confidence: Detection confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
            enable_segmentation: Enable person segmentation.
            smooth_landmarks: Enable landmark smoothing.
            device: Device (MediaPipe runs on CPU).
        """
        super().__init__(f"mediapipe_complexity_{model_complexity}", device, min_detection_confidence)

        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe not installed. Install with: pip install mediapipe"
            )

        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.smooth_landmarks = smooth_landmarks

        self.load_model()

    def load_model(self):
        """Load MediaPipe Pose model."""
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
                smooth_segmentation=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            logger.info(f"MediaPipe Pose model loaded (complexity={self.model_complexity})")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe Pose model: {e}")
            raise

    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
    ) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Estimate pose using MediaPipe.

        Args:
            image: Input image (BGR format).
            return_image: Whether to return annotated image.

        Returns:
            Tuple of (pose_data, annotated_image).
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image
            results = self.model.process(image_rgb)

            # Check if pose detected
            if not results or not results.pose_landmarks:
                return None, image if return_image else None

            # Extract keypoints
            pose_data = self._format_output(results, image.shape)

            # Annotate image if requested
            annotated_image = None
            if return_image:
                annotated_image = image.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            return pose_data, annotated_image

        except Exception as e:
            logger.error(f"MediaPipe pose estimation failed: {e}")
            return None, image if return_image else None

    def _format_output(self, results, image_shape: Tuple[int, int, int]) -> Dict:
        """Format MediaPipe output to standard format.

        Args:
            results: MediaPipe results.
            image_shape: Image shape (height, width, channels).

        Returns:
            Formatted pose data.
        """
        height, width, _ = image_shape

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        keypoint_names = []

        # MediaPipe has 33 landmarks
        landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky', 'right_pinky',
            'left_index', 'right_index',
            'left_thumb', 'right_thumb',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

        for landmark, name in zip(landmarks, landmark_names):
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * width
            y = landmark.y * height
            confidence = landmark.visibility  # MediaPipe uses visibility as confidence

            keypoints.append([x, y, confidence])
            keypoint_names.append(name)

        keypoints = np.array(keypoints)

        # Calculate bounding box
        bbox = self._calculate_bbox(keypoints)

        return {
            'keypoints': keypoints,
            'keypoint_names': keypoint_names,
            'bbox': bbox,
            'person_id': 0,  # MediaPipe is single-person
            'format': self.get_keypoint_format(),
            'metadata': {
                'model': 'mediapipe',
                'complexity': self.model_complexity,
                'world_landmarks': self._extract_world_landmarks(results),
            },
        }

    def _extract_world_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract 3D world landmarks if available.

        Args:
            results: MediaPipe results.

        Returns:
            3D world landmarks (Nx3) or None.
        """
        if not results.pose_world_landmarks:
            return None

        world_landmarks = []
        for landmark in results.pose_world_landmarks.landmark:
            world_landmarks.append([landmark.x, landmark.y, landmark.z])

        return np.array(world_landmarks)

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
        return KeypointFormat.MEDIAPIPE_33

    def supports_3d(self) -> bool:
        """MediaPipe supports 3D world landmarks."""
        return True

    def supports_multi_person(self) -> bool:
        """MediaPipe is single-person only."""
        return False

    def get_3d_landmarks(self, pose_data: Dict) -> Optional[np.ndarray]:
        """Get 3D world landmarks from pose data.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            3D landmarks (Nx3) or None.
        """
        return pose_data.get('metadata', {}).get('world_landmarks')

    def convert_to_coco17(self, pose_data: Dict) -> Dict:
        """Convert MediaPipe landmarks to COCO-17 format.

        Args:
            pose_data: MediaPipe pose data.

        Returns:
            Pose data in COCO-17 format.
        """
        keypoints = pose_data['keypoints']
        coco17_keypoints = map_keypoints_to_coco17(
            keypoints,
            KeypointFormat.MEDIAPIPE_33,
        )

        return {
            **pose_data,
            'keypoints': coco17_keypoints,
            'keypoint_names': [
                'nose',
                'left_eye', 'right_eye',
                'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hip', 'right_hip',
                'left_knee', 'right_knee',
                'left_ankle', 'right_ankle',
            ],
            'format': KeypointFormat.COCO_17,
        }

    def close(self):
        """Explicitly close MediaPipe resources."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.close()
                logger.debug("MediaPipe model closed successfully")
            except Exception as e:
                logger.warning(f"Error closing MediaPipe model: {e}")
            finally:
                self.model = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Cleanup MediaPipe resources on deletion."""
        # Safer cleanup that handles interpreter shutdown
        try:
            self.close()
        except Exception:
            # Ignore errors during interpreter shutdown
            pass
