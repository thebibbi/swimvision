"""AlphaPose wrapper for pose estimation.

AlphaPose is a multi-person pose estimation system with high accuracy.
Supports 133 keypoints (COCO-WholeBody format).

Install: https://github.com/MVIG-SJTU/AlphaPose
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import os

from src.pose.base_estimator import (
    BasePoseEstimator,
    KeypointFormat,
    PoseModel,
)

try:
    # AlphaPose imports (if installed)
    import torch
    from alphapose.models import builder
    from alphapose.utils.config import update_config
    from alphapose.utils.transforms import get_func_heatmap_to_coord
    ALPHAPOSE_AVAILABLE = True
except ImportError:
    ALPHAPOSE_AVAILABLE = False


class AlphaPoseEstimator(BasePoseEstimator):
    """AlphaPose wrapper for high-accuracy multi-person pose estimation."""

    def __init__(
        self,
        model_name: str = "halpe26",  # or "coco", "coco_wholebody"
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        confidence: float = 0.5,
    ):
        """Initialize AlphaPose estimator.

        Args:
            model_name: Model variant (halpe26, coco, coco_wholebody).
            config_path: Path to AlphaPose config.
            weights_path: Path to model weights.
            device: Device (cpu, cuda).
            confidence: Confidence threshold.
        """
        super().__init__(model_name, device, confidence)

        if not ALPHAPOSE_AVAILABLE:
            raise ImportError(
                "AlphaPose not installed. Install from: "
                "https://github.com/MVIG-SJTU/AlphaPose"
            )

        self.config_path = config_path
        self.weights_path = weights_path

        self.load_model()

    def load_model(self):
        """Load AlphaPose model."""
        # Load configuration
        if self.config_path:
            cfg = update_config(self.config_path)
        else:
            # Use default config for model variant
            cfg = self._get_default_config()

        # Build model
        self.model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        # Load weights
        if self.weights_path:
            self.model.load_state_dict(torch.load(self.weights_path))
        else:
            # Download pretrained weights
            self._download_weights()

        self.model.to(self.device)
        self.model.eval()

        # Get coordinate transform function
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
    ) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Estimate pose using AlphaPose.

        Args:
            image: Input image (BGR).
            return_image: Whether to return annotated image.

        Returns:
            Tuple of (pose_data, annotated_image).
        """
        # Preprocess image
        input_tensor = self._preprocess(image)

        # Run inference
        with torch.no_grad():
            heatmaps = self.model(input_tensor)

        # Convert heatmaps to keypoints
        keypoints = self.heatmap_to_coord(heatmaps)

        # Format output
        pose_data = self._format_output(keypoints[0])  # First person

        # Annotate image if requested
        annotated_image = None
        if return_image:
            annotated_image = self._draw_keypoints(image.copy(), pose_data)

        return pose_data, annotated_image

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for AlphaPose.

        Args:
            image: Input image (BGR).

        Returns:
            Preprocessed tensor.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size (typically 256x192 or 320x256)
        input_size = getattr(self, 'input_size', (256, 192))  # (height, width)
        resized = cv2.resize(image_rgb, (input_size[1], input_size[0]))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self.device)

        return tensor

    def _format_output(self, keypoints: np.ndarray) -> Dict:
        """Format AlphaPose output to standard format.

        Args:
            keypoints: AlphaPose keypoints.

        Returns:
            Formatted pose data.
        """
        return {
            'keypoints': keypoints,
            'keypoint_names': self._get_keypoint_names(),
            'bbox': None,  # TODO: Add bbox if available
            'person_id': 0,
            'format': self.get_keypoint_format(),
            'metadata': {
                'model': 'alphapose',
                'variant': self.model_name,
            },
        }

    def _get_keypoint_names(self) -> List[str]:
        """Get keypoint names for model variant."""
        # Return names based on model variant
        if self.model_name == "coco":
            # COCO 17 keypoints
            return [
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
        elif self.model_name == "halpe26":
            # Halpe 26 keypoints (full body)
            return [
                'nose',
                'left_eye', 'right_eye',
                'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hip', 'right_hip',
                'left_knee', 'right_knee',
                'left_ankle', 'right_ankle',
                'head', 'neck',
                'hip',
                'left_big_toe', 'right_big_toe',
                'left_small_toe', 'right_small_toe',
                'left_heel', 'right_heel',
            ]
        else:
            # COCO-WholeBody 133 keypoints (body + face + hands + feet)
            # Body (17) + Face (68) + Hands (42) + Feet (6)
            names = []
            # Body keypoints
            names.extend([
                'nose',
                'left_eye', 'right_eye',
                'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hip', 'right_hip',
                'left_knee', 'right_knee',
                'left_ankle', 'right_ankle',
            ])
            # Face keypoints (68)
            names.extend([f'face_{i}' for i in range(68)])
            # Left hand keypoints (21)
            names.extend([f'left_hand_{i}' for i in range(21)])
            # Right hand keypoints (21)
            names.extend([f'right_hand_{i}' for i in range(21)])
            # Foot keypoints (6)
            names.extend(['left_big_toe', 'left_small_toe', 'left_heel',
                         'right_big_toe', 'right_small_toe', 'right_heel'])
            return names

    def _get_default_config(self):
        """Get default configuration."""
        class Config:
            MODEL = {
                'TYPE': 'FastPose',
                'PRETRAINED': '',
                'NUM_JOINTS': self._get_num_joints(),
                'IMAGE_SIZE': [256, 192],
            }
            DATA_PRESET = {
                'TYPE': 'simple',
                'HEATMAP_SIZE': [64, 48],
                'SIGMA': 2,
            }

        self.input_size = (256, 192)
        return Config()

    def _get_num_joints(self) -> int:
        """Get number of joints for model variant."""
        if self.model_name == "coco":
            return 17
        elif self.model_name == "halpe26":
            return 26
        else:  # coco_wholebody
            return 133

    def _download_weights(self):
        """Download pretrained weights."""
        # Create weights directory
        weights_dir = Path("models/alphapose")
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Weight URLs for different models
        weight_urls = {
            "halpe26": "https://github.com/MVIG-SJTU/AlphaPose/releases/download/v0.5.0/halpe26_fast_res50_256x192.pth",
            "coco": "https://github.com/MVIG-SJTU/AlphaPose/releases/download/v0.5.0/fast_res50_256x192.pth",
            "coco_wholebody": "https://github.com/MVIG-SJTU/AlphaPose/releases/download/v0.5.0/fast_421_res152_256x192.pth",
        }

        if self.model_name not in weight_urls:
            print(f"Warning: No pretrained weights available for {self.model_name}")
            return

        # Download weights
        weight_path = weights_dir / f"{self.model_name}_weights.pth"

        if not weight_path.exists():
            print(f"Downloading AlphaPose {self.model_name} weights...")
            try:
                urllib.request.urlretrieve(weight_urls[self.model_name], str(weight_path))
                print(f"Downloaded weights to {weight_path}")
            except Exception as e:
                print(f"Failed to download weights: {e}")
                print(f"Please manually download from {weight_urls[self.model_name]}")
                return

        # Load weights
        if weight_path.exists():
            try:
                self.model.load_state_dict(torch.load(str(weight_path), map_location=self.device))
                print(f"Loaded weights from {weight_path}")
            except Exception as e:
                print(f"Failed to load weights: {e}")

    def _draw_keypoints(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw keypoints on image.

        Args:
            image: Image to draw on.
            pose_data: Pose data with keypoints.

        Returns:
            Annotated image.
        """
        keypoints = pose_data['keypoints']
        keypoint_names = pose_data['keypoint_names']

        # Define skeleton connections based on model variant
        if self.model_name == "coco" or len(keypoints) == 17:
            # COCO-17 skeleton
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6),  # Shoulders
                (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            ]
        elif self.model_name == "halpe26":
            # Halpe-26 skeleton
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
                (15, 19), (15, 20), (15, 21),  # Left foot
                (16, 22), (16, 23), (16, 24),  # Right foot
            ]
        else:
            # Only draw body keypoints for whole-body model
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            ]

        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue

            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            # Check confidence
            if start_point[2] > self.confidence and end_point[2] > self.confidence:
                start_pos = (int(start_point[0]), int(start_point[1]))
                end_pos = (int(end_point[0]), int(end_point[1]))

                # Draw line
                cv2.line(image, start_pos, end_pos, (0, 255, 0), 2)

        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if keypoint[2] > self.confidence:
                x, y = int(keypoint[0]), int(keypoint[1])

                # Color based on confidence
                confidence = keypoint[2]
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))

                # Draw circle
                cv2.circle(image, (x, y), 4, color, -1)
                cv2.circle(image, (x, y), 4, (255, 255, 255), 1)

        return image

    def get_keypoint_format(self) -> KeypointFormat:
        """Get keypoint format."""
        if self.model_name == "coco":
            return KeypointFormat.COCO_17
        else:
            return KeypointFormat.COCO_133

    def supports_3d(self) -> bool:
        """AlphaPose is 2D only."""
        return False

    def supports_multi_person(self) -> bool:
        """AlphaPose supports multi-person detection."""
        return True
