"""AlphaPose wrapper for pose estimation.

AlphaPose is a multi-person pose estimation system with high accuracy.
Supports 133 keypoints (COCO-WholeBody format).

Install: https://github.com/MVIG-SJTU/AlphaPose
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

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
        image_rgb = image[:, :, ::-1]

        # Resize and normalize (model-specific)
        # TODO: Implement proper preprocessing based on model config

        # Convert to tensor
        tensor = torch.from_numpy(image_rgb).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
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
        # Placeholder - should match actual AlphaPose format
        return [f"kp_{i}" for i in range(133)]

    def _get_default_config(self):
        """Get default configuration."""
        # Placeholder - should return actual config
        class Config:
            MODEL = {}
            DATA_PRESET = {}
        return Config()

    def _download_weights(self):
        """Download pretrained weights."""
        # Placeholder - implement weight download
        pass

    def _draw_keypoints(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw keypoints on image."""
        # Placeholder - implement visualization
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
