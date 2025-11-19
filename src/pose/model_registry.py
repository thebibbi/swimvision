"""Model Registry System for SwimVision Pro.

Centralized management of all pose estimation models with unified interface,
automatic model loading, caching, and configuration management.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import yaml
import json
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ModelFramework(Enum):
    """Supported model frameworks."""
    MMPOSE = "mmpose"
    ULTRALYTICS = "ultralytics"
    MEDIAPIPE = "mediapipe"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Model capabilities."""
    POSE_2D = "pose_2d"
    POSE_3D = "pose_3d"
    MULTI_PERSON = "multi_person"
    HAND_DETECTION = "hand_detection"
    FACE_DETECTION = "face_detection"
    TEMPORAL = "temporal"
    MESH_RECOVERY = "mesh_recovery"


@dataclass
class ModelConfig:
    """Configuration for a pose estimation model."""
    name: str
    framework: ModelFramework
    config_file: Optional[str] = None
    checkpoint_path: Optional[str] = None
    keypoint_format: str = "COCO_17"
    num_keypoints: int = 17
    capabilities: List[ModelCapability] = field(default_factory=list)
    fps_target: int = 30
    input_size: tuple = (256, 192)
    device_preference: str = "cuda"  # cuda, cpu, mps
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Central registry for all pose estimation models.

    Features:
    - Unified model configuration
    - Lazy loading (models loaded on first use)
    - Model caching
    - Capability querying
    - Performance tracking
    """

    # Model definitions
    MODELS: Dict[str, ModelConfig] = {
        # ===== RTMPose Models (MMPose) =====
        "rtmpose-t": ModelConfig(
            name="rtmpose-t",
            framework=ModelFramework.MMPOSE,
            config_file="rtmpose-t_8xb256-420e_coco-256x192.py",
            checkpoint_path="rtmpose-t_simcc-coco_pt-aic-coco_420e-256x192.pth",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=90,
            input_size=(256, 192),
            metadata={
                'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-coco_pt-aic-coco_420e-256x192-aff1f1fb_20230126.pth',
                'description': 'Tiny RTMPose model for maximum speed',
                'accuracy_ap': 68.5,
            }
        ),
        "rtmpose-s": ModelConfig(
            name="rtmpose-s",
            framework=ModelFramework.MMPOSE,
            config_file="rtmpose-s_8xb256-420e_coco-256x192.py",
            checkpoint_path="rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=60,
            input_size=(256, 192),
            metadata={
                'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth',
                'description': 'Small RTMPose model for balanced speed/accuracy',
                'accuracy_ap': 72.2,
            }
        ),
        "rtmpose-m": ModelConfig(
            name="rtmpose-m",
            framework=ModelFramework.MMPOSE,
            config_file="rtmpose-m_8xb256-420e_coco-256x192.py",
            checkpoint_path="rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=45,
            input_size=(256, 192),
            metadata={
                'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
                'description': 'Medium RTMPose model for balanced performance',
                'accuracy_ap': 75.8,
            }
        ),
        "rtmpose-l": ModelConfig(
            name="rtmpose-l",
            framework=ModelFramework.MMPOSE,
            config_file="rtmpose-l_8xb256-420e_coco-256x192.py",
            checkpoint_path="rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192.pth",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=30,
            input_size=(256, 192),
            metadata={
                'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth',
                'description': 'Large RTMPose model for high accuracy',
                'accuracy_ap': 76.7,
            }
        ),

        # ===== ViTPose Models (MMPose) =====
        "vitpose-b": ModelConfig(
            name="vitpose-b",
            framework=ModelFramework.MMPOSE,
            config_file="td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
            checkpoint_path="vitpose-b.pth",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=25,
            input_size=(256, 192),
            metadata={
                'description': 'ViTPose Base - Vision Transformer backbone',
                'accuracy_ap': 75.8,
            }
        ),
        "vitpose-l": ModelConfig(
            name="vitpose-l",
            framework=ModelFramework.MMPOSE,
            config_file="td-hm_ViTPose-large_8xb64-210e_coco-256x192.py",
            checkpoint_path="vitpose-l.pth",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=20,
            input_size=(256, 192),
            metadata={
                'description': 'ViTPose Large - Higher accuracy',
                'accuracy_ap': 78.3,
            }
        ),
        "vitpose-h": ModelConfig(
            name="vitpose-h",
            framework=ModelFramework.MMPOSE,
            config_file="td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py",
            checkpoint_path="vitpose-h.pth",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=15,
            input_size=(256, 192),
            metadata={
                'description': 'ViTPose Huge - Maximum accuracy',
                'accuracy_ap': 81.1,
            }
        ),

        # ===== YOLO Models (Ultralytics) =====
        "yolo11n-pose": ModelConfig(
            name="yolo11n-pose",
            framework=ModelFramework.ULTRALYTICS,
            checkpoint_path="yolo11n-pose.pt",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=60,
            input_size=(640, 640),
            metadata={
                'description': 'YOLO11 Nano - Fastest YOLO',
                'accuracy_ap': 50.5,
            }
        ),
        "yolo11s-pose": ModelConfig(
            name="yolo11s-pose",
            framework=ModelFramework.ULTRALYTICS,
            checkpoint_path="yolo11s-pose.pt",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=50,
            input_size=(640, 640),
            metadata={
                'description': 'YOLO11 Small - Good balance',
                'accuracy_ap': 59.2,
            }
        ),
        "yolo11m-pose": ModelConfig(
            name="yolo11m-pose",
            framework=ModelFramework.ULTRALYTICS,
            checkpoint_path="yolo11m-pose.pt",
            keypoint_format="COCO_17",
            num_keypoints=17,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=40,
            input_size=(640, 640),
            metadata={
                'description': 'YOLO11 Medium - Higher accuracy',
                'accuracy_ap': 66.0,
            }
        ),

        # ===== MediaPipe (Google) =====
        "mediapipe": ModelConfig(
            name="mediapipe",
            framework=ModelFramework.MEDIAPIPE,
            keypoint_format="MEDIAPIPE_33",
            num_keypoints=33,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.POSE_3D],
            fps_target=30,
            input_size=(256, 256),
            metadata={
                'description': 'MediaPipe Pose - Real-time 3D',
                'complexity_levels': [0, 1, 2],
            }
        ),

        # ===== Custom Models =====
        "openpose": ModelConfig(
            name="openpose",
            framework=ModelFramework.CUSTOM,
            keypoint_format="COCO_18",
            num_keypoints=18,
            capabilities=[
                ModelCapability.POSE_2D,
                ModelCapability.MULTI_PERSON,
                ModelCapability.HAND_DETECTION,
                ModelCapability.FACE_DETECTION,
            ],
            fps_target=20,
            input_size=(368, 368),
            metadata={
                'description': 'OpenPose - Multi-person with hands/face',
            }
        ),
        "alphapose": ModelConfig(
            name="alphapose",
            framework=ModelFramework.CUSTOM,
            keypoint_format="COCO_133",
            num_keypoints=133,
            capabilities=[ModelCapability.POSE_2D, ModelCapability.MULTI_PERSON],
            fps_target=15,
            input_size=(256, 192),
            metadata={
                'description': 'AlphaPose - Wholebody detection',
                'variants': ['halpe26', 'coco', 'coco_wholebody'],
            }
        ),
        "smpl-x": ModelConfig(
            name="smpl-x",
            framework=ModelFramework.CUSTOM,
            keypoint_format="SMPL_X_127",
            num_keypoints=127,
            capabilities=[
                ModelCapability.POSE_3D,
                ModelCapability.MESH_RECOVERY,
            ],
            fps_target=10,
            input_size=(224, 224),
            metadata={
                'description': 'SMPL-X - 3D body mesh',
                'mesh_vertices': 10475,
            }
        ),
        "wham": ModelConfig(
            name="wham",
            framework=ModelFramework.CUSTOM,
            keypoint_format="SMPL_24",
            num_keypoints=24,
            capabilities=[
                ModelCapability.POSE_3D,
                ModelCapability.MESH_RECOVERY,
                ModelCapability.TEMPORAL,
            ],
            fps_target=10,
            input_size=(224, 224),
            metadata={
                'description': 'WHAM - World-grounded temporal pose estimation',
                'requires_video': True,
            }
        ),
    }

    def __init__(self, models_dir: str = "models"):
        """Initialize model registry.

        Args:
            models_dir: Base directory for model checkpoints.
        """
        self.models_dir = Path(models_dir)
        self.loaded_models = {}  # Cache for loaded models
        self.performance_stats = {}  # Track model performance

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of all available model names."""
        return list(cls.MODELS.keys())

    @classmethod
    def get_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Model configuration.

        Raises:
            KeyError: If model not found.
        """
        if model_name not in cls.MODELS:
            raise KeyError(
                f"Model '{model_name}' not found. Available models: {cls.list_available_models()}"
            )
        return cls.MODELS[model_name]

    @classmethod
    def filter_by_capability(cls, capability: ModelCapability) -> List[str]:
        """Filter models by capability.

        Args:
            capability: Required capability.

        Returns:
            List of model names with that capability.
        """
        return [
            name for name, config in cls.MODELS.items()
            if capability in config.capabilities
        ]

    @classmethod
    def filter_by_speed(cls, min_fps: int) -> List[str]:
        """Filter models by minimum FPS target.

        Args:
            min_fps: Minimum FPS requirement.

        Returns:
            List of model names meeting FPS target.
        """
        return [
            name for name, config in cls.MODELS.items()
            if config.fps_target >= min_fps
        ]

    @classmethod
    def get_best_model(cls, criteria: str = "balanced") -> str:
        """Get recommended model based on criteria.

        Args:
            criteria: Selection criteria
                - "fastest": Maximum speed
                - "accurate": Maximum accuracy
                - "balanced": Best speed/accuracy balance
                - "3d": Best 3D capability

        Returns:
            Model name.
        """
        if criteria == "fastest":
            return max(cls.MODELS.items(), key=lambda x: x[1].fps_target)[0]

        elif criteria == "accurate":
            # Use metadata accuracy if available
            accurate_models = [
                (name, config.metadata.get('accuracy_ap', 0))
                for name, config in cls.MODELS.items()
                if 'accuracy_ap' in config.metadata
            ]
            if accurate_models:
                return max(accurate_models, key=lambda x: x[1])[0]
            return "vitpose-h"

        elif criteria == "balanced":
            return "rtmpose-m"

        elif criteria == "3d":
            # Get models with 3D capability
            models_3d = cls.filter_by_capability(ModelCapability.POSE_3D)
            if models_3d:
                return "mediapipe"  # Best real-time 3D
            return "wham"  # Best offline 3D

        else:
            raise ValueError(f"Unknown criteria: {criteria}")

    def load_model(self, model_name: str, device: str = "cuda") -> Any:
        """Load a model (lazy loading with caching).

        Args:
            model_name: Name of the model to load.
            device: Device to load model on.

        Returns:
            Loaded model instance.
        """
        cache_key = f"{model_name}_{device}"

        # Check cache
        if cache_key in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[cache_key]

        # Load model
        config = self.get_config(model_name)
        logger.info(f"Loading model: {model_name} on {device}")

        if config.framework == ModelFramework.MMPOSE:
            model = self._load_mmpose_model(config, device)
        elif config.framework == ModelFramework.ULTRALYTICS:
            model = self._load_yolo_model(config, device)
        elif config.framework == ModelFramework.MEDIAPIPE:
            model = self._load_mediapipe_model(config, device)
        elif config.framework == ModelFramework.CUSTOM:
            model = self._load_custom_model(config, device)
        else:
            raise ValueError(f"Unsupported framework: {config.framework}")

        # Cache model
        self.loaded_models[cache_key] = model
        logger.info(f"Model loaded successfully: {model_name}")

        return model

    def _load_mmpose_model(self, config: ModelConfig, device: str) -> Any:
        """Load MMPose model."""
        try:
            from mmpose.apis import init_model
            from mmpose.structures import merge_data_samples

            config_path = self.models_dir / "mmpose" / config.config_file
            checkpoint_path = self.models_dir / "mmpose" / config.checkpoint_path

            model = init_model(str(config_path), str(checkpoint_path), device=device)
            return model

        except ImportError:
            raise ImportError(
                "MMPose not installed. Install with: "
                "mim install mmengine mmcv mmpose"
            )

    def _load_yolo_model(self, config: ModelConfig, device: str) -> Any:
        """Load YOLO model."""
        from src.pose.yolo_estimator import YOLOPoseEstimator

        checkpoint_path = self.models_dir / config.checkpoint_path
        return YOLOPoseEstimator(
            model_name=str(checkpoint_path),
            device=device,
            confidence=0.5,
        )

    def _load_mediapipe_model(self, config: ModelConfig, device: str) -> Any:
        """Load MediaPipe model."""
        from src.pose.mediapipe_estimator import MediaPipeEstimator

        return MediaPipeEstimator(
            model_complexity=1,
            min_detection_confidence=0.5,
            device=device,
        )

    def _load_custom_model(self, config: ModelConfig, device: str) -> Any:
        """Load custom model."""
        if config.name == "openpose":
            from src.pose.openpose_estimator import OpenPoseEstimator
            return OpenPoseEstimator(device=device, confidence=0.5)

        elif config.name == "alphapose":
            from src.pose.alphapose_estimator import AlphaPoseEstimator
            return AlphaPoseEstimator(device=device, confidence=0.5)

        elif config.name == "smpl-x":
            from src.pose.smpl_estimator import SMPLEstimator
            return SMPLEstimator(model_type="smplx", device=device)

        elif config.name == "wham":
            # WHAM requires special handling (video-based)
            raise NotImplementedError("WHAM requires video buffer - use TemporalRefiner")

        else:
            raise ValueError(f"Unknown custom model: {config.name}")

    def save_config(self, output_path: str):
        """Save registry configuration to file.

        Args:
            output_path: Path to save configuration.
        """
        config_data = {
            name: {
                'framework': config.framework.value,
                'keypoint_format': config.keypoint_format,
                'num_keypoints': config.num_keypoints,
                'capabilities': [c.value for c in config.capabilities],
                'fps_target': config.fps_target,
                'input_size': config.input_size,
                'metadata': config.metadata,
            }
            for name, config in self.MODELS.items()
        }

        with open(output_path, 'w') as f:
            if output_path.endswith('.json'):
                json.dump(config_data, f, indent=2)
            else:
                yaml.dump(config_data, f, default_flow_style=False)

        logger.info(f"Configuration saved to: {output_path}")


# Convenience function
def get_model(model_name: str, device: str = "cuda") -> Any:
    """Convenience function to load a model.

    Args:
        model_name: Name of the model.
        device: Device to load on.

    Returns:
        Loaded model instance.
    """
    registry = ModelRegistry()
    return registry.load_model(model_name, device)


if __name__ == "__main__":
    # Demo usage
    print("SwimVision Pro - Model Registry")
    print("=" * 60)
    print("\nAvailable models:")
    for name in ModelRegistry.list_available_models():
        config = ModelRegistry.get_config(name)
        print(f"  - {name:20s} ({config.framework.value:12s}) {config.fps_target:3d} FPS")

    print("\nFast models (>=45 FPS):")
    for name in ModelRegistry.filter_by_speed(45):
        print(f"  - {name}")

    print("\n3D capable models:")
    for name in ModelRegistry.filter_by_capability(ModelCapability.POSE_3D):
        print(f"  - {name}")

    print("\nRecommended models:")
    print(f"  Fastest: {ModelRegistry.get_best_model('fastest')}")
    print(f"  Most Accurate: {ModelRegistry.get_best_model('accurate')}")
    print(f"  Balanced: {ModelRegistry.get_best_model('balanced')}")
    print(f"  Best 3D: {ModelRegistry.get_best_model('3d')}")
