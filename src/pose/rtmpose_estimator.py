"""RTMPose estimator wrapper using MMPose framework.

RTMPose is a high-performance real-time pose estimation model from OpenMMLab.
Achieves excellent speed/accuracy tradeoff for swimming analysis.

Variants:
- rtmpose-t: Tiny (90+ FPS, 68.5 AP)
- rtmpose-s: Small (60+ FPS, 72.2 AP)
- rtmpose-m: Medium (45+ FPS, 75.8 AP) - RECOMMENDED
- rtmpose-l: Large (30+ FPS, 76.7 AP)
"""

from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from src.pose.base_estimator import (
    BasePoseEstimator,
    KeypointFormat,
)
from src.utils.device_utils import get_optimal_device

try:
    import mmcv
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples, split_instances

    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False


class RTMPoseEstimator(BasePoseEstimator):
    """RTMPose wrapper for real-time pose estimation."""

    # Model configurations
    MODEL_CONFIGS = {
        "rtmpose-t": {
            "config": "rtmpose-t_8xb256-420e_coco-256x192.py",
            "checkpoint": "rtmpose-t_simcc-coco_pt-aic-coco_420e-256x192.pth",
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-coco_pt-aic-coco_420e-256x192-aff1f1fb_20230126.pth",
        },
        "rtmpose-s": {
            "config": "rtmpose-s_8xb256-420e_coco-256x192.py",
            "checkpoint": "rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth",
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth",
        },
        "rtmpose-m": {
            "config": "rtmpose-m_8xb256-420e_coco-256x192.py",
            "checkpoint": "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth",
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
        },
        "rtmpose-l": {
            "config": "rtmpose-l_8xb256-420e_coco-256x192.py",
            "checkpoint": "rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192.pth",
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth",
        },
    }

    def __init__(
        self,
        model_variant: str = "rtmpose-m",
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str = "auto",
        confidence: float = 0.5,
        models_dir: str = "models/rtmpose",
    ):
        """Initialize RTMPose estimator.

        Args:
            model_variant: Model variant (rtmpose-t/s/m/l)
            config_path: Path to config file (auto-resolved if None)
            checkpoint_path: Path to checkpoint (auto-resolved if None)
            device: Device to run on ('cuda', 'mps', 'cpu', or 'auto' for auto-detection)
            confidence: Confidence threshold for keypoints
            models_dir: Directory containing model files
        """
        # Auto-detect optimal device if not specified
        if device == "auto":
            device = get_optimal_device()
        else:
            device = get_optimal_device(preferred=device)

        super().__init__(f"rtmpose-{model_variant}", device, confidence)

        if not MMPOSE_AVAILABLE:
            raise ImportError(
                "MMPose not installed. Install with:\n"
                "  pip install -U openmim\n"
                "  mim install mmengine mmcv mmpose"
            )

        self.model_variant = model_variant
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Resolve config and checkpoint paths
        if model_variant in self.MODEL_CONFIGS:
            model_info = self.MODEL_CONFIGS[model_variant]

            if config_path is None:
                # Try to find config in MMPose installation
                config_path = self._find_config(model_info["config"])

            if checkpoint_path is None:
                checkpoint_path = self.models_dir / model_info["checkpoint"]
                # Download if not exists
                if not checkpoint_path.exists():
                    self._download_checkpoint(model_info["url"], checkpoint_path)
        else:
            if config_path is None or checkpoint_path is None:
                raise ValueError(
                    f"Unknown model variant: {model_variant}. "
                    f"Available: {list(self.MODEL_CONFIGS.keys())}"
                )

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        self.load_model()

    def _find_config(self, config_name: str) -> str:
        """Find config file in MMPose installation.

        Args:
            config_name: Config file name

        Returns:
            Path to config file
        """
        try:
            import mmpose

            mmpose_path = Path(mmpose.__file__).parent

            # Search for config
            possible_paths = [
                mmpose_path
                / ".mim"
                / "configs"
                / "body_2d_keypoint"
                / "rtmpose"
                / "coco"
                / config_name,
                mmpose_path / "configs" / "body_2d_keypoint" / "rtmpose" / "coco" / config_name,
            ]

            for path in possible_paths:
                if path.exists():
                    return str(path)

            # If not found, return name and let MMPose handle it
            return f"rtmpose/{config_name}"

        except Exception:
            # Fall back to config name
            return f"rtmpose/{config_name}"

    def _download_checkpoint(self, url: str, save_path: Path):
        """Download model checkpoint.

        Args:
            url: Download URL
            save_path: Path to save checkpoint
        """
        import urllib.request

        from tqdm import tqdm

        parsed = urlparse(url)
        if parsed.scheme not in {"https", "http"}:
            raise ValueError(f"Unsupported checkpoint URL scheme: {parsed.scheme}")
        if not parsed.netloc:
            raise ValueError("Checkpoint URL must include a network location")

        print(f"Downloading {self.model_variant} checkpoint...")
        print(f"  URL: {url}")
        print(f"  Saving to: {save_path}")

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1) as t:
            urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)  # nosec B310

        print(f"✅ Downloaded checkpoint to {save_path}")

    def load_model(self):
        """Load RTMPose model."""
        print(f"Loading {self.model_variant}...")
        print(f"  Config: {self.config_path}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Device: {self.device}")

        self.model = init_model(
            str(self.config_path), str(self.checkpoint_path), device=self.device
        )

        print(f"✅ {self.model_variant} loaded successfully")

    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
        bbox: np.ndarray | None = None,
    ) -> tuple[list[dict] | None, np.ndarray | None]:
        """Estimate pose using RTMPose.

        Args:
            image: Input image (BGR format)
            return_image: Whether to return annotated image
            bbox: Optional bounding box(es) for top-down inference
                  Format: (N, 4) or (N, 5) where each row is [x1, y1, x2, y2] or [x1, y1, x2, y2, score]

        Returns:
            Tuple of (pose_data_list, annotated_image)
            pose_data_list: List of pose dictionaries (one per person)
            annotated_image: Image with pose overlay (if return_image=True)
        """
        # If no bbox provided, use whole image (single person mode)
        if bbox is None:
            h, w = image.shape[:2]
            bbox = np.array([[0, 0, w, h, 1.0]])

        # Ensure bbox has score column
        if bbox.shape[1] == 4:
            scores = np.ones((bbox.shape[0], 1))
            bbox = np.hstack([bbox, scores])

        # Run inference
        results = inference_topdown(self.model, image, bboxes=bbox)

        # Extract pose data
        pose_data_list = []

        for idx, result in enumerate(results):
            # Get keypoints from result
            if hasattr(result, "pred_instances"):
                pred_instances = result.pred_instances

                keypoints = pred_instances.keypoints[0]  # (17, 2)
                scores = pred_instances.keypoint_scores[0]  # (17,)

                # Combine into (17, 3) format
                keypoints_with_scores = np.concatenate([keypoints, scores.reshape(-1, 1)], axis=1)

                # Get bounding box
                if hasattr(pred_instances, "bboxes"):
                    bbox_pred = pred_instances.bboxes[0].cpu().numpy()
                else:
                    bbox_pred = bbox[idx, :4]

                pose_data = self._format_output(keypoints_with_scores, bbox_pred, idx, image.shape)

                # Filter by confidence
                if np.mean(scores) > self.confidence:
                    pose_data_list.append(pose_data)

        # Return None if no detections
        if len(pose_data_list) == 0:
            return None, image if return_image else None

        # Create annotated image
        annotated_image = None
        if return_image:
            annotated_image = self._draw_poses(image.copy(), pose_data_list)

        return pose_data_list, annotated_image

    def _format_output(
        self,
        keypoints: np.ndarray,
        bbox: np.ndarray,
        person_id: int,
        image_shape: tuple[int, int, int],
    ) -> dict:
        """Format RTMPose output to standard format.

        Args:
            keypoints: Keypoints with scores (17, 3)
            bbox: Bounding box [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
            person_id: Person ID
            image_shape: Image shape

        Returns:
            Formatted pose data
        """
        # COCO-17 keypoint names
        keypoint_names = [
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

        # Build dictionary format compatible with YOLO
        keypoints_dict = {}
        for i, name in enumerate(keypoint_names):
            keypoints_dict[name] = {
                "x": float(keypoints[i, 0]),
                "y": float(keypoints[i, 1]),
                "confidence": float(keypoints[i, 2]),
            }

        return {
            "keypoints": keypoints_dict,  # Dictionary format for compatibility
            "keypoints_array": keypoints,  # Array format for legacy code
            "keypoint_names": keypoint_names,
            "bbox": bbox[:4].tolist() if len(bbox) >= 4 else bbox.tolist(),
            "person_id": person_id,
            "format": self.get_keypoint_format(),
            "confidence": float(np.mean(keypoints[:, 2])),
            "metadata": {
                "model": self.model_variant,
                "confidence_threshold": self.confidence,
                "image_shape": image_shape,
            },
        }

    def _draw_poses(
        self,
        image: np.ndarray,
        pose_data_list: list[dict],
    ) -> np.ndarray:
        """Draw poses on image.

        Args:
            image: Input image
            pose_data_list: List of pose data dictionaries

        Returns:
            Annotated image
        """
        # COCO skeleton connections
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

        # Colors for different people
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]

        for idx, pose_data in enumerate(pose_data_list):
            # Use array format for drawing
            keypoints = pose_data.get("keypoints_array", pose_data["keypoints"])
            color = colors[idx % len(colors)]

            # Draw skeleton
            for start_idx, end_idx in skeleton:
                if (
                    keypoints[start_idx, 2] > self.confidence
                    and keypoints[end_idx, 2] > self.confidence
                ):
                    pt1 = tuple(keypoints[start_idx, :2].astype(int))
                    pt2 = tuple(keypoints[end_idx, :2].astype(int))

                    cv2.line(image, pt1, pt2, color, 2)

            # Draw keypoints
            for kp in keypoints:
                if kp[2] > self.confidence:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(image, (x, y), 4, color, -1)
                    cv2.circle(image, (x, y), 4, (255, 255, 255), 1)

            # Draw bounding box
            bbox = pose_data["bbox"]
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw person ID
            cv2.putText(
                image, f"Person {idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return image

    def get_keypoint_format(self) -> KeypointFormat:
        """Get keypoint format."""
        return KeypointFormat.COCO_17

    def supports_3d(self) -> bool:
        """RTMPose is 2D only."""
        return False

    def supports_multi_person(self) -> bool:
        """RTMPose supports multi-person detection."""
        return True

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "name": self.model_variant,
            "framework": "MMPose",
            "keypoint_format": "COCO-17",
            "num_keypoints": 17,
            "supports_3d": False,
            "supports_multi_person": True,
            "config": str(self.config_path),
            "checkpoint": str(self.checkpoint_path),
        }


def demo_rtmpose():
    """Demo RTMPose estimator."""
    print("RTMPose Estimator Demo")
    print("=" * 60)

    # Create estimator
    print("\nInitializing RTMPose (medium variant)...")
    estimator = RTMPoseEstimator(
        model_variant="rtmpose-m",
        device="cpu",  # Use CPU for demo
        confidence=0.5,
    )

    print("\nModel info:")
    info = estimator.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Create test image
    print("\nCreating test image...")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 120, 140)  # Gray background

    # Run inference
    print("\nRunning inference...")
    poses, annotated = estimator.estimate_pose(test_image, return_image=True)

    if poses:
        print(f"✅ Detected {len(poses)} person(s)")
        for idx, pose in enumerate(poses):
            avg_conf = np.mean(pose["keypoints"][:, 2])
            print(
                f"  Person {idx}: {len(pose['keypoints'])} keypoints, avg confidence: {avg_conf:.3f}"
            )
    else:
        print("  No poses detected (expected on blank image)")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_rtmpose()
