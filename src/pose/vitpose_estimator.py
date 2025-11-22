"""ViTPose estimator wrapper using the MMPose framework."""

from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from src.pose.base_estimator import BasePoseEstimator, KeypointFormat
from src.utils.device_utils import get_optimal_device, normalize_device_for_framework

try:  # pragma: no cover - runtime dependency
    from mmpose.apis import inference_topdown, init_model

    MMPOSE_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    MMPOSE_AVAILABLE = False


class ViTPoseEstimator(BasePoseEstimator):
    """ViTPose wrapper for transformer-based 2D pose estimation."""

    MODEL_CONFIGS = {
        "vitpose-b": {
            "config": "td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
            "checkpoint": "vitpose-b.pth",
            "url": "https://download.openmmlab.com/mmpose/v1/projects/vitpose/vitpose-b.pth",
        },
        "vitpose-l": {
            "config": "td-hm_ViTPose-large_8xb64-210e_coco-256x192.py",
            "checkpoint": "vitpose-l.pth",
            "url": "https://download.openmmlab.com/mmpose/v1/projects/vitpose/vitpose-l.pth",
        },
        "vitpose-h": {
            "config": "td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py",
            "checkpoint": "vitpose-h.pth",
            "url": "https://download.openmmlab.com/mmpose/v1/projects/vitpose/vitpose-h.pth",
        },
    }

    def __init__(
        self,
        model_variant: str = "vitpose-b",
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str = "auto",
        confidence: float = 0.5,
        models_dir: str = "models/vitpose",
    ) -> None:
        if device == "auto":
            device = get_optimal_device()
        else:
            device = get_optimal_device(preferred=device)

        super().__init__(model_variant, device, confidence)

        if not MMPOSE_AVAILABLE:
            raise ImportError(
                "MMPose not installed. Install with:\n"
                "  pip install -U openmim\n"
                "  mim install mmengine mmcv mmpose"
            )

        self.model_variant = model_variant
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        model_info = self.MODEL_CONFIGS.get(model_variant)
        if model_info is None:
            raise ValueError(
                f"Unknown ViTPose variant: {model_variant}. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.config_path = (
            self._resolve_config_path(model_info["config"]) if config_path is None else config_path
        )
        self.checkpoint_path = (
            self._resolve_checkpoint(model_info["checkpoint"], model_info["url"])
            if checkpoint_path is None
            else Path(checkpoint_path)
        )

        # Normalize device for MMPose (needs "cuda" not "cuda:0")
        self.mmpose_device = normalize_device_for_framework(self.device, "mmpose")

        self.load_model()

    def _resolve_config_path(self, config_name: str) -> str:
        try:
            import mmpose

            mmpose_root = Path(mmpose.__file__).parent
            candidate_paths = [
                mmpose_root
                / ".mim"
                / "configs"
                / "body_2d_keypoint"
                / "vitpose"
                / "coco"
                / config_name,
                mmpose_root / "configs" / "body_2d_keypoint" / "vitpose" / "coco" / config_name,
            ]

            for path in candidate_paths:
                if path.exists():
                    return str(path)

            return f"vitpose/{config_name}"
        except Exception:  # pragma: no cover - fallback
            return f"vitpose/{config_name}"

    def _resolve_checkpoint(self, checkpoint_name: str, url: str) -> Path:
        checkpoint_path = self.models_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path

        self._download_checkpoint(url, checkpoint_path)
        return checkpoint_path

    @staticmethod
    def _download_checkpoint(url: str, save_path: Path) -> None:
        import urllib.request

        from tqdm import tqdm

        parsed = urlparse(url)
        if parsed.scheme not in {"https", "http"}:
            raise ValueError(f"Unsupported checkpoint URL scheme: {parsed.scheme}")
        if not parsed.netloc:
            raise ValueError("Checkpoint URL must include a network location")

        print(f"Downloading {save_path.name}...")
        print(f"  URL: {url}")
        print(f"  Saving to: {save_path}")

        class DownloadProgressBar(tqdm):
            def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1) as progress:
            urllib.request.urlretrieve(url, save_path, reporthook=progress.update_to)  # nosec B310

        print(f"✅ Downloaded checkpoint to {save_path}")

    def load_model(self) -> None:
        print(f"Loading {self.model_variant}...")
        print(f"  Config: {self.config_path}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Device: {self.device}")

        self.model = init_model(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.mmpose_device,
        )

        print(f"✅ {self.model_variant} loaded successfully")

    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
        bbox: np.ndarray | None = None,
    ) -> tuple[list[dict] | None, np.ndarray | None]:
        if bbox is None:
            height, width = image.shape[:2]
            bbox = np.array([[0, 0, width, height, 1.0]])

        if bbox.shape[1] == 4:
            scores = np.ones((bbox.shape[0], 1))
            bbox = np.hstack([bbox, scores])

        results = inference_topdown(self.model, image, bboxes=bbox)

        pose_data_list: list[dict] = []
        for idx, result in enumerate(results):
            if not hasattr(result, "pred_instances"):
                continue

            pred_instances = result.pred_instances
            keypoints = pred_instances.keypoints[0]
            scores = pred_instances.keypoint_scores[0]
            keypoints_with_scores = np.concatenate([keypoints, scores.reshape(-1, 1)], axis=1)

            if hasattr(pred_instances, "bboxes"):
                bbox_pred = pred_instances.bboxes[0].cpu().numpy()
            else:
                bbox_pred = bbox[idx, :4]

            pose_data = self._format_output(keypoints_with_scores, bbox_pred, idx, image.shape)
            if float(np.mean(scores)) > self.confidence:
                pose_data_list.append(pose_data)

        annotated_image = None
        if return_image and pose_data_list:
            annotated_image = self._draw_poses(image.copy(), pose_data_list)

        if not pose_data_list:
            return None, annotated_image if return_image else None

        return pose_data_list, annotated_image

    def _format_output(
        self,
        keypoints: np.ndarray,
        bbox: np.ndarray,
        person_id: int,
        image_shape: tuple[int, int, int],
    ) -> dict:
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

        keypoints_dict = {
            name: {
                "x": float(keypoints[i, 0]),
                "y": float(keypoints[i, 1]),
                "confidence": float(keypoints[i, 2]),
            }
            for i, name in enumerate(keypoint_names)
        }

        return {
            "keypoints": keypoints_dict,
            "keypoints_array": keypoints,
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

    def _draw_poses(self, image: np.ndarray, pose_data_list: list[dict]) -> np.ndarray:
        skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ]

        for idx, pose_data in enumerate(pose_data_list):
            keypoints = pose_data.get("keypoints_array", pose_data["keypoints"])
            color = colors[idx % len(colors)]

            for start_idx, end_idx in skeleton:
                if (
                    keypoints[start_idx, 2] > self.confidence
                    and keypoints[end_idx, 2] > self.confidence
                ):
                    pt1 = tuple(keypoints[start_idx, :2].astype(int))
                    pt2 = tuple(keypoints[end_idx, :2].astype(int))
                    cv2.line(image, pt1, pt2, color, 2)

            for kp in keypoints:
                if kp[2] > self.confidence:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(image, (x, y), 4, color, -1)
                    cv2.circle(image, (x, y), 4, (255, 255, 255), 1)

            x1, y1, x2, y2 = map(int, pose_data["bbox"][:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"Person {idx}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return image

    def get_keypoint_format(self) -> KeypointFormat:
        return KeypointFormat.COCO_17

    def supports_3d(self) -> bool:
        return False

    def supports_multi_person(self) -> bool:
        return True
