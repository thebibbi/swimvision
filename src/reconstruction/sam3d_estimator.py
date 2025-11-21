"""
SAM3D Body Estimator
Wrapper for Meta's SAM3D Body model for 3D human mesh reconstruction.

SAM3D Body advantages for swimming:
- Robust to occlusions (underwater limbs)
- Single-image reconstruction (no multi-view required)
- Handles extreme poses (swimming positions)
- Can use 2D keypoint prompts from RTMPose
- Multi-modal outputs (mesh, depth, normals, masks)

Author: SwimVision Pro Team
Date: 2025-01-20
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check if SAM3D is available
try:
    from sam_3d_body import SAM3DBodyEstimator as SAM3DEstimatorBase
    from sam_3d_body import load_sam_3d_body_hf

    SAM3D_AVAILABLE = True
except ImportError:
    SAM3DEstimatorBase = object
    SAM3D_AVAILABLE = False
    logger.warning("SAM3D Body not installed. Install with: pip install sam-3d-body")


@dataclass
class SAM3DOutput:
    """Output from SAM3D Body model."""

    # 3D mesh
    vertices: np.ndarray  # (N, 3) 3D vertex positions
    faces: np.ndarray  # (M, 3) triangle faces

    # MHR parameters
    mhr_pose: np.ndarray  # Skeletal pose parameters
    mhr_shape: np.ndarray  # Shape parameters

    # Additional outputs
    depth: np.ndarray | None = None  # (H, W) depth map
    normals: np.ndarray | None = None  # (H, W, 3) surface normals
    mask: np.ndarray | None = None  # (H, W) segmentation mask

    # Joint positions
    joints_3d: np.ndarray | None = None  # (J, 3) 3D joint positions

    # Metadata
    confidence: float = 1.0
    processing_time: float = 0.0


class SAM3DBodyEstimator:
    """
    Wrapper for Meta's SAM3D Body model.

    Integrates with SwimVision pipeline to provide 3D mesh reconstruction
    from 2D pose estimates.

    Features:
    - Single-image 3D reconstruction
    - Optional 2D keypoint prompts from RTMPose
    - Robust to occlusions (underwater limbs)
    - Multi-modal outputs (mesh, depth, normals)
    - Temporal smoothing for video

    Example:
        # Basic usage
        estimator = SAM3DBodyEstimator(model_name="facebook/sam-3d-body-dinov3")
        output = estimator.estimate(image)

        # With 2D keypoint prompts
        output = estimator.estimate(image, keypoints_2d=rtmpose_keypoints)

        # Process video
        outputs = estimator.process_video(video_path, keypoints_sequence)
    """

    # Supported models
    MODELS = {
        "dinov3": "facebook/sam-3d-body-dinov3",  # 840M params, 54.8 MPJPE
        "vit-h": "facebook/sam-3d-body-vit-h",  # 631M params, 54.8 MPJPE (recommended)
    }

    def __init__(
        self,
        model_name: str = "vit-h",
        device: str = "auto",
        use_hand_refinement: bool = True,
        cache_dir: str | None = None,
    ):
        """
        Initialize SAM3D Body estimator.

        Args:
            model_name: Model variant ("dinov3" or "vit-h")
            device: Device to use ("cuda", "mps", "cpu", or "auto")
            use_hand_refinement: Enable hand detail refinement
            cache_dir: Directory to cache model checkpoints
        """
        if not SAM3D_AVAILABLE:
            raise ImportError(
                "SAM3D Body not installed. Install with:\n"
                "  pip install sam-3d-body\n"
                "  or\n"
                "  pip install git+https://github.com/facebookresearch/sam-3d-body.git"
            )

        # Resolve model name
        if model_name in self.MODELS:
            model_path = self.MODELS[model_name]
        else:
            model_path = model_name  # Assume full Hugging Face path

        # Resolve device
        from src.utils.device_utils import get_optimal_device

        if device == "auto":
            self.device = get_optimal_device()
        else:
            self.device = get_optimal_device(preferred=device)

        logger.info(f"Loading SAM3D Body model: {model_path}")
        logger.info(f"Device: {self.device}")

        # Load model from Hugging Face
        try:
            self.model, self.model_cfg = load_sam_3d_body_hf(model_path, cache_dir=cache_dir)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Create estimator
            self.estimator = SAM3DEstimatorBase(
                sam_3d_body_model=self.model, model_cfg=self.model_cfg
            )

            self.use_hand_refinement = use_hand_refinement

            logger.info("✅ SAM3D Body model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load SAM3D Body model: {e}")
            raise

    def estimate(
        self,
        image: np.ndarray,
        keypoints_2d: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        return_all: bool = True,
    ) -> SAM3DOutput:
        """
        Reconstruct 3D mesh from single image.

        Args:
            image: RGB image (H, W, 3) in range [0, 255] or [0, 1]
            keypoints_2d: Optional 2D keypoints from RTMPose (17, 3) COCO format
            mask: Optional segmentation mask (H, W)
            return_all: Return all outputs (mesh, depth, normals, mask)

        Returns:
            SAM3DOutput with 3D mesh and optional additional outputs
        """
        import time

        start_time = time.time()

        # Prepare image
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Prepare prompts
        prompts = {}
        if keypoints_2d is not None:
            # Convert COCO-17 keypoints to SAM3D format
            prompts["keypoints"] = self._convert_keypoints_to_sam3d(keypoints_2d)

        if mask is not None:
            prompts["mask"] = mask

        # Run inference
        try:
            with torch.no_grad():
                outputs = self.estimator.process_one_image(
                    image,
                    **prompts,
                    return_depth=return_all,
                    return_normals=return_all,
                    return_mask=return_all,
                    use_hand_refinement=self.use_hand_refinement,
                )

            processing_time = time.time() - start_time

            # Parse outputs
            result = SAM3DOutput(
                vertices=outputs["vertices"],
                faces=outputs["faces"],
                mhr_pose=outputs["mhr_pose"],
                mhr_shape=outputs["mhr_shape"],
                depth=outputs.get("depth"),
                normals=outputs.get("normals"),
                mask=outputs.get("mask"),
                joints_3d=outputs.get("joints_3d"),
                confidence=outputs.get("confidence", 1.0),
                processing_time=processing_time,
            )

            logger.debug(f"SAM3D inference: {processing_time*1000:.1f}ms")

            return result

        except Exception as e:
            logger.error(f"SAM3D inference failed: {e}")
            raise

    def process_video(
        self,
        video_path: str,
        keypoints_sequence: list[np.ndarray] | None = None,
        smooth_temporal: bool = True,
        batch_size: int = 1,
        skip_frames: int = 1,
    ) -> list[SAM3DOutput]:
        """
        Process video with 3D reconstruction.

        Args:
            video_path: Path to video file
            keypoints_sequence: Optional list of 2D keypoints per frame from RTMPose
            smooth_temporal: Apply temporal smoothing to reduce jitter
            batch_size: Number of frames to process together (1 for now)
            skip_frames: Process every Nth frame (1 = all frames)

        Returns:
            List of SAM3DOutput for each processed frame
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing {total_frames} frames (skip={skip_frames})")

        results = []
        frame_idx = 0
        processed_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if requested
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue

                # Get keypoints for this frame if available
                kpts = None
                if keypoints_sequence is not None and processed_idx < len(keypoints_sequence):
                    kpts = keypoints_sequence[processed_idx]

                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output = self.estimate(frame_rgb, keypoints_2d=kpts)
                results.append(output)

                if processed_idx % 10 == 0:
                    logger.info(f"Processed {processed_idx}/{total_frames//skip_frames} frames")

                frame_idx += 1
                processed_idx += 1

        finally:
            cap.release()

        logger.info(f"Processed {len(results)} frames")

        # Apply temporal smoothing if requested
        if smooth_temporal and len(results) > 1:
            results = self._smooth_temporal(results)

        return results

    def _convert_keypoints_to_sam3d(self, keypoints_coco: np.ndarray) -> np.ndarray:
        """
        Convert COCO-17 keypoints from RTMPose to SAM3D format.

        Args:
            keypoints_coco: (17, 3) COCO keypoints [x, y, confidence]

        Returns:
            SAM3D format keypoints
        """
        # TODO: Implement proper conversion based on SAM3D's expected format
        # For now, pass through COCO format (SAM3D likely accepts it)
        return keypoints_coco

    def _smooth_temporal(
        self, outputs: list[SAM3DOutput], window_size: int = 5
    ) -> list[SAM3DOutput]:
        """
        Apply temporal smoothing to reduce mesh jitter.

        Args:
            outputs: List of SAM3DOutput from video
            window_size: Smoothing window size (frames)

        Returns:
            Smoothed outputs
        """
        if len(outputs) < window_size:
            return outputs

        logger.info(f"Applying temporal smoothing (window={window_size})")

        # Smooth MHR parameters
        smoothed = []
        for i, output in enumerate(outputs):
            # Get window of poses
            start = max(0, i - window_size // 2)
            end = min(len(outputs), i + window_size // 2 + 1)

            poses = [o.mhr_pose for o in outputs[start:end]]
            shapes = [o.mhr_shape for o in outputs[start:end]]

            # Average parameters
            smooth_pose = np.mean(poses, axis=0)
            smooth_shape = np.mean(shapes, axis=0)

            # Create smoothed output
            smoothed_output = SAM3DOutput(
                vertices=output.vertices,  # Will be updated based on smooth params
                faces=output.faces,
                mhr_pose=smooth_pose,
                mhr_shape=smooth_shape,
                depth=output.depth,
                normals=output.normals,
                mask=output.mask,
                joints_3d=output.joints_3d,
                confidence=output.confidence,
                processing_time=output.processing_time,
            )

            smoothed.append(smoothed_output)

        return smoothed

    def visualize_mesh(
        self,
        output: SAM3DOutput,
        image: np.ndarray | None = None,
        show_joints: bool = True,
        show_skeleton: bool = True,
    ) -> np.ndarray:
        """
        Visualize 3D mesh overlaid on image.

        Args:
            output: SAM3DOutput to visualize
            image: Optional background image
            show_joints: Show 3D joint positions
            show_skeleton: Show skeleton connections

        Returns:
            Rendered image with mesh overlay
        """
        # TODO: Implement mesh rendering using pyrender or pytorch3d
        # For now, placeholder
        if image is not None:
            return image
        else:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def export_mesh(self, output: SAM3DOutput, output_path: str, format: str = "obj"):
        """
        Export 3D mesh to file.

        Args:
            output: SAM3DOutput to export
            output_path: Path to save mesh
            format: File format ("obj", "ply", "fbx", "glb")
        """
        import trimesh

        mesh = trimesh.Trimesh(vertices=output.vertices, faces=output.faces)

        mesh.export(output_path, file_type=format)
        logger.info(f"Exported mesh to {output_path}")


def demo_sam3d():
    """Demo SAM3D Body reconstruction."""
    import cv2

    # Check if SAM3D is available
    if not SAM3D_AVAILABLE:
        print("❌ SAM3D Body not installed")
        print("Install with: pip install sam-3d-body")
        return

    # Load image
    image_path = "data/images/swimmer_test.jpg"
    if not Path(image_path).exists():
        print(f"Test image not found: {image_path}")
        print("Please provide a swimming image")
        return

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize estimator
    print("Loading SAM3D Body model...")
    estimator = SAM3DBodyEstimator(model_name="vit-h", device="auto")

    # Run reconstruction
    print("Running 3D reconstruction...")
    output = estimator.estimate(image_rgb)

    print("\nResults:")
    print(f"  Vertices: {output.vertices.shape}")
    print(f"  Faces: {output.faces.shape}")
    print(f"  Processing time: {output.processing_time*1000:.1f}ms")

    # Export mesh
    output_path = "results/swimmer_mesh.obj"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    estimator.export_mesh(output, output_path)
    print(f"  Mesh saved to: {output_path}")


if __name__ == "__main__":
    demo_sam3d()
