"""
Unified 3D Reconstruction Pipeline
===================================

Orchestrates all 3D reconstruction components:
- MotionAGFormer: Fast temporal 2D→3D lifting
- PoseFormerV2: Robust temporal 2D→3D lifting (frequency domain)
- SAM3D Body: Detailed 3D mesh reconstruction
- FreeMoCap: Multi-camera 3D triangulation

Provides flexible configuration for different use cases:
- Real-time 3D pose (MotionAGFormer-XS)
- High-quality offline 3D (MotionAGFormer-B + SAM3D)
- Multi-camera 3D (FreeMoCap triangulation)
- Combined approaches (temporal + multi-camera)

Usage:
    from src.reconstruction.pipeline_3d import Pipeline3D, ReconstructionConfig

    # Real-time configuration
    config = ReconstructionConfig(
        temporal_model='motionagformer',
        temporal_variant='xs',
        enable_mesh=False
    )

    # Create pipeline
    pipeline = Pipeline3D(config)

    # Process video
    results = pipeline.process_video("swimming.mp4", poses_2d)
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TemporalModel(str, Enum):
    """Temporal 2D→3D lifting models."""
    MOTIONAGFORMER = "motionagformer"
    POSEFORMERV2 = "poseformerv2"
    NONE = "none"


class MeshModel(str, Enum):
    """3D mesh reconstruction models."""
    SAM3D = "sam3d"
    NONE = "none"


@dataclass
class ReconstructionConfig:
    """Configuration for 3D reconstruction pipeline."""

    # Temporal 3D lifting
    temporal_model: str = "motionagformer"  # motionagformer, poseformerv2, or none
    temporal_variant: str = "xs"  # xs, s, b, l (for MotionAGFormer)
    sequence_length: int = 27  # 27, 81, or 243 frames

    # Mesh reconstruction
    enable_mesh: bool = False
    mesh_model: str = "sam3d"
    mesh_keyframe_interval: int = 30  # Process every Nth frame for mesh

    # Multi-camera
    enable_multicamera: bool = False
    camera_count: int = 1
    calibration_path: Optional[str] = None

    # Device
    device: str = "auto"

    # Processing
    batch_size: int = 32
    confidence_threshold: float = 0.3

    # Output
    export_format: str = "npz"  # npz, bvh, fbx, c3d
    export_freemocap_compatible: bool = True


class Pipeline3D:
    """
    Unified 3D reconstruction pipeline.

    Orchestrates multiple 3D reconstruction approaches and
    provides a simple API for processing swimming videos.
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize 3D reconstruction pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or ReconstructionConfig()

        # Initialize components
        self.temporal_lifter = None
        self.mesh_estimator = None
        self.freemocap_bridge = None

        self._initialize_components()

        logger.info(
            f"Pipeline3D initialized: temporal={self.config.temporal_model}, "
            f"mesh={self.config.enable_mesh}, "
            f"multi-cam={self.config.enable_multicamera}"
        )

    def _initialize_components(self):
        """Initialize reconstruction components based on configuration."""

        # Temporal lifter
        if self.config.temporal_model == TemporalModel.MOTIONAGFORMER:
            try:
                from src.reconstruction.motionagformer_estimator import MotionAGFormerEstimator
                self.temporal_lifter = MotionAGFormerEstimator(
                    model_variant=self.config.temporal_variant,
                    sequence_length=self.config.sequence_length,
                    device=self.config.device
                )
                logger.info("✓ MotionAGFormer loaded")
            except Exception as e:
                logger.error(f"Failed to load MotionAGFormer: {e}")

        elif self.config.temporal_model == TemporalModel.POSEFORMERV2:
            try:
                from src.reconstruction.poseformerv2_estimator import PoseFormerV2Estimator
                self.temporal_lifter = PoseFormerV2Estimator(
                    sequence_length=self.config.sequence_length,
                    device=self.config.device
                )
                logger.info("✓ PoseFormerV2 loaded")
            except Exception as e:
                logger.error(f"Failed to load PoseFormerV2: {e}")

        # Mesh estimator
        if self.config.enable_mesh:
            if self.config.mesh_model == MeshModel.SAM3D:
                try:
                    from src.reconstruction.sam3d_estimator import SAM3DBodyEstimator
                    self.mesh_estimator = SAM3DBodyEstimator(
                        model_name="vit-h",
                        device=self.config.device
                    )
                    logger.info("✓ SAM3D Body loaded")
                except Exception as e:
                    logger.error(f"Failed to load SAM3D: {e}")

        # FreeMoCap bridge for multi-camera
        if self.config.enable_multicamera:
            try:
                from src.integration.freemocap_bridge import FreeMoCapBridge
                self.freemocap_bridge = FreeMoCapBridge(
                    camera_count=self.config.camera_count,
                    calibration_path=self.config.calibration_path
                )
                logger.info("✓ FreeMoCap bridge initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FreeMoCap: {e}")

    def process_frame(
        self,
        image: np.ndarray,
        pose_2d: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process a single frame through the 3D reconstruction pipeline.

        Args:
            image: Input image (H x W x 3)
            pose_2d: 2D pose keypoints (17 x 3)

        Returns:
            Dictionary with 3D reconstruction results
        """
        result = {
            'has_2d': pose_2d is not None,
            'has_3d_temporal': False,
            'has_3d_mesh': False,
        }

        # Temporal 3D lifting
        if self.temporal_lifter is not None and pose_2d is not None:
            try:
                pose_3d = self.temporal_lifter.add_frame_2d(pose_2d)
                if pose_3d is not None:
                    result['pose_3d_temporal'] = pose_3d
                    result['has_3d_temporal'] = True
            except Exception as e:
                logger.error(f"Temporal lifting failed: {e}")

        # Mesh reconstruction (keyframes only)
        # Note: Mesh is expensive, only do on keyframes
        result['is_keyframe'] = False

        return result

    def process_video(
        self,
        video_path: Optional[str] = None,
        poses_2d: Optional[List[Dict]] = None,
        images: Optional[List[np.ndarray]] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a video through the complete 3D reconstruction pipeline.

        Args:
            video_path: Path to video file (alternative to images)
            poses_2d: List of 2D pose dictionaries per frame
            images: List of image frames (alternative to video_path)
            show_progress: Show progress bar

        Returns:
            List of per-frame 3D reconstruction results
        """
        from tqdm import tqdm

        # Load video if path provided
        if video_path is not None:
            images = self._load_video_frames(video_path)

        if images is None:
            raise ValueError("Either video_path or images must be provided")

        num_frames = len(images)
        logger.info(f"Processing {num_frames} frames")

        results = []

        # Process with temporal lifter
        if self.temporal_lifter is not None and poses_2d is not None:
            logger.info("Phase 1: Temporal 3D lifting")
            temporal_results = self.temporal_lifter.process_video(
                poses_2d,
                show_progress=show_progress
            )
        else:
            temporal_results = [None] * num_frames

        # Process mesh reconstruction on keyframes
        mesh_results = [None] * num_frames
        if self.mesh_estimator is not None:
            logger.info(
                f"Phase 2: Mesh reconstruction (every {self.config.mesh_keyframe_interval} frames)"
            )

            keyframe_indices = range(0, num_frames, self.config.mesh_keyframe_interval)
            pbar = tqdm(
                total=len(keyframe_indices),
                desc="Mesh reconstruction",
                disable=not show_progress
            )

            for idx in keyframe_indices:
                if idx < len(images):
                    try:
                        kpts_2d = poses_2d[idx]['keypoints'] if poses_2d and idx < len(poses_2d) else None
                        mesh_output = self.mesh_estimator.estimate(
                            images[idx],
                            keypoints_2d=kpts_2d
                        )
                        mesh_results[idx] = mesh_output
                    except Exception as e:
                        logger.error(f"Mesh reconstruction failed at frame {idx}: {e}")

                pbar.update(1)

            pbar.close()

        # Combine results
        logger.info("Phase 3: Combining results")
        for i in range(num_frames):
            frame_result = {
                'frame_idx': i,
                'has_2d': poses_2d is not None and i < len(poses_2d) and poses_2d[i] is not None,
            }

            # Add 2D data
            if poses_2d and i < len(poses_2d) and poses_2d[i]:
                frame_result['keypoints_2d'] = poses_2d[i]['keypoints']
                frame_result['confidence_2d'] = np.mean(poses_2d[i]['keypoints'][:, 2])

            # Add temporal 3D
            if temporal_results[i] is not None:
                frame_result['has_3d_temporal'] = temporal_results[i]['has_3d']
                if temporal_results[i]['has_3d']:
                    frame_result['pose_3d_temporal'] = temporal_results[i]['keypoints_3d']

            # Add mesh 3D (keyframes)
            if mesh_results[i] is not None:
                frame_result['has_3d_mesh'] = True
                frame_result['mesh_vertices'] = mesh_results[i].vertices
                frame_result['mesh_faces'] = mesh_results[i].faces
                frame_result['is_keyframe'] = True
            else:
                frame_result['has_3d_mesh'] = False
                frame_result['is_keyframe'] = False

            results.append(frame_result)

        logger.info(f"✓ Processed {num_frames} frames")
        return results

    def process_multicamera(
        self,
        images_per_camera: Dict[int, List[np.ndarray]],
        poses_2d_per_camera: Dict[int, List[Dict]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multi-camera footage with 3D triangulation.

        Args:
            images_per_camera: Dict mapping camera_id -> list of frames
            poses_2d_per_camera: Dict mapping camera_id -> list of 2D poses
            show_progress: Show progress bar

        Returns:
            List of per-frame 3D reconstruction results (triangulated)
        """
        if self.freemocap_bridge is None:
            raise ValueError("Multi-camera not enabled (enable_multicamera=False)")

        from tqdm import tqdm

        # Get number of frames (assume all cameras have same count)
        num_frames = len(next(iter(images_per_camera.values())))
        logger.info(f"Processing {num_frames} frames from {len(images_per_camera)} cameras")

        results = []
        pbar = tqdm(total=num_frames, desc="Multi-camera 3D", disable=not show_progress)

        for frame_idx in range(num_frames):
            # Collect poses from all cameras for this frame
            frame_poses_2d = {}
            for cam_id, poses_list in poses_2d_per_camera.items():
                if frame_idx < len(poses_list) and poses_list[frame_idx] is not None:
                    frame_poses_2d[cam_id] = poses_list[frame_idx]

            # Triangulate to 3D
            if len(frame_poses_2d) >= 2:
                pose_3d = self.freemocap_bridge.triangulate_frames(frame_poses_2d)
            else:
                pose_3d = None

            result = {
                'frame_idx': frame_idx,
                'has_3d_multicam': pose_3d is not None,
                'n_cameras_detected': len(frame_poses_2d),
            }

            if pose_3d is not None:
                result['pose_3d_multicam'] = pose_3d['points_3d']
                result['confidence_3d'] = pose_3d['confidence']

            results.append(result)
            pbar.update(1)

        pbar.close()
        logger.info(f"✓ Triangulated {num_frames} frames")
        return results

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load all frames from video file."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        return frames

    def export_results(
        self,
        results: List[Dict],
        output_dir: str,
        session_name: str = "reconstruction"
    ):
        """
        Export 3D reconstruction results in various formats.

        Args:
            results: List of per-frame results
            output_dir: Output directory path
            session_name: Name for this session
        """
        output_path = Path(output_dir) / session_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as NPZ (default)
        self._save_npz(results, output_path)

        # Save FreeMoCap-compatible format if requested
        if self.config.export_freemocap_compatible and self.freemocap_bridge:
            try:
                self.freemocap_bridge.export_to_freemocap(
                    results,
                    output_dir,
                    session_name
                )
            except Exception as e:
                logger.error(f"FreeMoCap export failed: {e}")

        logger.info(f"✓ Results exported to {output_path}")

    def _save_npz(self, results: List[Dict], output_path: Path):
        """Save results as NPZ file."""
        # Extract arrays from results
        data = {}

        # Collect 2D poses
        poses_2d = []
        for r in results:
            if r.get('keypoints_2d') is not None:
                poses_2d.append(r['keypoints_2d'])
            else:
                poses_2d.append(np.zeros((17, 3)))

        data['poses_2d'] = np.array(poses_2d)

        # Collect temporal 3D poses
        if any(r.get('has_3d_temporal') for r in results):
            poses_3d_temporal = []
            for r in results:
                if r.get('pose_3d_temporal') is not None:
                    poses_3d_temporal.append(r['pose_3d_temporal'])
                else:
                    poses_3d_temporal.append(np.zeros((17, 3)))
            data['poses_3d_temporal'] = np.array(poses_3d_temporal)

        # Collect multi-camera 3D poses
        if any(r.get('has_3d_multicam') for r in results):
            poses_3d_multicam = []
            for r in results:
                if r.get('pose_3d_multicam') is not None:
                    poses_3d_multicam.append(r['pose_3d_multicam'])
                else:
                    poses_3d_multicam.append(np.zeros((17, 3)))
            data['poses_3d_multicam'] = np.array(poses_3d_multicam)

        # Save
        output_file = output_path / "reconstruction_results.npz"
        np.savez(output_file, **data)
        logger.info(f"Saved NPZ: {output_file}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about loaded pipeline components."""
        info = {
            'temporal_model': self.config.temporal_model,
            'temporal_enabled': self.temporal_lifter is not None,
            'mesh_enabled': self.mesh_estimator is not None,
            'multicamera_enabled': self.freemocap_bridge is not None,
            'device': self.config.device,
        }

        if self.temporal_lifter is not None and hasattr(self.temporal_lifter, 'get_model_info'):
            info['temporal_info'] = self.temporal_lifter.get_model_info()

        return info

    def close(self):
        """Cleanup resources."""
        if self.temporal_lifter is not None:
            if hasattr(self.temporal_lifter, 'close'):
                self.temporal_lifter.close()
        if self.mesh_estimator is not None:
            if hasattr(self.mesh_estimator, 'close'):
                self.mesh_estimator.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Convenience presets
PRESET_REALTIME = ReconstructionConfig(
    temporal_model="motionagformer",
    temporal_variant="xs",
    sequence_length=27,
    enable_mesh=False,
)

PRESET_HIGH_QUALITY = ReconstructionConfig(
    temporal_model="motionagformer",
    temporal_variant="b",
    sequence_length=243,
    enable_mesh=True,
    mesh_keyframe_interval=15,
)

PRESET_UNDERWATER = ReconstructionConfig(
    temporal_model="poseformerv2",  # Frequency domain better for noise
    sequence_length=81,
    enable_mesh=False,
)

PRESET_MULTICAM = ReconstructionConfig(
    temporal_model="none",
    enable_multicamera=True,
    camera_count=4,
)


def create_pipeline(preset: str = "realtime", **kwargs) -> Pipeline3D:
    """
    Create a 3D reconstruction pipeline with preset configuration.

    Args:
        preset: Preset name ('realtime', 'high_quality', 'underwater', 'multicam')
        **kwargs: Override configuration parameters

    Returns:
        Initialized Pipeline3D

    Example:
        >>> pipeline = create_pipeline('realtime', device='cuda')
        >>> results = pipeline.process_video("swim.mp4", poses_2d)
    """
    presets = {
        'realtime': PRESET_REALTIME,
        'high_quality': PRESET_HIGH_QUALITY,
        'underwater': PRESET_UNDERWATER,
        'multicam': PRESET_MULTICAM,
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset]

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return Pipeline3D(config)
