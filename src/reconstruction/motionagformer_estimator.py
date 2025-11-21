"""
MotionAGFormer 2D→3D Pose Lifter
=================================

MotionAGFormer is a state-of-the-art temporal 2D-to-3D pose lifting model
that uses attention-guided mechanisms and graph convolutions.

Paper: "MotionAGFormer: Enhancing 3D Human Pose Estimation with
        Attention-Guided Graph Convolutions" (WACV 2024)

Key Features:
- Dual-stream architecture (Transformer + GCN)
- Temporal modeling with 27, 81, or 243 frame windows
- 4 model variants: XS (2.2M params, 300 FPS), S (4.8M, 200 FPS),
                     B (11.7M, 100 FPS), L (19.0M, 50 FPS)
- 38.4mm MPJPE on Human3.6M dataset
- Robust to occlusions and missing keypoints

Usage:
    from src.reconstruction.motionagformer_estimator import MotionAGFormerEstimator

    # Create lifter
    lifter = MotionAGFormerEstimator(
        model_variant='xs',
        sequence_length=27,
        device='cuda'
    )

    # Process sequence of 2D poses
    poses_2d = [...]  # List of 2D pose arrays
    pose_3d = lifter.lift_to_3d(poses_2d)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Conditional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


@dataclass
class MotionAGFormerConfig:
    """Configuration for MotionAGFormer model."""
    model_variant: str = "xs"  # xs, s, b, l
    sequence_length: int = 27  # 27, 81, or 243 frames
    num_joints: int = 17  # COCO-17 format
    device: str = "auto"
    model_path: Optional[str] = None
    confidence_threshold: float = 0.3

    # Model-specific parameters
    variant_params = {
        'xs': {'hidden_dim': 128, 'num_layers': 4, 'expected_fps': 300},
        's': {'hidden_dim': 256, 'num_layers': 6, 'expected_fps': 200},
        'b': {'hidden_dim': 512, 'num_layers': 8, 'expected_fps': 100},
        'l': {'hidden_dim': 768, 'num_layers': 12, 'expected_fps': 50},
    }

    def get_params(self) -> Dict:
        """Get parameters for the selected variant."""
        return self.variant_params.get(self.model_variant.lower(), self.variant_params['xs'])


class SequenceBuffer:
    """
    Buffer for maintaining temporal sequences of 2D poses.

    Handles frame buffering, interpolation for missing frames,
    and sequence extraction for temporal models.
    """

    def __init__(
        self,
        sequence_length: int = 27,
        num_joints: int = 17,
        interpolate_missing: bool = True
    ):
        """
        Initialize sequence buffer.

        Args:
            sequence_length: Number of frames in sequence
            num_joints: Number of keypoints per frame
            interpolate_missing: Whether to interpolate missing frames
        """
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.interpolate_missing = interpolate_missing

        # Use deque for efficient FIFO operations
        self.buffer = deque(maxlen=sequence_length)
        self.frame_count = 0

    def add_frame(self, keypoints: Optional[np.ndarray]):
        """
        Add a frame to the buffer.

        Args:
            keypoints: 2D keypoints (num_joints x 3) or None if no detection
        """
        if keypoints is None:
            # Add placeholder for missing frame
            keypoints = np.zeros((self.num_joints, 3))
            keypoints[:, 2] = 0.0  # Zero confidence

        self.buffer.append(keypoints)
        self.frame_count += 1

    def get_sequence(self, center_index: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get a sequence of frames centered at specified index.

        Args:
            center_index: Index to center sequence around (default: latest frame)

        Returns:
            Sequence of 2D poses (sequence_length x num_joints x 3) or None
        """
        if len(self.buffer) < self.sequence_length:
            return None

        # Convert deque to array
        sequence = np.array(list(self.buffer))

        if self.interpolate_missing:
            sequence = self._interpolate_missing_frames(sequence)

        return sequence

    def _interpolate_missing_frames(self, sequence: np.ndarray) -> np.ndarray:
        """
        Interpolate missing frames based on confidence scores.

        Args:
            sequence: Input sequence (T x num_joints x 3)

        Returns:
            Interpolated sequence
        """
        # Find frames with low confidence
        confidences = np.mean(sequence[:, :, 2], axis=1)
        missing_mask = confidences < 0.1

        if not np.any(missing_mask):
            return sequence

        # Linear interpolation for missing frames
        for joint_idx in range(self.num_joints):
            for dim in range(2):  # x, y coordinates
                values = sequence[:, joint_idx, dim]
                valid_mask = ~missing_mask

                if np.any(valid_mask):
                    # Interpolate missing values
                    valid_indices = np.where(valid_mask)[0]
                    missing_indices = np.where(missing_mask)[0]

                    if len(valid_indices) > 1:
                        interpolated = np.interp(
                            missing_indices,
                            valid_indices,
                            values[valid_indices]
                        )
                        sequence[missing_indices, joint_idx, dim] = interpolated

        return sequence

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for a sequence."""
        return len(self.buffer) >= self.sequence_length

    def reset(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.frame_count = 0


class MotionAGFormerEstimator:
    """
    MotionAGFormer wrapper for 2D→3D pose lifting.

    Lifts sequences of 2D poses to 3D using temporal attention mechanisms.
    """

    def __init__(
        self,
        model_variant: str = "xs",
        sequence_length: int = 27,
        device: str = "auto",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize MotionAGFormer estimator.

        Args:
            model_variant: Model size ('xs', 's', 'b', 'l')
            sequence_length: Temporal window size (27, 81, or 243)
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
            model_path: Path to model weights (None to download)
            confidence_threshold: Minimum confidence for input poses
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = MotionAGFormerConfig(
            model_variant=model_variant,
            sequence_length=sequence_length,
            device=device,
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        # Determine device
        self.device = self._get_device()

        # Create sequence buffer
        self.sequence_buffer = SequenceBuffer(
            sequence_length=sequence_length,
            num_joints=17,  # COCO-17
            interpolate_missing=True
        )

        # Load model (placeholder - actual implementation needed)
        self.model = None
        self._load_model()

        logger.info(
            f"MotionAGFormer initialized: variant={model_variant}, "
            f"seq_len={sequence_length}, device={self.device}"
        )

    def _get_device(self) -> str:
        """Determine the device to use."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device

    def _load_model(self):
        """
        Load MotionAGFormer model with pre-trained weights.
        """
        try:
            from src.reconstruction.motionagformer_loader import load_motionagformer_model

            self.model = load_motionagformer_model(
                variant=self.config.model_variant,
                device=self.device,
                checkpoint_path=self.config.model_path,
                download_if_missing=False  # User should run download script
            )
            logger.info(f"MotionAGFormer model loaded successfully")

        except FileNotFoundError as e:
            logger.warning(
                f"Pre-trained weights not found: {e}\n"
                f"To download weights, run: bash scripts/download_motionagformer_weights.sh\n"
                f"Model will run in placeholder mode (zero depth estimation)"
            )
            self.model = None

        except ImportError as e:
            logger.warning(
                f"Failed to load MotionAGFormer: {e}\n"
                f"Make sure the repository is cloned at models/motionagformer/\n"
                f"Model will run in placeholder mode"
            )
            self.model = None

        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            self.model = None

    def add_frame_2d(self, pose_2d: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Add a 2D pose frame and get 3D pose if sequence is ready.

        Args:
            pose_2d: 2D keypoints (17 x 3) or None

        Returns:
            3D pose (17 x 3) if sequence is ready, None otherwise
        """
        # Add frame to buffer
        self.sequence_buffer.add_frame(pose_2d)

        # Check if we can process a sequence
        if not self.sequence_buffer.is_ready():
            return None

        # Get sequence and lift to 3D
        sequence_2d = self.sequence_buffer.get_sequence()
        pose_3d = self._lift_sequence(sequence_2d)

        return pose_3d

    def lift_to_3d(
        self,
        poses_2d: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Lift a sequence of 2D poses to 3D.

        Args:
            poses_2d: Sequence of 2D poses (T x 17 x 3) or list of arrays

        Returns:
            3D poses for each frame (T x 17 x 3)
        """
        if isinstance(poses_2d, list):
            poses_2d = np.array(poses_2d)

        if len(poses_2d.shape) != 3:
            raise ValueError(f"Expected 3D array, got shape {poses_2d.shape}")

        # Process sequences with sliding window
        num_frames = poses_2d.shape[0]
        poses_3d = np.zeros((num_frames, 17, 3))

        # Reset buffer
        self.sequence_buffer.reset()

        for i in range(num_frames):
            pose_3d = self.add_frame_2d(poses_2d[i])
            if pose_3d is not None:
                # Center frame index in sequence
                center_idx = i
                poses_3d[center_idx] = pose_3d

        return poses_3d

    def _lift_sequence(self, sequence_2d: np.ndarray) -> np.ndarray:
        """
        Lift a sequence of 2D poses to 3D using MotionAGFormer.

        Args:
            sequence_2d: Sequence of 2D poses (T x 17 x 3)

        Returns:
            3D pose for center frame (17 x 3)
        """
        if self.model is None:
            # Placeholder: return simple depth estimation
            logger.warning("Using placeholder depth estimation (model not loaded)")
            center_idx = len(sequence_2d) // 2
            pose_3d = np.zeros((17, 3))
            pose_3d[:, :2] = sequence_2d[center_idx, :, :2]  # x, y from 2D
            pose_3d[:, 2] = 0.0  # Zero depth
            return pose_3d

        # Actual model inference
        try:
            # Prepare input: extract x,y coordinates only (T x 17 x 2)
            sequence_xy = sequence_2d[:, :, :2]

            # Reshape for model: (1, T, 17, 2) -> (1, T, 17*2)
            batch_size = 1
            T, J, _ = sequence_xy.shape
            sequence_flat = sequence_xy.reshape(batch_size, T, -1)

            # Convert to torch tensor
            sequence_tensor = torch.from_numpy(sequence_flat).float().to(self.device)

            # Run inference
            with torch.no_grad():
                # Model outputs 3D pose for center frame
                pose_3d_tensor = self.model(sequence_tensor)

                # Reshape output: (1, 17*3) -> (17, 3)
                pose_3d = pose_3d_tensor.cpu().numpy().reshape(J, 3)

            return pose_3d

        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            # Fallback to placeholder
            center_idx = len(sequence_2d) // 2
            pose_3d = np.zeros((17, 3))
            pose_3d[:, :2] = sequence_2d[center_idx, :, :2]
            pose_3d[:, 2] = 0.0
            return pose_3d

    def batch_lift(
        self,
        sequences_2d: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Lift multiple sequences in batches for efficiency.

        Args:
            sequences_2d: Batch of sequences (B x T x 17 x 3)
            batch_size: Batch size for processing

        Returns:
            3D poses (B x 17 x 3)
        """
        num_sequences = sequences_2d.shape[0]
        poses_3d = np.zeros((num_sequences, 17, 3))

        for i in range(0, num_sequences, batch_size):
            batch = sequences_2d[i:i+batch_size]
            # Process batch
            for j, sequence in enumerate(batch):
                poses_3d[i+j] = self._lift_sequence(sequence)

        return poses_3d

    def process_video(
        self,
        poses_2d: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process a video's 2D poses to 3D.

        Args:
            poses_2d: List of per-frame 2D pose dictionaries
            show_progress: Whether to show progress bar

        Returns:
            List of per-frame 3D pose dictionaries
        """
        from tqdm import tqdm

        results = []
        self.sequence_buffer.reset()

        pbar = tqdm(
            total=len(poses_2d),
            desc="Lifting to 3D",
            disable=not show_progress
        )

        for frame_data in poses_2d:
            if frame_data is None:
                pose_3d = self.add_frame_2d(None)
            else:
                pose_3d = self.add_frame_2d(frame_data['keypoints'])

            result = {
                'keypoints_3d': pose_3d,
                'has_3d': pose_3d is not None,
                'sequence_ready': self.sequence_buffer.is_ready(),
            }

            if frame_data is not None:
                result['keypoints_2d'] = frame_data['keypoints']
                result['confidence'] = np.mean(frame_data['keypoints'][:, 2])

            results.append(result)
            pbar.update(1)

        pbar.close()
        return results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        params = self.config.get_params()
        return {
            'variant': self.config.model_variant,
            'sequence_length': self.config.sequence_length,
            'expected_fps': params['expected_fps'],
            'hidden_dim': params['hidden_dim'],
            'num_layers': params['num_layers'],
            'device': self.device,
            'num_parameters': self._count_parameters(),
        }

    def _count_parameters(self) -> int:
        """Count model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def close(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None
        self.sequence_buffer.reset()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def create_lifter(
    variant: str = "xs",
    sequence_length: int = 27,
    **kwargs
) -> MotionAGFormerEstimator:
    """
    Convenience function to create a MotionAGFormer lifter.

    Args:
        variant: Model variant ('xs', 's', 'b', 'l')
        sequence_length: Temporal window size
        **kwargs: Additional arguments

    Returns:
        Initialized MotionAGFormerEstimator

    Example:
        >>> lifter = create_lifter('xs', sequence_length=27)
        >>> pose_3d = lifter.lift_to_3d(poses_2d_sequence)
    """
    return MotionAGFormerEstimator(
        model_variant=variant,
        sequence_length=sequence_length,
        **kwargs
    )
