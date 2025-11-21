"""
3D Reconstruction Module
Handles 3D human mesh reconstruction from 2D pose estimates.

Supported approaches:
- **Temporal 2D→3D Lifting**:
  - MotionAGFormer: Fast attention-guided temporal lifting (27-243 frames)
  - PoseFormerV2: Frequency-domain temporal lifting (robust to noise)

- **Single-Image 3D Mesh**:
  - SAM3D Body (Meta): Single-image 3D mesh reconstruction

- **Multi-Camera 3D**:
  - FreeMoCap integration: 3D triangulation from multiple views

- **Unified Pipeline**:
  - Pipeline3D: Orchestrates all 3D reconstruction approaches
  - Configurable presets for different use cases
"""

from src.reconstruction.motionagformer_estimator import MotionAGFormerEstimator, SequenceBuffer
from src.reconstruction.poseformerv2_estimator import PoseFormerV2Estimator
from src.reconstruction.sam3d_estimator import SAM3DBodyEstimator, SAM3DOutput
from src.reconstruction.pipeline_3d import (
    Pipeline3D,
    ReconstructionConfig,
    create_pipeline,
    PRESET_REALTIME,
    PRESET_HIGH_QUALITY,
    PRESET_UNDERWATER,
    PRESET_MULTICAM,
)

__all__ = [
    # Temporal lifters
    "MotionAGFormerEstimator",
    "PoseFormerV2Estimator",
    "SequenceBuffer",
    # Mesh reconstruction
    "SAM3DBodyEstimator",
    "SAM3DOutput",
    # Unified pipeline
    "Pipeline3D",
    "ReconstructionConfig",
    "create_pipeline",
    # Presets
    "PRESET_REALTIME",
    "PRESET_HIGH_QUALITY",
    "PRESET_UNDERWATER",
    "PRESET_MULTICAM",
]
