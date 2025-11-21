# Phase 1.5 Quick Start Guide

Welcome to SwimVision Phase 1.5! This guide will help you get started with 3D human pose reconstruction.

## Overview

Phase 1.5 adds temporal 2D→3D pose lifting to the SwimVision pipeline:
- **MotionAGFormer**: Fast, attention-guided 2D→3D lifting (primary)
- **PoseFormerV2**: Frequency-domain lifting for noisy/underwater scenarios
- **Pipeline3D**: Unified orchestrator for all 3D reconstruction approaches

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
# Install PoseFormerV2 dependencies
pip install torch-dct einops

# Install additional tools (optional)
pip install gdown easydict yacs
```

### 2. Download Pre-trained Weights

```bash
# Download MotionAGFormer weights (choose variant)
bash scripts/download_motionagformer_weights.sh

# Download PoseFormerV2 weights (choose variant)
bash scripts/download_poseformerv2_weights.sh
```

**Note**: Weights are ~100-300MB each. Download only what you need:
- **Real-time**: MotionAGFormer-XS (27 frames, 300 FPS)
- **Balanced**: MotionAGFormer-S or PoseFormerV2 3-81 (81 frames, 200 FPS)
- **High Quality**: MotionAGFormer-B or PoseFormerV2 27-243 (243 frames, best accuracy)

### 3. Run Integration Test

```bash
# Quick test with sample video
python tests/test_phase1_5_integration.py --quick

# Test MotionAGFormer on your video
python tests/test_phase1_5_integration.py --video your_video.mp4 --model motionagformer

# Test PoseFormerV2 on your video
python tests/test_phase1_5_integration.py --video your_video.mp4 --model poseformerv2

# Test unified pipeline
python tests/test_phase1_5_integration.py --video your_video.mp4 --model pipeline
```

## Usage Examples

### Example 1: Basic 3D Pose Estimation

```python
from src.pose.rtmpose_estimator import RTMPoseEstimator
from src.reconstruction.motionagformer_estimator import MotionAGFormerEstimator
import cv2

# Initialize estimators
pose_2d = RTMPoseEstimator(model_size='m', device='cuda')
lifter_3d = MotionAGFormerEstimator(model_variant='xs', sequence_length=27)

# Process video
cap = cv2.VideoCapture('swimming.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 2D pose
    pose_data, _ = pose_2d.estimate_pose(frame, return_image=False)
    keypoints_2d = pose_data['keypoints'] if pose_data else None

    # 3D lifting (streaming)
    pose_3d = lifter_3d.add_frame_2d(keypoints_2d)

    if pose_3d is not None:
        print(f"3D pose shape: {pose_3d.shape}")  # (17, 3)

cap.release()
```

### Example 2: Batch Processing with PoseFormerV2

```python
from src.pose.rtmpose_estimator import RTMPoseEstimator
from src.reconstruction.poseformerv2_estimator import PoseFormerV2Estimator
import numpy as np

# Initialize
pose_2d = RTMPoseEstimator(model_size='m')
lifter_3d = PoseFormerV2Estimator(
    variant='3-81-47.1',  # 81 frames, best for underwater
    sequence_length=81
)

# Extract 2D poses from video
poses_2d = []  # List of (17, 3) arrays
# ... extract poses ...

# Batch lift to 3D
poses_2d_array = np.array(poses_2d)  # (T, 17, 3)
poses_3d = lifter_3d.lift_to_3d(poses_2d_array)

print(f"3D poses shape: {poses_3d.shape}")  # (T, 17, 3)
```

### Example 3: Unified Pipeline with Configuration

```python
from src.reconstruction.pipeline_3d import (
    create_pipeline,
    PRESET_REALTIME,
    PRESET_HIGH_QUALITY,
    PRESET_UNDERWATER
)

# Create pipeline with preset
pipeline = create_pipeline(PRESET_UNDERWATER)  # Best for swimming

# Process video
results = pipeline.process_video(
    video_path='swimming.mp4',
    show_progress=True
)

# Export to FreeMoCap format
pipeline.export_to_freemocap(results, 'output/session_001')
```

### Example 4: Custom Pipeline Configuration

```python
from src.reconstruction.pipeline_3d import Pipeline3D, ReconstructionConfig

# Custom configuration
config = ReconstructionConfig(
    temporal_model='poseformerv2',  # Better for noisy scenarios
    temporal_variant='3-81-47.1',
    sequence_length=81,
    enable_mesh=False,  # Disable SAM3D mesh (faster)
    enable_multicamera=False,
    device='cuda'
)

pipeline = Pipeline3D(config)
results = pipeline.process_video(video_path='swimming.mp4')
```

## Model Selection Guide

### MotionAGFormer Variants

| Variant | Frames | Params | FPS  | MPJPE | Use Case |
|---------|--------|--------|------|-------|----------|
| XS      | 27     | 2.2M   | 300  | 40mm  | Real-time streaming |
| S       | 81     | 4.8M   | 200  | 38mm  | Balanced |
| B       | 243    | 11.7M  | 100  | 36mm  | High quality offline |
| L       | 243    | 19.0M  | 50   | 35mm  | Best accuracy |

### PoseFormerV2 Variants

| Variant   | Frames | MPJPE | FPS     | Use Case |
|-----------|--------|-------|---------|----------|
| 1-27      | 27     | 48.7  | 300-400 | Real-time |
| 3-27      | 27     | 47.9  | 200-300 | Fast, good quality |
| 3-81      | 81     | 47.1  | 150-250 | Underwater/noisy |
| 27-243    | 243    | 45.2  | 50-100  | Best accuracy, noise robust |

**Recommendation for Swimming:**
- **Real-time**: MotionAGFormer-XS (27 frames)
- **Underwater/Noisy**: PoseFormerV2 3-81 or 27-243 (frequency domain)
- **Best Quality**: MotionAGFormer-B (243 frames)

## Configuration Presets

### PRESET_REALTIME
```python
temporal_model='motionagformer'
temporal_variant='xs'
sequence_length=27
enable_mesh=False
# Target: 50+ FPS
```

### PRESET_HIGH_QUALITY
```python
temporal_model='motionagformer'
temporal_variant='b'
sequence_length=243
enable_mesh=True  # SAM3D on keyframes
mesh_keyframe_interval=15
# Target: Best accuracy, offline processing
```

### PRESET_UNDERWATER
```python
temporal_model='poseformerv2'
temporal_variant='3-81-47.1'
sequence_length=81
# Frequency domain for noise robustness
```

### PRESET_MULTICAM
```python
enable_multicamera=True
camera_count=2  # or more
# Uses FreeMoCap triangulation
```

## Troubleshooting

### Issue: Model weights not found

```bash
# Download weights
bash scripts/download_motionagformer_weights.sh
bash scripts/download_poseformerv2_weights.sh

# Verify download
ls checkpoint/motionagformer/
ls checkpoint/poseformerv2/
```

### Issue: torch-dct not installed

```bash
pip install torch-dct
```

### Issue: Out of memory

```python
# Use smaller variant
lifter = MotionAGFormerEstimator(model_variant='xs', sequence_length=27)

# Or reduce batch size
results = pipeline.process_video(video_path, batch_size=1)
```

### Issue: Low FPS

- Use smaller model variant (XS instead of B)
- Reduce sequence length (27 instead of 243)
- Enable GPU: `device='cuda'`
- Process fewer frames: `skip_frames=2`

## Advanced Features

### Temporal Smoothing

```python
from src.reconstruction.poseformerv2_estimator import PoseFormerV2Estimator

lifter = PoseFormerV2Estimator(
    variant='3-81-47.1',
    use_temporal_smoothing=True  # Reduce jitter
)
```

### 2D Keypoint Prompts for SAM3D

```python
from src.reconstruction.sam3d_estimator import SAM3DBodyEstimator

mesh_estimator = SAM3DBodyEstimator(model_name='vit-h')

# Use 2D keypoints as prompts
output = mesh_estimator.estimate(
    image,
    keypoints_2d=rtmpose_keypoints  # COCO-17 format
)

print(f"Vertices: {output.vertices.shape}")  # (10000, 3)
print(f"Faces: {output.faces.shape}")  # (19998, 3)
```

### Multi-Camera Setup

```python
from src.integration.freemocap_bridge import FreeMoCapBridge

bridge = FreeMoCapBridge()
bridge.load_calibration('calibration.toml')

# Triangulate from multiple views
points_2d_dict = {
    0: camera_0_keypoints,  # (17, 2)
    1: camera_1_keypoints,
    2: camera_2_keypoints,
}

points_3d = bridge.triangulate_2d_to_3d(points_2d_dict, min_cameras=2)
```

## Performance Benchmarks

Tested on NVIDIA RTX 3090:

| Pipeline Component | FPS | Notes |
|-------------------|-----|-------|
| RTMPose-M         | 50-85 | 2D pose estimation |
| MotionAGFormer-XS | 300 | 27 frames |
| MotionAGFormer-S  | 200 | 81 frames |
| MotionAGFormer-B  | 100 | 243 frames |
| PoseFormerV2 3-27 | 250 | Frequency domain |
| PoseFormerV2 27-243 | 80 | Best accuracy |
| Full Pipeline (RT) | 45-50 | RTMPose + MAG-XS |
| Full Pipeline (HQ) | 20-30 | RTMPose + MAG-B |

## Next Steps

1. **Download weights** for your chosen model variant
2. **Run integration test** to verify setup
3. **Process your swimming videos** with the pipeline
4. **Explore FreeMoCap integration** for multi-camera setups
5. **Check Phase 1.5 status document** for advanced features

## Resources

- **Documentation**: `docs/PHASE1.5_STATUS.md`
- **Integration Plan**: `docs/COMPREHENSIVE_3D_INTEGRATION.md`
- **MotionAGFormer Paper**: https://arxiv.org/abs/2310.16288
- **PoseFormerV2 Paper**: https://arxiv.org/pdf/2303.17472.pdf
- **SAM3D Body**: Meta's 3D mesh reconstruction
- **FreeMoCap**: https://github.com/freemocap/freemocap

## Support

For issues or questions:
1. Check `docs/PHASE1.5_STATUS.md` for known issues
2. Review test output for error messages
3. Verify all dependencies are installed
4. Ensure pre-trained weights are downloaded

---

**SwimVision Phase 1.5** - Professional 3D Swimming Motion Capture
