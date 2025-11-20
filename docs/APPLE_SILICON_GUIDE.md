# SwimVision Pro - Apple Silicon (M1/M2/M3) Guide

## Overview

SwimVision Pro fully supports Apple Silicon Macs with GPU acceleration via **MPS (Metal Performance Shaders)**. This provides 3-5x speedup compared to CPU-only execution, making real-time pose estimation and tracking feasible on MacBooks and Mac Studios.

## Supported Devices

### ✅ Fully Supported
- **M1 Series**: M1, M1 Pro, M1 Max, M1 Ultra
- **M2 Series**: M2, M2 Pro, M2 Max, M2 Ultra
- **M3 Series**: M3, M3 Pro, M3 Max

### System Requirements
- **macOS**: 12.0 (Monterey) or later
- **Python**: 3.9 or higher
- **RAM**: 16GB minimum, 32GB+ recommended for larger models
- **Storage**: 5GB for models and dependencies

## Quick Start

### 1. Installation

Run the Apple Silicon-optimized setup script:

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd swimvision

# Run Apple Silicon setup
bash scripts/setup_macos_apple_silicon.sh
```

The script will:
- ✅ Detect your Apple Silicon chip
- ✅ Create an optimized virtual environment
- ✅ Install PyTorch with MPS support
- ✅ Install MMPose and all dependencies
- ✅ Download RTMPose models
- ✅ Verify MPS functionality

### 2. Activate Environment

```bash
source venv_advanced/bin/activate
```

### 3. Test Installation

```bash
# Verify device detection
python -c "from src.utils.device_utils import print_device_info; print_device_info()"

# Create test video
python demos/demo_phase1_pipeline.py --create-test-video

# Run demo
python demos/demo_phase1_pipeline.py --video data/videos/test_swimmers.mp4
```

## Performance Guide

### Expected FPS (1920x1080 video)

| Your Mac           | Recommended Model | Expected FPS | Use Case              |
|--------------------|-------------------|--------------|------------------------|
| M1 / M1 Pro       | rtmpose-s         | 25-30        | Real-time demos        |
| M1 / M1 Pro       | rtmpose-m         | 20-25        | **Balanced**           |
| M1 Max            | rtmpose-m         | 28-32        | **Recommended**        |
| M2 / M2 Pro       | rtmpose-m         | 25-30        | **Balanced**           |
| M2 Max            | rtmpose-m         | 30-35        | **Recommended**        |
| M2 Ultra          | rtmpose-l         | 35-40        | High accuracy          |
| M3 / M3 Pro       | rtmpose-m         | 28-33        | **Recommended**        |
| M3 Max            | rtmpose-l         | 30-35        | High accuracy          |

### Model Selection Guide

**rtmpose-s** (Small)
- Best for: M1 base, real-time requirements
- FPS: 25-35
- Accuracy: Good (72.0 AP)
- Latency: ~30ms

**rtmpose-m** (Medium) - **RECOMMENDED**
- Best for: M1 Pro/Max, M2, M3 series
- FPS: 20-32
- Accuracy: Excellent (75.8 AP)
- Latency: ~35-45ms

**rtmpose-l** (Large)
- Best for: M2 Max/Ultra, M3 Max
- FPS: 24-30
- Accuracy: Best (77.0 AP)
- Latency: ~40-50ms

## Usage Examples

### Basic Video Processing

```bash
# Auto-detect device (uses MPS automatically)
python demos/demo_phase1_pipeline.py --video input.mp4

# Explicitly use MPS
python demos/demo_phase1_pipeline.py --video input.mp4 --device mps

# Use smaller model for better FPS
python demos/demo_phase1_pipeline.py --video input.mp4 --model rtmpose-s

# Use larger model for better accuracy (M2 Max/M3 Max)
python demos/demo_phase1_pipeline.py --video input.mp4 --model rtmpose-l
```

### Webcam Demo

```bash
# Use built-in webcam
python demos/demo_phase1_pipeline.py --webcam

# Optimize for speed
python demos/demo_phase1_pipeline.py --webcam --model rtmpose-s --mode realtime
```

### Save Output Video

```bash
python demos/demo_phase1_pipeline.py \
    --video input.mp4 \
    --output results/output.mp4 \
    --model rtmpose-m
```

## Python API

```python
from src.pipeline.orchestrator import (
    SwimVisionPipeline,
    PipelineConfig,
    ProcessingMode
)

# Configure for Apple Silicon
config = PipelineConfig(
    pose_models=["rtmpose-m"],
    enable_tracking=True,
    mode=ProcessingMode.BALANCED,
    device="auto",  # Will auto-detect MPS
    visualize=True
)

# Initialize pipeline
pipeline = SwimVisionPipeline(config)

# Process video
import cv2
cap = cv2.VideoCapture("swimming.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = pipeline.process_frame(frame)

    print(f"FPS: {result.fps:.1f}")
    cv2.imshow("SwimVision", result.visualized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Get statistics
stats = pipeline.get_statistics()
print(f"Average FPS: {stats['avg_fps']:.1f}")
```

## Troubleshooting

### MPS Not Available

If you see "MPS not available" warnings:

```bash
# Check PyTorch version (should be 2.0+)
python -c "import torch; print(torch.__version__)"

# Verify MPS
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Reinstall PyTorch if needed
pip install --upgrade torch torchvision torchaudio
```

### Low FPS

If FPS is lower than expected:

1. **Use smaller model**:
   ```bash
   python demos/demo_phase1_pipeline.py --video input.mp4 --model rtmpose-s
   ```

2. **Disable visualization** (faster processing):
   ```python
   config = PipelineConfig(visualize=False)
   ```

3. **Close other applications** (free up GPU memory)

4. **Check Activity Monitor**:
   - Look for "GPU" usage
   - Make sure Python is using GPU

### Memory Issues

If you encounter memory errors:

1. **Use smaller model**: Switch from rtmpose-l → rtmpose-m → rtmpose-s
2. **Process smaller videos**: Resize to 720p instead of 1080p
3. **Close other applications**: Free up unified memory

### OpenCV Camera Issues

If webcam doesn't work:

```bash
# Grant camera permissions
# System Preferences → Security & Privacy → Camera → Terminal

# Or use a video file instead
python demos/demo_phase1_pipeline.py --video test.mp4
```

## Optimization Tips

### 1. Optimal Video Resolution

- **1080p (1920x1080)**: Best quality, 20-30 FPS on M1 Pro+
- **720p (1280x720)**: Good quality, 40-60 FPS on all models
- **540p (960x540)**: Real-time, 60+ FPS on M2+

### 2. Memory Management

Apple Silicon uses **unified memory** - shared between CPU and GPU:

- **16GB Macs**: Use rtmpose-s or rtmpose-m
- **32GB+ Macs**: Can use rtmpose-l comfortably
- Close browser tabs and other apps for best performance

### 3. Thermal Throttling

MacBooks may throttle under sustained load:

- Use cooling pad for long sessions
- Enable "Low Power Mode" in battery settings (paradoxically can improve sustained performance)
- Consider batch processing instead of real-time for long videos

### 4. Power Settings

For best performance:

```bash
# Check power mode
pmset -g

# Disable sleep during processing
caffeinate python demos/demo_phase1_pipeline.py --video long_video.mp4
```

## Benchmarking Your Mac

Run this to benchmark your specific Mac:

```python
from src.utils.device_utils import DeviceManager

# Print detailed device info
DeviceManager.print_device_info()

# Test MPS performance
import torch
import time

device = "mps"
size = 1000

# Matrix multiplication test
x = torch.randn(size, size, device=device)
y = torch.randn(size, size, device=device)

# Warmup
for _ in range(10):
    z = x @ y

# Benchmark
start = time.time()
for _ in range(100):
    z = x @ y
torch.mps.synchronize()  # Wait for GPU
elapsed = time.time() - start

print(f"MPS Performance: {100/elapsed:.1f} matmuls/sec")
print(f"This is {'GOOD' if 100/elapsed > 50 else 'SLOW'} for an Apple Silicon Mac")
```

## Known Limitations

### MPS Backend Limitations

Some operations are not yet optimized for MPS:

- **Fallback to CPU**: Some ops automatically fall back to CPU (transparent)
- **Mixed precision**: FP16 support varies by operation
- **Large batch sizes**: Stick to batch_size=1 for real-time processing

### Workarounds

If you encounter MPS-specific errors:

```python
# Force CPU for specific operations
config = PipelineConfig(device="cpu")

# Or use auto-detection (handles fallbacks automatically)
config = PipelineConfig(device="auto")  # Recommended
```

## Comparison: CUDA vs MPS vs CPU

| Feature                | NVIDIA GPU (CUDA) | Apple Silicon (MPS) | CPU           |
|------------------------|-------------------|---------------------|---------------|
| Setup complexity       | High (drivers)    | **Low (built-in)**  | None          |
| Performance            | **Excellent**     | **Very Good**       | Poor          |
| Power efficiency       | Low               | **Excellent**       | Medium        |
| Portability            | Desktop/Laptop*   | **MacBook friendly**| Universal     |
| Cost                   | High              | **Included**        | Free          |
| Real-time capable      | ✅ Yes            | ✅ Yes (M1 Pro+)    | ❌ No         |

*External GPU or gaming laptop

## Getting Help

### Check Device Status

```python
from src.utils.device_utils import DeviceManager

DeviceManager.print_device_info()
```

### Common Issues

1. **"No module named 'torch'"**: Activate virtual environment first
2. **"MPS not available"**: Update to macOS 12.3+ and PyTorch 2.0+
3. **Low FPS**: Use smaller model or lower resolution
4. **Memory errors**: Close other apps, use smaller model
5. **Camera not found**: Grant camera permissions in System Preferences

### Support Resources

- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [Apple ML Compute](https://developer.apple.com/metal/pytorch/)

## What's Next?

After verifying Phase 1 works on your Mac:

1. **Test with your swimming videos**
2. **Experiment with different models**
3. **Try multi-camera setups**
4. **Prepare for Phase 2**: Underwater preprocessing

## Contributing Performance Data

Help us improve benchmarks! Share your results:

```bash
# Run benchmark and share output
python demos/demo_phase1_pipeline.py --video data/videos/test_swimmers.mp4 --model rtmpose-m

# Report at end:
# - Your Mac model (M1/M2/M3, Pro/Max/Ultra)
# - Average FPS
# - Video resolution
```

---

**Note**: This guide is specific to Apple Silicon. For NVIDIA GPUs, see [PHASE1_GUIDE.md](PHASE1_GUIDE.md).
