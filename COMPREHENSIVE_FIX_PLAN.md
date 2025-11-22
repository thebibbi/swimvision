# Comprehensive Fix Plan for SwimVision

**Date:** 2025-11-22  
**Status:** In Progress  
**Priority:** Critical

## 🔴 Critical Issues Identified

### 1. Abstract Method Missing
- **Issue:** `YOLOPoseEstimator` missing `supports_multi_person()` method
- **Impact:** Cannot instantiate YOLO estimator, blocking all pose estimation
- **Status:** ✅ FIXED

### 2. Model Loading Failures
- **Issue:** Models failing to load across multiple estimators
- **Root Causes:**
  - Missing dependencies in wrong environment
  - Incorrect device string formats
  - Model file paths not resolved correctly
  - Missing model downloads
- **Status:** 🔄 IN PROGRESS

### 3. Dual Environment Complexity
- **Issue:** `venv` vs `venv_advanced` causing confusion and testing difficulties
- **Impact:**
  - Users don't know which environment to use
  - Dependencies split across environments
  - Testing requires switching environments
- **Status:** 📋 PLANNED

### 4. FreeMoCap Not Visible
- **Issue:** FreeMoCap integration exists but not exposed in Streamlit UI
- **Impact:** Users can't access multi-camera 3D reconstruction
- **Status:** 📋 PLANNED

### 5. Slow Package Installation
- **Issue:** pip is slow, especially for large ML packages
- **Solution:** Migrate to `uv` (10-100x faster than pip)
- **Status:** 📋 PLANNED

---

## 📋 Action Plan

### Phase 1: Critical Fixes (Immediate)
- [x] Fix `YOLOPoseEstimator.supports_multi_person()`
- [ ] Audit all estimators for missing abstract methods
- [ ] Fix model loading in each estimator:
  - [ ] YOLO11 - verify ultralytics installation
  - [ ] MediaPipe - check mediapipe package
  - [ ] RTMPose - verify MMPose + checkpoints
  - [ ] ViTPose - verify MMPose + checkpoints
  - [ ] AlphaPose - check installation
  - [ ] OpenPose - check installation
  - [ ] SMPL-X - check model files

### Phase 2: Environment Consolidation (High Priority)
- [ ] Create unified `pyproject.toml` with optional dependencies
- [ ] Migrate to `uv` package manager
- [ ] Create single environment with feature flags:
  ```toml
  [project.optional-dependencies]
  basic = ["ultralytics", "mediapipe", ...]
  advanced = ["mmpose", "mmcv", "mmengine", ...]
  reconstruction = ["smplx", "trimesh", "pyrender", ...]
  freemocap = ["freemocap", "aniposelib", ...]
  ```
- [ ] Update setup scripts to use `uv`
- [ ] Document environment setup clearly

### Phase 3: FreeMoCap Integration (Medium Priority)
- [ ] Add FreeMoCap to Streamlit UI pose model list
- [ ] Create FreeMoCap estimator wrapper
- [ ] Add multi-camera setup UI
- [ ] Document FreeMoCap workflow

### Phase 4: Testing & Validation (High Priority)
- [ ] Create integration tests for each estimator
- [ ] Test model loading on fresh environment
- [ ] Verify all models work end-to-end
- [ ] Performance benchmarks

---

## 🔧 Implementation Details

### Environment Consolidation with `uv`

#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

#### Create pyproject.toml
```toml
[project]
name = "swimvision"
version = "2.0.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0,<2.0.0",
    "opencv-python>=4.8.0",
    "streamlit>=1.28.0",
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    # ... core deps
]

[project.optional-dependencies]
yolo = ["ultralytics>=8.3.0"]
mediapipe = ["mediapipe>=0.10.0"]
mmpose = [
    "openmim>=0.3.9",
    "mmengine>=0.10.0",
    "mmcv>=2.0.0",
    "mmpose>=1.2.0",
]
tracking = ["lap>=0.4.0", "filterpy>=1.4.5"]
reconstruction = ["smplx>=0.1.28", "trimesh>=4.0.0", "pyrender>=0.1.45"]
freemocap = ["freemocap>=0.3.0", "aniposelib>=0.4.0"]

# Convenience groups
basic = ["swimvision[yolo,mediapipe]"]
advanced = ["swimvision[mmpose,tracking,reconstruction]"]
all = ["swimvision[basic,advanced,freemocap]"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

#### New Setup Commands
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install basic features
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[basic]"

# Install advanced features
uv pip install -e ".[advanced]"

# Install everything
uv pip install -e ".[all]"
```

### Model Loading Fixes

#### 1. YOLO11 (Already Working)
```python
# Verify installation
uv pip install ultralytics

# Test
from ultralytics import YOLO
model = YOLO("yolo11n-pose.pt")
```

#### 2. MediaPipe
```python
# Install
uv pip install mediapipe

# Test
import mediapipe as mp
mp_pose = mp.solutions.pose
```

#### 3. RTMPose/ViTPose (MMPose)
```bash
# Install MMPose properly
uv pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose>=1.2.0"

# Download checkpoints
mkdir -p models/rtmpose models/vitpose
# Add checkpoint download logic
```

#### 4. AlphaPose
```bash
# Clone and install
git clone https://github.com/MVIG-SJTU/AlphaPose.git external/AlphaPose
cd external/AlphaPose
uv pip install -r requirements.txt
python setup.py install
```

#### 5. OpenPose
```bash
# OpenPose requires system-level installation
# Document in setup guide
```

#### 6. SMPL-X
```bash
uv pip install smplx
# Download models from https://smpl-x.is.tue.mpg.de/
```

### FreeMoCap Integration

#### Add to app.py
```python
pose_models = [
    "YOLO11",
    "MediaPipe",
    "OpenPose",
    "AlphaPose",
    "RTMPose",
    "ViTPose",
    "SMPL-X",
    "FreeMoCap",  # NEW
    "Multi-Model Fusion",
]

# Add FreeMoCap initialization
elif pose_model_type == "FreeMoCap":
    camera_count = st.number_input("Number of Cameras", min_value=1, max_value=8, value=4)
    calibration_file = st.file_uploader("Upload Calibration File", type=["toml", "json"])

    if calibration_file:
        from src.integration.freemocap_bridge import FreeMoCapBridge
        bridge = FreeMoCapBridge(
            camera_count=camera_count,
            calibration_path=calibration_file.name
        )
        st.success(f"FreeMoCap initialized with {camera_count} cameras")
```

---

## 🧪 Testing Strategy

### 1. Unit Tests for Each Estimator
```python
# tests/test_estimators.py
import pytest
from src.pose import *

@pytest.mark.parametrize("estimator_class,kwargs", [
    (YOLOPoseEstimator, {"model_name": "yolo11n-pose.pt"}),
    (MediaPipeEstimator, {"model_complexity": 1}),
    (RTMPoseEstimator, {"model_variant": "rtmpose-m"}),
    (ViTPoseEstimator, {"model_variant": "vitpose-b"}),
])
def test_estimator_instantiation(estimator_class, kwargs):
    """Test that each estimator can be instantiated."""
    estimator = estimator_class(**kwargs)
    assert estimator is not None
    assert hasattr(estimator, 'estimate_pose')
    assert hasattr(estimator, 'supports_multi_person')
    assert hasattr(estimator, 'supports_3d')
    assert hasattr(estimator, 'get_keypoint_format')

def test_estimator_inference(estimator_class, kwargs, test_image):
    """Test that each estimator can run inference."""
    estimator = estimator_class(**kwargs)
    pose_data, annotated = estimator.estimate_pose(test_image)
    assert pose_data is not None or annotated is not None
```

### 2. Integration Tests
```python
# tests/test_integration.py
def test_streamlit_app_loads():
    """Test that Streamlit app loads without errors."""
    # Use streamlit testing framework
    pass

def test_pipeline_end_to_end():
    """Test full pipeline from video to analysis."""
    from src.pipeline.orchestrator import SwimVisionPipeline
    pipeline = SwimVisionPipeline(pose_model="yolo11n-pose.pt")
    results = pipeline.process_video("test_video.mp4")
    assert results is not None
```

---

## 📊 Success Criteria

- [ ] All estimators can be instantiated without errors
- [ ] All estimators can run inference on test image
- [ ] Streamlit app loads and displays all model options
- [ ] FreeMoCap visible and functional in UI
- [ ] Single environment setup (no venv vs venv_advanced confusion)
- [ ] Setup time reduced by 50%+ with uv
- [ ] Comprehensive documentation updated
- [ ] All tests passing

---

## 📝 Documentation Updates Needed

1. **README.md** - Update installation instructions for uv
2. **SETUP.md** - Simplify to single environment
3. **MODELS.md** - Document each model's requirements and capabilities
4. **FREEMOCAP.md** - Add FreeMoCap setup and usage guide
5. **TROUBLESHOOTING.md** - Common issues and solutions

---

## 🚀 Migration Path for Existing Users

```bash
# 1. Backup current environment
mv venv venv_backup
mv venv_advanced venv_advanced_backup

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create new unified environment
uv venv
source .venv/bin/activate

# 4. Install desired features
uv pip install -e ".[all]"  # or [basic], [advanced]

# 5. Test
python -c "from src.pose import YOLOPoseEstimator; print('Success!')"
streamlit run app.py

# 6. Remove backups once verified
rm -rf venv_backup venv_advanced_backup
```

---

## ⏱️ Timeline

- **Phase 1 (Critical Fixes):** 2-4 hours
- **Phase 2 (Environment Consolidation):** 4-6 hours
- **Phase 3 (FreeMoCap Integration):** 2-3 hours
- **Phase 4 (Testing & Validation):** 3-4 hours

**Total Estimated Time:** 11-17 hours

---

## 🎯 Next Immediate Actions

1. ✅ Fix YOLOPoseEstimator.supports_multi_person() - DONE
2. Audit all other estimators for missing methods
3. Create pyproject.toml with optional dependencies
4. Install and test uv package manager
5. Create unified setup script
6. Test each estimator individually
7. Add FreeMoCap to UI
8. Write integration tests
9. Update documentation
