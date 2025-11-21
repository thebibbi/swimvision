# SwimVision Pro - Integration Compatibility Guide

## Executive Summary

This document provides a detailed analysis of compatibility issues and solutions for integrating all advanced features into SwimVision Pro.

## ✅ Compatible Components (No Issues)

| Component | Framework | Version | Notes |
|-----------|-----------|---------|-------|
| RTMPose | MMPose (PyTorch) | 1.2.0+ | ✅ Direct integration |
| ViTPose++ | MMPose (PyTorch) | 1.2.0+ | ✅ Same as RTMPose |
| ByteTrack | PyTorch | Any | ✅ Pure Python + NumPy |
| WHAM | PyTorch | 2.0+ | ✅ Compatible with our stack |
| 4D Gaussian Splatting | PyTorch + CUDA | 2.0+ | ✅ CUDA 11.8+ required |
| Existing YOLO11 | Ultralytics | Latest | ✅ Already integrated |
| Existing MediaPipe | Google | Latest | ✅ Already integrated |

## ⚠️  Components Requiring Adaptation

### 1. WaterNet/UWCNN (TensorFlow → PyTorch)

**Problem**: Original implementations use TensorFlow

**Solutions** (ranked by preference):

#### Option A: ONNX Conversion (RECOMMENDED)
```python
# Convert TensorFlow model to ONNX
import tf2onnx
import onnx

spec = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input"),)
output_path = "waternet.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    waternet_model,
    input_signature=spec,
    opset=13,
    output_path=output_path
)

# Use via ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("waternet.onnx")
output = session.run(None, {"input": image})
```

**Pros**:
- No rewrite needed
- Fast inference via ONNX Runtime
- Can use TensorRT backend

**Cons**:
- Need original TF checkpoint
- May have operator compatibility issues

#### Option B: PyTorch Reimplementation
```python
# Implement WaterNet architecture in PyTorch
class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Reimplement based on paper architecture
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def forward(self, x):
        features = self.encoder(x)
        enhanced = self.decoder(features)
        return enhanced
```

**Pros**:
- Full control over implementation
- Native PyTorch integration
- Can modify architecture

**Cons**:
- Requires reimplementation effort
- Need to retrain or convert weights

#### Option C: Separate TF Process (NOT RECOMMENDED)
Keep TensorFlow in separate process, communicate via pipes/sockets.

**Pros**: No modification needed

**Cons**:
- Performance overhead (IPC)
- Deployment complexity
- Memory overhead (two frameworks)

**DECISION**: Use Option A (ONNX) for initial implementation, Option B for future optimization.

---

### 2. OpenSim (Separate Ecosystem)

**Problem**: OpenSim is C++ library with Python bindings, separate from deep learning stack

**Integration Strategy**:

```python
# Wrapper class that bridges SMPL and OpenSim
class OpenSimBridge:
    def __init__(self, model_path='swimmer_model.osim'):
        import opensim
        self.model = opensim.Model(model_path)
        self.state = self.model.initSystem()

        # Define mapping from SMPL to OpenSim markers
        self.joint_mapping = {
            'pelvis': 'pelvis',
            'spine1': 'lumbar',
            'spine2': 'thorax',
            # ... complete mapping
        }

    def smpl_to_opensim_motion(self, smpl_vertices, smpl_joints):
        """Convert SMPL output to OpenSim motion format"""
        # 1. Extract joint positions from SMPL
        # 2. Map to OpenSim coordinate system
        # 3. Create OpenSim Storage object
        pass

    def compute_inverse_kinematics(self, motion):
        """Run OpenSim IK solver"""
        ik_tool = opensim.InverseKinematicsTool()
        ik_tool.setModel(self.model)
        ik_tool.run()
        return coordinates

    def compute_inverse_dynamics(self, coordinates):
        """Compute joint torques and forces"""
        id_tool = opensim.InverseDynamicsTool()
        # ...
        return forces, moments
```

**Key Challenges**:

1. **Coordinate System Differences**
   - SMPL: Y-up, meters
   - OpenSim: Y-up, meters (usually compatible)
   - Solution: May need axis transformation

2. **Joint Definition Differences**
   - SMPL: 24 joints, fixed hierarchy
   - OpenSim: Custom skeleton, variable DOF
   - Solution: Create swimmer-specific OpenSim model

3. **Missing Ground Contact**
   - Swimming has no ground reaction forces
   - Solution: Implement water resistance forces model

**DECISION**: Create `OpenSimBridge` class, start with basic IK, add ID later.

---

### 3. WHAM Real-time Adaptation

**Problem**: WHAM designed for offline video processing (processes entire video)

**Current WHAM Pipeline**:
```python
# Offline processing
video = load_video("swim.mp4")  # Load all frames
keypoints_2d = extract_keypoints(video)  # All frames
poses_3d = wham.predict(video, keypoints_2d)  # Batch processing
```

**Real-time Adaptation Strategy**:

```python
class RealtimeWHAM:
    def __init__(self, buffer_size=30):  # 1 second at 30 FPS
        self.wham = load_wham_model()
        self.frame_buffer = deque(maxlen=buffer_size)
        self.keypoint_buffer = deque(maxlen=buffer_size)

    def add_frame(self, frame, keypoints_2d):
        """Add frame to sliding window"""
        self.frame_buffer.append(frame)
        self.keypoint_buffer.append(keypoints_2d)

    def get_current_pose_3d(self):
        """Get 3D pose for most recent frame"""
        if len(self.frame_buffer) < self.buffer_size:
            return None  # Need full buffer

        # Process buffer (overlapping windows)
        video_tensor = torch.stack(list(self.frame_buffer))
        keypoints = np.array(list(self.keypoint_buffer))

        # Run WHAM on sliding window
        with torch.no_grad():
            results = self.wham.predict(video_tensor, keypoints)

        # Return only the middle frame (most context)
        return results[self.buffer_size // 2]
```

**Latency Analysis**:
- Buffer fill time: 1 second (30 frames at 30 FPS)
- WHAM inference: ~100-200ms for 30 frames (with GPU)
- Total latency: ~1.2 seconds

**Optimization Options**:
1. Reduce buffer size to 15 frames (500ms context)
2. TensorRT optimization for faster inference
3. Use lightweight temporal model (TCMR) for real-time

**DECISION**: Start with 15-frame buffer, optimize with TensorRT, add fallback to 2D-only mode.

---

## 🔄 Data Format Conversions

### COCO-17 → SMPL-24 Mapping

```python
def coco17_to_smpl24(coco_keypoints):
    """
    COCO-17: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
             l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip,
             l_knee, r_knee, l_ankle, r_ankle

    SMPL-24: pelvis, l_hip, r_hip, spine1, l_knee, r_knee, spine2,
             l_ankle, r_ankle, spine3, l_foot, r_foot, neck,
             l_collar, r_collar, head, l_shoulder, r_shoulder,
             l_elbow, r_elbow, l_wrist, r_wrist, l_hand, r_hand
    """
    smpl_keypoints = np.zeros((24, 3))

    # Direct mappings
    smpl_keypoints[16] = coco_keypoints[5]  # l_shoulder
    smpl_keypoints[17] = coco_keypoints[6]  # r_shoulder
    smpl_keypoints[18] = coco_keypoints[7]  # l_elbow
    smpl_keypoints[19] = coco_keypoints[8]  # r_elbow
    smpl_keypoints[20] = coco_keypoints[9]  # l_wrist
    smpl_keypoints[21] = coco_keypoints[10] # r_wrist
    smpl_keypoints[1] = coco_keypoints[11]  # l_hip
    smpl_keypoints[2] = coco_keypoints[12]  # r_hip
    smpl_keypoints[4] = coco_keypoints[13]  # l_knee
    smpl_keypoints[5] = coco_keypoints[14]  # r_knee
    smpl_keypoints[7] = coco_keypoints[15]  # l_ankle
    smpl_keypoints[8] = coco_keypoints[16]  # r_ankle

    # Interpolated joints
    smpl_keypoints[0] = (coco_keypoints[11] + coco_keypoints[12]) / 2  # pelvis (mid hips)
    smpl_keypoints[3] = smpl_keypoints[0] + np.array([0, 0.1, 0])  # spine1 (approximate)
    smpl_keypoints[6] = smpl_keypoints[3] + np.array([0, 0.15, 0])  # spine2
    smpl_keypoints[9] = smpl_keypoints[6] + np.array([0, 0.15, 0])  # spine3
    smpl_keypoints[12] = (coco_keypoints[0] + smpl_keypoints[9]) / 2  # neck
    smpl_keypoints[15] = coco_keypoints[0]  # head (use nose as proxy)

    # Hands and feet (approximate from wrists/ankles)
    smpl_keypoints[22] = coco_keypoints[9] + np.array([0, -0.1, 0])  # l_hand
    smpl_keypoints[23] = coco_keypoints[10] + np.array([0, -0.1, 0])  # r_hand
    smpl_keypoints[10] = coco_keypoints[15] + np.array([0, -0.1, 0])  # l_foot
    smpl_keypoints[11] = coco_keypoints[16] + np.array([0, -0.1, 0])  # r_foot

    return smpl_keypoints
```

**Note**: WHAM has built-in regression for this, but useful for standalone conversion.

### SMPL Mesh → OpenSim Markers

```python
def smpl_mesh_to_opensim_markers(smpl_vertices, smpl_faces):
    """
    Place virtual markers on SMPL mesh for OpenSim tracking
    """
    marker_vertices = {
        # Head markers
        'head_top': 411,
        'head_front': 337,
        'head_back': 2158,

        # Torso markers
        'sternum': 3040,
        'c7': 3056,
        't10': 1781,
        'sacrum': 3021,

        # Shoulder markers
        'r_shoulder': 4380,
        'l_shoulder': 1350,
        'r_acromion': 4406,
        'l_acromion': 1376,

        # Arm markers
        'r_elbow_lat': 4587,
        'r_elbow_med': 4609,
        'l_elbow_lat': 1557,
        'l_elbow_med': 1579,
        'r_wrist': 4795,
        'l_wrist': 1765,

        # Hip markers
        'r_asis': 3170,
        'l_asis': 736,
        'r_psis': 3115,
        'l_psis': 681,

        # Leg markers
        'r_knee_lat': 4142,
        'r_knee_med': 4074,
        'l_knee_lat': 1112,
        'l_knee_med': 1044,
        'r_ankle_lat': 4328,
        'r_ankle_med': 4315,
        'l_ankle_lat': 1298,
        'l_ankle_med': 1285,

        # Foot markers
        'r_heel': 4381,
        'l_heel': 1351,
        'r_toe': 4417,
        'l_toe': 1387,
    }

    markers = {}
    for name, vertex_idx in marker_vertices.items():
        markers[name] = smpl_vertices[vertex_idx]

    return markers
```

---

## 🔧 Required Infrastructure Components

### 1. Model Registry System

```python
class ModelRegistry:
    """Central registry for all pose estimation models"""

    AVAILABLE_MODELS = {
        'rtmpose-s': {
            'framework': 'mmpose',
            'config': 'rtmpose-s_8xb256-420e_coco-256x192.py',
            'checkpoint': 'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth',
            'keypoint_format': 'COCO_17',
            'fps_target': 60,
        },
        'rtmpose-m': {
            'framework': 'mmpose',
            'config': 'rtmpose-m_8xb256-420e_coco-256x192.py',
            'checkpoint': 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth',
            'keypoint_format': 'COCO_17',
            'fps_target': 45,
        },
        'vitpose-h': {
            'framework': 'mmpose',
            'config': 'td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py',
            'checkpoint': 'vitpose-h.pth',
            'keypoint_format': 'COCO_17',
            'fps_target': 15,
        },
        # ... existing models
    }

    @staticmethod
    def get_model(name, device='cuda'):
        """Load model by name"""
        config = ModelRegistry.AVAILABLE_MODELS[name]

        if config['framework'] == 'mmpose':
            return load_mmpose_model(config, device)
        elif config['framework'] == 'ultralytics':
            return load_yolo_model(config, device)
        # ...
```

### 2. Pipeline Orchestrator

```python
class SwimVisionPipeline:
    """Orchestrates the entire processing pipeline"""

    def __init__(self, config):
        # Load components based on config
        self.preprocessor = self._init_preprocessor(config)
        self.pose_estimator = self._init_pose_estimator(config)
        self.tracker = self._init_tracker(config)
        self.temporal_refiner = self._init_temporal_refiner(config)
        self.biomechanics = self._init_biomechanics(config)
        self.ai_coach = self._init_ai_coach(config)
        # ...

    def process_frame(self, frame):
        """Process single frame through pipeline"""
        # 1. Preprocess
        enhanced = self.preprocessor.enhance(frame)

        # 2. Pose estimation
        poses_2d = self.pose_estimator.estimate(enhanced)

        # 3. Track swimmers
        tracked = self.tracker.update(poses_2d)

        # 4. Temporal refinement (if buffer full)
        if self.temporal_refiner.can_refine():
            poses_3d = self.temporal_refiner.refine(tracked)
        else:
            poses_3d = None

        # 5. Biomechanics (if 3D available)
        if poses_3d:
            biomech = self.biomechanics.analyze(poses_3d)
        else:
            biomech = None

        # 6. AI Coach
        feedback = self.ai_coach.analyze(poses_2d, poses_3d, biomech)

        return {
            'poses_2d': poses_2d,
            'poses_3d': poses_3d,
            'tracked_ids': tracked,
            'biomechanics': biomech,
            'feedback': feedback,
        }
```

---

## 📦 Docker Configuration for Isolation

```dockerfile
# Base image with CUDA
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install PyTorch ecosystem
RUN pip3 install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MMPose
RUN pip3 install -U openmim && \
    mim install mmengine mmcv mmpose

# Install ByteTrack
RUN pip3 install lap filterpy

# Install WHAM (assume we have repo)
COPY wham_requirements.txt /tmp/
RUN pip3 install -r /tmp/wham_requirements.txt

# Install OpenSim (via conda)
RUN apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    /opt/conda/bin/conda install -c opensim-org opensim=4.4.1

# Install ONNX Runtime for WaterNet
RUN pip3 install onnxruntime-gpu

# Install other dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Copy application
COPY . /app
WORKDIR /app

CMD ["python3", "app.py"]
```

---

## 🧪 Testing Strategy

### Unit Tests
```python
# Test each component independently
def test_rtmpose_integration():
    model = load_rtmpose('rtmpose-s')
    image = create_test_image()
    poses = model.estimate(image)
    assert poses.shape == (17, 3)

def test_coco_to_smpl_conversion():
    coco_kps = create_test_coco_keypoints()
    smpl_kps = coco17_to_smpl24(coco_kps)
    assert smpl_kps.shape == (24, 3)
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    pipeline = SwimVisionPipeline(test_config)
    video = load_test_video()

    results = []
    for frame in video:
        result = pipeline.process_frame(frame)
        results.append(result)

    assert len(results) == len(video)
    assert all(r['poses_2d'] is not None for r in results)
```

### Performance Tests
```python
def test_real_time_performance():
    pipeline = SwimVisionPipeline(real_time_config)

    latencies = []
    for _ in range(100):
        frame = generate_test_frame()
        start = time.time()
        pipeline.process_frame(frame)
        latency = time.time() - start
        latencies.append(latency)

    avg_latency = np.mean(latencies)
    assert avg_latency < 0.033  # <33ms for 30 FPS
```

---

## 📊 Compatibility Matrix Summary

| Feature | PyTorch | TensorFlow | ONNX | OpenSim | Status |
|---------|---------|------------|------|---------|--------|
| RTMPose | ✅ Native | ❌ | ✅ Export | ❌ | ✅ Ready |
| ViTPose | ✅ Native | ❌ | ✅ Export | ❌ | ✅ Ready |
| ByteTrack | ✅ Native | ❌ | ✅ Export | ❌ | ✅ Ready |
| WHAM | ✅ Native | ❌ | ⚠️  Partial | ❌ | ⚠️  Needs adaptation |
| WaterNet | ⚠️  Convert | ✅ Native | ✅ Convert | ❌ | ⚠️  Needs conversion |
| OpenSim | ❌ | ❌ | ❌ | ✅ Native | ⚠️  Needs bridge |
| 4DGS | ✅ Native | ❌ | ❌ | ❌ | ✅ Ready |

---

## 🎯 Next Steps

1. **Set up clean Python 3.10 environment**
2. **Install core dependencies (PyTorch, MMPose)**
3. **Implement ModelRegistry and format converters**
4. **Test RTMPose integration**
5. **Implement ByteTrack wrapper**
6. **Create pipeline orchestrator**
7. **Begin Phase 1 implementation**
