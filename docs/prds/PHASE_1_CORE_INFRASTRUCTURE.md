# PRD: Phase 1 - Core Infrastructure

**Version:** 1.0
**Status:** Ready for Development
**Priority:** P0 (Critical)
**Timeline:** Week 1
**Dependencies:** None

---

## Executive Summary

Phase 1 establishes the foundational infrastructure for SwimVision Pro by implementing video input capabilities, pose estimation, and basic visualization. This phase is critical as all subsequent features depend on reliable pose detection and video processing.

**Goal:** Enable real-time pose estimation from swimming videos with a basic Streamlit interface.

---

## Success Criteria

### Functional Requirements
✅ Process video at minimum 15 FPS on target hardware (GPU: RTX 3060 or equivalent)
✅ Extract 17 COCO keypoints with minimum 0.5 confidence threshold
✅ Support 3 video input sources: webcam, video files, Intel RealSense D455
✅ Display skeleton overlay on video in real-time
✅ Streamlit UI functional with all three modes (Live, Upload, Compare)

### Performance Requirements
- **Latency:** <67ms per frame (15 FPS minimum)
- **Accuracy:** Keypoint detection mAP ≥75% on test videos
- **Resource Usage:** <4GB VRAM on GPU, <8GB RAM on CPU
- **Startup Time:** <5 seconds to initialize models

### Quality Requirements
- **Test Coverage:** ≥80% for core modules
- **Code Quality:** Pass Ruff linting with zero errors
- **Type Safety:** Pass mypy type checking
- **Documentation:** Docstrings for all public functions

---

## User Stories

### US1.1: As a Swimmer, I Want to Process My Swimming Videos
**Acceptance Criteria:**
- User can upload MP4/AVI/MOV video files
- System processes video and displays pose overlay
- User sees progress indicator during processing
- Processing completes in reasonable time (<2x video duration)

### US1.2: As a Coach, I Want Real-Time Pose Analysis During Training
**Acceptance Criteria:**
- User can select webcam or RealSense camera
- Live video feed displays with <1 second latency
- Skeleton overlay renders in real-time
- FPS counter shows current performance
- User can pause/resume live feed

### US1.3: As a Developer, I Want to Switch Between Pose Models
**Acceptance Criteria:**
- Configuration file allows model selection (YOLO11n/s/m, MediaPipe)
- Models download automatically on first use
- System falls back to MediaPipe if GPU unavailable
- Model performance metrics logged

---

## Technical Architecture

### Component Diagram
```
┌─────────────────────────────────────────────┐
│           Streamlit UI (app.py)             │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │  Live   │ │  Upload  │ │   Compare    │ │
│  │ Camera  │ │  Video   │ │  (Skeleton)  │ │
│  └────┬────┘ └────┬─────┘ └──────┬───────┘ │
└───────┼───────────┼──────────────┼─────────┘
        │           │              │
┌───────▼───────────▼──────────────▼─────────┐
│       Camera Abstraction Layer             │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │  Webcam  │ │VideoFile │ │ RealSense  │ │
│  └──────────┘ └──────────┘ └────────────┘ │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│       Pose Estimation Engine                │
│  ┌──────────────┐    ┌──────────────────┐  │
│  │ YOLO11-Pose  │    │ MediaPipe Pose   │  │
│  │  (Primary)   │    │    (Backup)      │  │
│  └──────────────┘    └──────────────────┘  │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│     Swimming Keypoint Mapping               │
│  • 17 COCO keypoints                        │
│  • Swimming joint groups                    │
│  • Angle calculations                       │
│  • Trajectory extraction                    │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│     Pose Overlay Visualization              │
│  • Draw skeleton                            │
│  • Color-code sides                         │
│  • Show confidence                          │
└─────────────────────────────────────────────┘
```

### Data Flow
```
1. Video Source → Camera Interface
2. Camera Interface → Frame (numpy array)
3. Frame → Pose Estimator → Keypoints [(x, y, conf), ...]
4. Keypoints → Swimming Keypoint Mapper → Swimming Joints
5. Frame + Keypoints → Pose Overlay → Annotated Frame
6. Annotated Frame → Streamlit Display
```

---

## Module Specifications

### 1. Camera Abstraction (`src/cameras/`)

#### `base_camera.py` (Abstract Base Class)
```python
class BaseCamera(ABC):
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera hardware/connection"""

    @abstractmethod
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """Get a single frame. Returns (success, frame)"""

    @abstractmethod
    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously"""

    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second"""

    @abstractmethod
    def release(self):
        """Release camera resources"""
```

**Design Decisions:**
- Use ABC to enforce consistent interface across cameras
- Generator pattern for memory efficiency with video streams
- Explicit resource management with `release()`

#### `webcam.py`
- Wrap `cv2.VideoCapture` with error handling
- Support camera selection by index
- Configurable resolution and FPS
- Reconnection logic for dropped connections

#### `video_file.py`
- Load video files with format validation
- Frame-by-frame iteration
- Progress tracking (frame count)
- Support for common formats (MP4, AVI, MOV)

#### `realsense_camera.py`
- Intel RealSense SDK integration
- RGB + Depth streams
- Camera calibration parameters
- Alignment between depth and color

---

### 2. Pose Estimation (`src/pose/`)

#### `yolo_estimator.py`
```python
class YOLOPoseEstimator:
    def __init__(self, model_size='m', device='cuda', conf_threshold=0.5):
        """
        Initialize YOLO11 Pose estimator

        Args:
            model_size: 'n', 's', 'm', 'l', 'x'
            device: 'cuda' or 'cpu'
            conf_threshold: Minimum confidence (0-1)
        """

    def estimate_pose(self, frame: np.ndarray) -> List[Dict]:
        """
        Estimate pose for single frame

        Returns:
            [{'keypoints': np.ndarray (17, 3), 'bbox': ..., 'conf': ...}]
        """

    def estimate_pose_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """Batch inference for efficiency"""
```

**Model Selection Strategy:**
- **YOLO11n:** For CPU or low-end GPU (fastest)
- **YOLO11m:** For mid-range GPU (balanced)
- **YOLO11l:** For high-end GPU (most accurate)

**Keypoint Format:**
```python
{
    'keypoints': np.ndarray,  # Shape: (17, 3) - (x, y, confidence)
    'bbox': [x1, y1, x2, y2],  # Bounding box
    'conf': float              # Overall detection confidence
}
```

#### `mediapipe_estimator.py`
- Backup option for CPU-only scenarios
- 33 landmarks → map to 17 COCO keypoints
- Optimized for real-time performance
- Lower accuracy but faster on CPU

---

### 3. Swimming Keypoint Mapping (`src/pose/`)

#### `swimming_keypoints.py`
```python
class SwimmingKeypoints:
    """Swimming-specific keypoint utilities"""

    # Joint groups
    UPPER_BODY = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
    LOWER_BODY = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
    LEFT_ARM = [5, 7, 9]
    RIGHT_ARM = [6, 8, 10]
    LEFT_LEG = [11, 13, 15]
    RIGHT_LEG = [12, 14, 16]

    @staticmethod
    def calculate_joint_angle(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> float:
        """Calculate angle at joint p2 formed by p1-p2-p3"""

    @staticmethod
    def get_body_angles(keypoints: np.ndarray) -> Dict[str, float]:
        """Extract all swimming-relevant angles"""

    @staticmethod
    def get_hand_path(
        keypoints_sequence: np.ndarray
    ) -> np.ndarray:
        """Extract hand trajectory over time"""
```

**Angle Calculations:**
- Use vector math: `arccos((v1 · v2) / (|v1| |v2|))`
- Return degrees (more intuitive than radians)
- Handle degenerate cases (collinear points, division by zero)
- Validate keypoint confidence before calculation

---

### 4. Visualization (`src/visualization/`)

#### `pose_overlay.py`
```python
class PoseOverlay:
    """Draw pose estimation on frames"""

    SKELETON_CONNECTIONS = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # legs
    ]

    def draw_skeleton(
        self,
        frame: np.ndarray,
        poses: List[Dict],
        show_confidence: bool = False
    ) -> np.ndarray:
        """Draw skeleton overlay on frame"""
```

**Visual Design:**
- Left side: Blue
- Right side: Red
- Low confidence (<0.5): Dashed lines
- High confidence (≥0.5): Solid lines
- Keypoints: Circles with gradient based on confidence

---

### 5. Streamlit UI (`app.py`)

#### Main Layout
```python
st.set_page_config(
    page_title="SwimVision Pro",
    page_icon="🏊",
    layout="wide"
)

# Sidebar
with st.sidebar:
    mode = st.radio("Mode", ["Live Camera", "Upload Video", "Compare"])
    stroke_type = st.selectbox("Stroke", ["Freestyle", "Backstroke", ...])
    # Settings...

# Main area
if mode == "Live Camera":
    # Live video with pose overlay
elif mode == "Upload Video":
    # File uploader and processing
elif mode == "Compare":
    # Side-by-side comparison (skeleton for Phase 1)
```

**Session State Management:**
```python
if 'pose_estimator' not in st.session_state:
    st.session_state.pose_estimator = YOLOPoseEstimator()

if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
```

---

## Configuration

### `config/pose_config.yaml`
```yaml
pose_estimation:
  primary_model: "yolo11m-pose"
  fallback_model: "mediapipe"
  device: "cuda"  # or "cpu"
  confidence_threshold: 0.5
  batch_size: 8

models:
  yolo11n:
    url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    size: "6.4 MB"
  yolo11m:
    url: "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt"
    size: "50 MB"
```

### `config/camera_config.yaml`
```yaml
cameras:
  webcam:
    device_id: 0
    width: 1280
    height: 720
    fps: 30

  realsense:
    width: 1280
    height: 720
    fps: 30
    enable_depth: true

  video_file:
    supported_formats: [".mp4", ".avi", ".mov", ".mkv"]
```

---

## Testing Strategy

### Unit Tests

**Test: Webcam Interface**
```python
def test_webcam_initialization():
    camera = WebcamCamera(device_id=0)
    assert camera.initialize()

def test_webcam_frame_capture():
    camera = WebcamCamera()
    camera.initialize()
    success, frame = camera.get_frame()
    assert success
    assert frame.shape == (720, 1280, 3)
    camera.release()
```

**Test: YOLO Pose Estimation**
```python
def test_yolo_pose_estimation():
    estimator = YOLOPoseEstimator(model_size='n')
    test_image = cv2.imread('tests/data/swimmer.jpg')
    poses = estimator.estimate_pose(test_image)

    assert len(poses) >= 1  # At least one person detected
    assert poses[0]['keypoints'].shape == (17, 3)
    assert all(poses[0]['keypoints'][:, 2] >= 0.5)  # Confidence check
```

**Test: Angle Calculations**
```python
def test_right_angle():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([1, 1])
    angle = SwimmingKeypoints.calculate_joint_angle(p1, p2, p3)
    assert np.isclose(angle, 90.0, atol=0.1)

def test_straight_angle():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([2, 0])
    angle = SwimmingKeypoints.calculate_joint_angle(p1, p2, p3)
    assert np.isclose(angle, 180.0, atol=0.1)
```

### Integration Tests

**Test: End-to-End Video Processing**
```python
def test_process_video_with_pose():
    camera = VideoFileCamera('tests/data/freestyle_10sec.mp4')
    camera.initialize()
    estimator = YOLOPoseEstimator()
    overlay = PoseOverlay()

    frames_processed = 0
    for frame in camera.stream_frames():
        poses = estimator.estimate_pose(frame)
        annotated = overlay.draw_skeleton(frame, poses)
        frames_processed += 1

        assert annotated.shape == frame.shape
        if frames_processed >= 10:
            break

    assert frames_processed == 10
```

### Performance Tests

**Test: FPS Measurement**
```python
def test_pose_estimation_fps():
    estimator = YOLOPoseEstimator(model_size='m', device='cuda')
    test_frames = load_test_frames(num=100)

    start_time = time.time()
    for frame in test_frames:
        poses = estimator.estimate_pose(frame)
    end_time = time.time()

    fps = len(test_frames) / (end_time - start_time)
    assert fps >= 15  # Minimum requirement
```

---

## Dependencies

### Python Packages
```python
# Core
ultralytics>=8.3.0
mediapipe>=0.10.9
opencv-python>=4.10.0
numpy>=2.0.0

# UI
streamlit>=1.39.0
streamlit-webrtc>=0.47.0

# Depth Camera
pyrealsense2>=2.55.0

# Utils
pyyaml>=6.0
python-dotenv>=1.0.0
```

### System Requirements
- Python 3.10+
- CUDA 11.8+ (optional, for GPU)
- 16GB+ RAM
- 4GB+ VRAM (for GPU inference)

---

## Deployment Checklist

- [ ] Install dependencies via requirements.txt
- [ ] Download YOLO11 models (run `python scripts/download_models.py`)
- [ ] Configure camera settings in `config/camera_config.yaml`
- [ ] Test webcam connectivity
- [ ] Test GPU availability (if applicable)
- [ ] Run unit tests: `pytest tests/`
- [ ] Start Streamlit app: `streamlit run app.py`
- [ ] Verify all three modes (Live, Upload, Compare skeleton)

---

## Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU not available | Medium | High | Fall back to MediaPipe (CPU-optimized) |
| YOLO model download fails | Low | Medium | Pre-download and cache models |
| Webcam access denied | Medium | Medium | Clear error messages, permission instructions |
| Low FPS on target hardware | Medium | High | Offer frame skipping, smaller models (nano) |
| Keypoint confidence too low | Medium | Medium | Adjust lighting recommendations, confidence threshold |

---

## Success Metrics (Phase 1)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Video Processing FPS | ≥15 | Automated performance test |
| Keypoint Detection Confidence | ≥0.5 | Average over test dataset |
| Unit Test Coverage | ≥80% | pytest-cov |
| Code Quality Score | A | Ruff linting |
| Startup Time | <5s | Manual timing |
| Memory Usage (GPU) | <4GB | nvidia-smi monitoring |

---

## Future Enhancements (Post-Phase 1)

- [ ] Multi-person tracking (currently single person)
- [ ] 3D pose estimation using depth camera
- [ ] Custom pose model trained on swimming data
- [ ] Pose smoothing/stabilization (Kalman filter)
- [ ] Automatic camera calibration

---

## Appendix

### COCO Keypoint Index Reference
```
0: nose
1: left_eye, 2: right_eye
3: left_ear, 4: right_ear
5: left_shoulder, 6: right_shoulder
7: left_elbow, 8: right_elbow
9: left_wrist, 10: right_wrist
11: left_hip, 12: right_hip
13: left_knee, 14: right_knee
15: left_ankle, 16: right_ankle
```

### Swimming-Relevant Joints
- **Shoulders:** Entry angle, rotation
- **Elbows:** Catch phase angle (90-110° optimal for freestyle)
- **Wrists:** Hand entry and exit points
- **Hips:** Body roll angle
- **Knees:** Kick mechanics
- **Ankles:** Ankle flexibility

---

**Document Status:** Approved for Development
**Next Review:** After Phase 1 Completion
**Owner:** Development Team
**Stakeholders:** Swimmers, Coaches, Data Scientists
