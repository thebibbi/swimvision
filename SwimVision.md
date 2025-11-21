SwimVision Pro: AI Coding Project Prompt
Project Overview
Build SwimVision Pro, a real-time computer vision system for swimming technique analysis, performance optimization, and injury prediction. This system combines pose estimation, depth camera integration, time-series similarity analysis, and machine learning to help both beginner and elite swimmers improve technique and prevent injuries.

Core Requirements
Primary Objectives

Real-time pose estimation from swimming videos (above-water analysis)
Biomechanical analysis using DTW, Frechet distance, and Levenshtein-inspired sequence matching
Stroke comparison against ideal technique templates
Injury risk prediction based on biomechanical patterns and training load
Interactive dashboard for coaches and swimmers
Spaghetti diagrams showing movement patterns in the pool

Key Features

Live video feed processing with pose overlay
Side-by-side comparison (swimmer vs. ideal technique)
Stroke phase detection and analysis
Real-time metrics: stroke rate, stroke length, body angles, symmetry
Historical performance tracking
Injury risk alerts
Export analysis reports

---

Technical Stack
Core Libraries

# Pose Estimation
ultralytics>=8.3.0           # YOLO11-Pose (actively maintained)
mediapipe>=0.10.9            # Backup/mobile option

# Depth Camera Support
pyrealsense2>=2.55.0         # Intel RealSense SDK
# opencv-contrib-python for other camera support

# Time Series Analysis
tslearn>=0.6.3               # DTW, soft-DTW, time series clustering
scipy>=1.11.0                # Frechet distance, signal processing
similaritymeasures>=0.9.0    # Additional similarity metrics

# Computer Vision & Video Processing
opencv-python>=4.9.0
opencv-contrib-python>=4.9.0
numpy>=1.24.0
pillow>=10.0.0

# Machine Learning
scikit-learn>=1.4.0
xgboost>=2.0.0
pandas>=2.1.0
imbalanced-learn>=0.11.0     # For handling class imbalance in injury data

# Deep Learning (if needed for custom models)
torch>=2.1.0
torchvision>=0.16.0

# Web Interface
streamlit>=1.31.0
streamlit-webrtc>=0.47.0     # Real-time video streaming
plotly>=5.18.0               # Interactive visualizations
altair>=5.2.0                # Additional plotting

# Data Management
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0       # PostgreSQL adapter
python-dotenv>=1.0.0

# Utilities
pyyaml>=6.0
tqdm>=4.66.0
joblib>=1.3.0
```

### System Requirements
- Python 3.9-3.11
- CUDA-capable GPU (optional but recommended for real-time processing)
- Webcam or depth camera (Intel RealSense D455 recommended)
- 16GB+ RAM for video processing

---

## Project Structure

Create the following directory structure:
```
swimvision-pro/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   ├── pose_config.yaml           # YOLO11 pose settings
│   ├── camera_config.yaml         # Camera parameters
│   ├── analysis_config.yaml       # DTW, thresholds, etc.
│   └── injury_model_config.yaml   # ML model parameters
├── src/
│   ├── __init__.py
│   ├── cameras/
│   │   ├── __init__.py
│   │   ├── base_camera.py         # Abstract camera interface
│   │   ├── realsense_camera.py    # Intel RealSense integration
│   │   ├── webcam.py              # Standard webcam
│   │   └── video_file.py          # Process recorded videos
│   ├── pose/
│   │   ├── __init__.py
│   │   ├── yolo_estimator.py      # YOLO11-Pose wrapper
│   │   ├── mediapipe_estimator.py # MediaPipe backup option
│   │   ├── skeleton_model.py      # Keypoint definitions
│   │   └── swimming_keypoints.py  # Swimming-specific mappings
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── dtw_analyzer.py        # Dynamic Time Warping
│   │   ├── frechet_analyzer.py    # Frechet distance
│   │   ├── stroke_similarity.py   # Combined similarity metrics
│   │   ├── features_extractor.py  # Biomechanical features
│   │   ├── stroke_phases.py       # Phase detection (entry, catch, pull, push, recovery)
│   │   └── symmetry_analyzer.py   # Left vs right comparison
│   ├── injury/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py # Injury risk features
│   │   ├── predictor.py           # ML models for injury prediction
│   │   ├── risk_scorer.py         # Real-time risk scoring
│   │   └── biomechanics_rules.py  # Rule-based injury indicators
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── pose_overlay.py        # Draw skeleton on video
│   │   ├── spaghetti_diagram.py   # Movement path visualization
│   │   ├── comparison_view.py     # Side-by-side comparisons
│   │   ├── metrics_dashboard.py   # Real-time metrics display
│   │   └── reports.py             # Generate analysis reports
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py            # SQLAlchemy models
│   │   ├── session_manager.py     # Manage training sessions
│   │   └── export.py              # Export data/reports
│   └── utils/
│       ├── __init__.py
│       ├── video_processing.py    # Video I/O utilities
│       ├── geometry.py            # Angle calculations, distances
│       ├── smoothing.py           # Kalman filter, moving average
│       └── metrics.py             # Performance metrics
├── models/
│   ├── ideal_techniques/
│   │   ├── freestyle_elite.pkl    # Reference stroke data
│   │   ├── backstroke_elite.pkl
│   │   ├── breaststroke_elite.pkl
│   │   └── butterfly_elite.pkl
│   ├── injury_models/
│   │   ├── shoulder_risk_rf.pkl   # Trained Random Forest
│   │   └── scaler.pkl             # Feature scaler
│   └── yolo11/
│       └── yolo11n-pose.pt        # YOLO11 pose weights
├── data/
│   ├── raw/                       # Raw video files
│   ├── processed/                 # Extracted pose data
│   ├── annotations/               # Manual annotations for training
│   └── sessions/                  # Saved training sessions
├── tests/
│   ├── __init__.py
│   ├── test_pose_estimation.py
│   ├── test_dtw.py
│   └── test_injury_prediction.py
├── notebooks/
│   ├── 01_pose_exploration.ipynb
│   ├── 02_dtw_analysis.ipynb
│   └── 03_injury_model_training.ipynb
├── scripts/
│   ├── download_models.py         # Download YOLO11 weights
│   ├── process_videos_batch.py    # Batch processing
│   └── train_injury_model.py      # Train ML models
└── app.py                         # Main Streamlit application

---

Development Phases
Phase 1: Core Infrastructure (Week 1)
1.1 Camera & Video Input
File: src/cameras/base_camera.py

from abc import ABC, abstractmethod
from typing import Generator, Tuple
import numpy as np

class BaseCamera(ABC):
    """Abstract base class for all camera types"""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera hardware/connection"""
        pass

    @abstractmethod
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """Get a single frame. Returns (success, frame)"""
        pass

    @abstractmethod
    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously"""
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Get frames per second"""
        pass

    @abstractmethod
    def release(self):
        """Release camera resources"""
        pass


 Requirements:

Download and cache YOLO11n-pose, YOLO11s-pose, YOLO11m-pose models
Implement single-frame and batch processing
Add GPU/CPU device selection
Include confidence filtering
Return standardized keypoint format (x, y, confidence)
Add visualization method to draw skeleton on frame

1.3 Swimming-Specific Keypoint Mapping
File: src/pose/swimming_keypoints.py

from typing import Dict, List
import numpy as np

class SwimmingKeypoints:
    """
    Swimming-specific keypoint utilities
    Maps COCO keypoints to swimming biomechanics
    """

    # Define swimming-relevant joint groups
    UPPER_BODY = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
    LOWER_BODY = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
    LEFT_ARM = [5, 7, 9]  # left shoulder, elbow, wrist
    RIGHT_ARM = [6, 8, 10]
    LEFT_LEG = [11, 13, 15]
    RIGHT_LEG = [12, 14, 16]

    @staticmethod
    def calculate_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at joint p2 formed by p1-p2-p3
        Returns angle in degrees
        """
        pass

    @staticmethod
    def get_body_angles(keypoints: np.ndarray) -> Dict[str, float]:
        """
        Extract swimming-relevant angles

        Returns: {
            'left_elbow': float,
            'right_elbow': float,
            'left_shoulder': float,
            'right_shoulder': float,
            'left_hip': float,
            'right_hip': float,
            'left_knee': float,
            'right_knee': float,
            'body_roll': float  # rotation around longitudinal axis
        }
        """
        pass

    @staticmethod
    def get_hand_path(keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        Extract hand trajectory over time
        Args:
            keypoints_sequence: (num_frames, 17, 3)
        Returns:
            (num_frames, 2, 2) - left and right hand (x, y) over time
        """
        pass

 Requirements:

Implement angle calculations using vector math
Add swimming-specific biomechanical measurements
Include body roll calculation
Compute hand/foot velocities
Add validation for missing/occluded keypoints

---

Phase 2: Time-Series Analysis (Week 2)
2.1 Dynamic Time Warping Analysis
File: src/analysis/dtw_analyzer.py

from tslearn.metrics import dtw, dtw_path, soft_dtw
from tslearn.barycenters import softdtw_barycenter
import numpy as np
from typing import Tuple, Dict

class DTWAnalyzer:
    """
    Dynamic Time Warping for swimming stroke comparison
    Handles multi-dimensional joint trajectories with timing variations
    """

    def __init__(self,
                 global_constraint: str = "sakoe_chiba",
                 sakoe_chiba_radius: int = 10):
        """
        Initialize DTW analyzer

        Args:
            global_constraint: 'sakoe_chiba' or 'itakura'
            sakoe_chiba_radius: Constraint window size
        """
        pass

    def compare_strokes(self,
                       stroke1: np.ndarray,
                       stroke2: np.ndarray) -> Dict:
        """
        Compare two swimming strokes using DTW

        Args:
            stroke1, stroke2: (num_frames, num_joints, 2) arrays

        Returns: {
            'dtw_distance': float,
            'normalized_distance': float,  # normalized by sequence length
            'alignment_path': List[Tuple[int, int]],
            'similarity_score': float  # 0-100, higher is better
        }
        """
        pass

    def compute_barycenter(self, strokes: List[np.ndarray]) -> np.ndarray:
        """
        Compute average stroke template from multiple examples
        Uses soft-DTW for differentiability
        """
        pass

    def detect_stroke_phases(self, stroke: np.ndarray) -> List[Dict]:
        """
        Detect stroke phases using DTW against phase templates

        Returns: [{
            'phase': str,  # 'entry', 'catch', 'pull', 'push', 'recovery'
            'start_frame': int,
            'end_frame': int,
            'quality_score': float
        }]
        """
        pass


   Requirements:

Implement DTW with Sakoe-Chiba band constraints
Add multi-dimensional DTW for full body tracking
Create stroke phase templates for each swimming style
Normalize distances for meaningful comparison
Visualize alignment paths


---

2.2 Frechet Distance Analysis
File: src/analysis/frechet_analyzer.py

from scipy.spatial.distance import directed_hausdorff
from similaritymeasures import frechet_dist
import numpy as np

class FrechetAnalyzer:
    """
    Frechet distance for continuous movement path comparison
    Better captures the "shape" of swimming motions
    """

    @staticmethod
    def compute_frechet_distance(path1: np.ndarray, path2: np.ndarray) -> float:
        """
        Compute discrete Frechet distance between two paths

        Args:
            path1, path2: (num_points, 2 or 3) trajectories

        Returns:
            Frechet distance value
        """
        pass

    def compare_hand_paths(self,
                          stroke1_hands: np.ndarray,
                          stroke2_hands: np.ndarray) -> Dict[str, float]:
        """
        Compare left and right hand paths separately

        Returns: {
            'left_hand_frechet': float,
            'right_hand_frechet': float,
            'combined_frechet': float
        }
        """
        pass

    def analyze_stroke_trajectory_shape(self,
                                       hand_path: np.ndarray) -> Dict:
        """
        Analyze the shape characteristics of a stroke

        Returns: {
            'path_length': float,
            'path_efficiency': float,  # straight_line_distance / path_length
            'curvature_stats': Dict,
            'pull_pattern': str  # 'S-pull', 'I-pull', 'sculling'
        }
        """
        pass



Requirements:

Implement discrete Frechet distance
Add path length and efficiency calculations
Classify pull patterns (S-pull, I-pull, straight)
Visualize path comparisons

---

2.3 Stroke Similarity Ensemble
File: src/analysis/stroke_similarity.py

import numpy as np
from typing import Dict, List
from .dtw_analyzer import DTWAnalyzer
from .frechet_analyzer import FrechetAnalyzer

class StrokeSimilarityAnalyzer:
    """
    Combines multiple similarity metrics for robust comparison
    DTW for timing, Frechet for shape, sequence matching for phases
    """

    def __init__(self):
        self.dtw_analyzer = DTWAnalyzer()
        self.frechet_analyzer = FrechetAnalyzer()

    def comprehensive_comparison(self,
                                swimmer_stroke: np.ndarray,
                                ideal_stroke: np.ndarray,
                                stroke_type: str) -> Dict:
        """
        Multi-metric stroke comparison

        Returns: {
            'overall_score': float (0-100),
            'dtw_score': float,
            'frechet_score': float,
            'phase_sequence_score': float,
            'timing_score': float,
            'technique_breakdown': {
                'entry': float,
                'catch': float,
                'pull': float,
                'push': float,
                'recovery': float
            },
            'recommendations': List[str]
        }
        """
        pass

    def progressive_analysis(self,
                           session_strokes: List[np.ndarray]) -> Dict:
        """
        Analyze technique changes over a session
        Detect fatigue-induced degradation
        """
        pass


Requirements:

Combine DTW, Frechet, and phase matching into unified score
Weight metrics appropriately for swimming
Generate actionable technique recommendations
Track improvement over time
Detect fatigue patterns

---

Phase 3: Feature Extraction & Biomechanics (Week 3)
3.1 Biomechanical Feature Engineering
File: src/analysis/features_extractor.py

import numpy as np
from typing import Dict, List
import scipy.signal as signal

class BiomechanicalFeatureExtractor:
    """
    Extract comprehensive biomechanical features from pose sequences
    Features used for both analysis and injury prediction
    """

    def extract_stroke_features(self,
                                keypoints_sequence: np.ndarray,
                                fps: float = 30.0) -> Dict:
        """
        Extract all features from a stroke sequence

        Args:
            keypoints_sequence: (num_frames, 17, 3) pose sequence
            fps: video frame rate

        Returns: {
            'temporal': {
                'stroke_rate': float,  # strokes per minute
                'stroke_length': float,  # meters per stroke (if calibrated)
                'cycle_time': float,  # seconds per stroke
                'tempo': float
            },
            'kinematic': {
                'hand_velocity_max': float,
                'hand_acceleration_max': float,
                'body_velocity': float,
                'velocity_fluctuation': float  # intra-cycle variation
            },
            'angular': {
                'elbow_angles': {'left': List[float], 'right': List[float]},
                'shoulder_angles': {'left': List[float], 'right': List[float]},
                'hip_angles': {'left': List[float], 'right': List[float]},
                'knee_angles': {'left': List[float], 'right': List[float]},
                'body_roll': List[float]
            },
            'symmetry': {
                'arm_symmetry': float,  # 0-1, 1 = perfect symmetry
                'leg_symmetry': float,
                'temporal_symmetry': float  # left vs right stroke timing
            },
            'spatial': {
                'hand_paths': {'left': np.ndarray, 'right': np.ndarray},
                'body_alignment': float,
                'streamline_score': float
            }
        }
        """
        pass

    def extract_injury_risk_features(self,
                                    keypoints_sequence: np.ndarray,
                                    fps: float = 30.0) -> Dict:
        """
        Extract features specifically for injury prediction

        Returns: {
            'shoulder_risk_features': {
                'humeral_hyperextension': float,  # critical for shoulder injury
                'shoulder_abduction_max': float,
                'shoulder_range_of_motion': float,
                'shoulder_load_asymmetry': float
            },
            'technique_degradation': {
                'early_vs_late_dtw': float,  # fatigue indicator
                'stroke_consistency': float,
                'form_breakdown_rate': float
            },
            'workload': {
                'total_strokes': int,
                'high_intensity_strokes': int,
                'session_duration': float
            },
            'asymmetry_metrics': {
                'arm_force_imbalance': float,
                'rotation_imbalance': float,
                'kick_asymmetry': float
            }
        }
        """
        pass

    def smooth_trajectories(self,
                           trajectories: np.ndarray,
                           method: str = 'kalman') -> np.ndarray:
        """
        Smooth noisy pose estimates
        Methods: 'kalman', 'savgol', 'moving_average'
        """
        pass

    def calculate_angular_velocity(self,
                                  angles: np.ndarray,
                                  fps: float) -> np.ndarray:
        """Calculate angular velocities from joint angles"""
        pass

Requirements:

Implement temporal feature extraction (rates, times)
Calculate kinematic features (velocities, accelerations)
Extract angular measurements for all major joints
Compute symmetry metrics (critical for injury)
Add Kalman filtering for trajectory smoothing
Use scipy.signal for velocity/acceleration computation  
---

3.2 Stroke Phase Detection
File: src/analysis/stroke_phases.py

from typing import List, Dict, Tuple
import numpy as np

class StrokePhaseDetector:
    """
    Detect and classify swimming stroke phases
    Uses keypoint velocities and positions
    """

    PHASES = ['entry', 'catch', 'pull', 'push', 'recovery']

    def detect_phases_freestyle(self,
                               keypoints_sequence: np.ndarray) -> List[Dict]:
        """
        Detect phases for freestyle stroke

        Returns: [{
            'phase': str,
            'start_frame': int,
            'end_frame': int,
            'duration': float,
            'key_events': List[str]  # e.g., ['max_elbow_flexion', 'hand_exit']
        }]
        """
        pass

    def detect_cycle_boundaries(self,
                               keypoints_sequence: np.ndarray) -> List[int]:
        """
        Detect start of each stroke cycle
        Returns frame indices where new cycle begins
        """
        pass

    def validate_phase_sequence(self, phases: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if detected phases follow expected sequence
        Returns (is_valid, issues)
        """
        pass

 Requirements:

Implement phase detection for freestyle (primary)
Add backstroke, breaststroke, butterfly variants
Use hand velocity and position as primary signals
Detect cycle boundaries automatically
Validate phase sequences for correctness

---

Phase 4: Injury Prediction (Week 4)
4.1 Injury Risk Model
File: src/injury/predictor.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from typing import Dict, List, Tuple
import joblib

class InjuryRiskPredictor:
    """
    Machine learning model for swimming injury prediction
    Focus on shoulder injuries (most common in swimmers)
    """

    def __init__(self, model_path: str = None):
        """Load pre-trained model or initialize new one"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        if model_path:
            self.load_model(model_path)

    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: np.ndarray,
                   model_type: str = 'xgboost') -> Dict:
        """
        Train injury prediction model

        Args:
            X_train: Features (biomechanics, training load, history)
            y_train: Binary labels (0=no injury, 1=injury within 4 weeks)
            model_type: 'random_forest', 'xgboost', 'gradient_boost'

        Returns: {
            'model': trained model,
            'cv_scores': cross-validation results,
            'feature_importance': Dict[str, float]
        }
        """
        pass

    def predict_risk(self,
                    features: Dict,
                    return_probability: bool = True) -> Dict:
        """
        Predict injury risk for current session

        Returns: {
            'risk_level': str,  # 'low', 'medium', 'high'
            'probability': float,  # 0-1
            'confidence': float,
            'contributing_factors': List[Tuple[str, float]],  # sorted by importance
            'recommendations': List[str]
        }
        """
        pass

    def analyze_risk_factors(self, features: Dict) -> Dict:
        """
        Detailed analysis of what's contributing to risk
        Uses SHAP values for interpretability
        """
        pass

Requirements:

Implement Random Forest and XGBoost classifiers
Handle class imbalance (injuries are rare)
Feature importance analysis
Real-time risk scoring
Generate specific recommendations
Save/load trained models


---

4.2 Real-time Risk Scoring
File: src/injury/risk_scorer.py

from typing import Dict, List
import numpy as np

class RealTimeRiskScorer:
    """
    Real-time injury risk assessment during swimming session
    Combines rule-based checks with ML predictions
    """

    RISK_THRESHOLDS = {
        'shoulder_hyperextension': 45.0,  # degrees
        'elbow_angle_min': 90.0,
        'elbow_angle_max': 170.0,
        'symmetry_deviation': 0.15,  # 15% asymmetry is concerning
        'velocity_drop': 0.20  # 20% drop indicates fatigue
    }

    def __init__(self, predictor: InjuryRiskPredictor):
        self.predictor = predictor
        self.session_history = []

    def update(self,
              stroke_features: Dict,
              biomechanics: Dict) -> Dict:
        """
        Update risk assessment with new stroke data

        Returns: {
            'instant_risk': float (0-1),
            'session_risk': float (0-1),
            'alerts': List[str],  # immediate concerns
            'warnings': List[str],  # building issues
            'trend': str  # 'improving', 'stable', 'degrading'
        }
        """
        pass

    def check_biomechanical_rules(self, biomechanics: Dict) -> List[str]:
        """
        Rule-based checks for immediate injury risk

        Returns list of triggered warnings:
        - Excessive shoulder hyperextension
        - Dangerous elbow angles
        - Severe asymmetry
        - Rapid technique degradation
        """
        pass

    def detect_fatigue(self) -> Tuple[bool, float]:
        """
        Detect fatigue-induced form breakdown
        Returns (is_fatigued, fatigue_level)
        """
        pass

 Requirements:

Implement rule-based safety checks
Track session history for trends
Detect dangerous biomechanical patterns
Combine with ML predictions
Generate real-time alerts

---

Phase 5: Visualization & UI (Week 5)
5.1 Streamlit Main Application
File: app.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from src.pose.yolo_estimator import YOLOPoseEstimator
from src.analysis.stroke_similarity import StrokeSimilarityAnalyzer
from src.visualization.pose_overlay import PoseOverlay
from src.visualization.metrics_dashboard import MetricsDashboard

st.set_page_config(
    page_title="SwimVision Pro",
    page_icon="🏊",
    layout="wide"
)

class VideoProcessor(VideoTransformerBase):
    """Real-time video processing with pose estimation"""

    def __init__(self):
        self.pose_estimator = YOLOPoseEstimator()
        self.overlay = PoseOverlay()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Estimate pose
        poses = self.pose_estimator.estimate_pose(img)

        # Draw overlay
        img = self.overlay.draw_skeleton(img, poses)

        return img

def main():
    st.title("🏊 SwimVision Pro")
    st.markdown("Real-time swimming technique analysis and injury prevention")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Live Camera", "Upload Video", "Compare"])
        stroke_type = st.selectbox("Stroke Type",
                                   ["Freestyle", "Backstroke", "Breaststroke", "Butterfly"])

        st.header("Analysis Options")
        show_skeleton = st.checkbox("Show Skeleton", value=True)
        show_angles = st.checkbox("Show Joint Angles", value=True)
        show_metrics = st.checkbox("Show Metrics", value=True)

        st.header("Thresholds")
        confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5)

    # Main area
    if mode == "Live Camera":
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Live Feed")
            webrtc_streamer(
                key="live",
                video_processor_factory=VideoProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

        with col2:
            st.subheader("Real-time Metrics")
            # Metrics display (placeholders)
            st.metric("Stroke Rate", "32 SPM")
            st.metric("Technique Score", "85/100")
            st.metric("Injury Risk", "Low", delta="-5%")

    elif mode == "Upload Video":
        uploaded_file = st.file_uploader("Choose video", type=['mp4', 'avi', 'mov'])

        if uploaded_file:
            # Process uploaded video
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original")
                st.video(uploaded_file)

            with col2:
                st.subheader("Analysis")
                # Show processed video with overlay
                process_btn = st.button("Analyze Video")

    elif mode == "Compare":
        st.subheader("Compare with Ideal Technique")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Your Stroke**")
            user_video = st.file_uploader("Upload your video", type=['mp4'])

        with col2:
            st.markdown("**Ideal Technique**")
            ideal = st.selectbox("Select reference", ["Elite Freestyle", "Elite Backstroke"])

        if st.button("Compare"):
            # Run comparison analysis
            pass

if __name__ == "__main__":
    main()


    Requirements:

Implement three main modes: Live, Upload, Compare
Real-time video streaming with streamlit-webrtc
Responsive layout with metrics sidebar
Video upload and processing
Export analysis results
Session history tracking

----

5.2 Visualization Components
File: src/visualization/pose_overlay.py

import cv2
import numpy as np
from typing import List, Dict

class PoseOverlay:
    """
    Draw pose estimation results on video frames
    """

    # COCO skeleton connections
    SKELETON_CONNECTIONS = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # legs
    ]

    COLORS = {
        'skeleton': (0, 255, 0),
        'keypoints': (0, 0, 255),
        'left_side': (255, 0, 0),
        'right_side': (0, 255, 255)
    }

    def draw_skeleton(self,
                     frame: np.ndarray,
                     poses: List[Dict],
                     show_confidence: bool = False) -> np.ndarray:
        """
        Draw skeleton overlay on frame

        Args:
            frame: RGB image
            poses: List of detected poses
            show_confidence: Display confidence scores
        """
        pass

    def draw_angles(self,
                   frame: np.ndarray,
                   angles: Dict[str, float],
                   keypoints: np.ndarray) -> np.ndarray:
        """Draw joint angles on frame"""
        pass

    def draw_trajectory(self,
                       frame: np.ndarray,
                       trajectory: np.ndarray,
                       color: tuple = (255, 0, 0)) -> np.ndarray:
        """Draw hand/foot trajectory path"""
        pass


5.3 Metrics Dashboard
File: src/visualization/metrics_dashboard.py

import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List

class MetricsDashboard:
    """
    Create interactive visualizations for swimming metrics
    """

    def create_stroke_comparison_chart(self,
                                      swimmer_data: Dict,
                                      ideal_data: Dict) -> go.Figure:
        """
        Radar chart comparing swimmer to ideal technique
        Metrics: timing, power, efficiency, form, symmetry
        """
        pass

    def create_session_progress_chart(self,
                                     session_metrics: List[Dict]) -> go.Figure:
        """
        Line chart showing metrics over session
        Detect fatigue patterns
        """
        pass

    def create_injury_risk_gauge(self, risk_score: float) -> go.Figure:
        """Gauge chart for injury risk level"""
        pass

    def create_angle_timeline(self,
                            angles_sequence: Dict[str, List[float]]) -> go.Figure:
        """Timeline of joint angles through stroke"""
        pass

Requirements:

Interactive Plotly charts
Real-time updates
Export functionality
Responsive design

---

5.4 Spaghetti Diagrams
File: src/visualization/spaghetti_diagram.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class SpaghettaDiagram:
    """
    Movement path visualization for pool utilization
    Similar to lean manufacturing spaghetti diagrams
    """

    def __init__(self, pool_dimensions: tuple = (50, 25)):
        """
        Args:
            pool_dimensions: (length, width) in meters
        """
        self.pool_dims = pool_dimensions

    def create_diagram(self,
                      trajectories: List[np.ndarray],
                      lane_width: float = 2.5) -> np.ndarray:
        """
        Create spaghetti diagram showing swimmer paths

        Args:
            trajectories: List of (num_frames, 2) position arrays

        Returns:
            Diagram image
        """
        pass

    def analyze_lane_usage(self, trajectory: np.ndarray) -> Dict:
        """
        Analyze how efficiently swimmer uses lane

        Returns: {
            'lane_deviation': float,  # meters from center
            'turn_efficiency': float,
            'straightness_score': float
        }
        """
        pass

  Phase 6: Data Management & Reports (Week 6)
6.1 Database Schema
File: src/data/database.py

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Swimmer(Base):
    __tablename__ = 'swimmers'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    experience_level = Column(String)  # beginner, intermediate, elite
    injury_history = Column(JSON)

    sessions = relationship("TrainingSession", back_populates="swimmer")

class TrainingSession(Base):
    __tablename__ = 'training_sessions'

    id = Column(Integer, primary_key=True)
    swimmer_id = Column(Integer, ForeignKey('swimmers.id'))
    session_date = Column(DateTime, default=datetime.datetime.utcnow)
    stroke_type = Column(String)
    video_path = Column(String)
    duration_minutes = Column(Float)
    total_strokes = Column(Integer)

    swimmer = relationship("Swimmer", back_populates="sessions")
    strokes = relationship("StrokeAnalysis", back_populates="session")
    metrics = relationship("SessionMetrics", back_populates="session")

class StrokeAnalysis(Base):
    __tablename__ = 'stroke_analysis'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('training_sessions.id'))
    stroke_number = Column(Integer)
    timestamp = Column(Float)

    # Pose data
    keypoints_data = Column(JSON)  # Serialized pose sequence

    # Metrics
    stroke_rate = Column(Float)
    technique_score = Column(Float)
    dtw_distance_to_ideal = Column(Float)

    # Biomechanics
    biomechanical_features = Column(JSON)

    session = relationship("TrainingSession", back_populates="strokes")

class InjuryRiskAssessment(Base):
    __tablename__ = 'injury_assessments'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('training_sessions.id'))
    assessment_time = Column(DateTime, default=datetime.datetime.utcnow)

    risk_level = Column(String)  # low, medium, high
    risk_probability = Column(Float)
    contributing_factors = Column(JSON)
    recommendations = Column(JSON)


    Requirements:

SQLite for local development
PostgreSQL support for production
Store pose data efficiently (consider compression)
Session and stroke-level metrics
Injury risk history

---


6.2 Report Generation
File: src/visualization/reports.py

from typing import Dict
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class ReportGenerator:
    """
    Generate PDF analysis reports for swimmers and coaches
    """

    def generate_session_report(self,
                                session_data: Dict,
                                output_path: str):
        """
        Create comprehensive session report

        Includes:
        - Summary statistics
        - Technique comparison charts
        - Stroke-by-stroke breakdown
        - Injury risk assessment
        - Personalized recommendations
        """
        pass

    def generate_progress_report(self,
                                swimmer_id: int,
                                date_range: tuple,
                                output_path: str):
        """
        Long-term progress report
        Shows improvement trends over time
        """
        pass


from typing import Dict
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class ReportGenerator:
    """
    Generate PDF analysis reports for swimmers and coaches
    """

    def generate_session_report(self,
                                session_data: Dict,
                                output_path: str):
        """
        Create comprehensive session report

        Includes:
        - Summary statistics
        - Technique comparison charts
        - Stroke-by-stroke breakdown
        - Injury risk assessment
        - Personalized recommendations
        """
        pass

    def generate_progress_report(self,
                                swimmer_id: int,
                                date_range: tuple,
                                output_path: str):
        """
        Long-term progress report
        Shows improvement trends over time
        """
        pass

 Testing Requirements
Create tests for:

Pose Estimation

Test YOLO inference on sample swimming videos
Verify keypoint format and confidence scores
Test batch processing


DTW Analysis

Test stroke comparison accuracy
Verify alignment path correctness
Test phase detection


Injury Prediction

Test feature extraction
Verify risk scoring
Test rule-based checks


End-to-End

Test full pipeline from video to report
Verify database operations
Test UI components

----

Deployment Considerations
Docker Setup
Create Dockerfile:
dockerfileFROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
Performance Optimization

Use GPU for YOLO inference if available
Implement frame skipping for real-time (process every 2-3 frames)
Cache pose estimation results
Use multiprocessing for batch video processing


Documentation Requirements
Each module should include:

Docstrings with type hints
Usage examples
Performance characteristics
Known limitations

Create:

README.md with setup instructions
CONTRIBUTING.md for development guidelines
docs/USER_GUIDE.md for end users
docs/API_REFERENCE.md for developers


Success Criteria
The system should:

✅ Process video at ≥15 FPS with pose estimation
✅ Achieve <10% error in stroke rate measurement
✅ Provide technique scores within 5% of expert ratings
✅ Detect injury risk with >70% sensitivity
✅ Generate actionable recommendations for improvement
✅ Handle multiple swimmers in frame
✅ Work with both RGB and depth cameras
✅ Export analysis reports in <10 seconds


Priority Order
Week 1 (Critical):

Camera input ✓
YOLO pose estimation ✓
Basic visualization ✓
Streamlit skeleton ✓

Week 2 (High Priority):

DTW implementation ✓
Stroke comparison ✓
Feature extraction ✓

Week 3-4 (Medium Priority):

Injury prediction model ✓
Database integration ✓
Advanced visualizations ✓

Week 5-6 (Nice to Have):

Report generation ✓
Progressive analysis ✓
Optimization ✓


Additional Notes

Swimming Domain Knowledge: The system needs reference data for "ideal" techniques. Consider collecting video from elite swimmers or using publicly available Olympic footage.
Camera Calibration: For accurate spatial measurements (stroke length, velocity), implement camera calibration to convert pixel distances to real-world meters.
Privacy: If deploying to cloud, ensure video data is encrypted and complies with privacy regulations.
Real-time Performance: For live analysis, consider using YOLO11n (nano) or YOLO11s (small) models for better FPS.
Depth Camera Limitations: Standard depth cameras don't work well underwater. Above-water analysis is recommended, or use specialized underwater cameras with custom pose models.


Getting Started
bash# Setup
git clone <repo>
cd swimvision-pro
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download YOLO models
python scripts/download_models.py

# Run app
streamlit run app.py
