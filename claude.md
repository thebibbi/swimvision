# SwimVision Pro - Claude AI Development Guide

This document provides guidance for Claude AI (or other AI assistants) working on the SwimVision Pro project.

## Project Overview

**SwimVision Pro** is a real-time computer vision system for swimming technique analysis, performance optimization, and injury prediction. It combines pose estimation, time-series analysis, biomechanical feature extraction, and machine learning to help swimmers and coaches improve technique and prevent injuries.

## Project Context

- **Repository:** thebibbi/swimvision
- **Primary Branch:** main
- **Development Model:** Feature branch workflow
- **Documentation:** See SwimVision.md for detailed specifications
- **Tech Review:** See TECHNOLOGY_REVIEW_2025.md for SOTA comparisons

## Architecture Overview

```
SwimVision Pro Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                          │
│          (Live Camera | Upload Video | Compare)             │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                 Video Input Layer                            │
│    (Webcam | RealSense D455 | Video Files)                  │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│              Pose Estimation Engine                          │
│    YOLO11-Pose (GPU) | MediaPipe (CPU/Edge)                 │
│         17 COCO Keypoints → Swimming Keypoints              │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│           Biomechanical Analysis Layer                       │
│  • DTW Stroke Comparison  • Frechet Path Analysis           │
│  • Feature Extraction     • Phase Detection                 │
│  • Symmetry Analysis      • Angle Measurements              │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│             Injury Prediction Layer                          │
│    CatBoost/XGBoost Models + Rule-Based Checks              │
│         Real-time Risk Scoring & Alerts                     │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│        Visualization & Reporting Layer                       │
│  • Pose Overlays    • Metrics Dashboards                    │
│  • Spaghetti Diagrams • Comparison Views                    │
│  • PDF Reports       • Historical Tracking                  │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│              Data Persistence Layer                          │
│     PostgreSQL/SQLite (SQLAlchemy ORM)                      │
│  Swimmers | Sessions | Strokes | Injury Assessments        │
└─────────────────────────────────────────────────────────────┘
```

## Development Phases

### Phase 1: Core Infrastructure (Week 1) - CRITICAL PATH
**Goal:** Get basic pose estimation working with video input/output

**Key Deliverables:**
- Camera input abstraction (webcam, RealSense, video files)
- YOLO11-Pose integration with GPU support
- Swimming-specific keypoint mapping (17 COCO → swimming joints)
- Basic Streamlit UI skeleton
- Pose overlay visualization

**Success Criteria:**
- Process video at ≥15 FPS
- Display skeleton overlay in real-time
- Extract 17 keypoints with ≥0.5 confidence

### Phase 2: Time-Series Analysis (Week 2) - HIGH PRIORITY
**Goal:** Compare swimmer technique to ideal templates

**Key Deliverables:**
- DTW analyzer for stroke comparison
- Frechet distance for hand path analysis
- Stroke phase detection (entry, catch, pull, push, recovery)
- Ideal technique templates for freestyle
- Stroke similarity scoring (0-100)

**Success Criteria:**
- Detect stroke phases with ≥80% accuracy
- Generate similarity scores within 5% of expert ratings
- Create stroke comparison visualizations

### Phase 3: Feature Extraction & Biomechanics (Week 3) - HIGH PRIORITY
**Goal:** Extract meaningful biomechanical features

**Key Deliverables:**
- Temporal features (stroke rate, cycle time)
- Kinematic features (velocities, accelerations)
- Angular measurements (elbow/shoulder/hip/knee angles)
- Symmetry metrics (left vs right)
- Trajectory smoothing (Kalman filter)

**Success Criteria:**
- Extract 30+ biomechanical features per stroke
- Measure stroke rate within 10% error
- Detect asymmetry >15% reliably

### Phase 4: Injury Prediction (Week 4) - MEDIUM PRIORITY
**Goal:** Predict injury risk from biomechanical patterns

**Key Deliverables:**
- Feature engineering for injury risk
- CatBoost/XGBoost models
- Rule-based safety checks
- Real-time risk scoring
- Alert system

**Success Criteria:**
- Achieve ≥70% sensitivity for injury prediction
- Generate actionable recommendations
- Real-time risk updates (<100ms latency)

### Phase 5: Visualization & UI (Week 5) - MEDIUM PRIORITY
**Goal:** Create production-ready user interface

**Key Deliverables:**
- Three modes: Live Camera, Upload Video, Compare
- Interactive dashboards with Plotly
- Spaghetti diagrams for movement patterns
- Side-by-side comparisons
- Session history tracking

**Success Criteria:**
- Responsive UI on desktop/tablet
- Real-time metrics updates
- Export analysis reports

### Phase 6: Data Management & Reports (Week 6) - NICE TO HAVE
**Goal:** Persist data and generate comprehensive reports

**Key Deliverables:**
- SQLAlchemy models (Swimmer, Session, Stroke, Assessment)
- Database migrations (Alembic)
- PDF report generation
- Progress tracking over time
- Data export functionality

**Success Criteria:**
- Store 1000+ sessions without performance degradation
- Generate reports in <10 seconds
- Track improvement trends

## Technology Stack (2025 Recommended)

### Core Dependencies
```python
# Pose Estimation
ultralytics>=8.3.0           # YOLO11-Pose
mediapipe>=0.10.9            # Backup/edge option

# Time Series Analysis
tslearn>=0.6.3               # DTW, soft-DTW
scipy>=1.14.0                # Frechet, signal processing

# Computer Vision
opencv-python>=4.10.0
opencv-contrib-python>=4.10.0
numpy>=2.0.0

# Machine Learning - Injury Prediction
scikit-learn>=1.5.0
catboost>=1.2.5              # ⭐ ADDED: Best for injury prediction
xgboost>=2.1.0
lightgbm>=4.5.0              # ⭐ ADDED: Fast experimentation
imbalanced-learn>=0.12.0

# Deep Learning
torch>=2.5.0
torchvision>=0.20.0

# Web Interface
streamlit>=1.39.0
streamlit-webrtc>=0.47.0
plotly>=5.24.0
altair>=5.4.0

# Data Management
sqlalchemy>=2.0.35
psycopg2-binary>=2.9.10
alembic>=1.13.0              # ⭐ ADDED: Migrations

# Model Interpretability
shap>=0.46.0                 # ⭐ ADDED: Injury factor interpretation

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.66.0
joblib>=1.3.0
```

### Development Dependencies
```python
pytest>=8.3.0
pytest-cov>=6.0.0
ruff>=0.7.0                  # Linting/formatting
mypy>=1.13.0                 # Type checking
pre-commit>=4.0.0            # Git hooks
```

## Code Organization Principles

### Directory Structure
```
swimvision-pro/
├── src/                     # Source code
│   ├── cameras/            # Camera abstraction
│   ├── pose/               # Pose estimation
│   ├── analysis/           # DTW, features, phases
│   ├── injury/             # Injury prediction
│   ├── visualization/      # Plotting, overlays
│   ├── data/               # Database, ORM
│   └── utils/              # Helpers
├── models/                 # Trained models, templates
├── data/                   # Raw videos, processed data
├── tests/                  # Unit & integration tests
├── notebooks/              # Jupyter experiments
├── scripts/                # Utility scripts
├── config/                 # YAML configurations
├── docs/                   # Documentation
│   ├── prds/              # Product requirement docs
│   └── guides/            # User/developer guides
└── app.py                  # Main Streamlit app
```

### Coding Standards

**Python Style:**
- Use Ruff for linting and formatting (replaces black, isort, flake8)
- Type hints for all function signatures
- Docstrings in Google format
- Maximum line length: 100 characters

**Example:**
```python
def calculate_joint_angle(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray
) -> float:
    """
    Calculate angle at joint p2 formed by p1-p2-p3.

    Args:
        p1: First point (x, y) or (x, y, z)
        p2: Joint point (vertex of angle)
        p3: Third point

    Returns:
        Angle in degrees (0-180)

    Raises:
        ValueError: If points are collinear or invalid

    Example:
        >>> shoulder = np.array([100, 100])
        >>> elbow = np.array([150, 150])
        >>> wrist = np.array([200, 150])
        >>> angle = calculate_joint_angle(shoulder, elbow, wrist)
        >>> print(f"Elbow angle: {angle:.1f}°")
    """
    # Implementation...
```

**Testing:**
- Aim for ≥80% code coverage
- Unit tests for all core functions
- Integration tests for pipelines
- Property-based testing with Hypothesis for math functions

**Configuration:**
- Use YAML files in `config/` for settings
- Environment variables via `.env` for secrets
- Never commit API keys or sensitive data

## Common Development Tasks

### Adding a New Pose Estimation Model
1. Create new file in `src/pose/` (e.g., `rtmpose_estimator.py`)
2. Inherit from abstract base class or implement required interface
3. Add tests in `tests/test_pose_estimation.py`
4. Update `config/pose_config.yaml`
5. Document in `docs/guides/pose_estimation.md`

### Adding a New Biomechanical Feature
1. Add extraction logic to `src/analysis/features_extractor.py`
2. Update feature schema in `src/injury/feature_engineering.py`
3. Add unit tests with known values
4. Retrain injury models if needed
5. Update documentation

### Adding a New Stroke Type
1. Create phase templates in `models/ideal_techniques/`
2. Update `src/analysis/stroke_phases.py`
3. Add stroke-specific logic to `src/analysis/dtw_analyzer.py`
4. Update UI dropdown in `app.py`
5. Add example video to `data/raw/examples/`

## Testing Strategy

### Unit Tests
```python
# tests/test_swimming_keypoints.py
import numpy as np
import pytest
from src.pose.swimming_keypoints import SwimmingKeypoints

def test_calculate_joint_angle_right_angle():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([1, 1])
    angle = SwimmingKeypoints.calculate_joint_angle(p1, p2, p3)
    assert np.isclose(angle, 90.0, atol=0.1)

def test_elbow_angle_freestyle_catch():
    # Known elbow angle during catch phase of elite freestyle
    # Test with real data from reference videos
    pass
```

### Integration Tests
```python
# tests/test_pipeline.py
def test_end_to_end_video_processing():
    """Test full pipeline from video to analysis"""
    video_path = "data/test/sample_freestyle.mp4"

    # 1. Load video
    # 2. Estimate pose
    # 3. Extract features
    # 4. Compare to ideal
    # 5. Generate report

    assert stroke_score >= 0 and stroke_score <= 100
```

## Performance Optimization

### Pose Estimation
- Use YOLO11n (nano) for real-time CPU scenarios
- Use YOLO11m (medium) for GPU with good accuracy/speed tradeoff
- Implement frame skipping (process every 2-3 frames) for live video
- Batch process frames when possible

### DTW Computation
- Use Sakoe-Chiba band constraints (radius=10)
- Cache computed DTW distances for reference templates
- Consider GPU-accelerated DTW for large datasets

### Database Queries
- Index frequently queried columns (swimmer_id, session_date)
- Use eager loading for relationships
- Implement pagination for large result sets
- Consider caching for read-heavy operations

## Deployment Considerations

### Docker Setup
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### Environment Variables
```bash
# .env.example
DATABASE_URL=postgresql://user:pass@localhost:5432/swimvision
YOLO_MODEL_PATH=./models/yolo11/yolo11m-pose.pt
DEVICE=cuda  # or cpu
LOG_LEVEL=INFO
```

## Common Pitfalls & Solutions

### Issue: YOLO model download fails
**Solution:** Pre-download models using `scripts/download_models.py`

### Issue: Pose estimation slow on CPU
**Solution:** Switch to MediaPipe or use YOLO11n, enable frame skipping

### Issue: DTW computation too slow
**Solution:** Use Sakoe-Chiba constraints, reduce number of keypoints, use soft-DTW approximation

### Issue: Streamlit session state issues
**Solution:** Use `st.session_state` consistently, avoid mixing session and global state

### Issue: Database connection pooling errors
**Solution:** Configure SQLAlchemy pool size, use connection pooling best practices

## Resources

### Documentation
- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/)
- [MediaPipe Pose Guide](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md)
- [tslearn DTW Tutorial](https://tslearn.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Research Papers
- RTMPose: Real-Time Multi-Person Pose Estimation (2023)
- BlazePose: On-device Real-time Body Pose Tracking (2020)
- Dynamic Time Warping for Gesture Recognition

### Example Swimming Analysis Projects
- Search GitHub for "swimming pose estimation"
- Check research papers on swimming biomechanics

## Support & Communication

### When Working on This Project

1. **Always start by reading:**
   - This file (claude.md)
   - SwimVision.md (detailed specifications)
   - TECHNOLOGY_REVIEW_2025.md (tech stack rationale)
   - TODO.md (current priorities)

2. **Before making changes:**
   - Check existing tests
   - Review related PRD in `docs/prds/`
   - Consider backward compatibility

3. **When stuck:**
   - Check issue tracker and documentation
   - Review similar implementations in codebase
   - Consult research papers for biomechanics questions

4. **Before committing:**
   - Run tests: `pytest`
   - Run linter: `ruff check .`
   - Run type checker: `mypy src/`
   - Update documentation if needed

## Quick Start Checklist

For AI assistants beginning work:

- [ ] Read this file completely
- [ ] Review SwimVision.md for project requirements
- [ ] Check TODO.md for current priorities
- [ ] Review TECHNOLOGY_REVIEW_2025.md for tech decisions
- [ ] Understand the current phase of development
- [ ] Check existing tests to understand expected behavior
- [ ] Review recent commits to understand context
- [ ] Set up development environment if needed
- [ ] Ask clarifying questions before major changes

## Version History

- **v1.0** (2025-11-18): Initial version created by Claude
  - Established architecture overview
  - Defined development phases
  - Documented technology stack (2025 SOTA)
  - Added coding standards and best practices

---

**Remember:** SwimVision Pro aims to help swimmers improve and stay injury-free. Every feature should contribute to these core goals. Quality, accuracy, and user safety are paramount.
