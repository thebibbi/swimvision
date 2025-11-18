# 🏊 SwimVision Pro

**Real-time computer vision system for swimming technique analysis, performance optimization, and injury prediction.**

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Project Overview

SwimVision Pro combines cutting-edge AI and computer vision to help swimmers and coaches:

- 📹 **Analyze technique** from video with real-time pose estimation
- 📊 **Compare strokes** against elite swimmer templates using Dynamic Time Warping
- 🔬 **Extract biomechanical insights** (angles, velocities, symmetry)
- ⚠️ **Predict injury risk** using machine learning models
- 📈 **Track progress** over time with comprehensive dashboards
- 📄 **Generate reports** with actionable recommendations

---

## ✨ Key Features

### Phase 1: Core Infrastructure (Week 1) ✅
- ✅ Real-time pose estimation with YOLO11-Pose & MediaPipe
- ✅ Support for webcam, video files, and Intel RealSense D455
- ✅ Basic Streamlit UI with pose overlay visualization
- ✅ Swimming-specific keypoint mapping and angle calculations

### Phase 2: Time-Series Analysis (Week 2) 🔄
- 🔄 Dynamic Time Warping for stroke comparison
- 🔄 Frechet distance for hand path analysis
- 🔄 Automated stroke phase detection (entry→catch→pull→push→recovery)
- 🔄 Ideal technique templates for all four strokes

### Phase 3: Feature Extraction (Week 3) ⏳
- ⏳ 30+ biomechanical features (temporal, kinematic, angular, symmetry)
- ⏳ Kalman filtering for trajectory smoothing
- ⏳ Stroke rate and cycle time measurement
- ⏳ Asymmetry detection

### Phase 4: Injury Prediction (Week 4) ⏳
- ⏳ CatBoost/XGBoost models for injury risk prediction
- ⏳ Rule-based safety checks (shoulder angles, asymmetry, fatigue)
- ⏳ Real-time risk scoring with SHAP interpretability
- ⏳ Actionable safety recommendations

### Phase 5: Visualization & UI (Week 5) ⏳
- ⏳ Enhanced Streamlit dashboard (Live, Upload, Compare modes)
- ⏳ Interactive charts (Plotly radar charts, progress tracking)
- ⏳ Spaghetti diagrams for movement patterns
- ⏳ Export analysis as PNG/CSV/PDF

### Phase 6: Data Management (Week 6) ⏳
- ⏳ PostgreSQL database with SQLAlchemy ORM
- ⏳ Session history and swimmer profiles
- ⏳ Comprehensive PDF report generation
- ⏳ Progress tracking over weeks/months

---

## 🏗️ Architecture

```
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

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (optional, for GPU acceleration)
- **16GB+ RAM** (8GB minimum)
- **4GB+ VRAM** (for GPU inference with YOLO11)
- **Webcam or video files** (Intel RealSense D455 optional)

### Installation

#### Option 1: Automated Setup Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/thebibbi/swimvision.git
cd swimvision

# Run setup script
# On macOS/Linux:
bash scripts/setup.sh

# On Windows:
scripts\setup.bat
```

#### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/thebibbi/swimvision.git
cd swimvision

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install

# Configure environment variables
cp .env.example .env
# Edit .env with your settings
```

#### Option 3: Docker (Easiest)

```bash
# Clone the repository
git clone https://github.com/thebibbi/swimvision.git
cd swimvision

# Start with Docker Compose (development)
docker-compose --profile dev up -d

# Access at http://localhost:8501

# View logs
docker-compose logs -f app-dev

# Stop
docker-compose down
```

#### Using Makefile Commands

We provide a `Makefile` for common development tasks:

```bash
# View all available commands
make help

# Setup development environment
make setup-dev

# Run tests
make test

# Lint and format code
make lint
make format

# Start application
make run

# Docker commands
make docker-up      # Start dev environment
make docker-down    # Stop containers
make docker-logs    # View logs
```

### Run the Application

```bash
# Start Streamlit app
streamlit run app.py

# App will open at http://localhost:8501
```

### First-Time Setup

1. **Test webcam access** - Go to "Live Camera" mode
2. **Upload a test video** - Try "Upload Video" mode with sample swimming video
3. **Check pose estimation** - Verify skeleton overlay appears
4. **Review settings** - Adjust confidence thresholds in sidebar

---

## 📂 Project Structure

```
swimvision-pro/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.py                       # Main Streamlit application
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore file
│
├── config/                      # Configuration files
│   ├── pose_config.yaml         # YOLO11/MediaPipe settings
│   ├── camera_config.yaml       # Camera parameters
│   ├── analysis_config.yaml     # DTW, thresholds, scoring weights
│   └── injury_model_config.yaml # ML model parameters
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── cameras/                 # Camera abstraction layer
│   │   ├── __init__.py
│   │   ├── base_camera.py       # Abstract interface
│   │   ├── webcam.py            # Webcam implementation
│   │   ├── video_file.py        # Video file processing
│   │   └── realsense_camera.py  # Intel RealSense integration
│   ├── pose/                    # Pose estimation
│   │   ├── __init__.py
│   │   ├── yolo_estimator.py    # YOLO11-Pose wrapper
│   │   ├── mediapipe_estimator.py # MediaPipe backup
│   │   ├── skeleton_model.py    # Keypoint definitions
│   │   └── swimming_keypoints.py # Swimming-specific mappings
│   ├── analysis/                # Time-series & biomechanical analysis
│   │   ├── __init__.py
│   │   ├── dtw_analyzer.py      # Dynamic Time Warping
│   │   ├── frechet_analyzer.py  # Frechet distance
│   │   ├── stroke_similarity.py # Combined similarity metrics
│   │   ├── features_extractor.py # Biomechanical features
│   │   ├── stroke_phases.py     # Phase detection
│   │   └── symmetry_analyzer.py # Symmetry analysis
│   ├── injury/                  # Injury prediction
│   │   ├── __init__.py
│   │   ├── feature_engineering.py # Injury-specific features
│   │   ├── predictor.py         # ML models (CatBoost, XGBoost)
│   │   ├── risk_scorer.py       # Real-time risk scoring
│   │   └── biomechanics_rules.py # Rule-based checks
│   ├── visualization/           # Visualization components
│   │   ├── __init__.py
│   │   ├── pose_overlay.py      # Draw skeleton on video
│   │   ├── spaghetti_diagram.py # Movement path visualization
│   │   ├── comparison_view.py   # Side-by-side comparisons
│   │   ├── metrics_dashboard.py # Interactive dashboards
│   │   └── reports.py           # PDF report generation
│   ├── data/                    # Data management
│   │   ├── __init__.py
│   │   ├── database.py          # SQLAlchemy models
│   │   ├── session_manager.py   # Session CRUD operations
│   │   └── export.py            # Data export (CSV, JSON, video)
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── video_processing.py  # Video I/O utilities
│       ├── geometry.py          # Angle calculations
│       ├── smoothing.py         # Kalman filter, moving average
│       ├── metrics.py           # Performance metrics
│       └── config.py            # Configuration loader
│
├── models/                      # Trained models & templates
│   ├── ideal_techniques/        # Reference stroke data
│   │   ├── freestyle_elite.pkl
│   │   ├── backstroke_elite.pkl
│   │   ├── breaststroke_elite.pkl
│   │   └── butterfly_elite.pkl
│   ├── injury_models/           # Trained ML models
│   │   ├── shoulder_risk_catboost.pkl
│   │   └── scaler.pkl
│   └── yolo11/                  # YOLO11 pose weights
│       └── yolo11m-pose.pt
│
├── data/                        # Data directory
│   ├── raw/                     # Raw video files
│   ├── processed/               # Extracted pose data
│   ├── annotations/             # Manual annotations
│   └── sessions/                # Saved training sessions
│
├── tests/                       # Unit & integration tests
│   ├── __init__.py
│   ├── test_pose_estimation.py
│   ├── test_dtw.py
│   ├── test_features.py
│   ├── test_injury_prediction.py
│   └── data/                    # Test fixtures
│       └── sample_videos/
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_pose_exploration.ipynb
│   ├── 02_dtw_analysis.ipynb
│   └── 03_injury_model_training.ipynb
│
├── scripts/                     # Utility scripts
│   ├── download_models.py       # Download YOLO11 weights
│   ├── create_ideal_template.py # Create reference templates
│   ├── process_videos_batch.py  # Batch video processing
│   └── train_injury_model.py    # Train ML models
│
└── docs/                        # Documentation
    ├── claude.md                # AI development guide
    ├── TODO.md                  # Development TODO tracker
    ├── TECHNOLOGY_REVIEW_2025.md # Tech stack review
    ├── SwimVision.md            # Original project specification
    └── prds/                    # Product requirement docs
        ├── PHASE_1_CORE_INFRASTRUCTURE.md
        ├── PHASE_2_TIME_SERIES_ANALYSIS.md
        └── PHASE_3_TO_6_SUMMARY.md
```

---

## 🧪 Testing

```bash
# Run all tests with coverage (recommended)
make test

# Or use pytest directly:
pytest tests/ -v --cov=src --cov-report=html

# Run unit tests only
make test-unit
# or: pytest tests/unit/ -v

# Run integration tests only
make test-integration
# or: pytest tests/integration/ -v

# Run specific test file
pytest tests/test_pose_estimation.py -v

# View coverage report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

---

## 🛠️ Development

### Code Quality

```bash
# Lint code (recommended: use make)
make lint
# or: ruff check src/ tests/

# Format code
make format
# or: ruff format src/ tests/

# Type checking
mypy src/ --ignore-missing-imports

# Run all quality checks
make lint && make test
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
make pre-commit
# or: pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Docker Development

```bash
# Build Docker images
make docker-build
# or: docker-compose build

# Start development environment (with hot reload)
make docker-up
# or: docker-compose --profile dev up -d

# Start production environment
make docker-up-prod
# or: docker-compose --profile prod up -d

# Start with GPU support (requires NVIDIA Docker)
make docker-up-gpu
# or: docker-compose --profile gpu up -d

# View logs
make docker-logs
# or: docker-compose logs -f app-dev

# Access shell in container
make docker-shell
# or: docker-compose exec app-dev bash

# Stop containers
make docker-down
# or: docker-compose down

# Clean up (remove volumes)
make docker-clean
# or: docker-compose down -v
```

### Database Management

```bash
# Access PostgreSQL shell
make db-shell
# or: docker-compose exec postgres psql -U swimvision

# Create database migration
make db-migrate message="your migration message"
# or: alembic revision --autogenerate -m "your message"

# Apply migrations
make db-upgrade
# or: alembic upgrade head

# Rollback migration
make db-downgrade
# or: alembic downgrade -1
```

### Adding New Features

1. Review relevant PRD in `docs/prds/`
2. Create feature branch: `git checkout -b feature/your-feature`
3. Write tests first (TDD approach)
4. Implement feature
5. Run tests and linting
6. Update documentation
7. Create pull request

---

## 📊 Performance Benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| Video Processing FPS | ≥15 | TBD |
| Pose Estimation Latency | <67ms | TBD |
| DTW Comparison Time | <1s | TBD |
| Feature Extraction Time | <50ms | TBD |
| Report Generation Time | <10s | TBD |

---

## 🔬 Technology Stack (2025 SOTA)

### Core Technologies
- **Pose Estimation:** YOLO11-Pose (Ultralytics), MediaPipe
- **Time-Series Analysis:** tslearn (DTW), scipy (Frechet distance)
- **Machine Learning:** CatBoost (primary), XGBoost, LightGBM
- **Deep Learning:** PyTorch 2.5+
- **Web Framework:** Streamlit (MVP), FastAPI (production roadmap)
- **Visualization:** Plotly, Altair, Matplotlib
- **Database:** PostgreSQL (production), SQLite (development)

### Why These Technologies?

See [TECHNOLOGY_REVIEW_2025.md](TECHNOLOGY_REVIEW_2025.md) for detailed analysis and comparisons with alternatives.

**Key Decisions:**
- **CatBoost** for injury prediction: 91%+ accuracy in sports injury research
- **YOLO11** over older models: Active maintenance, 89.4% mAP, real-time performance
- **Streamlit** for MVP: Fastest time-to-prototype, easy iteration
- **Planned migration to FastAPI**: Production scalability and WebSocket support

---

## 🎓 Research & References

### Swimming Biomechanics
- Maglischo, E. W. (2003). *Swimming Fastest*. Human Kinetics.
- Psycharakis, S. G., & McCabe, C. B. (2011). Shoulder and hip roll changes during 200-m front crawl swimming. *Medicine & Science in Sports & Exercise*.

### Computer Vision & Pose Estimation
- RTMPose: Real-Time Multi-Person Pose Estimation (2023)
- BlazePose: On-device Real-time Body Pose Tracking (2020)
- Ultralytics YOLO11 Documentation

### Time-Series Analysis
- Dynamic Time Warping for Gesture Recognition
- Discrete Frechet Distance for Trajectory Comparison

### Sports Injury Prediction
- Machine learning approaches to injury risk prediction in sport (2024)
- CatBoost for reinjury risk prediction in elite soccer (2025)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run quality checks: `ruff check . && mypy src/ && pytest`
5. Submit pull request

### Code Style

- Follow PEP 8 (enforced by Ruff)
- Type hints for all functions
- Docstrings in Google format
- Maximum line length: 100 characters
- Test coverage ≥80%

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLO11-Pose
- **Google** for MediaPipe
- **OpenMMLab** for RTMPose research
- **Swimming community** for domain expertise
- **Open-source contributors** for amazing libraries

---

## 📧 Contact

- **Repository:** [github.com/thebibbi/swimvision](https://github.com/thebibbi/swimvision)
- **Issues:** [Report a bug or request a feature](https://github.com/thebibbi/swimvision/issues)
- **Email:** ahmed.eldinayoub@gmail.com

---

## 🗺️ Roadmap

### Phase 1 (Week 1) - Core Infrastructure 🔄
- [x] Pose estimation with YOLO11
- [x] Basic Streamlit UI
- [ ] Camera abstraction layer
- [ ] Initial testing framework

### Phase 2 (Week 2) - Time-Series Analysis ⏳
- [ ] DTW implementation
- [ ] Stroke phase detection
- [ ] Ideal technique templates
- [ ] Comparison scoring

### Phase 3 (Week 3) - Feature Extraction ⏳
- [ ] 30+ biomechanical features
- [ ] Trajectory smoothing
- [ ] Symmetry analysis

### Phase 4 (Week 4) - Injury Prediction ⏳
- [ ] CatBoost model training
- [ ] Real-time risk scoring
- [ ] SHAP interpretability

### Phase 5 (Week 5) - Enhanced UI ⏳
- [ ] Interactive dashboards
- [ ] Spaghetti diagrams
- [ ] Export functionality

### Phase 6 (Week 6) - Data & Reports ⏳
- [ ] Database integration
- [ ] PDF report generation
- [ ] Progress tracking

### Future (Post-MVP)
- [ ] Multi-person tracking
- [ ] Mobile app
- [ ] Custom pose model trained on swimming
- [ ] FastAPI + React production deployment
- [ ] 3D pose estimation with depth camera

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for swimmers, coaches, and sports scientists worldwide.**
