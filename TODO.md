# SwimVision Pro - Development TODO

**Last Updated:** November 18, 2025
**Current Phase:** Pre-Development (Phase 0)
**Next Milestone:** Phase 1 - Core Infrastructure

---

## Project Status Overview

```
Phase 0: Planning & Setup          [################] 100% ✅ COMPLETE
Phase 1: Core Infrastructure        [                ]   0% 🔄 NEXT
Phase 2: Time-Series Analysis       [                ]   0% ⏳ PENDING
Phase 3: Feature Extraction         [                ]   0% ⏳ PENDING
Phase 4: Injury Prediction          [                ]   0% ⏳ PENDING
Phase 5: Visualization & UI         [                ]   0% ⏳ PENDING
Phase 6: Data Management            [                ]   0% ⏳ PENDING
```

---

## Phase 0: Planning & Setup ✅ COMPLETE

### Documentation
- [x] Create comprehensive project specification (SwimVision.md)
- [x] Research and document 2025 SOTA technologies (TECHNOLOGY_REVIEW_2025.md)
- [x] Create Claude AI development guide (claude.md)
- [x] Create TODO tracker (this file)
- [x] Create PRD documents for all phases

### Repository Setup
- [ ] Initialize Python project structure
- [ ] Create virtual environment
- [ ] Set up Git workflow (.gitignore, .gitattributes)
- [ ] Create requirements.txt with pinned versions
- [ ] Set up pre-commit hooks (ruff, mypy, pytest)
- [ ] Create Docker development environment
- [ ] Set up CI/CD pipeline (GitHub Actions)

### Development Environment
- [ ] Install Python 3.10+ and dependencies
- [ ] Download YOLO11 pose models (n, s, m variants)
- [ ] Test CUDA/GPU setup (if available)
- [ ] Set up PostgreSQL database (local dev)
- [ ] Create sample swimming videos for testing
- [ ] Set up Jupyter for experiments

---

## Phase 1: Core Infrastructure (Week 1) 🔄 NEXT

**Priority:** CRITICAL
**Goal:** Get basic pose estimation working with video I/O
**Due:** Week 1

### 1.1 Camera & Video Input
- [ ] Implement `src/cameras/base_camera.py`
  - [ ] Abstract camera interface (ABC)
  - [ ] Frame streaming generator
  - [ ] FPS measurement
- [ ] Implement `src/cameras/webcam.py`
  - [ ] OpenCV VideoCapture wrapper
  - [ ] Resolution/FPS configuration
  - [ ] Error handling
- [ ] Implement `src/cameras/video_file.py`
  - [ ] Video file loading
  - [ ] Frame extraction
  - [ ] Progress tracking
- [ ] Implement `src/cameras/realsense_camera.py`
  - [ ] Intel RealSense SDK integration
  - [ ] Depth + RGB streams
  - [ ] Camera calibration
- [ ] Write tests for camera abstraction
  - [ ] Test with sample videos
  - [ ] Test webcam fallback
  - [ ] Test error conditions

### 1.2 YOLO11 Pose Estimation
- [ ] Implement `src/pose/yolo_estimator.py`
  - [ ] Load YOLO11n/s/m-pose models
  - [ ] Single-frame inference
  - [ ] Batch inference
  - [ ] GPU/CPU device selection
  - [ ] Confidence filtering (threshold 0.5)
  - [ ] Return standardized keypoint format
- [ ] Create model download script
  - [ ] `scripts/download_models.py`
  - [ ] Cache models locally
  - [ ] Verify checksums
- [ ] Implement `src/pose/mediapipe_estimator.py` (backup)
  - [ ] MediaPipe Pose initialization
  - [ ] 33 landmarks → 17 COCO mapping
  - [ ] CPU-optimized inference
- [ ] Write tests for pose estimation
  - [ ] Test with known images
  - [ ] Verify keypoint format
  - [ ] Test batch processing
  - [ ] Compare YOLO vs MediaPipe

### 1.3 Swimming Keypoint Mapping
- [ ] Implement `src/pose/swimming_keypoints.py`
  - [ ] Define joint groups (arms, legs, torso)
  - [ ] `calculate_joint_angle()` function
  - [ ] `get_body_angles()` for all swimming joints
  - [ ] `get_hand_path()` trajectory extraction
  - [ ] Body roll calculation
  - [ ] Validation for missing/occluded keypoints
- [ ] Implement `src/utils/geometry.py`
  - [ ] Vector operations
  - [ ] Angle calculations
  - [ ] Distance measurements
  - [ ] 2D/3D point utilities
- [ ] Write comprehensive tests
  - [ ] Test angle calculations with known values
  - [ ] Test with occluded keypoints
  - [ ] Property-based tests (Hypothesis)

### 1.4 Basic Streamlit UI
- [ ] Create `app.py` skeleton
  - [ ] Page configuration
  - [ ] Sidebar with mode selection
  - [ ] Three modes: Live, Upload, Compare
  - [ ] Session state management
- [ ] Implement Live Camera mode
  - [ ] Video capture display
  - [ ] Real-time pose overlay
  - [ ] FPS counter
- [ ] Implement Upload Video mode
  - [ ] File uploader
  - [ ] Video player
  - [ ] Process button
- [ ] Basic pose overlay visualization
  - [ ] `src/visualization/pose_overlay.py`
  - [ ] Draw skeleton on frame
  - [ ] Color-code left/right sides
  - [ ] Show confidence scores

### 1.5 Configuration Management
- [ ] Create `config/pose_config.yaml`
  - [ ] Model selection
  - [ ] Confidence thresholds
  - [ ] Device (GPU/CPU)
- [ ] Create `config/camera_config.yaml`
  - [ ] Resolution settings
  - [ ] FPS targets
  - [ ] Camera IDs
- [ ] Implement config loader
  - [ ] `src/utils/config.py`
  - [ ] YAML parsing
  - [ ] Environment variable override

### Phase 1 Acceptance Criteria
- [ ] Process video at ≥15 FPS on target hardware
- [ ] Display skeleton overlay in real-time
- [ ] Extract 17 keypoints with ≥0.5 confidence
- [ ] Support webcam, video files, and RealSense
- [ ] Streamlit UI functional with all three modes
- [ ] All tests passing with ≥80% coverage

---

## Phase 2: Time-Series Analysis (Week 2) ⏳ PENDING

**Priority:** HIGH
**Goal:** Compare swimmer strokes to ideal templates
**Dependencies:** Phase 1 complete

### 2.1 Dynamic Time Warping
- [ ] Implement `src/analysis/dtw_analyzer.py`
  - [ ] DTW with Sakoe-Chiba constraints
  - [ ] `compare_strokes()` method
  - [ ] Multi-dimensional DTW for full body
  - [ ] `compute_barycenter()` for templates
  - [ ] `detect_stroke_phases()` method
  - [ ] Normalize distances for comparison
- [ ] Create ideal stroke templates
  - [ ] Record/obtain elite freestyle videos
  - [ ] Extract pose sequences
  - [ ] Save to `models/ideal_techniques/freestyle_elite.pkl`
  - [ ] Add backstroke, breaststroke, butterfly
- [ ] Write tests
  - [ ] Test DTW distance calculation
  - [ ] Test alignment path correctness
  - [ ] Test template creation

### 2.2 Frechet Distance Analysis
- [ ] Implement `src/analysis/frechet_analyzer.py`
  - [ ] `compute_frechet_distance()` method
  - [ ] `compare_hand_paths()` left vs right
  - [ ] `analyze_stroke_trajectory_shape()`
  - [ ] Classify pull patterns (S-pull, I-pull, straight)
  - [ ] Path efficiency metrics
- [ ] Write tests
  - [ ] Test with known trajectories
  - [ ] Test pull pattern classification

### 2.3 Stroke Similarity Ensemble
- [ ] Implement `src/analysis/stroke_similarity.py`
  - [ ] Combine DTW + Frechet + phase matching
  - [ ] `comprehensive_comparison()` method
  - [ ] Weight metrics appropriately
  - [ ] Generate technique recommendations
  - [ ] `progressive_analysis()` for sessions
  - [ ] Fatigue detection
- [ ] Create recommendation engine
  - [ ] Map scores to actionable feedback
  - [ ] Prioritize issues (most impactful first)
- [ ] Write tests
  - [ ] Test ensemble scoring
  - [ ] Test recommendation generation

### 2.4 Stroke Phase Detection
- [ ] Implement `src/analysis/stroke_phases.py`
  - [ ] `detect_phases_freestyle()` method
  - [ ] Phase transitions: entry→catch→pull→push→recovery
  - [ ] `detect_cycle_boundaries()`
  - [ ] `validate_phase_sequence()`
  - [ ] Use hand velocity + position
- [ ] Add phase templates
  - [ ] Create reference phase data
  - [ ] Save to models directory
- [ ] Write tests
  - [ ] Test phase detection accuracy
  - [ ] Test cycle boundary detection

### Phase 2 Acceptance Criteria
- [ ] Detect stroke phases with ≥80% accuracy
- [ ] Generate similarity scores (0-100)
- [ ] Produce technique recommendations
- [ ] Handle 4 stroke types (free, back, breast, fly)
- [ ] Comparison completes in <1 second
- [ ] All tests passing with ≥80% coverage

---

## Phase 3: Feature Extraction & Biomechanics (Week 3) ⏳ PENDING

**Priority:** HIGH
**Goal:** Extract meaningful biomechanical features
**Dependencies:** Phase 1-2 complete

### 3.1 Biomechanical Feature Engineering
- [ ] Implement `src/analysis/features_extractor.py`
  - [ ] `extract_stroke_features()` method
    - [ ] Temporal features (stroke rate, cycle time, tempo)
    - [ ] Kinematic features (velocities, accelerations)
    - [ ] Angular features (all joint angles)
    - [ ] Symmetry metrics (arm/leg/temporal)
    - [ ] Spatial features (hand paths, alignment)
  - [ ] `extract_injury_risk_features()` method
    - [ ] Shoulder risk features
    - [ ] Technique degradation metrics
    - [ ] Workload tracking
    - [ ] Asymmetry metrics
  - [ ] `smooth_trajectories()` method
    - [ ] Kalman filter implementation
    - [ ] Savitzky-Golay filter
    - [ ] Moving average
  - [ ] `calculate_angular_velocity()` method
- [ ] Write tests
  - [ ] Test feature extraction with synthetic data
  - [ ] Test smoothing algorithms
  - [ ] Validate against known biomechanical values

### 3.2 Symmetry Analysis
- [ ] Implement `src/analysis/symmetry_analyzer.py`
  - [ ] Left vs right arm comparison
  - [ ] Left vs right leg comparison
  - [ ] Temporal symmetry (stroke timing)
  - [ ] Force imbalance estimation
  - [ ] Rotation imbalance
- [ ] Write tests
  - [ ] Test with symmetric/asymmetric data
  - [ ] Validate thresholds

### 3.3 Smoothing & Filtering
- [ ] Implement `src/utils/smoothing.py`
  - [ ] Kalman filter for pose sequences
  - [ ] Savitzky-Golay filter
  - [ ] Moving average filter
  - [ ] Outlier detection and removal
- [ ] Write tests
  - [ ] Test noise reduction
  - [ ] Test filter parameters

### Phase 3 Acceptance Criteria
- [ ] Extract 30+ features per stroke
- [ ] Measure stroke rate within ±10% error
- [ ] Detect asymmetry >15% reliably
- [ ] Smooth trajectories reduce jitter visibly
- [ ] Feature extraction <50ms per stroke
- [ ] All tests passing with ≥80% coverage

---

## Phase 4: Injury Prediction (Week 4) ⏳ PENDING

**Priority:** MEDIUM
**Goal:** Predict injury risk from biomechanics
**Dependencies:** Phase 3 complete

### 4.1 Injury Risk Models
- [ ] Implement `src/injury/predictor.py`
  - [ ] Load/save trained models
  - [ ] `train_model()` method (CatBoost, XGBoost, LightGBM)
  - [ ] `predict_risk()` method
  - [ ] `analyze_risk_factors()` with SHAP
  - [ ] Handle class imbalance (SMOTE)
  - [ ] Cross-validation
  - [ ] Feature importance analysis
- [ ] Create `src/injury/feature_engineering.py`
  - [ ] Engineer injury-specific features
  - [ ] Feature selection
  - [ ] Feature scaling/normalization
- [ ] Create training script
  - [ ] `scripts/train_injury_model.py`
  - [ ] Load training data
  - [ ] Train multiple models
  - [ ] Evaluate and compare
  - [ ] Save best model
- [ ] Write tests
  - [ ] Test model loading/saving
  - [ ] Test predictions
  - [ ] Test feature importance

### 4.2 Real-time Risk Scoring
- [ ] Implement `src/injury/risk_scorer.py`
  - [ ] `RealTimeRiskScorer` class
  - [ ] `update()` method for new stroke data
  - [ ] `check_biomechanical_rules()`
  - [ ] `detect_fatigue()` method
  - [ ] Session history tracking
  - [ ] Trend analysis (improving/stable/degrading)
- [ ] Implement `src/injury/biomechanics_rules.py`
  - [ ] Rule-based safety checks
  - [ ] Shoulder hyperextension thresholds
  - [ ] Elbow angle limits
  - [ ] Symmetry deviation thresholds
  - [ ] Velocity drop detection
- [ ] Write tests
  - [ ] Test rule-based alerts
  - [ ] Test fatigue detection
  - [ ] Test trend analysis

### 4.3 Model Training Data
- [ ] Collect/create injury dataset
  - [ ] Swimming injury case studies
  - [ ] Biomechanical measurements
  - [ ] Training load data
  - [ ] Label: injury within 4 weeks (binary)
- [ ] Annotate videos
  - [ ] Mark dangerous techniques
  - [ ] Note injury outcomes
- [ ] Create synthetic data if needed
  - [ ] Augment with variations
  - [ ] Balance classes

### Phase 4 Acceptance Criteria
- [ ] Achieve ≥70% sensitivity for injury prediction
- [ ] False positive rate <30%
- [ ] Generate actionable recommendations
- [ ] Real-time risk updates (<100ms)
- [ ] SHAP values explain predictions
- [ ] All tests passing with ≥80% coverage

---

## Phase 5: Visualization & UI (Week 5) ⏳ PENDING

**Priority:** MEDIUM
**Goal:** Production-ready user interface
**Dependencies:** Phase 1-4 complete

### 5.1 Enhanced Streamlit UI
- [ ] Refactor `app.py`
  - [ ] Improve layout and responsiveness
  - [ ] Add session history
  - [ ] Implement settings persistence
  - [ ] Add export functionality
- [ ] Implement Live Camera mode enhancements
  - [ ] Real-time metrics sidebar
  - [ ] Live alerts for injury risk
  - [ ] Recording functionality
- [ ] Implement Upload Video mode enhancements
  - [ ] Progress bar for processing
  - [ ] Frame-by-frame scrubbing
  - [ ] Comparison to ideal technique
- [ ] Implement Compare mode
  - [ ] Side-by-side video comparison
  - [ ] Synchronized playback
  - [ ] Difference highlighting

### 5.2 Pose Overlay Enhancements
- [ ] Enhance `src/visualization/pose_overlay.py`
  - [ ] `draw_skeleton()` improvements
  - [ ] `draw_angles()` on frame
  - [ ] `draw_trajectory()` hand paths
  - [ ] Color-code by risk level
  - [ ] Highlight problem areas
- [ ] Add confidence visualization
  - [ ] Show low-confidence keypoints differently
  - [ ] Fade uncertain connections

### 5.3 Metrics Dashboard
- [ ] Implement `src/visualization/metrics_dashboard.py`
  - [ ] `create_stroke_comparison_chart()` (radar)
  - [ ] `create_session_progress_chart()` (line)
  - [ ] `create_injury_risk_gauge()` (gauge)
  - [ ] `create_angle_timeline()` (line)
  - [ ] Interactive Plotly charts
  - [ ] Real-time updates
- [ ] Add export functionality
  - [ ] Export charts as PNG/SVG
  - [ ] Export data as CSV/JSON

### 5.4 Spaghetti Diagrams
- [ ] Implement `src/visualization/spaghetti_diagram.py`
  - [ ] Pool layout visualization
  - [ ] `create_diagram()` method
  - [ ] `analyze_lane_usage()` method
  - [ ] Turn efficiency metrics
  - [ ] Straightness scoring
- [ ] Add calibration
  - [ ] Camera calibration for real-world measurements
  - [ ] Pixel-to-meter conversion

### 5.5 Comparison View
- [ ] Implement `src/visualization/comparison_view.py`
  - [ ] Side-by-side layout
  - [ ] Synchronized video playback
  - [ ] Overlay both skeletons
  - [ ] Highlight differences
  - [ ] Difference metrics panel

### Phase 5 Acceptance Criteria
- [ ] Responsive UI on desktop/tablet
- [ ] Real-time metrics update smoothly
- [ ] All visualizations render correctly
- [ ] Export functionality works
- [ ] UI handles errors gracefully
- [ ] Session history persists
- [ ] All tests passing

---

## Phase 6: Data Management & Reports (Week 6) ⏳ PENDING

**Priority:** NICE-TO-HAVE
**Goal:** Persist data and generate reports
**Dependencies:** Phase 1-5 complete

### 6.1 Database Schema
- [ ] Implement `src/data/database.py`
  - [ ] SQLAlchemy Base
  - [ ] `Swimmer` model
  - [ ] `TrainingSession` model
  - [ ] `StrokeAnalysis` model
  - [ ] `SessionMetrics` model
  - [ ] `InjuryRiskAssessment` model
  - [ ] Relationships and foreign keys
- [ ] Set up Alembic migrations
  - [ ] Initial migration
  - [ ] Migration scripts
  - [ ] Upgrade/downgrade paths
- [ ] Create database utilities
  - [ ] Connection management
  - [ ] Session factory
  - [ ] Context managers
- [ ] Write tests
  - [ ] Test CRUD operations
  - [ ] Test relationships
  - [ ] Test migrations

### 6.2 Session Manager
- [ ] Implement `src/data/session_manager.py`
  - [ ] Create new session
  - [ ] Add strokes to session
  - [ ] Update session metrics
  - [ ] Query session history
  - [ ] Get swimmer statistics
- [ ] Write tests
  - [ ] Test session creation
  - [ ] Test data retrieval

### 6.3 Data Export
- [ ] Implement `src/data/export.py`
  - [ ] Export to CSV
  - [ ] Export to JSON
  - [ ] Export to Excel
  - [ ] Export videos with overlays
  - [ ] Batch export
- [ ] Write tests
  - [ ] Test export formats
  - [ ] Test data integrity

### 6.4 Report Generation
- [ ] Implement `src/visualization/reports.py`
  - [ ] `generate_session_report()` PDF
  - [ ] `generate_progress_report()` PDF
  - [ ] Include charts and visualizations
  - [ ] Include recommendations
  - [ ] Swimmer profile page
  - [ ] Technique breakdown
  - [ ] Injury risk summary
- [ ] Create report templates
  - [ ] Header/footer design
  - [ ] Page layouts
  - [ ] Branding
- [ ] Write tests
  - [ ] Test PDF generation
  - [ ] Test report content

### Phase 6 Acceptance Criteria
- [ ] Store 1000+ sessions without degradation
- [ ] Generate reports in <10 seconds
- [ ] Track improvement trends accurately
- [ ] Export works for all formats
- [ ] Database migrations run successfully
- [ ] All tests passing

---

## Testing & Quality Assurance

### Unit Tests
- [ ] Pose estimation modules (≥80% coverage)
- [ ] Time-series analysis (≥80% coverage)
- [ ] Feature extraction (≥80% coverage)
- [ ] Injury prediction (≥80% coverage)
- [ ] Visualization utilities (≥70% coverage)
- [ ] Database operations (≥80% coverage)

### Integration Tests
- [ ] End-to-end video processing
- [ ] Full analysis pipeline
- [ ] Database integration
- [ ] UI workflows

### Performance Tests
- [ ] Video processing speed (≥15 FPS)
- [ ] DTW computation time (<1s)
- [ ] Feature extraction time (<50ms/stroke)
- [ ] Database query performance
- [ ] Report generation speed (<10s)

### Manual Testing
- [ ] Test with real swimming videos
- [ ] Test all four stroke types
- [ ] Test with beginner/intermediate/elite swimmers
- [ ] Test on different hardware (CPU/GPU)
- [ ] Test edge cases (partial occlusion, multiple swimmers)

---

## Documentation

### User Documentation
- [ ] Installation guide
- [ ] User manual
- [ ] Video tutorials
- [ ] FAQ
- [ ] Troubleshooting guide

### Developer Documentation
- [ ] API reference (auto-generated from docstrings)
- [ ] Architecture overview
- [ ] Contribution guidelines
- [ ] Code style guide
- [ ] Testing guide

### Research Documentation
- [ ] Swimming biomechanics reference
- [ ] Injury prevention guidelines
- [ ] Technical approach explanation
- [ ] Bibliography

---

## Deployment & DevOps

### Containerization
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Multi-stage builds
- [ ] Optimize image size

### CI/CD
- [ ] GitHub Actions workflow
- [ ] Automated testing
- [ ] Code quality checks
- [ ] Automated deployment

### Production Deployment
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Database backup strategy
- [ ] Monitoring and logging
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring

---

## Future Enhancements (Post-MVP)

### Phase 7: Advanced Features
- [ ] Multi-person tracking
- [ ] Underwater pose estimation
- [ ] Mobile app (React Native/Flutter)
- [ ] Real-time coaching mode
- [ ] Social features (share progress)
- [ ] Leaderboards and challenges

### Phase 8: AI Improvements
- [ ] Custom pose model for swimming
- [ ] Generative AI for technique suggestions
- [ ] Predictive training plans
- [ ] Automated video editing
- [ ] Voice coaching feedback

### Phase 9: Enterprise Features
- [ ] Team/coach dashboards
- [ ] Multi-swimmer comparison
- [ ] Training plan management
- [ ] Competition readiness assessment
- [ ] Integration with wearables

---

## Known Issues & Risks

### Technical Risks
- [ ] **Pose estimation accuracy in water** - Above-water analysis recommended
- [ ] **Streamlit WebRTC scalability** - Plan FastAPI migration for production
- [ ] **Limited swimming injury datasets** - May need synthetic data or manual collection
- [ ] **Camera calibration complexity** - May need simplified approach for MVP

### Mitigation Strategies
- [ ] Start with above-water analysis only
- [ ] Prototype with Streamlit, architect for FastAPI
- [ ] Begin dataset creation early
- [ ] Use default calibration for MVP, improve later

---

## Questions & Decisions Needed

### Open Questions
- [ ] What swimming events to prioritize? (Sprint vs distance)
- [ ] Indoor pool only or outdoor too?
- [ ] Target users: recreational, competitive, or both?
- [ ] Pricing model (free, freemium, enterprise)?
- [ ] Privacy/data retention policies?

### Decisions Made
- ✅ Tech stack finalized (see TECHNOLOGY_REVIEW_2025.md)
- ✅ Development phases defined (6 phases, ~6 weeks)
- ✅ Primary pose model: YOLO11-Pose
- ✅ Primary ML model for injury: CatBoost
- ✅ MVP UI framework: Streamlit

---

## Success Metrics

### Phase 1 Success
- [ ] Video processing at ≥15 FPS
- [ ] Keypoint detection confidence ≥0.5
- [ ] UI functional and responsive

### Phase 2 Success
- [ ] Stroke phase detection ≥80% accuracy
- [ ] Technique scores within 5% of expert ratings

### Phase 3 Success
- [ ] Stroke rate measurement within ±10% error
- [ ] 30+ features extracted successfully

### Phase 4 Success
- [ ] Injury prediction ≥70% sensitivity
- [ ] Actionable recommendations generated

### Phase 5 Success
- [ ] Polished UI with all features
- [ ] Export functionality working

### Phase 6 Success
- [ ] Reports generated in <10 seconds
- [ ] Database handles 1000+ sessions

### Overall Project Success
- [ ] ✅ Process video at ≥15 FPS
- [ ] ✅ Stroke rate error <10%
- [ ] ✅ Technique scores within 5% of experts
- [ ] ✅ Injury prediction sensitivity >70%
- [ ] ✅ Actionable recommendations
- [ ] ✅ Multi-swimmer support
- [ ] ✅ RGB and depth camera support
- [ ] ✅ Report generation <10 seconds

---

**Note:** This TODO is a living document. Update it regularly as tasks are completed and new requirements emerge.

**Last Review:** 2025-11-18
**Next Review:** After Phase 1 completion
