# PRDs: Phases 3-6 Summary

This document provides streamlined PRDs for Phases 3-6 of SwimVision Pro. For detailed Phase 1 and 2 specifications, see their individual PRD documents.

---

# Phase 3: Feature Extraction & Biomechanics

**Version:** 1.0
**Priority:** P1 (High)
**Timeline:** Week 3
**Dependencies:** Phases 1-2 Complete

## Success Criteria

✅ Extract 30+ biomechanical features per stroke
✅ Measure stroke rate within ±10% error
✅ Detect asymmetry >15% reliably
✅ Smooth trajectories reduce jitter by ≥50%
✅ Feature extraction completes in <50ms per stroke

## Key Deliverables

### 1. Biomechanical Feature Extractor (`src/analysis/features_extractor.py`)

**Temporal Features:**
- Stroke rate (strokes per minute)
- Cycle time (seconds per stroke)
- Tempo/rhythm consistency

**Kinematic Features:**
- Hand velocity (max, average, at each phase)
- Hand acceleration (max, average)
- Body velocity estimation
- Velocity fluctuation (intra-cycle variation)

**Angular Features:**
- Elbow angles (left/right, min/max/average)
- Shoulder angles and range of motion
- Hip angles and body roll
- Knee angles (for kick analysis)

**Symmetry Metrics:**
- Arm symmetry index (0-1, 1=perfect)
- Leg symmetry index
- Temporal symmetry (left vs right timing)
- Force imbalance estimation

**Spatial Features:**
- Hand path trajectories
- Body alignment score
- Streamline score
- Pull pattern classification

### 2. Smoothing & Filtering (`src/utils/smoothing.py`)

**Algorithms:**
- **Kalman Filter:** For trajectory smoothing
- **Savitzky-Golay:** For velocity/acceleration calculation
- **Moving Average:** Simple smoothing
- **Outlier Detection:** Remove erroneous keypoints

**Configuration:**
```yaml
smoothing:
  method: "kalman"  # or "savgol", "moving_average"
  kalman:
    process_noise: 0.01
    measurement_noise: 0.1
  savgol:
    window_length: 5
    poly_order: 2
  moving_average:
    window_size: 3
```

### 3. Symmetry Analyzer (`src/analysis/symmetry_analyzer.py`)

**Metrics:**
```python
{
    'arm_symmetry': 0.92,  # 0-1, higher is better
    'leg_symmetry': 0.88,
    'temporal_symmetry': 0.85,
    'rotation_imbalance': 5.2,  # degrees
    'arm_force_imbalance': 0.12,  # 12% difference
    'issues': ['Right arm pulls slightly shorter than left']
}
```

## Testing

```python
def test_stroke_rate_calculation():
    keypoints_30fps = load_test_data('freestyle_60sec.pkl')
    features = extractor.extract_stroke_features(keypoints_30fps, fps=30)
    expected_rate = 32  # SPM from manual count
    assert abs(features['temporal']['stroke_rate'] - expected_rate) / expected_rate < 0.10

def test_kalman_smoothing():
    noisy_trajectory = add_gaussian_noise(clean_trajectory, sigma=2.0)
    smoothed = smoother.smooth_trajectories(noisy_trajectory, method='kalman')
    rmse = np.sqrt(np.mean((smoothed - clean_trajectory)**2))
    assert rmse < 1.0  # pixels
```

---

# Phase 4: Injury Prediction

**Version:** 1.0
**Priority:** P2 (Medium)
**Timeline:** Week 4
**Dependencies:** Phase 3 Complete

## Success Criteria

✅ Achieve ≥70% sensitivity for injury prediction
✅ False positive rate <30%
✅ Generate 3-5 actionable safety recommendations
✅ Real-time risk updates in <100ms
✅ SHAP values explain top 5 contributing factors

## Key Deliverables

### 1. Injury Risk Predictor (`src/injury/predictor.py`)

**Model Stack:**
```python
models = {
    'catboost': CatBoostClassifier(),  # Primary (91%+ accuracy in research)
    'xgboost': XGBClassifier(),        # Secondary
    'lightgbm': LGBMClassifier(),      # Fast experimentation
    'random_forest': RandomForestClassifier()  # Baseline
}
```

**Training Process:**
```python
def train_injury_model(data, labels):
    # 1. Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(data, labels)

    # 2. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # 3. Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y_resampled, cv=5,
                                 scoring='recall')  # Prioritize sensitivity

    # 4. Train final model
    model.fit(X_scaled, y_resampled)

    # 5. Feature importance with SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return model, scaler, explainer
```

**Injury-Specific Features (30+):**
- Shoulder hyperextension angle (max, frequency)
- Shoulder abduction max
- Shoulder range of motion
- Elbow angle min/max (dangerous ranges)
- Asymmetry metrics (all)
- Technique degradation (early vs late session DTW)
- Stroke consistency (variance)
- Workload (total strokes, high-intensity %)
- Session duration
- Historical injury flags

### 2. Real-Time Risk Scorer (`src/injury/risk_scorer.py`)

**Rule-Based Safety Checks:**
```python
RISK_THRESHOLDS = {
    'shoulder_hyperextension': 45.0,    # degrees (CRITICAL)
    'elbow_angle_min': 90.0,            # degrees
    'elbow_angle_max': 170.0,           # degrees
    'symmetry_deviation': 0.15,         # 15% (WARNING)
    'velocity_drop': 0.20,              # 20% (FATIGUE)
    'stroke_rate_spike': 0.30           # 30% increase (OVEREXERTION)
}

def check_biomechanical_rules(biomechanics):
    alerts = []
    if biomechanics['shoulder_hyperextension'] > 45.0:
        alerts.append({
            'level': 'CRITICAL',
            'message': 'Excessive shoulder hyperextension detected',
            'value': biomechanics['shoulder_hyperextension'],
            'risk': 'Shoulder impingement'
        })
    # More checks...
    return alerts
```

**Fatigue Detection:**
```python
def detect_fatigue(session_history):
    if len(session_history) < 10:
        return False, 0.0

    early_strokes = session_history[:5]
    late_strokes = session_history[-5:]

    # Compare DTW distance to ideal
    early_avg = np.mean([s['dtw_distance'] for s in early_strokes])
    late_avg = np.mean([s['dtw_distance'] for s in late_strokes])

    degradation = (late_avg - early_avg) / early_avg

    is_fatigued = degradation > 0.15  # 15% degradation
    fatigue_level = min(degradation, 1.0)

    return is_fatigued, fatigue_level
```

### 3. Training Data Collection

**Dataset Requirements (Minimum):**
- 500+ swimming sessions
- 50+ injury cases (within 4 weeks)
- Biomechanical measurements
- Training load data
- Injury outcomes

**Synthetic Data Generation:**
```python
def generate_synthetic_injury_data(n_samples=1000):
    """Generate synthetic data if real data unavailable"""
    # Based on biomechanics research:
    # - High shoulder angles → higher injury risk
    # - High asymmetry → higher injury risk
    # - Low technique consistency → higher injury risk
    # - High workload + poor technique → higher injury risk
```

## UI Integration

```python
# Real-time risk display
col1, col2 = st.columns([3, 1])
with col1:
    st.video(video_stream)
with col2:
    risk_score = st.session_state.get('risk_score', 0)

    # Gauge chart
    fig = create_risk_gauge(risk_score)
    st.plotly_chart(fig)

    # Alerts
    if alerts:
        for alert in alerts:
            if alert['level'] == 'CRITICAL':
                st.error(f"⚠️ {alert['message']}")
            else:
                st.warning(f"⚡ {alert['message']}")

    # Recommendations
    st.subheader("Safety Tips")
    for rec in recommendations:
        st.info(rec)
```

---

# Phase 5: Visualization & UI

**Version:** 1.0
**Priority:** P2 (Medium)
**Timeline:** Week 5
**Dependencies:** Phases 1-4 Complete

## Success Criteria

✅ Responsive UI on desktop and tablet
✅ Real-time metrics update at ≥10 FPS
✅ All visualizations render correctly
✅ Export functionality (PNG, CSV, PDF)
✅ Session history persists across restarts

## Key Deliverables

### 1. Enhanced Streamlit UI

**Three Modes (Fully Implemented):**

#### Live Camera Mode
- Real-time video with pose overlay
- Live metrics sidebar (stroke rate, technique score, risk level)
- Recording functionality
- Alert banners for injury risk
- Session timer

#### Upload Video Mode
- File uploader (MP4, AVI, MOV)
- Progress bar during processing
- Frame-by-frame scrubber
- Download processed video with overlay
- Generate PDF report

#### Compare Mode
- Side-by-side synchronized playback
- Difference highlighting
- Radar chart comparison
- Detailed breakdown table
- Export comparison report

### 2. Metrics Dashboard (`src/visualization/metrics_dashboard.py`)

**Radar Chart (Technique Comparison):**
```python
categories = ['Timing', 'Power', 'Efficiency', 'Form', 'Symmetry']
swimmer_values = [75, 68, 82, 71, 65]
ideal_values = [95, 90, 95, 92, 95]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=swimmer_values, theta=categories, name='You'))
fig.add_trace(go.Scatterpolar(r=ideal_values, theta=categories, name='Ideal'))
```

**Session Progress Chart:**
```python
# Line chart showing metrics over time
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stroke_numbers,
    y=technique_scores,
    name='Technique Score',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=stroke_numbers,
    y=fatigue_indicators,
    name='Fatigue Level',
    line=dict(color='red', dash='dash')
))
```

**Injury Risk Gauge:**
```python
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=risk_score,
    domain={'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 30], 'color': "lightgreen"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 70
        }
    }
))
```

### 3. Spaghetti Diagram (`src/visualization/spaghetti_diagram.py`)

**Pool Layout Visualization:**
```python
class SpaghettaDiagram:
    def create_diagram(self, trajectories, pool_dims=(50, 25)):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Draw pool outline
        pool = plt.Rectangle((0, 0), pool_dims[0], pool_dims[1],
                             fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(pool)

        # Draw lane lines
        lane_width = pool_dims[1] / 8
        for i in range(1, 8):
            ax.plot([0, pool_dims[0]], [i*lane_width, i*lane_width],
                   'k--', alpha=0.3)

        # Draw trajectories
        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=2)

        # Analyze lane usage
        metrics = self.analyze_lane_usage(trajectories[0])
        ax.set_title(f"Lane Deviation: {metrics['lane_deviation']:.2f}m | "
                    f"Turn Efficiency: {metrics['turn_efficiency']:.1%}")

        return fig
```

### 4. Comparison View (`src/visualization/comparison_view.py`)

**Synchronized Playback:**
```python
def display_comparison(video1, video2, poses1, poses2):
    col1, col2 = st.columns(2)

    frame_idx = st.slider("Frame", 0, min(len(poses1), len(poses2))-1)

    with col1:
        st.subheader("Your Stroke")
        annotated1 = overlay.draw_skeleton(video1[frame_idx], poses1[frame_idx])
        st.image(annotated1)

    with col2:
        st.subheader("Ideal Technique")
        annotated2 = overlay.draw_skeleton(video2[frame_idx], poses2[frame_idx])
        st.image(annotated2)

    # Metrics comparison
    st.subheader("At This Frame")
    comparison_table = pd.DataFrame({
        'Metric': ['Elbow Angle L', 'Elbow Angle R', 'Body Roll'],
        'You': [angles1['left_elbow'], angles1['right_elbow'], angles1['body_roll']],
        'Ideal': [angles2['left_elbow'], angles2['right_elbow'], angles2['body_roll']],
        'Difference': [...]
    })
    st.table(comparison_table)
```

---

# Phase 6: Data Management & Reports

**Version:** 1.0
**Priority:** P3 (Nice-to-Have)
**Timeline:** Week 6
**Dependencies:** Phases 1-5 Complete

## Success Criteria

✅ Store 1000+ sessions without performance degradation
✅ Generate PDF reports in <10 seconds
✅ Track improvement trends accurately over time
✅ Export works for all formats (CSV, JSON, Excel, PDF)
✅ Database migrations run successfully

## Key Deliverables

### 1. Database Schema (`src/data/database.py`)

```python
class Swimmer(Base):
    __tablename__ = 'swimmers'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    experience_level = Column(String)  # beginner, intermediate, elite
    injury_history = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("TrainingSession", back_populates="swimmer")

class TrainingSession(Base):
    __tablename__ = 'training_sessions'
    id = Column(Integer, primary_key=True)
    swimmer_id = Column(Integer, ForeignKey('swimmers.id'))
    session_date = Column(DateTime, default=datetime.utcnow)
    stroke_type = Column(String)
    video_path = Column(String)
    duration_minutes = Column(Float)
    total_strokes = Column(Integer)
    average_technique_score = Column(Float)
    injury_risk_level = Column(String)  # low, medium, high
    notes = Column(Text)
    swimmer = relationship("Swimmer", back_populates="sessions")
    strokes = relationship("StrokeAnalysis", back_populates="session")

class StrokeAnalysis(Base):
    __tablename__ = 'stroke_analysis'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('training_sessions.id'))
    stroke_number = Column(Integer)
    timestamp = Column(Float)  # seconds into session
    keypoints_data = Column(JSON)  # Compressed pose sequence
    stroke_rate = Column(Float)
    technique_score = Column(Float)
    dtw_distance_to_ideal = Column(Float)
    biomechanical_features = Column(JSON)
    phase_durations = Column(JSON)
    session = relationship("TrainingSession", back_populates="strokes")

class InjuryRiskAssessment(Base):
    __tablename__ = 'injury_assessments'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('training_sessions.id'))
    assessment_time = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(String)  # low, medium, high
    risk_probability = Column(Float)  # 0-1
    contributing_factors = Column(JSON)
    recommendations = Column(JSON)
    biomechanical_warnings = Column(JSON)
```

### 2. Session Manager (`src/data/session_manager.py`)

```python
class SessionManager:
    def create_session(self, swimmer_id, stroke_type):
        session = TrainingSession(
            swimmer_id=swimmer_id,
            stroke_type=stroke_type,
            session_date=datetime.utcnow()
        )
        db.session.add(session)
        db.session.commit()
        return session.id

    def add_stroke_analysis(self, session_id, stroke_data):
        stroke = StrokeAnalysis(session_id=session_id, **stroke_data)
        db.session.add(stroke)
        db.session.commit()

    def get_swimmer_progress(self, swimmer_id, date_range):
        sessions = db.session.query(TrainingSession).filter(
            TrainingSession.swimmer_id == swimmer_id,
            TrainingSession.session_date.between(*date_range)
        ).all()
        # Compute trends...
        return progress_data
```

### 3. Report Generation (`src/visualization/reports.py`)

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

class ReportGenerator:
    def generate_session_report(self, session_id, output_path):
        """
        Generate comprehensive PDF report:
        - Page 1: Summary stats, technique score, injury risk
        - Page 2: Stroke-by-stroke breakdown chart
        - Page 3: Technique comparison radar chart
        - Page 4: Recommendations and action items
        - Page 5: Detailed biomechanical analysis
        """
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        # Page 1: Summary
        c.setFont("Helvetica-Bold", 24)
        c.drawString(inch, height - inch, "SwimVision Pro - Session Report")

        c.setFont("Helvetica", 12)
        y = height - 2*inch
        c.drawString(inch, y, f"Swimmer: {session.swimmer.name}")
        y -= 0.3*inch
        c.drawString(inch, y, f"Date: {session.session_date}")
        y -= 0.3*inch
        c.drawString(inch, y, f"Stroke Type: {session.stroke_type}")
        y -= 0.3*inch
        c.drawString(inch, y, f"Duration: {session.duration_minutes:.1f} min")
        y -= 0.3*inch
        c.drawString(inch, y, f"Total Strokes: {session.total_strokes}")
        y -= 0.5*inch

        # Technique score (large)
        c.setFont("Helvetica-Bold", 48)
        c.drawString(inch, y, f"{session.average_technique_score:.0f}/100")
        c.setFont("Helvetica", 14)
        c.drawString(inch, y - 0.4*inch, "Technique Score")

        # Embed charts (convert matplotlib/plotly to image)
        # ...

        c.showPage()
        c.save()

    def generate_progress_report(self, swimmer_id, date_range, output_path):
        """Long-term progress report (multi-week/month)"""
        # Similar structure, focus on trends
```

### 4. Data Export (`src/data/export.py`)

```python
def export_session_csv(session_id, output_path):
    """Export stroke analysis to CSV"""
    strokes = db.session.query(StrokeAnalysis).filter_by(session_id=session_id).all()
    df = pd.DataFrame([{
        'stroke_number': s.stroke_number,
        'timestamp': s.timestamp,
        'stroke_rate': s.stroke_rate,
        'technique_score': s.technique_score,
        'dtw_distance': s.dtw_distance_to_ideal,
        **json.loads(s.biomechanical_features)
    } for s in strokes])
    df.to_csv(output_path, index=False)

def export_session_video(session_id, output_path):
    """Export video with pose overlay"""
    video_path = db.session.query(TrainingSession).get(session_id).video_path
    strokes = db.session.query(StrokeAnalysis).filter_by(session_id=session_id).all()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))

    for stroke in strokes:
        keypoints = json.loads(stroke.keypoints_data)
        ret, frame = cap.read()
        if not ret:
            break
        annotated = overlay.draw_skeleton(frame, keypoints)
        # Add metrics overlay
        cv2.putText(annotated, f"Score: {stroke.technique_score:.0f}/100",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(annotated)

    cap.release()
    out.release()
```

---

## Cross-Phase Integration Notes

### Data Flow Across Phases
```
Phase 1 (Video/Pose) → Phase 2 (DTW/Phases) → Phase 3 (Features) →
Phase 4 (Injury Risk) → Phase 5 (Visualization) → Phase 6 (Storage/Reports)
```

### Shared Data Structures

**Stroke Analysis Result:**
```python
{
    'session_id': int,
    'stroke_number': int,
    'timestamp': float,

    # Phase 1: Pose data
    'keypoints_sequence': np.ndarray,  # (frames, 17, 3)
    'fps': float,

    # Phase 2: Time-series analysis
    'dtw_distance': float,
    'similarity_score': float,
    'phases': [{'phase': str, 'start': int, 'end': int}, ...],
    'technique_breakdown': {...},

    # Phase 3: Features
    'biomechanical_features': {
        'temporal': {...},
        'kinematic': {...},
        'angular': {...},
        'symmetry': {...}
    },

    # Phase 4: Injury risk
    'injury_risk': {
        'level': str,  # low, medium, high
        'probability': float,
        'contributing_factors': [...],
        'recommendations': [...]
    },

    # Phase 5: (Visualization metadata)
    'charts_generated': bool,

    # Phase 6: (Database IDs)
    'db_id': int
}
```

### Performance Optimization Across Phases

**Pipeline Optimization:**
1. **Batch pose estimation** (Phase 1) - Process multiple frames at once
2. **Cache DTW templates** (Phase 2) - Avoid recomputing ideal templates
3. **Vectorize feature extraction** (Phase 3) - Use NumPy operations
4. **Pre-load ML models** (Phase 4) - Keep models in memory
5. **Lazy chart rendering** (Phase 5) - Only generate when displayed
6. **Database connection pooling** (Phase 6) - Reuse connections

---

## Final Success Metrics (All Phases)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Performance** |
| Video processing FPS | ≥15 | Automated test |
| End-to-end latency | <2s | Full pipeline test |
| Report generation | <10s | PDF export test |
| **Accuracy** |
| Stroke rate error | <10% | vs manual count |
| Phase detection | ≥80% | vs annotations |
| Technique score correlation | r≥0.8 | vs expert ratings |
| Injury prediction sensitivity | ≥70% | On test dataset |
| **Quality** |
| Test coverage | ≥80% | pytest-cov |
| Code quality grade | A | Ruff |
| **Usability** |
| UI responsiveness | <100ms | User interaction |
| Crash rate | <1% | Error monitoring |

---

**All PRDs Status:** Ready for Implementation
**Next Step:** Begin Phase 1 Development
