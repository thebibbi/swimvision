# PRD: Phase 2 - Time-Series Analysis

**Version:** 1.0
**Status:** Ready for Development
**Priority:** P1 (High)
**Timeline:** Week 2
**Dependencies:** Phase 1 Complete

---

## Executive Summary

Phase 2 implements time-series analysis capabilities to compare swimmer strokes against ideal technique templates. This includes Dynamic Time Warping (DTW) for temporal alignment, Frechet distance for path shape comparison, and automated stroke phase detection.

**Goal:** Enable quantitative technique comparison and generate actionable feedback scores.

---

## Success Criteria

✅ Detect stroke phases with ≥80% accuracy
✅ Generate similarity scores (0-100) within 5% of expert ratings
✅ Process stroke comparison in <1 second
✅ Support 4 stroke types (freestyle, backstroke, breaststroke, butterfly)
✅ Produce 3-5 actionable technique recommendations per analysis

---

## Key Features

### 1. Dynamic Time Warping (DTW) Analysis
**Purpose:** Compare two stroke sequences with different timings
**Algorithm:** DTW with Sakoe-Chiba band constraints
**Output:** Similarity score (0-100), alignment path, phase-by-phase breakdown

### 2. Frechet Distance Analysis
**Purpose:** Compare continuous movement paths (hand trajectories)
**Algorithm:** Discrete Frechet distance
**Output:** Path similarity, efficiency metrics, pull pattern classification

### 3. Stroke Phase Detection
**Purpose:** Automatically segment strokes into biomechanical phases
**Phases:** Entry → Catch → Pull → Push → Recovery
**Method:** Velocity-based + template matching

### 4. Ideal Technique Templates
**Purpose:** Reference data for comparison
**Sources:** Elite swimmer videos, biomechanics research
**Formats:** Pickled numpy arrays with metadata

---

## Technical Specifications

### DTW Analyzer Module (`src/analysis/dtw_analyzer.py`)

```python
class DTWAnalyzer:
    def __init__(self, global_constraint="sakoe_chiba", sakoe_chiba_radius=10):
        """Initialize with time warping constraints"""

    def compare_strokes(
        self,
        stroke1: np.ndarray,  # (num_frames, num_joints, 2)
        stroke2: np.ndarray
    ) -> Dict:
        """
        Returns:
            {
                'dtw_distance': float,
                'normalized_distance': float,
                'alignment_path': List[Tuple[int, int]],
                'similarity_score': float  # 0-100
            }
        """

    def compute_barycenter(self, strokes: List[np.ndarray]) -> np.ndarray:
        """Compute average stroke template (soft-DTW)"""

    def detect_stroke_phases(self, stroke: np.ndarray) -> List[Dict]:
        """Detect phases using DTW against phase templates"""
```

**Key Algorithms:**
- **Standard DTW:** `tslearn.metrics.dtw()`
- **Soft-DTW:** `tslearn.metrics.soft_dtw()` for differentiability
- **DTW Barycenter Averaging:** For creating templates from multiple examples

**Normalization:**
```python
normalized_distance = dtw_distance / max(len(stroke1), len(stroke2))
similarity_score = 100 * (1 - min(normalized_distance / threshold, 1.0))
```

### Frechet Distance Module (`src/analysis/frechet_analyzer.py`)

```python
class FrechetAnalyzer:
    @staticmethod
    def compute_frechet_distance(
        path1: np.ndarray,  # (num_points, 2 or 3)
        path2: np.ndarray
    ) -> float:
        """Compute discrete Frechet distance"""

    def compare_hand_paths(
        self,
        stroke1_hands: np.ndarray,
        stroke2_hands: np.ndarray
    ) -> Dict[str, float]:
        """
        Returns:
            {
                'left_hand_frechet': float,
                'right_hand_frechet': float,
                'combined_frechet': float
            }
        """

    def analyze_stroke_trajectory_shape(self, hand_path: np.ndarray) -> Dict:
        """
        Returns:
            {
                'path_length': float,
                'path_efficiency': float,  # straight_line / actual_path
                'curvature_stats': {...},
                'pull_pattern': str  # 'S-pull', 'I-pull', 'sculling'
            }
        """
```

**Pull Pattern Classification:**
- **S-Pull:** Curved inward then outward (traditional freestyle)
- **I-Pull:** Straight pull directly backward (modern freestyle)
- **Sculling:** Multiple direction changes (breaststroke, artistic swimming)

**Classification Method:**
- Analyze curvature at multiple points along path
- Compare to reference patterns using Frechet distance
- Classify based on minimum distance to templates

### Stroke Similarity Ensemble (`src/analysis/stroke_similarity.py`)

```python
class StrokeSimilarityAnalyzer:
    def comprehensive_comparison(
        self,
        swimmer_stroke: np.ndarray,
        ideal_stroke: np.ndarray,
        stroke_type: str
    ) -> Dict:
        """
        Combines multiple metrics:
        - DTW score (timing accuracy): 40% weight
        - Frechet score (path shape): 30% weight
        - Phase sequence score: 20% weight
        - Symmetry score: 10% weight

        Returns:
            {
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

    def progressive_analysis(
        self,
        session_strokes: List[np.ndarray]
    ) -> Dict:
        """Analyze technique changes over session (fatigue detection)"""
```

**Recommendation Generation:**
```python
def generate_recommendations(analysis: Dict) -> List[str]:
    recommendations = []

    if analysis['technique_breakdown']['catch'] < 60:
        recommendations.append(
            "Improve catch phase: Elbow should be higher than hand "
            "at water entry. Current score: {:.1f}/100".format(
                analysis['technique_breakdown']['catch']
            )
        )

    if analysis['symmetry_score'] < 70:
        recommendations.append(
            "Asymmetry detected: Left and right strokes differ by >15%. "
            "Focus on balanced technique."
        )

    # More rules...
    return recommendations[:5]  # Top 5 recommendations
```

### Stroke Phase Detection (`src/analysis/stroke_phases.py`)

```python
class StrokePhaseDetector:
    PHASES = ['entry', 'catch', 'pull', 'push', 'recovery']

    def detect_phases_freestyle(
        self,
        keypoints_sequence: np.ndarray  # (num_frames, 17, 3)
    ) -> List[Dict]:
        """
        Detection logic:
        1. Extract hand positions (wrist keypoints 9, 10)
        2. Compute hand velocities
        3. Identify key events:
           - Entry: Hand crosses water plane (y-coordinate minimum)
           - Catch: Hand velocity changes from downward to backward
           - Pull: Maximum backward velocity
           - Push: Hand passes hip (x-coordinate alignment)
           - Recovery: Hand exits water (y-coordinate maximum)

        Returns: [{
            'phase': str,
            'start_frame': int,
            'end_frame': int,
            'duration': float,
            'key_events': List[str]
        }]
        """

    def detect_cycle_boundaries(
        self,
        keypoints_sequence: np.ndarray
    ) -> List[int]:
        """
        Detect start of each stroke cycle
        Method: Find peaks in hand y-coordinate (recovery phase)
        Returns: Frame indices of cycle starts
        """

    def validate_phase_sequence(
        self,
        phases: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that phases follow expected order
        Returns: (is_valid, issues)
        """
```

---

## Ideal Technique Templates

### Template Structure
```python
{
    'stroke_type': 'freestyle',
    'level': 'elite',
    'source': 'Olympic 2024 100m Free Final',
    'athlete': 'Anonymous',
    'fps': 30,
    'keypoints_sequence': np.ndarray,  # (num_frames, 17, 3)
    'phases': [
        {'phase': 'entry', 'start': 0, 'end': 5},
        {'phase': 'catch', 'start': 5, 'end': 12},
        # ...
    ],
    'metrics': {
        'stroke_rate': 48,  # strokes per minute
        'cycle_time': 1.25,  # seconds
        'hand_path_efficiency': 0.85
    }
}
```

### Template Creation Process
1. **Source videos:** Obtain high-quality swimming videos
   - Olympic/World Championship footage
   - Coaching demonstration videos
   - Research lab motion capture data

2. **Process videos:**
   ```bash
   python scripts/create_ideal_template.py \
       --video data/elite/freestyle_elite.mp4 \
       --stroke-type freestyle \
       --output models/ideal_techniques/freestyle_elite.pkl
   ```

3. **Manual validation:**
   - Review detected phases
   - Verify keypoint quality
   - Annotate any corrections

4. **Create variants:**
   - Sprint vs distance technique
   - Different body types
   - Male vs female (optional)

### Required Templates (Minimum)
- [ ] Freestyle (elite, intermediate, beginner)
- [ ] Backstroke (elite, intermediate)
- [ ] Breaststroke (elite, intermediate)
- [ ] Butterfly (elite, intermediate)

---

## Testing Strategy

### Unit Tests

```python
def test_dtw_symmetric():
    """DTW distance should be symmetric"""
    stroke1 = generate_synthetic_stroke()
    stroke2 = generate_synthetic_stroke()
    dist1 = dtw_analyzer.compare_strokes(stroke1, stroke2)['dtw_distance']
    dist2 = dtw_analyzer.compare_strokes(stroke2, stroke1)['dtw_distance']
    assert np.isclose(dist1, dist2)

def test_dtw_identical_strokes():
    """Identical strokes should have distance 0"""
    stroke = generate_synthetic_stroke()
    result = dtw_analyzer.compare_strokes(stroke, stroke)
    assert result['dtw_distance'] == 0.0
    assert result['similarity_score'] == 100.0

def test_frechet_straight_line():
    """Straight line paths should have high efficiency"""
    path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    analysis = frechet_analyzer.analyze_stroke_trajectory_shape(path)
    assert analysis['path_efficiency'] > 0.99

def test_phase_detection_order():
    """Phases should be detected in correct order"""
    keypoints = load_test_stroke('freestyle_correct.pkl')
    phases = phase_detector.detect_phases_freestyle(keypoints)
    phase_names = [p['phase'] for p in phases]
    assert phase_names == ['entry', 'catch', 'pull', 'push', 'recovery']
```

### Integration Tests

```python
def test_full_stroke_comparison():
    """End-to-end stroke comparison"""
    swimmer_stroke = load_test_stroke('swimmer_freestyle.pkl')
    ideal_stroke = load_ideal_template('freestyle_elite.pkl')

    analyzer = StrokeSimilarityAnalyzer()
    result = analyzer.comprehensive_comparison(
        swimmer_stroke,
        ideal_stroke,
        stroke_type='freestyle'
    )

    assert 0 <= result['overall_score'] <= 100
    assert len(result['recommendations']) >= 1
    assert all(k in result['technique_breakdown']
               for k in ['entry', 'catch', 'pull', 'push', 'recovery'])
```

---

## Performance Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| DTW computation | <500ms | Real-time feedback |
| Frechet computation | <100ms | Simple algorithm |
| Phase detection | <200ms | Heuristic-based |
| Full comparison | <1s | Combined analysis |
| Template loading | <50ms | Cached in memory |

**Optimization Strategies:**
- Use Sakoe-Chiba constraints (radius=10) to limit DTW search space
- Cache DTW distances for frequently used templates
- Vectorize operations with NumPy
- Consider numba JIT compilation for hot paths

---

## UI Integration

### Streamlit Compare Mode Updates

```python
elif mode == "Compare":
    st.subheader("Compare with Ideal Technique")

    col1, col2 = st.columns(2)

    with col1:
        user_video = st.file_uploader("Your stroke", type=['mp4'])

    with col2:
        ideal = st.selectbox("Reference",
            ["Elite Freestyle", "Elite Backstroke", ...])

    if user_video and st.button("Analyze"):
        with st.spinner("Processing..."):
            # 1. Extract poses from user video
            # 2. Load ideal template
            # 3. Run comparison
            # 4. Display results

        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Score", f"{result['overall_score']:.1f}/100")
        col2.metric("Timing", f"{result['dtw_score']:.1f}/100")
        col3.metric("Path Shape", f"{result['frechet_score']:.1f}/100")

        # Breakdown
        st.subheader("Technique Breakdown")
        breakdown_df = pd.DataFrame([result['technique_breakdown']])
        st.bar_chart(breakdown_df.T)

        # Recommendations
        st.subheader("Recommendations")
        for i, rec in enumerate(result['recommendations'], 1):
            st.info(f"{i}. {rec}")
```

---

## Configuration

### `config/analysis_config.yaml`
```yaml
dtw:
  global_constraint: "sakoe_chiba"
  sakoe_chiba_radius: 10
  normalization: "max_length"
  similarity_threshold: 0.3  # For 0-100 scaling

frechet:
  distance_metric: "euclidean"
  path_efficiency_threshold: 0.7

stroke_phases:
  min_phase_duration: 0.1  # seconds
  velocity_smoothing_window: 5  # frames
  confidence_threshold: 0.6

scoring:
  weights:
    dtw: 0.4
    frechet: 0.3
    phase_sequence: 0.2
    symmetry: 0.1
```

---

## Dependencies

```python
tslearn>=0.6.3
scipy>=1.14.0
similaritymeasures>=0.9.0  # Or use scipy for Frechet
pandas>=2.1.0
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Phase detection accuracy | ≥80% | Manual annotation comparison |
| Similarity score correlation | r≥0.8 | vs expert ratings |
| Processing time | <1s | Automated tests |
| False positive phases | <10% | Validation dataset |

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Limited elite templates | High | Partner with swim clubs, use public videos |
| DTW too slow | Medium | Optimize with Sakoe-Chiba, consider GPU DTW |
| Phase detection fails | High | Provide manual phase annotation tool |
| Score doesn't match expert | High | Tune weights with expert feedback |

---

## Future Enhancements

- [ ] 3D DTW using depth data
- [ ] Multi-stroke cycle analysis
- [ ] Personalized templates (compare to own best)
- [ ] Video synchronization visualization
- [ ] Export DTW alignment as video

---

**Next Phase:** Phase 3 - Feature Extraction & Biomechanics
