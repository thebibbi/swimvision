# SwimVision Pro - Phase 2 Demo Guide

This guide explains how to run the Phase 2 demonstration and use the enhanced Streamlit application.

## 🎯 Quick Start

### Run the Phase 2 Demonstration

The demonstration script showcases all Phase 2 time-series analysis features with synthetic swimming data:

```bash
# Activate your virtual environment first
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Run the demonstration
python scripts/demo_phase2.py
```

**What the demo shows:**
- ✅ DTW Analysis with similarity scoring
- ✅ 6 different similarity measures (DTW, Soft-DTW, LCSS, Cosine, Cross-correlation, Euclidean)
- ✅ Fréchet distance for trajectory comparison
- ✅ Stroke phase detection (5 phases)
- ✅ Pull pattern classification (S-pull, I-pull, straight, irregular)
- ✅ Comprehensive ensemble scoring
- ✅ Progressive analysis with fatigue detection
- ✅ Visual comparison (saved to `data/exports/phase2_demonstration.png`)

### Run the Streamlit Application

```bash
# Make sure you're in the virtual environment
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📱 Using the Streamlit App

### Mode 1: Upload Video

**Purpose:** Analyze a swimming video and extract stroke data

**Steps:**
1. Select "Upload Video" from the mode dropdown
2. Upload a swimming video (MP4, AVI, MOV, MKV)
3. Check options:
   - ✅ Detect Stroke Phases
   - ✅ Extract Hand Trajectories
4. Click "Initialize/Reload Model" in sidebar (first time only)
5. Click "🎬 Process Video"
6. View results:
   - Detected stroke phases table
   - Average phase durations chart
   - Hand trajectory plots (left and right)

**What it extracts:**
- Pose sequence for every frame
- Hand trajectories (left and right wrists)
- Joint angles over time (elbows, shoulders)
- Stroke phases with timings
- Detection rate and performance metrics

### Mode 2: Compare Strokes

**Purpose:** Compare two analyzed videos to measure similarity

**Prerequisites:** At least 2 videos must be analyzed first in "Upload Video" mode

**Steps:**
1. Select "Compare Strokes" from the mode dropdown
2. Choose "Reference Video" (ideal technique)
3. Choose "Comparison Video" (your stroke)
4. Click "🔍 Compare Strokes"
5. View results:
   - **Overall Similarity Score** (0-100)
   - **Rating:** Excellent (90+), Good (75+), Needs Work (60+), Significant Differences (<60)
   - **Radar Chart:** Visual comparison across metrics
   - **Individual Scores:** DTW, Fréchet, Phase Timing, Angles
   - **Recommendations:** Actionable technique advice
   - **Additional Measures:** Soft-DTW, LCSS, Cross-correlation

**Interpretation:**
- **DTW Score:** How similar the hand paths are (accounts for timing variations)
- **Fréchet Score:** Trajectory shape similarity
- **Phase Timing Score:** How well stroke phases align
- **Angle Similarity:** Joint angle sequence matching

### Mode 3: Stroke Analysis

**Purpose:** Deep dive into a single video's biomechanics

**Steps:**
1. Select "Stroke Analysis" from the mode dropdown
2. Choose a previously analyzed video
3. View detailed analysis:
   - Summary metrics (frames, FPS, duration)
   - Joint angles over time (interactive charts)
   - Trajectory information
   - Phase breakdown

## 🎨 Visualization Features

### In the App:
- **Real-time skeleton overlay** during processing
- **Joint angle annotations** on video frames
- **Hand trajectory trails** (last 30 frames)
- **Interactive Plotly charts** (zoom, pan, hover)
- **Radar charts** for multi-metric comparison
- **Progress bars** and real-time FPS counter

### In the Demo:
- **4-panel comparison figure**:
  1. Hand trajectories overlay
  2. Elbow angle time series
  3. Similarity scores bar chart
  4. Summary text panel
- Saved to `data/exports/phase2_demonstration.png`

## 📊 Understanding the Metrics

### DTW (Dynamic Time Warping)
- **Range:** 0-100 (higher = more similar)
- **Best for:** Comparing strokes with timing variations
- **Accounts for:** Speed differences, phase shifts

### Fréchet Distance
- **Range:** 0-infinity (lower = more similar)
- **Best for:** Trajectory shape comparison
- **Accounts for:** Path geometry, spatial similarity

### LCSS (Longest Common Subsequence)
- **Range:** 0-100
- **Best for:** Noisy data with outliers
- **Accounts for:** Matching subsequences, robust to noise

### Soft-DTW
- **Range:** 0-infinity (lower = more similar)
- **Best for:** Differentiable optimization
- **Accounts for:** Smooth alignment

### Cosine Similarity
- **Range:** -1 to 1 (1 = identical direction)
- **Best for:** Angle sequences
- **Accounts for:** Directional similarity

### Cross-Correlation
- **Range:** -1 to 1 (1 = perfect correlation)
- **Best for:** Phase alignment detection
- **Accounts for:** Temporal synchronization

## 🎬 Demo Output Example

```
════════════════════════════════════════════════════════════════════════════════
  Dynamic Time Warping (DTW) Analysis
════════════════════════════════════════════════════════════════════════════════

DTW Distance: 2.3456
Similarity Score: 76.54/100
Interpretation: Good match

Warping path length: 95
Alignment efficiency: 1.06

════════════════════════════════════════════════════════════════════════════════
  Stroke Phase Detection
════════════════════════════════════════════════════════════════════════════════

Detected 8 stroke phases:

Phase        Start    End      Duration (s)
──────────────────────────────────────────────────────
entry        0        15       0.500
catch        15       25       0.333
pull         25       45       0.667
push         45       60       0.500
recovery     60       75       0.500
...

Average Phase Durations:
  entry: 0.500s
  catch: 0.333s
  pull: 0.667s
  push: 0.500s
  recovery: 0.500s

Detected 2 complete stroke cycles
```

## 🔧 Troubleshooting

### "Please initialize the model first"
**Solution:** Click "Initialize/Reload Model" in the sidebar before processing videos

### "Need at least 2 videos for comparison"
**Solution:** Upload and analyze at least 2 videos in "Upload Video" mode first

### Slow processing
**Solutions:**
- Use smaller videos (shorter duration, lower resolution)
- Select "cpu" device if GPU is not available
- Try "yolo11n-pose.pt" (fastest) instead of larger models
- Lower the confidence threshold slightly

### No phases detected
**Possible causes:**
- Video too short (need at least 10 frames with hand visible)
- Poor pose detection (low confidence)
- Hand not visible in most frames

**Solutions:**
- Use longer video clips
- Improve lighting and camera angle
- Increase confidence threshold if getting false positives

## 📈 Best Practices

### For Video Upload:
1. **Camera position:** Side view of swimmer, perpendicular to swimming direction
2. **Lighting:** Good, even lighting
3. **Resolution:** At least 640x480
4. **Duration:** 5-30 seconds for single stroke analysis
5. **Single swimmer:** Best results with one person in frame

### For Comparison:
1. **Same camera angle:** Compare videos shot from similar perspectives
2. **Same stroke type:** Compare freestyle to freestyle, etc.
3. **Similar duration:** Videos of roughly same length
4. **Consistent conditions:** Similar lighting, water conditions

### For Best Results:
- Process multiple strokes from same swimmer to track progress
- Use an elite swimmer video as reference
- Focus on one aspect at a time (e.g., hand path, then angles)
- Take recommendations one at a time

## 🎓 Next Steps

After exploring Phase 2:
1. Try with your own swimming videos
2. Compare your technique to an elite swimmer
3. Track your progress over multiple sessions
4. Focus on specific recommendations
5. Monitor fatigue with progressive analysis

## 🆘 Need Help?

- **Documentation:** See `README.md` and `TODO.md`
- **Code examples:** Check `tests/unit/` for usage examples
- **Issues:** Report at GitHub issues page
- **Configuration:** Adjust settings in `config/*.yaml`

---

**Have fun analyzing swimming technique! 🏊‍♂️**
