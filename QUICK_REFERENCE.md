# SwimVision Pro - Quick Reference Card

**Last Updated:** 2025-11-22

---

## 🚀 Quick Start

```bash
# Current environment (works now)
source venv_advanced/bin/activate
streamlit run app.py

# New setup (recommended for fresh install)
./scripts/setup_with_uv.sh
streamlit run app.py
```

---

## ✅ Verification Commands

```bash
# Check all estimators implement interface
python scripts/audit_estimators.py

# Run unit tests
pytest tests/test_pose_estimators.py -v

# Test YOLO instantiation
python -c "from src.pose import YOLOPoseEstimator; e = YOLOPoseEstimator(); print('✅ OK')"
```

---

## 📦 Installation Options

### Basic (Fast, ~2 min)
```bash
uv pip install -e ".[basic]"
```
Includes: YOLO11, MediaPipe, Streamlit

### Advanced (Recommended, ~5 min)
```bash
uv pip install -e ".[advanced]"
```
Includes: Basic + RTMPose, ViTPose, ByteTrack, SMPL-X

### Complete (~10 min)
```bash
uv pip install -e ".[complete]"
```
Includes: Advanced + FreeMoCap, RealSense, Export

---

## 🔧 Common Issues & Fixes

### Issue: "Can't instantiate YOLOPoseEstimator"
**Fix:** ✅ Already fixed - update to latest code

### Issue: Models not loading
**Fix:** Check environment activated:
```bash
which python  # Should show .venv/bin/python
```

### Issue: Pose not displaying
**Fix:** ✅ Already fixed in app.py - handles list[dict] returns

### Issue: Slow installation
**Fix:** Use uv instead of pip:
```bash
pip install uv
uv pip install -e ".[advanced]"  # 10-100x faster!
```

---

## 📊 Available Models

| Model | Multi-Person | 3D | Speed | Accuracy | Status |
|-------|--------------|----|----|----------|--------|
| YOLO11 | ✅ | ❌ | ⚡⚡⚡ | ⭐⭐⭐ | ✅ Working |
| MediaPipe | ❌ | ✅ | ⚡⚡⚡ | ⭐⭐ | ✅ Working |
| RTMPose | ✅ | ❌ | ⚡⚡ | ⭐⭐⭐⭐ | ✅ Working |
| ViTPose | ✅ | ❌ | ⚡ | ⭐⭐⭐⭐⭐ | ✅ Working |
| AlphaPose | ✅ | ❌ | ⚡ | ⭐⭐⭐⭐ | ⚠️ Needs setup |
| OpenPose | ✅ | ❌ | ⚡ | ⭐⭐⭐ | ⚠️ Needs setup |
| SMPL-X | ❌ | ✅ | ⚡ | ⭐⭐⭐⭐ | ✅ Working |
| FreeMoCap | ✅ | ✅ | ⚡ | ⭐⭐⭐⭐⭐ | ✅ UI Added |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pose_estimators.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Quick smoke test
python -c "from src.pose import *; print('✅ All imports OK')"
```

---

## 📁 Key Files

### Code
- `src/pose/yolo_estimator.py` - YOLO11 implementation
- `src/pose/base_estimator.py` - Abstract base class
- `app.py` - Streamlit UI (lines 139-182 handle list[dict])

### Setup
- `pyproject.toml` - Package configuration
- `scripts/setup_with_uv.sh` - Unified setup
- `scripts/audit_estimators.py` - Verification tool

### Tests
- `tests/test_pose_estimators.py` - Unit tests

### Documentation
- `QUICKSTART_UV.md` - Quick start guide
- `COMPREHENSIVE_FIX_PLAN.md` - Detailed plan
- `SESSION_SUMMARY_2025-11-22.md` - Session summary
- `QUICK_REFERENCE.md` - This file

---

## 🐛 Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Model Loading
```python
from src.pose import YOLOPoseEstimator
est = YOLOPoseEstimator()
print(f"Model: {est.model}")
print(f"Device: {est.device}")
print(f"Supports multi-person: {est.supports_multi_person()}")
```

### Test Inference
```python
import numpy as np
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
pose_data, _ = est.estimate_pose(test_img, return_image=False)
print(f"Result type: {type(pose_data)}")
if pose_data:
    print(f"Detections: {len(pose_data) if isinstance(pose_data, list) else 1}")
```

---

## 💡 Tips

1. **Start with Basic:** Install `[basic]` first, test, then upgrade
2. **Use GPU:** Set device to `cuda` or `mps` for 5-10x speedup
3. **Multi-person:** Use YOLO11, RTMPose, or ViTPose
4. **3D:** Use SMPL-X (single-person) or FreeMoCap (multi-camera)
5. **Combine models:** Use Multi-Model Fusion for best accuracy

---

## 📞 Get Help

1. **Check docs:** Start with `QUICKSTART_UV.md`
2. **Run audit:** `python scripts/audit_estimators.py`
3. **Check logs:** Look in `logs/` directory
4. **Run tests:** `pytest tests/ -v`
5. **GitHub Issues:** Report bugs with full error trace

---

## 🎯 Quick Commands Cheat Sheet

```bash
# Setup
./scripts/setup_with_uv.sh              # New setup
source venv_advanced/bin/activate        # Activate current env

# Verify
python scripts/audit_estimators.py       # Check estimators
pytest tests/test_pose_estimators.py -v  # Run tests

# Run
streamlit run app.py                     # Start UI
python demos/demo_phase1_pipeline.py     # Run demo

# Install
uv pip install -e ".[basic]"            # Basic features
uv pip install -e ".[advanced]"         # Advanced features
uv pip install -e ".[complete]"         # Everything

# Debug
python -c "from src.pose import *"       # Test imports
pytest tests/ --pdb                      # Debug on failure
streamlit run app.py --logger.level=debug # Debug Streamlit
```

---

**Version:** 1.0  
**Status:** ✅ All major fixes complete  
**Next:** Test with real video, fix remaining issues
