# SwimVision Pro - Quick Start with uv

**Last Updated:** 2025-11-22  
**Setup Time:** ~5-10 minutes (10-100x faster than pip!)

## 🚀 What's New?

- ✅ **Single unified environment** (no more `venv` vs `venv_advanced` confusion)
- ✅ **10-100x faster installs** with `uv` package manager
- ✅ **Modular installation** - install only what you need
- ✅ **All estimators verified** - every model implements the correct interface
- ✅ **FreeMoCap integration** - multi-camera 3D motion capture now available in UI

---

## 📦 Installation Options

### Option 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/thebibbi/swimvision.git
cd swimvision

# Run setup script (installs uv automatically)
chmod +x scripts/setup_with_uv.sh
./scripts/setup_with_uv.sh

# Follow the prompts to select features:
# 1) Basic (YOLO11, MediaPipe, Streamlit)
# 2) Advanced (Basic + RTMPose, ViTPose, ByteTrack, SMPL-X) [Recommended]
# 3) Complete (Advanced + FreeMoCap, RealSense, Export tools)
# 4) All (Complete + Development tools)
```

### Option 2: Manual Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment
uv venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# 3. Install PyTorch (choose based on your platform)
# For Apple Silicon (MPS):
uv pip install torch torchvision torchaudio

# For CUDA 11.8:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install SwimVision with desired features
# Basic (fast, ~2 min):
uv pip install -e ".[basic]"

# Advanced (recommended, ~5 min):
uv pip install -e ".[advanced]"

# Complete (everything, ~10 min):
uv pip install -e ".[complete]"

# 5. Install MMPose (if using advanced/complete)
uv pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose>=1.2.0"

# 6. Verify installation
python scripts/audit_estimators.py
```

---

## 🎯 Feature Sets Explained

### Basic (`[basic]`)
**Install time:** ~2 minutes  
**Includes:**
- ✅ YOLO11 pose estimation
- ✅ MediaPipe pose estimation
- ✅ Streamlit web interface
- ✅ Video processing
- ✅ Swimming analysis
- ✅ Injury prediction

**Best for:** Quick start, single-swimmer analysis, CPU-only systems

### Advanced (`[advanced]`)
**Install time:** ~5 minutes  
**Includes:** Everything in Basic, plus:
- ✅ RTMPose (MMPose)
- ✅ ViTPose (MMPose)
- ✅ ByteTrack multi-person tracking
- ✅ SMPL-X 3D body models
- ✅ ONNX optimization
- ✅ GPU acceleration

**Best for:** Multi-swimmer tracking, advanced pose estimation, GPU systems

### Complete (`[complete]`)
**Install time:** ~10 minutes  
**Includes:** Everything in Advanced, plus:
- ✅ FreeMoCap multi-camera 3D
- ✅ RealSense camera support
- ✅ PDF/Excel export
- ✅ All advanced features

**Best for:** Professional setups, multi-camera systems, research

### All (`[all]`)
**Includes:** Everything in Complete, plus:
- ✅ Development tools (pytest, ruff, mypy)
- ✅ Documentation tools
- ✅ Profiling tools

**Best for:** Contributors, developers

---

## 🏃 Running the App

```bash
# Activate environment
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Run Streamlit app
streamlit run app.py

# The app will open in your browser at http://localhost:8501
```

---

## 🧪 Verify Installation

```bash
# Check all estimators implement the interface correctly
python scripts/audit_estimators.py

# Expected output:
# ✅ PASS YOLOPoseEstimator
# ✅ PASS MediaPipeEstimator
# ✅ PASS RTMPoseEstimator
# ✅ PASS ViTPoseEstimator
# ✅ PASS AlphaPoseEstimator
# ✅ PASS OpenPoseEstimator
# ✅ PASS SMPLEstimator
# ✅ All estimators implement the required interface!
```

---

## 📊 Available Models in UI

After installation, you'll see these models in the Streamlit UI:

1. **YOLO11** - Fast, accurate, multi-person (✅ Works out of the box)
2. **MediaPipe** - Fast, single-person, 3D landmarks (✅ Works out of the box)
3. **OpenPose** - Multi-person, 2D (⚠️ Requires system installation)
4. **AlphaPose** - Multi-person, whole-body (⚠️ Requires advanced setup)
5. **RTMPose** - Fast, accurate, multi-person (✅ With `[advanced]`)
6. **ViTPose** - High accuracy, multi-person (✅ With `[advanced]`)
7. **SMPL-X** - 3D body mesh, hands, face (✅ With `[advanced]`)
8. **FreeMoCap (Multi-Camera 3D)** - Multi-camera 3D (✅ With `[complete]`)
9. **Multi-Model Fusion** - Combine multiple models (✅ Always available)

---

## 🔧 Troubleshooting

### Issue: "Can't instantiate abstract class YOLOPoseEstimator"
**Status:** ✅ FIXED  
**Solution:** This was fixed by adding the missing `supports_multi_person()` method.

### Issue: Models not loading
**Check:**
1. Correct environment activated? `which python` should show `.venv/bin/python`
2. Correct feature set installed? Run `python scripts/audit_estimators.py`
3. For MMPose models, did you run `mim install mmpose`?

### Issue: Slow installation
**Solution:** You're probably using pip. Switch to uv:
```bash
pip install uv  # Install uv once
uv pip install -e ".[advanced]"  # 10-100x faster!
```

### Issue: CUDA out of memory
**Solution:** Use smaller models or CPU:
- YOLO: Use `yolo11n-pose.pt` instead of `yolo11m-pose.pt`
- RTMPose: Use `rtmpose-s` instead of `rtmpose-l`
- Set device to `cpu` in UI

### Issue: FreeMoCap not visible
**Solution:** Install complete feature set:
```bash
uv pip install -e ".[complete]"
```

---

## 📚 Next Steps

1. **Upload a video** in the Streamlit UI
2. **Select a pose model** (start with YOLO11)
3. **Adjust confidence threshold** if needed
4. **Enable tracking** for multi-swimmer analysis
5. **View analysis** - stroke detection, angles, trajectory

### Advanced Usage

- **Multi-camera 3D:** Select "FreeMoCap (Multi-Camera 3D)" and upload calibration
- **Multi-model fusion:** Select "Multi-Model Fusion" to combine predictions
- **Export results:** Use the export buttons to save analysis as PDF/Excel

---

## 🆚 Old vs New Setup

| Aspect | Old Setup | New Setup |
|--------|-----------|-----------|
| **Environments** | 2 (`venv`, `venv_advanced`) | 1 (`.venv`) |
| **Package Manager** | pip | uv (10-100x faster) |
| **Install Time** | 30-60 min | 5-10 min |
| **Complexity** | High (which env?) | Low (one env, feature flags) |
| **Flexibility** | Low (all or nothing) | High (install what you need) |
| **FreeMoCap** | Hidden | Visible in UI |

---

## 🔄 Migrating from Old Setup

If you have existing `venv` or `venv_advanced`:

```bash
# 1. Backup old environments (optional)
mv venv venv_backup
mv venv_advanced venv_advanced_backup

# 2. Run new setup
./scripts/setup_with_uv.sh

# 3. Test
streamlit run app.py

# 4. Remove backups once verified (optional)
rm -rf venv_backup venv_advanced_backup
```

---

## 💡 Tips

1. **Start with Basic:** Install `[basic]` first, test, then upgrade to `[advanced]`
2. **Use GPU:** Set device to `cuda` or `mps` for 5-10x speedup
3. **Multi-person:** Use YOLO11, RTMPose, or ViTPose for multi-swimmer tracking
4. **3D reconstruction:** Use SMPL-X for single-person or FreeMoCap for multi-camera
5. **Combine models:** Use Multi-Model Fusion for best accuracy

---

## 📖 Documentation

- **Full Setup Guide:** [COMPREHENSIVE_FIX_PLAN.md](COMPREHENSIVE_FIX_PLAN.md)
- **Apple Silicon:** [docs/APPLE_SILICON_GUIDE.md](docs/APPLE_SILICON_GUIDE.md)
- **API Reference:** [docs/API.md](docs/API.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🐛 Found a Bug?

1. Check [COMPREHENSIVE_FIX_PLAN.md](COMPREHENSIVE_FIX_PLAN.md) for known issues
2. Run `python scripts/audit_estimators.py` to verify estimators
3. Check logs in `logs/` directory
4. Open an issue on GitHub with:
   - Python version (`python --version`)
   - Platform (macOS/Linux/Windows)
   - Feature set installed (`[basic]`/`[advanced]`/`[complete]`)
   - Error message and traceback

---

## ✅ Success Checklist

- [ ] uv installed (`uv --version`)
- [ ] Virtual environment created (`.venv/` exists)
- [ ] SwimVision installed (`python -c "import src.pose; print('OK')"`)
- [ ] Estimators verified (`python scripts/audit_estimators.py` passes)
- [ ] Streamlit runs (`streamlit run app.py` opens in browser)
- [ ] Can select models in UI
- [ ] Can upload and process video

If all checked, you're ready to go! 🎉

---

**Questions?** See [COMPREHENSIVE_FIX_PLAN.md](COMPREHENSIVE_FIX_PLAN.md) or open an issue.
