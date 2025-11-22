#!/bin/bash

################################################################################
# SwimVision Pro - Apple Silicon (M1/M2/M3) Setup Script
################################################################################
#
# This script sets up the SwimVision Pro advanced features on macOS with
# Apple Silicon (M1, M2, M3) chips using MPS (Metal Performance Shaders)
# backend for GPU acceleration.
#
# Usage:
#   bash scripts/setup_macos_apple_silicon.sh
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3)
#   - Python 3.9 or higher
#   - Homebrew (optional, for OpenCV dependencies)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   SwimVision Pro - Apple Silicon Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Check if running on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo -e "${RED}❌ Error: This script is for macOS only${NC}"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Not running on Apple Silicon (detected: $ARCH)${NC}"
    echo -e "${YELLOW}   This script is optimized for M1/M2/M3 Macs${NC}"
    echo -e "${YELLOW}   Continuing anyway...${NC}\n"
fi

# Check Python version
echo -e "${BLUE}[1/8] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    echo -e "${RED}❌ Python 3.9+ required, found $PYTHON_VERSION${NC}"
    echo -e "${YELLOW}💡 Install Python 3.9+ from https://www.python.org/${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}\n"

# Display system info
echo -e "${BLUE}[2/8] System Information${NC}"
echo "  macOS Version:  $(sw_vers -productVersion)"
echo "  Architecture:   $ARCH"
echo "  Chip:           $(sysctl -n machdep.cpu.brand_string)"
echo ""

# Create virtual environment
echo -e "${BLUE}[3/8] Creating virtual environment...${NC}"
if [ -d "venv_advanced" ]; then
    echo -e "${YELLOW}⚠️  venv_advanced already exists, removing...${NC}"
    rm -rf venv_advanced
fi

python3 -m venv venv_advanced
source venv_advanced/bin/activate

echo -e "${GREEN}✅ Virtual environment created${NC}\n"

# Upgrade pip, setuptools, wheel
echo -e "${BLUE}[4/8] Upgrading pip, setuptools, wheel...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✅ Tools upgraded${NC}\n"

# Install PyTorch with MPS support
echo -e "${BLUE}[5/8] Installing PyTorch with MPS support...${NC}"
echo -e "${YELLOW}ℹ️  Using PyTorch 2.5.1 (Python 3.12 compatible) with Apple Silicon MPS backend${NC}"

# For Apple Silicon, we use the default PyTorch which has MPS support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo -e "${GREEN}✅ PyTorch installed${NC}\n"

# Verify MPS availability
echo -e "${BLUE}[6/8] Verifying MPS availability...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✅ MPS backend is available!')
    try:
        # Test MPS
        x = torch.randn(10, 10, device='mps')
        y = x @ x
        print('✅ MPS backend is working!')
    except Exception as e:
        print(f'⚠️  MPS available but not working: {e}')
else:
    print('⚠️  MPS not available, will use CPU')
"
echo ""

# Install MMPose ecosystem
echo -e "${BLUE}[7/8] Installing MMPose ecosystem...${NC}"
pip install -U openmim

# openmim pulls in openxlab which downgrades setuptools to 60.x, but
# Python 3.12's pkg_resources requires a modern release. Re-upgrade
# setuptools BEFORE invoking mim (which imports setuptools).
pip install --no-deps --upgrade "setuptools>=75.0"

# xtcocotools (mmpose dependency) has a broken source dist. Use pycocotools instead,
# which is API-compatible and has working wheels for Apple Silicon.
pip install "pycocotools>=2.0.6"

mim install mmengine
mim install "mmcv>=2.0.0"

# Install mmpose without dependencies first, then install remaining deps manually
# This avoids the broken xtcocotools dependency
pip install --no-deps mmpose
pip install chumpy json-tricks munkres scipy
echo -e "${GREEN}✅ MMPose installed${NC}\n"

# Install core requirements
echo -e "${BLUE}[8/8] Installing core requirements...${NC}"

# Install from requirements_advanced.txt but skip CUDA-specific packages and packages with build issues on Apple Silicon
cat requirements_advanced.txt | grep -v "cupy" | grep -v "onnxruntime-gpu" | grep -v "decord" | grep -v "onnxsim" | grep -v "tf2onnx" > requirements_macos.txt

# Add onnxruntime (CPU version for macOS)
echo "onnxruntime>=1.16.0" >> requirements_macos.txt

pip install -r requirements_macos.txt

# Install time-series analysis dependencies required by DTW analyzer
pip install "dtaidistance>=2.3.12" "similaritymeasures>=1.1.0"

# Install ByteTrack
echo -e "${BLUE}Installing ByteTrack...${NC}"
mkdir -p external
cd external

if [ -d "ByteTrack" ]; then
    echo -e "${YELLOW}⚠️  ByteTrack already exists, updating...${NC}"
    cd ByteTrack
    git pull
else
    git clone https://github.com/ifzhang/ByteTrack.git
    cd ByteTrack
fi

pip install -r requirements.txt
# Install ByteTrack in editable mode without build isolation to avoid
# torch import issues inside pip's temporary build environment
pip install --no-build-isolation -e .
cd ../..

echo -e "${GREEN}✅ Core requirements installed${NC}\n"

# Download RTMPose models
echo -e "${BLUE}Downloading RTMPose models...${NC}"
mkdir -p models/rtmpose

# Download RTMPose-m (recommended for Apple Silicon)
RTMPOSE_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
RTMPOSE_FILE="models/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth"

if [ ! -f "$RTMPOSE_FILE" ]; then
    echo "Downloading RTMPose-m..."
    curl -L "$RTMPOSE_URL" -o "$RTMPOSE_FILE"
    echo -e "${GREEN}✅ RTMPose-m downloaded${NC}"
else
    echo -e "${YELLOW}RTMPose-m already exists${NC}"
fi

echo ""

# Cleanup
rm -f requirements_macos.txt

# Test installation
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Testing Installation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

python3 -c "
import sys
import torch
from src.utils.device_utils import DeviceManager

print('Testing SwimVision components...\n')

# Device info
DeviceManager.print_device_info()

# Test imports
try:
    from src.pose.rtmpose_estimator import RTMPoseEstimator
    print('✅ RTMPose estimator: OK')
except Exception as e:
    print(f'❌ RTMPose estimator: {e}')

try:
    from src.tracking.bytetrack_tracker import ByteTrackTracker
    print('✅ ByteTrack tracker: OK')
except Exception as e:
    print(f'❌ ByteTrack tracker: {e}')

try:
    from src.utils.format_converters import FormatConverter
    print('✅ Format converters: OK')
except Exception as e:
    print(f'❌ Format converters: {e}')

try:
    from src.pipeline.orchestrator import SwimVisionPipeline
    print('✅ Pipeline orchestrator: OK')
except Exception as e:
    print(f'❌ Pipeline orchestrator: {e}')

print('')
" || echo -e "${YELLOW}⚠️  Some components failed to import (this is OK if dependencies are missing)${NC}"

# Success message
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ Installation Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Activate the virtual environment:"
echo -e "   ${BLUE}source venv_advanced/bin/activate${NC}"
echo -e ""
echo -e "2. Test the pipeline:"
echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --create-test-video${NC}"
echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video data/videos/test_swimmers.mp4${NC}"
echo -e ""
echo -e "3. Use your webcam:"
echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --webcam${NC}"
echo -e ""
echo -e "${YELLOW}💡 Tips for Apple Silicon:${NC}"
echo -e "   • Use '--device mps' to explicitly use Metal Performance Shaders"
echo -e "   • Use '--device auto' for automatic device detection (default)"
echo -e "   • Recommended model for M1/M2/M3: rtmpose-m or rtmpose-s"
echo -e "   • Expected FPS on M1 Pro: 20-30 FPS (rtmpose-m @ 1080p)"
echo -e ""

deactivate
