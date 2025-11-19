#!/bin/bash
# SwimVision Pro - Advanced Features Setup Script
# This script installs all dependencies for advanced features
# Tested on Ubuntu 22.04 with CUDA 11.8

set -e  # Exit on error

echo "=================================="
echo "SwimVision Pro - Advanced Setup"
echo "=================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"
echo ""

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✅ CUDA version: $CUDA_VERSION"
else
    echo "⚠️  CUDA not found. GPU acceleration will not be available."
    echo "   Install CUDA 11.8+ for best performance."
fi
echo ""

# Create virtual environment (recommended)
echo "📦 Setting up virtual environment..."
if [ ! -d "venv_advanced" ]; then
    python3 -m venv venv_advanced
    echo "✅ Created virtual environment: venv_advanced"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv_advanced/bin/activate
echo "✅ Activated virtual environment"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install PyTorch (CUDA 11.8 version)
echo "📦 Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
echo "✅ PyTorch installed"
echo ""

# Install MMPose ecosystem
echo "📦 Installing MMPose ecosystem..."
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpose>=1.2.0"
echo "✅ MMPose installed"
echo ""

# Install core requirements
echo "📦 Installing core requirements..."
pip install -r requirements_advanced.txt
echo "✅ Core requirements installed"
echo ""

# Install ByteTrack
echo "📦 Installing ByteTrack..."
if [ ! -d "external/ByteTrack" ]; then
    mkdir -p external
    cd external
    git clone https://github.com/ifzhang/ByteTrack.git
    cd ByteTrack
    pip install -r requirements.txt
    python setup.py develop
    cd ../..
    echo "✅ ByteTrack installed"
else
    echo "✅ ByteTrack already installed"
fi
echo ""

# Install WHAM
echo "📦 Installing WHAM..."
if [ ! -d "external/WHAM" ]; then
    mkdir -p external
    cd external
    git clone https://github.com/yohanshin/WHAM.git
    cd WHAM
    pip install -r requirements.txt
    cd ../..
    echo "✅ WHAM installed"
    echo "⚠️  Note: Download WHAM pretrained models from:"
    echo "   https://github.com/yohanshin/WHAM#download-models"
else
    echo "✅ WHAM already installed"
fi
echo ""

# Install 4D Gaussian Splatting
echo "📦 Installing 4D Gaussian Splatting..."
if [ ! -d "external/4DGaussians" ]; then
    mkdir -p external
    cd external
    git clone https://github.com/hustvl/4DGaussians.git --recursive
    cd 4DGaussians
    pip install submodules/diff-gaussian-rasterization
    pip install submodules/simple-knn
    cd ../..
    echo "✅ 4D Gaussian Splatting installed"
else
    echo "✅ 4D Gaussian Splatting already installed"
fi
echo ""

# Install OpenSim (optional, via conda)
echo "📦 Checking for OpenSim..."
if python -c "import opensim" 2>/dev/null; then
    echo "✅ OpenSim already installed"
else
    echo "⚠️  OpenSim not found."
    echo "   Install via conda (recommended):"
    echo "   conda install -c opensim-org opensim=4.4.1"
    echo ""
    read -p "   Install OpenSim via conda now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v conda &> /dev/null; then
            conda install -c opensim-org opensim=4.4.1 -y
            echo "✅ OpenSim installed"
        else
            echo "❌ Conda not found. Please install Miniconda/Anaconda first."
        fi
    fi
fi
echo ""

# Download RTMPose models
echo "📦 Downloading RTMPose models..."
mkdir -p models/rtmpose
cd models/rtmpose

# RTMPose-s (small, fast)
if [ ! -f "rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth" ]; then
    echo "   Downloading RTMPose-s..."
    wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth -O rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192.pth
fi

# RTMPose-m (medium, balanced)
if [ ! -f "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth" ]; then
    echo "   Downloading RTMPose-m..."
    wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth -O rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth
fi

cd ../..
echo "✅ RTMPose models downloaded"
echo ""

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/videos
mkdir -p data/swimming_dataset
mkdir -p models/opensim
mkdir -p models/wham
mkdir -p models/4dgs
mkdir -p results/biomechanics
mkdir -p results/3d_reconstruction
mkdir -p results/coaching
mkdir -p logs
mkdir -p cache
echo "✅ Directories created"
echo ""

# Verify installation
echo "🧪 Verifying installation..."
python3 - << 'EOF'
import sys

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

try:
    import mmpose
    print(f"✅ MMPose: {mmpose.__version__}")
except ImportError:
    print("❌ MMPose not installed")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV not installed")

try:
    import streamlit
    print(f"✅ Streamlit: {streamlit.__version__}")
except ImportError:
    print("❌ Streamlit not installed")

try:
    import opensim
    print(f"✅ OpenSim: {opensim.GetVersion()}")
except ImportError:
    print("⚠️  OpenSim not installed (optional)")

print("\n✅ Core installation verified!")
EOF

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv_advanced/bin/activate"
echo "2. Download SMPL models: https://smpl-x.is.tue.mpg.de/"
echo "3. Download WHAM pretrained models"
echo "4. Run tests: pytest tests/"
echo "5. Start application: streamlit run app.py"
echo ""
echo "For development:"
echo "  - Format code: black ."
echo "  - Run linting: flake8 src/"
echo "  - Type checking: mypy src/"
echo ""
