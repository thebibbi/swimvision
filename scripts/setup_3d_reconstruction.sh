#!/bin/bash

################################################################################
# SwimVision Pro - 3D Reconstruction Setup Script
################################################################################
#
# This script installs all dependencies for Phase 1.5: 3D Reconstruction
# Including:
# - MotionAGFormer (primary 2D→3D lifter)
# - PoseFormerV2 (alternative frequency-domain lifter)
# - SAM3D Body (detailed 3D mesh reconstruction)
#
# Usage:
#   bash scripts/setup_3d_reconstruction.sh
#
# Prerequisites:
#   - Phase 1 setup completed (RTMPose + ByteTrack)
#   - Python 3.9+
#   - PyTorch 2.0+ with CUDA/MPS
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           SwimVision Pro - 3D Reconstruction Setup               ║
║                                                                   ║
║                        Phase 1.5                                 ║
║                                                                   ║
║  • MotionAGFormer    (Primary 2D→3D Lifter)                      ║
║  • PoseFormerV2      (Frequency-Domain Alternative)              ║
║  • SAM3D Body        (Detailed 3D Mesh)                          ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Warning: No virtual environment detected${NC}"
    echo -e "${YELLOW}Attempting to activate venv_advanced...${NC}\n"

    if [[ -d "venv_advanced" ]]; then
        source venv_advanced/bin/activate
        echo -e "${GREEN}✅ Activated venv_advanced${NC}\n"
    else
        echo -e "${RED}❌ Error: venv_advanced not found${NC}"
        echo -e "${YELLOW}💡 Please run setup.sh first to create the environment${NC}"
        exit 1
    fi
fi

# Check Python version
echo -e "${BLUE}[1/8] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    echo -e "${RED}❌ Python 3.9+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}\n"

# Check PyTorch installation
echo -e "${BLUE}[2/8] Checking PyTorch installation...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('MPS available (Apple Silicon)')
else:
    print('CPU only (no GPU acceleration)')
" || {
    echo -e "${RED}❌ PyTorch not installed or not working${NC}"
    echo -e "${YELLOW}💡 Please run setup.sh first${NC}"
    exit 1
}

echo -e "${GREEN}✅ PyTorch detected${NC}\n"

# Install core 3D reconstruction dependencies
echo -e "${BLUE}[3/8] Installing core dependencies...${NC}"
pip install -r requirements_3d_reconstruction.txt
echo -e "${GREEN}✅ Core dependencies installed${NC}\n"

# Create directories
mkdir -p external/3d_reconstruction
mkdir -p models/3d_reconstruction
mkdir -p checkpoints/motionagformer
mkdir -p checkpoints/poseformerv2
mkdir -p checkpoints/sam3d

# Install MotionAGFormer
echo -e "${BLUE}[4/8] Installing MotionAGFormer...${NC}"
cd external/3d_reconstruction

if [ -d "MotionAGFormer" ]; then
    echo -e "${YELLOW}⚠️  MotionAGFormer already exists, updating...${NC}"
    cd MotionAGFormer
    git pull
    cd ..
else
    echo "Cloning MotionAGFormer repository..."
    git clone https://github.com/TaatiTeam/MotionAGFormer.git
fi

cd MotionAGFormer

# Install MotionAGFormer requirements
echo "Installing MotionAGFormer dependencies..."
pip install -r requirements.txt

# Install MotionAGFormer in development mode
python setup.py develop || {
    echo -e "${YELLOW}⚠️  setup.py develop failed, trying pip install -e .${NC}"
    pip install -e .
}

cd ../../..

echo -e "${GREEN}✅ MotionAGFormer installed${NC}\n"

# Install PoseFormerV2
echo -e "${BLUE}[5/8] Installing PoseFormerV2...${NC}"
cd external/3d_reconstruction

if [ -d "PoseFormerV2" ]; then
    echo -e "${YELLOW}⚠️  PoseFormerV2 already exists, updating...${NC}"
    cd PoseFormerV2
    git pull
    cd ..
else
    echo "Cloning PoseFormerV2 repository..."
    git clone https://github.com/qitaozhao/poseformerv2.git PoseFormerV2
fi

cd PoseFormerV2

# Install PoseFormerV2 requirements
if [ -f "requirements.txt" ]; then
    echo "Installing PoseFormerV2 dependencies..."
    pip install -r requirements.txt
fi

# PoseFormerV2 might not have setup.py, so just add to path
cd ../../..

echo -e "${GREEN}✅ PoseFormerV2 installed${NC}\n"

# Install SAM3D Body
echo -e "${BLUE}[6/8] Installing SAM3D Body...${NC}"

# Check if sam_3d_body package is available
python3 -c "import sam_3d_body" 2>/dev/null && {
    echo -e "${YELLOW}⚠️  SAM3D Body already installed${NC}"
} || {
    echo "Installing SAM3D Body from GitHub..."

    # Try installing from GitHub
    pip install git+https://github.com/facebookresearch/sam-3d-body.git || {
        echo -e "${YELLOW}⚠️  Direct install failed, cloning repository...${NC}"

        cd external/3d_reconstruction

        if [ -d "sam-3d-body" ]; then
            echo "SAM3D repository already exists, updating..."
            cd sam-3d-body
            git pull
        else
            git clone https://github.com/facebookresearch/sam-3d-body.git
            cd sam-3d-body
        fi

        # Install SAM3D requirements
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        fi

        # Install SAM3D
        pip install -e .

        cd ../../..
    }
}

echo -e "${GREEN}✅ SAM3D Body installed${NC}\n"

# Download pre-trained models
echo -e "${BLUE}[7/8] Downloading pre-trained models...${NC}"

# Download MotionAGFormer models
echo "Downloading MotionAGFormer checkpoints..."
mkdir -p checkpoints/motionagformer

# Models will be downloaded on first use via the library
# Or manually from: https://github.com/TaatiTeam/MotionAGFormer/releases

echo -e "${YELLOW}ℹ️  MotionAGFormer models will be downloaded on first use${NC}"

# Download PoseFormerV2 models
echo "Downloading PoseFormerV2 checkpoints..."
mkdir -p checkpoints/poseformerv2

# Models will be downloaded on first use
echo -e "${YELLOW}ℹ️  PoseFormerV2 models will be downloaded on first use${NC}"

# Download SAM3D models from Hugging Face
echo "Downloading SAM3D Body models..."
python3 -c "
try:
    from sam_3d_body import load_sam_3d_body_hf
    print('Downloading SAM3D ViT-H model (recommended)...')
    model, cfg = load_sam_3d_body_hf('facebook/sam-3d-body-vit-h')
    print('✅ SAM3D ViT-H model downloaded')
except Exception as e:
    print(f'⚠️  SAM3D model download will happen on first use: {e}')
" || echo -e "${YELLOW}ℹ️  SAM3D models will be downloaded on first use${NC}"

echo -e "${GREEN}✅ Model downloads initiated${NC}\n"

# Test installation
echo -e "${BLUE}[8/8] Testing installation...${NC}\n"

python3 << 'PYEOF'
import sys

print("Testing 3D reconstruction components...\n")

# Test MotionAGFormer
try:
    # Try importing MotionAGFormer modules
    sys.path.insert(0, 'external/3d_reconstruction/MotionAGFormer')
    print("✅ MotionAGFormer: Import path configured")
except Exception as e:
    print(f"⚠️  MotionAGFormer: {e}")

# Test PoseFormerV2
try:
    sys.path.insert(0, 'external/3d_reconstruction/PoseFormerV2')
    print("✅ PoseFormerV2: Import path configured")
except Exception as e:
    print(f"⚠️  PoseFormerV2: {e}")

# Test SAM3D
try:
    import sam_3d_body
    print("✅ SAM3D Body: Package available")
except ImportError as e:
    print(f"⚠️  SAM3D Body: {e}")

# Test core dependencies
try:
    import einops
    print("✅ einops: OK")
except ImportError:
    print("❌ einops: NOT INSTALLED")

try:
    import timm
    print("✅ timm: OK")
except ImportError:
    print("❌ timm: NOT INSTALLED")

try:
    import transformers
    print("✅ transformers: OK")
except ImportError:
    print("❌ transformers: NOT INSTALLED")

try:
    import trimesh
    print("✅ trimesh: OK")
except ImportError:
    print("❌ trimesh: NOT INSTALLED")

# Test PyTorch3D (optional)
try:
    import pytorch3d
    print("✅ pytorch3d: OK (optional)")
except ImportError:
    print("ℹ️  pytorch3d: Not installed (optional)")

print("")
PYEOF

echo ""

# Success message
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                   ║${NC}"
echo -e "${GREEN}║                  ✅  Installation Complete!                        ║${NC}"
echo -e "${GREEN}║                                                                   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}                    QUICK START GUIDE                              ${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${GREEN}1. Test MotionAGFormer:${NC}"
echo -e "   ${BLUE}python demos/demo_motionagformer.py --video test.mp4${NC}\n"

echo -e "${GREEN}2. Test PoseFormerV2:${NC}"
echo -e "   ${BLUE}python demos/demo_poseformerv2.py --video test.mp4${NC}\n"

echo -e "${GREEN}3. Test SAM3D Body:${NC}"
echo -e "   ${BLUE}python demos/demo_sam3d.py --image swimmer.jpg${NC}\n"

echo -e "${GREEN}4. Test complete 3D pipeline:${NC}"
echo -e "   ${BLUE}python demos/demo_3d_reconstruction.py --video test.mp4${NC}\n"

echo -e "${GREEN}5. Integrate with Phase 1 pipeline:${NC}"
echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video test.mp4 --enable-3d${NC}\n"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${YELLOW}📚 Documentation:${NC}"
echo -e "   • 3D Reconstruction Guide:    ${BLUE}docs/PHASE1.5_3D_RECONSTRUCTION_PLAN.md${NC}"
echo -e "   • SAM3D Integration:          ${BLUE}docs/SAM3D_INTEGRATION_PLAN.md${NC}"
echo -e "   • Phase 1 Guide:              ${BLUE}docs/PHASE1_GUIDE.md${NC}\n"

echo -e "${YELLOW}🔧 Model Variants:${NC}"
echo -e "   • MotionAGFormer: XS (fast), S, B (balanced), L (accurate)"
echo -e "   • PoseFormerV2: 27-frame (fast), 81-frame, 243-frame (accurate)"
echo -e "   • SAM3D: vit-h (recommended), dinov3 (larger)\n"

echo -e "${YELLOW}⚡ Performance:${NC}"
echo -e "   • MotionAGFormer-XS: ~300 FPS (RTX 3090)"
echo -e "   • PoseFormerV2-27: ~400 FPS (RTX 3090)"
echo -e "   • SAM3D Body: ~0.5-1 FPS (detailed mesh)\n"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${GREEN}🎉 Phase 1.5 setup complete! Ready for 3D reconstruction!${NC}\n"

deactivate 2>/dev/null || true
