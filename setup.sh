#!/bin/bash

################################################################################
# SwimVision Pro - Master Setup Script
################################################################################
#
# This script automatically detects your system and runs the appropriate
# setup process. It handles:
# - Linux/Windows with NVIDIA GPU (CUDA)
# - macOS with Apple Silicon (M1/M2/M3)
# - CPU-only systems
#
# Usage:
#   bash setup.sh              # Interactive setup
#   bash setup.sh --auto       # Automatic setup (no prompts)
#   bash setup.sh --demo       # Setup + run demo
#   bash setup.sh --test       # Setup + run tests
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

# Parse arguments
AUTO_MODE=false
RUN_DEMO=false
RUN_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --demo)
            RUN_DEMO=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --help)
            echo "SwimVision Pro - Master Setup Script"
            echo ""
            echo "Usage:"
            echo "  bash setup.sh              # Interactive setup"
            echo "  bash setup.sh --auto       # Automatic setup (no prompts)"
            echo "  bash setup.sh --demo       # Setup + run demo"
            echo "  bash setup.sh --test       # Setup + run tests"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Banner
clear
echo -e "${MAGENTA}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║              ███████╗██╗    ██╗██╗███╗   ███╗                    ║
║              ██╔════╝██║    ██║██║████╗ ████║                    ║
║              ███████╗██║ █╗ ██║██║██╔████╔██║                    ║
║              ╚════██║██║███╗██║██║██║╚██╔╝██║                    ║
║              ███████║╚███╔███╔╝██║██║ ╚═╝ ██║                    ║
║              ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝     ╚═╝                    ║
║                                                                   ║
║                    VISION PRO - Phase 1 Setup                    ║
║                                                                   ║
║           Advanced Pose Estimation & Tracking System             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Detect system
echo -e "${BLUE}[1/6] Detecting system configuration...${NC}\n"

OS=$(uname -s)
ARCH=$(uname -m)

echo -e "  Operating System: ${CYAN}$OS${NC}"
echo -e "  Architecture:     ${CYAN}$ARCH${NC}"

# Detect hardware acceleration
HAS_NVIDIA=false
HAS_APPLE_SILICON=false
HAS_GPU=false

if [[ "$OS" == "Darwin" ]]; then
    echo -e "  Platform:         ${CYAN}macOS${NC}"

    if [[ "$ARCH" == "arm64" ]]; then
        HAS_APPLE_SILICON=true
        HAS_GPU=true
        CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
        echo -e "  Chip:             ${GREEN}$CHIP${NC}"
        echo -e "  GPU:              ${GREEN}Apple Silicon (MPS)${NC}"
    else
        echo -e "  Chip:             ${YELLOW}Intel${NC}"
        echo -e "  GPU:              ${YELLOW}Not detected${NC}"
    fi

elif [[ "$OS" == "Linux" ]]; then
    echo -e "  Platform:         ${CYAN}Linux${NC}"

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        HAS_NVIDIA=true
        HAS_GPU=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
        echo -e "  GPU:              ${GREEN}NVIDIA $GPU_NAME${NC}"
    else
        echo -e "  GPU:              ${YELLOW}NVIDIA GPU not detected${NC}"
    fi

else
    echo -e "  Platform:         ${CYAN}$OS${NC}"
    echo -e "  GPU:              ${YELLOW}Detection not supported${NC}"
fi

echo ""

# Recommend setup path
echo -e "${BLUE}[2/6] Determining optimal setup...${NC}\n"

SETUP_SCRIPT=""
DEVICE_TYPE=""

if [[ "$HAS_APPLE_SILICON" == true ]]; then
    SETUP_SCRIPT="scripts/setup_macos_apple_silicon.sh"
    DEVICE_TYPE="Apple Silicon (MPS)"
    EXPECTED_FPS="20-35 FPS"
    RECOMMENDED_MODEL="rtmpose-m"
    echo -e "  ✅ Detected: ${GREEN}Apple Silicon Mac${NC}"
    echo -e "  📦 Setup:    ${CYAN}$SETUP_SCRIPT${NC}"
    echo -e "  🚀 Device:   ${GREEN}MPS (Metal Performance Shaders)${NC}"
    echo -e "  ⚡ Expected: ${GREEN}$EXPECTED_FPS @ 1080p${NC}"

elif [[ "$HAS_NVIDIA" == true ]]; then
    SETUP_SCRIPT="scripts/setup_advanced_features.sh"
    DEVICE_TYPE="NVIDIA GPU (CUDA)"
    EXPECTED_FPS="40-85 FPS"
    RECOMMENDED_MODEL="rtmpose-m"
    echo -e "  ✅ Detected: ${GREEN}NVIDIA GPU${NC}"
    echo -e "  📦 Setup:    ${CYAN}$SETUP_SCRIPT${NC}"
    echo -e "  🚀 Device:   ${GREEN}CUDA${NC}"
    echo -e "  ⚡ Expected: ${GREEN}$EXPECTED_FPS @ 1080p${NC}"

else
    SETUP_SCRIPT="scripts/setup_advanced_features.sh"
    DEVICE_TYPE="CPU"
    EXPECTED_FPS="5-10 FPS"
    RECOMMENDED_MODEL="rtmpose-s"
    echo -e "  ⚠️  Detected: ${YELLOW}CPU only (no GPU acceleration)${NC}"
    echo -e "  📦 Setup:    ${CYAN}$SETUP_SCRIPT${NC}"
    echo -e "  🚀 Device:   ${YELLOW}CPU${NC}"
    echo -e "  ⚡ Expected: ${YELLOW}$EXPECTED_FPS @ 1080p${NC}"
    echo -e "  ${YELLOW}💡 Consider using a system with GPU for real-time performance${NC}"
fi

echo ""

# Confirm before proceeding
if [[ "$AUTO_MODE" == false ]]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}This will install:${NC}"
    echo -e "  • Python virtual environment"
    echo -e "  • PyTorch 2.1.0 (with $DEVICE_TYPE support)"
    echo -e "  • MMPose ecosystem"
    echo -e "  • ByteTrack multi-object tracker"
    echo -e "  • RTMPose models (~100MB)"
    echo -e "  • All required dependencies"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

    read -p "Continue with installation? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation cancelled${NC}"
        exit 1
    fi
    echo ""
fi

# Check if setup script exists
if [[ ! -f "$SETUP_SCRIPT" ]]; then
    echo -e "${RED}❌ Error: Setup script not found: $SETUP_SCRIPT${NC}"
    exit 1
fi

# Run setup script
echo -e "${BLUE}[3/6] Running installation...${NC}\n"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

bash "$SETUP_SCRIPT"

SETUP_EXIT_CODE=$?

if [[ $SETUP_EXIT_CODE -ne 0 ]]; then
    echo -e "\n${RED}❌ Installation failed (exit code: $SETUP_EXIT_CODE)${NC}"
    echo -e "${YELLOW}💡 Check the error messages above for details${NC}"
    exit 1
fi

echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Activate virtual environment for remaining steps
source venv_advanced/bin/activate

# Verify installation
echo -e "${BLUE}[4/6] Verifying installation...${NC}\n"

VERIFICATION_FAILED=false

# Test imports
python3 << 'PYEOF'
import sys

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch: NOT INSTALLED")
    sys.exit(1)

try:
    from src.utils.device_utils import DeviceManager
    device = DeviceManager.detect_device()
    print(f"✅ Device Detection: {device.upper()}")
except Exception as e:
    print(f"❌ Device Detection: FAILED ({e})")
    sys.exit(1)

try:
    from src.pose.rtmpose_estimator import RTMPoseEstimator
    print("✅ RTMPose Estimator: OK")
except ImportError as e:
    print(f"❌ RTMPose Estimator: FAILED ({e})")
    sys.exit(1)

try:
    from src.tracking.bytetrack_tracker import ByteTrackTracker
    print("✅ ByteTrack Tracker: OK")
except ImportError as e:
    print(f"❌ ByteTrack Tracker: FAILED ({e})")
    sys.exit(1)

try:
    from src.pipeline.orchestrator import SwimVisionPipeline
    print("✅ Pipeline Orchestrator: OK")
except ImportError as e:
    print(f"❌ Pipeline Orchestrator: FAILED ({e})")
    sys.exit(1)

print("\n✅ All core components verified!")
PYEOF

if [[ $? -ne 0 ]]; then
    VERIFICATION_FAILED=true
fi

echo ""

if [[ "$VERIFICATION_FAILED" == true ]]; then
    echo -e "${RED}❌ Verification failed${NC}"
    echo -e "${YELLOW}💡 Some components could not be imported${NC}"
    exit 1
fi

# Show device info
echo -e "${BLUE}[5/6] System capabilities:${NC}\n"

python3 -c "from src.utils.device_utils import DeviceManager; DeviceManager.print_device_info()"

# Create test video
echo -e "${BLUE}[6/6] Setting up demo...${NC}\n"

echo "Creating synthetic test video..."
python3 demos/demo_phase1_pipeline.py --create-test-video 2>/dev/null || echo -e "${YELLOW}⚠️  Test video creation skipped${NC}"

echo ""

# Success banner
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║                    ✅  INSTALLATION COMPLETE!                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Show next steps
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}                         QUICK START GUIDE                        ${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${GREEN}1. Activate the environment:${NC}"
echo -e "   ${BLUE}source venv_advanced/bin/activate${NC}\n"

echo -e "${GREEN}2. Run a quick test:${NC}"
if [[ -f "data/videos/test_swimmers.mp4" ]]; then
    echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video data/videos/test_swimmers.mp4${NC}\n"
else
    echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --create-test-video${NC}"
    echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video data/videos/test_swimmers.mp4${NC}\n"
fi

echo -e "${GREEN}3. Try with webcam:${NC}"
echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --webcam${NC}\n"

echo -e "${GREEN}4. Process your own video:${NC}"
echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video YOUR_VIDEO.mp4 --output results/output.mp4${NC}\n"

echo -e "${GREEN}5. Optimize for your hardware:${NC}"
if [[ "$HAS_APPLE_SILICON" == true ]]; then
    echo -e "   ${BLUE}# Your Mac - use MPS acceleration${NC}"
    echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video input.mp4 --device mps --model $RECOMMENDED_MODEL${NC}\n"
elif [[ "$HAS_NVIDIA" == true ]]; then
    echo -e "   ${BLUE}# Your GPU - use CUDA acceleration${NC}"
    echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video input.mp4 --device cuda --model $RECOMMENDED_MODEL${NC}\n"
else
    echo -e "   ${BLUE}# CPU only - use smaller model for better speed${NC}"
    echo -e "   ${BLUE}python demos/demo_phase1_pipeline.py --video input.mp4 --device cpu --model rtmpose-s${NC}\n"
fi

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${YELLOW}📚 Documentation:${NC}"
echo -e "   • Phase 1 Guide:        ${BLUE}docs/PHASE1_GUIDE.md${NC}"
if [[ "$HAS_APPLE_SILICON" == true ]]; then
    echo -e "   • Apple Silicon Guide:  ${BLUE}docs/APPLE_SILICON_GUIDE.md${NC}"
fi
echo -e "   • Architecture:         ${BLUE}docs/ADVANCED_FEATURES_ARCHITECTURE.py${NC}"
echo -e "   • API Reference:        ${BLUE}src/pipeline/orchestrator.py${NC}\n"

echo -e "${YELLOW}🔧 Configuration:${NC}"
echo -e "   • Model variants:  rtmpose-t (fastest), rtmpose-s, rtmpose-m (balanced), rtmpose-l (accurate)"
echo -e "   • Devices:         auto (recommended), cuda, mps, cpu"
echo -e "   • Modes:           realtime, balanced, accuracy"
echo -e "   • Output formats:  coco17, smpl24, opensim\n"

echo -e "${YELLOW}⚡ Performance Tips:${NC}"
if [[ "$HAS_APPLE_SILICON" == true ]]; then
    echo -e "   • Use --device mps for GPU acceleration (3-5x faster)"
    echo -e "   • Expected: $EXPECTED_FPS with $RECOMMENDED_MODEL @ 1080p"
    echo -e "   • For higher FPS, use rtmpose-s or reduce resolution to 720p"
elif [[ "$HAS_NVIDIA" == true ]]; then
    echo -e "   • Use --device cuda for GPU acceleration"
    echo -e "   • Expected: $EXPECTED_FPS with $RECOMMENDED_MODEL @ 1080p"
    echo -e "   • For maximum FPS, use rtmpose-t"
else
    echo -e "   • Use --model rtmpose-s for better CPU performance"
    echo -e "   • Expected: $EXPECTED_FPS with CPU @ 1080p"
    echo -e "   • Consider reducing resolution to 720p or 540p"
fi
echo ""

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Run demo if requested
if [[ "$RUN_DEMO" == true ]]; then
    echo -e "${GREEN}Running demo as requested...${NC}\n"

    if [[ -f "data/videos/test_swimmers.mp4" ]]; then
        python demos/demo_phase1_pipeline.py --video data/videos/test_swimmers.mp4 --device auto --model $RECOMMENDED_MODEL
    else
        echo -e "${YELLOW}Test video not found. Creating it now...${NC}"
        python demos/demo_phase1_pipeline.py --create-test-video
        python demos/demo_phase1_pipeline.py --video data/videos/test_swimmers.mp4 --device auto --model $RECOMMENDED_MODEL
    fi
fi

# Run tests if requested
if [[ "$RUN_TESTS" == true ]]; then
    echo -e "${GREEN}Running tests as requested...${NC}\n"

    if command -v pytest &> /dev/null; then
        pytest tests/test_phase1_integration.py -v
    else
        echo -e "${YELLOW}pytest not installed, installing...${NC}"
        pip install pytest
        pytest tests/test_phase1_integration.py -v
    fi
fi

# Final message
echo -e "${GREEN}🎉 You're all set! Happy swimming analysis!${NC}\n"

deactivate 2>/dev/null || true
