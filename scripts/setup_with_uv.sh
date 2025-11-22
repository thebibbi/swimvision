#!/bin/bash
# SwimVision Pro - Unified Setup with uv Package Manager
# This script replaces both setup.sh and setup_advanced_features.sh
# Uses uv for 10-100x faster package installation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_warning "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add uv to PATH for this session
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            print_error "Failed to install uv. Please install manually:"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi

        print_success "uv installed successfully"
    else
        print_success "uv is already installed ($(uv --version))"
    fi
}

# Detect platform
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            PLATFORM="macos_apple_silicon"
            print_info "Detected: macOS Apple Silicon"
        else
            PLATFORM="macos_intel"
            print_info "Detected: macOS Intel"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        PLATFORM="linux"
        print_info "Detected: Linux"
    else
        PLATFORM="unknown"
        print_warning "Unknown platform: $OSTYPE"
    fi
}

# Main setup function
main() {
    print_header "SwimVision Pro - Unified Setup with uv"

    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Python version: $python_version"

    # Check uv
    check_uv

    # Detect platform
    detect_platform

    # Ask user what features they want
    print_header "Feature Selection"
    echo "What features do you want to install?"
    echo ""
    echo "1) Basic (YOLO11, MediaPipe, Streamlit)"
    echo "2) Advanced (Basic + RTMPose, ViTPose, ByteTrack, SMPL-X)"
    echo "3) Complete (Advanced + FreeMoCap, RealSense, Export tools)"
    echo "4) All (Complete + Development tools)"
    echo ""
    read -p "Enter choice [1-4] (default: 2): " choice
    choice=${choice:-2}

    case $choice in
        1) FEATURE_SET="basic" ;;
        2) FEATURE_SET="advanced" ;;
        3) FEATURE_SET="complete" ;;
        4) FEATURE_SET="all" ;;
        *)
            print_error "Invalid choice. Using 'advanced' as default."
            FEATURE_SET="advanced"
            ;;
    esac

    print_info "Selected feature set: $FEATURE_SET"

    # Create virtual environment
    print_header "Creating Virtual Environment"
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists at .venv"
        read -p "Remove and recreate? [y/N]: " recreate
        if [[ $recreate == "y" || $recreate == "Y" ]]; then
            rm -rf .venv
            print_info "Removed existing .venv"
        else
            print_info "Using existing .venv"
        fi
    fi

    if [ ! -d ".venv" ]; then
        uv venv .venv
        print_success "Created virtual environment at .venv"
    fi

    # Activate virtual environment
    source .venv/bin/activate
    print_success "Activated virtual environment"

    # Install PyTorch first (platform-specific)
    print_header "Installing PyTorch"
    if [[ "$PLATFORM" == "macos_apple_silicon" ]]; then
        print_info "Installing PyTorch with MPS support for Apple Silicon..."
        uv pip install torch torchvision torchaudio
    elif [[ "$PLATFORM" == "linux" ]] && command -v nvidia-smi &> /dev/null; then
        print_info "Installing PyTorch with CUDA 11.8 support..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_info "Installing PyTorch (CPU version)..."
        uv pip install torch torchvision torchaudio
    fi
    print_success "PyTorch installed"

    # Install SwimVision with selected features
    print_header "Installing SwimVision [$FEATURE_SET]"
    uv pip install -e ".[$FEATURE_SET]"
    print_success "SwimVision installed with $FEATURE_SET features"

    # Install MMPose if advanced or complete
    if [[ "$FEATURE_SET" == "advanced" || "$FEATURE_SET" == "complete" || "$FEATURE_SET" == "all" ]]; then
        print_header "Installing MMPose"
        print_info "Installing MMPose via mim (this may take a few minutes)..."

        # Install mim if not already installed
        uv pip install openmim

        # Install MMPose components
        mim install mmengine
        mim install "mmcv>=2.0.0"
        mim install "mmpose>=1.2.0"

        print_success "MMPose installed"
    fi

    # Install ByteTrack if advanced or complete
    if [[ "$FEATURE_SET" == "advanced" || "$FEATURE_SET" == "complete" || "$FEATURE_SET" == "all" ]]; then
        print_header "Installing ByteTrack"

        if [ ! -d "external/ByteTrack" ]; then
            print_info "Cloning ByteTrack..."
            mkdir -p external
            git clone https://github.com/ifzhang/ByteTrack.git external/ByteTrack
        fi

        cd external/ByteTrack

        # Comment out problematic pins
        if grep -q "^onnx==" requirements.txt; then
            sed -i.bak 's/^onnx==/#onnx==/g' requirements.txt
            print_info "Commented out onnx pin in ByteTrack requirements"
        fi

        # Install without build isolation
        uv pip install -e . --no-build-isolation
        cd ../..

        print_success "ByteTrack installed"
    fi

    # Create necessary directories
    print_header "Creating Directories"
    mkdir -p data/videos data/models logs models/rtmpose models/vitpose models/alphapose models/smpl
    print_success "Created data directories"

    # Download YOLO models
    print_header "Downloading YOLO Models"
    if [ ! -f "yolo11n-pose.pt" ]; then
        print_info "Downloading yolo11n-pose.pt..."
        python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
        print_success "Downloaded yolo11n-pose.pt"
    else
        print_info "yolo11n-pose.pt already exists"
    fi

    # Run audit
    print_header "Running Estimator Audit"
    python3 scripts/audit_estimators.py

    # Print summary
    print_header "Setup Complete!"
    echo ""
    print_success "SwimVision Pro installed with $FEATURE_SET features"
    echo ""
    print_info "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
    print_info "To run the Streamlit app:"
    echo "  streamlit run app.py"
    echo ""
    print_info "To run tests:"
    echo "  pytest tests/"
    echo ""

    if [[ "$FEATURE_SET" == "basic" ]]; then
        print_warning "Note: Advanced models (RTMPose, ViTPose, SMPL-X) not installed."
        print_info "To install advanced features later:"
        echo "  source .venv/bin/activate"
        echo "  uv pip install -e '.[advanced]'"
    fi

    echo ""
    print_info "For more information, see:"
    echo "  - README.md"
    echo "  - COMPREHENSIVE_FIX_PLAN.md"
    echo "  - docs/SETUP.md"
    echo ""
}

# Run main function
main "$@"
