#!/bin/bash
# SwimVision Pro - Development Environment Setup Script

set -e  # Exit on error

echo "========================================="
echo "SwimVision Pro - Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt -r requirements-dev.txt
echo "✅ Dependencies installed"
echo ""

# Install package in editable mode
echo "Installing SwimVision in editable mode..."
pip install -e .
echo "✅ Package installed"
echo ""

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install
echo "✅ Pre-commit hooks installed"
echo ""

# Create .env file
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created from .env.example"
    echo "⚠️  Please update .env with your settings"
else
    echo "⚠️  .env file already exists"
fi
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/videos data/exports models/pose_models models/ideal_techniques logs
echo "✅ Directories created"
echo ""

# Download YOLO models (optional)
read -p "Download YOLO11 pose models now? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading YOLO11 models..."
    python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt'); YOLO('yolo11s-pose.pt')"
    echo "✅ Models downloaded"
else
    echo "⏭️  Skipping model download"
fi
echo ""

echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Update .env with your settings"
echo "  3. Run tests: make test"
echo "  4. Start the app: make run"
echo "  5. Or use Docker: make docker-up"
echo ""
echo "For more commands, run: make help"
echo ""
