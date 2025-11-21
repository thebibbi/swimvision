#!/bin/bash
# Download PoseFormerV2 pre-trained weights for various configurations

set -e

CHECKPOINT_DIR="/home/user/swimvision/checkpoint/poseformerv2"
mkdir -p "$CHECKPOINT_DIR"

echo "Downloading PoseFormerV2 pre-trained weights..."

# Best model: 27-243-45.2 (243 frames, 27 frames kept, 45.2mm MPJPE)
echo "Downloading best model (27-243-45.2)..."
gdown "14SpqPyq9yiblCzTH5CorymKCUsXapmkg" -O "$CHECKPOINT_DIR/27_243_45.2.bin"

# Balanced: 3-81-47.1 (81 frames, 3 frames kept)
echo "Downloading balanced model (3-81-47.1)..."
gdown "13rXCkYnVnkbT-cz4XCo0QkUnUEYiSeoi" -O "$CHECKPOINT_DIR/3_81_47.1.bin"

# Real-time: 3-27-47.9 (27 frames, 3 frames kept)
echo "Downloading real-time model (3-27-47.9)..."
gdown "13oJz5-aBVvvPVFvTU_PrLG_m6kdbQkYs" -O "$CHECKPOINT_DIR/3_27_47.9.bin"

# Alternative models (optional)
echo "Downloading additional variants..."

# 1-27-48.7
gdown "14J0GYIzk_rGKSMxAPI2ydzX76QB70-g3" -O "$CHECKPOINT_DIR/1_27_48.7.bin"

# 1-81-47.6
gdown "14WgFFBsP0DtTq61XZWI9X2TzvFLCWEnd" -O "$CHECKPOINT_DIR/1_81_47.6.bin"

# 9-81-46.0
gdown "13wla4b5RgJGKX5zVehv4qKhCrQEFhfzG" -O "$CHECKPOINT_DIR/9_81_46.0.bin"

echo "All PoseFormerV2 weights downloaded to $CHECKPOINT_DIR"
ls -lh "$CHECKPOINT_DIR"
