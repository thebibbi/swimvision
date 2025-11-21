#!/bin/bash
# Download MotionAGFormer pre-trained weights for all variants

set -e

CHECKPOINT_DIR="/home/user/swimvision/checkpoint/motionagformer"
mkdir -p "$CHECKPOINT_DIR"

echo "Downloading MotionAGFormer pre-trained weights..."

# MotionAGFormer-XS (H3.6M) - 27 frames
echo "Downloading MotionAGFormer-XS..."
gdown "1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH" -O "$CHECKPOINT_DIR/motionagformer-xs-h36m.pth.tr"

# MotionAGFormer-S (H3.6M) - 81 frames
echo "Downloading MotionAGFormer-S..."
gdown "1DrF7WZdDvRPsH12gQm5DPXbviZ4waYFf" -O "$CHECKPOINT_DIR/motionagformer-s-h36m.pth.tr"

# MotionAGFormer-B (H3.6M) - 243 frames
echo "Downloading MotionAGFormer-B..."
gdown "1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP" -O "$CHECKPOINT_DIR/motionagformer-b-h36m.pth.tr"

# MotionAGFormer-L (H3.6M) - 243 frames
echo "Downloading MotionAGFormer-L..."
gdown "1WI8QSsD84wlXIdK1dLp6hPZq4FPozmVZ" -O "$CHECKPOINT_DIR/motionagformer-l-h36m.pth.tr"

echo "All MotionAGFormer weights downloaded to $CHECKPOINT_DIR"
ls -lh "$CHECKPOINT_DIR"
