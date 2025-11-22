"""Pose estimation module."""

from src.pose.base_estimator import BasePoseEstimator, KeypointFormat, PoseModel
from src.pose.yolo_estimator import YOLOPoseEstimator

# Optional imports with graceful fallback
try:
    from src.pose.mediapipe_estimator import MediaPipeEstimator
except ImportError:
    MediaPipeEstimator = None

try:
    from src.pose.rtmpose_estimator import RTMPoseEstimator
except ImportError:
    RTMPoseEstimator = None

try:
    from src.pose.vitpose_estimator import ViTPoseEstimator
except ImportError:
    ViTPoseEstimator = None

try:
    from src.pose.alphapose_estimator import AlphaPoseEstimator
except ImportError:
    AlphaPoseEstimator = None

try:
    from src.pose.openpose_estimator import OpenPoseEstimator
except ImportError:
    OpenPoseEstimator = None

try:
    from src.pose.smpl_estimator import SMPLEstimator
except ImportError:
    SMPLEstimator = None

__all__ = [
    "BasePoseEstimator",
    "KeypointFormat",
    "PoseModel",
    "YOLOPoseEstimator",
    "MediaPipeEstimator",
    "RTMPoseEstimator",
    "ViTPoseEstimator",
    "AlphaPoseEstimator",
    "OpenPoseEstimator",
    "SMPLEstimator",
]
