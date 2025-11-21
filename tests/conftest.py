"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame for testing.

    Returns:
        Numpy array representing a 640x480 BGR image.
    """
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_pose_data():
    """Create sample pose data for testing.

    Returns:
        Dictionary with mock pose data.
    """
    return {
        "keypoints": {
            "nose": {"x": 320.0, "y": 100.0, "confidence": 0.9},
            "left_shoulder": {"x": 280.0, "y": 150.0, "confidence": 0.85},
            "right_shoulder": {"x": 360.0, "y": 150.0, "confidence": 0.85},
            "left_elbow": {"x": 260.0, "y": 200.0, "confidence": 0.8},
            "right_elbow": {"x": 380.0, "y": 200.0, "confidence": 0.8},
            "left_wrist": {"x": 240.0, "y": 250.0, "confidence": 0.75},
            "right_wrist": {"x": 400.0, "y": 250.0, "confidence": 0.75},
            "left_hip": {"x": 290.0, "y": 300.0, "confidence": 0.9},
            "right_hip": {"x": 350.0, "y": 300.0, "confidence": 0.9},
            "left_knee": {"x": 285.0, "y": 380.0, "confidence": 0.85},
            "right_knee": {"x": 355.0, "y": 380.0, "confidence": 0.85},
            "left_ankle": {"x": 280.0, "y": 450.0, "confidence": 0.8},
            "right_ankle": {"x": 360.0, "y": 450.0, "confidence": 0.8},
        },
        "bbox": {
            "x1": 200.0,
            "y1": 80.0,
            "x2": 440.0,
            "y2": 470.0,
            "confidence": 0.92,
        },
        "confidence": 0.92,
    }


@pytest.fixture
def sample_trajectory():
    """Create sample trajectory for testing.

    Returns:
        List of (x, y) tuples representing a hand path.
    """
    return [
        (100, 200),
        (110, 195),
        (120, 190),
        (130, 188),
        (140, 190),
        (150, 195),
        (160, 200),
    ]
