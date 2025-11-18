"""Webcam camera implementation."""

from typing import Optional, Tuple

import cv2
import numpy as np

from src.cameras.base_camera import BaseCamera
from src.utils.config import load_camera_config


class WebcamCamera(BaseCamera):
    """Webcam camera using OpenCV."""

    def __init__(
        self,
        camera_id: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ):
        """Initialize webcam camera.

        Args:
            camera_id: Camera device ID (default from config).
            width: Frame width (default from config).
            height: Frame height (default from config).
            fps: Target FPS (default from config).
        """
        super().__init__()

        # Load configuration
        config = load_camera_config()
        webcam_config = config.get("webcam", {})

        self.camera_id = camera_id if camera_id is not None else webcam_config.get("camera_id", 0)
        self.width = width if width is not None else webcam_config.get("width", 1280)
        self.height = height if height is not None else webcam_config.get("height", 720)
        self.target_fps = fps if fps is not None else webcam_config.get("fps", 30)

        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open webcam connection.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self._cap = cv2.VideoCapture(self.camera_id)

            if not self._cap.isOpened():
                return False

            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

            # Verify settings
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)

            # Update dimensions if different
            if actual_width != self.width or actual_height != self.height:
                self.width = actual_width
                self.height = actual_height

            self._is_opened = True
            return True

        except Exception as e:
            print(f"Error opening webcam: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from webcam.

        Returns:
            Tuple of (success, frame).
        """
        if not self.is_opened() or self._cap is None:
            return False, None

        success, frame = self._cap.read()
        return success, frame if success else None

    def release(self) -> None:
        """Release webcam resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        """Check if webcam is opened.

        Returns:
            True if webcam is opened.
        """
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def get_fps(self) -> float:
        """Get webcam FPS.

        Returns:
            Frames per second.
        """
        return self._fps

    def get_resolution(self) -> Tuple[int, int]:
        """Get webcam resolution.

        Returns:
            Tuple of (width, height).
        """
        return (self.width, self.height)
