"""Base camera interface for SwimVision."""

from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import numpy as np


class BaseCamera(ABC):
    """Abstract base class for camera sources."""

    def __init__(self):
        """Initialize camera."""
        self._is_opened = False
        self._frame_count = 0
        self._fps = 0.0

    @abstractmethod
    def open(self) -> bool:
        """Open camera connection.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from camera.

        Returns:
            Tuple of (success, frame). Frame is None if read fails.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is opened.

        Returns:
            True if camera is opened.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Get camera FPS.

        Returns:
            Frames per second.
        """
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Get camera resolution.

        Returns:
            Tuple of (width, height).
        """
        pass

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Stream frames from camera.

        Yields:
            Frame as numpy array (BGR format).

        Example:
            >>> camera = WebcamCamera()
            >>> camera.open()
            >>> for frame in camera.stream_frames():
            >>>     # Process frame
            >>>     if some_condition:
            >>>         break
            >>> camera.release()
        """
        if not self.is_opened():
            raise RuntimeError("Camera not opened. Call open() first.")

        while True:
            success, frame = self.read()
            if not success or frame is None:
                break

            self._frame_count += 1
            yield frame

    def get_frame_count(self) -> int:
        """Get number of frames read so far.

        Returns:
            Frame count.
        """
        return self._frame_count

    def reset_frame_count(self) -> None:
        """Reset frame counter."""
        self._frame_count = 0

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
