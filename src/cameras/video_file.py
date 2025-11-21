"""Video file camera implementation."""

from pathlib import Path

import cv2
import numpy as np

from src.cameras.base_camera import BaseCamera


class VideoFileCamera(BaseCamera):
    """Video file reader using OpenCV."""

    def __init__(self, video_path: str, start_frame: int = 0, loop: bool = False):
        """Initialize video file camera.

        Args:
            video_path: Path to video file.
            start_frame: Frame number to start from (0-indexed).
            loop: Whether to loop video playback.
        """
        super().__init__()

        self.video_path = Path(video_path)
        self.start_frame = start_frame
        self.loop = loop

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap: cv2.VideoCapture | None = None
        self._total_frames = 0
        self._current_frame = 0

    def open(self) -> bool:
        """Open video file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self._cap = cv2.VideoCapture(str(self.video_path))

            if not self._cap.isOpened():
                return False

            # Get video properties
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (width, height)

            # Seek to start frame
            if self.start_frame > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self._current_frame = self.start_frame

            self._is_opened = True
            return True

        except Exception as e:
            print(f"Error opening video file: {e}")
            return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame from video file.

        Returns:
            Tuple of (success, frame).
        """
        if not self.is_opened() or self._cap is None:
            return False, None

        success, frame = self._cap.read()

        if success:
            self._current_frame += 1
            return True, frame

        # Handle end of video
        if self.loop and self._total_frames > 0:
            # Reset to start
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._current_frame = self.start_frame
            success, frame = self._cap.read()
            if success:
                self._current_frame += 1
            return success, frame if success else None

        return False, None

    def release(self) -> None:
        """Release video file resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def is_opened(self) -> bool:
        """Check if video file is opened.

        Returns:
            True if video file is opened.
        """
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def get_fps(self) -> float:
        """Get video FPS.

        Returns:
            Frames per second.
        """
        return self._fps

    def get_resolution(self) -> tuple[int, int]:
        """Get video resolution.

        Returns:
            Tuple of (width, height).
        """
        return self._resolution

    def get_total_frames(self) -> int:
        """Get total number of frames in video.

        Returns:
            Total frame count.
        """
        return self._total_frames

    def get_current_frame_number(self) -> int:
        """Get current frame number.

        Returns:
            Current frame number (0-indexed).
        """
        return self._current_frame

    def get_duration_seconds(self) -> float:
        """Get video duration in seconds.

        Returns:
            Duration in seconds.
        """
        if self._fps > 0:
            return self._total_frames / self._fps
        return 0.0

    def seek(self, frame_number: int) -> bool:
        """Seek to specific frame.

        Args:
            frame_number: Frame number to seek to (0-indexed).

        Returns:
            True if successful.
        """
        if not self.is_opened() or self._cap is None:
            return False

        if 0 <= frame_number < self._total_frames:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._current_frame = frame_number
            return True

        return False

    def get_progress(self) -> float:
        """Get playback progress as percentage.

        Returns:
            Progress percentage (0-100).
        """
        if self._total_frames > 0:
            return (self._current_frame / self._total_frames) * 100
        return 0.0
