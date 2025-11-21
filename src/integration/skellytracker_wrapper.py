"""
SkellyTracker Integration Wrapper
===================================

Wraps our existing pose estimators (RTMPose, MediaPipe, YOLO) to be compatible
with SkellyTracker's unified API.

SkellyTracker provides a consistent interface for multiple pose estimation models,
making it easy to switch between different trackers and integrate with FreeMoCap.

Usage:
    from src.integration.skellytracker_wrapper import SwimVisionTracker

    # Create tracker with RTMPose backend
    tracker = SwimVisionTracker(backend="rtmpose", model="m")

    # Process frame
    results = tracker.track_frame(frame)

    # Get 3D data if available
    data_3d = tracker.get_3d_data()
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from skellytracker import BaseSkellyTracker
    SKELLYTRACKER_AVAILABLE = True
except ImportError:
    SKELLYTRACKER_AVAILABLE = False
    logger.warning("SkellyTracker not available. Install with: pip install skellytracker")
    # Create dummy base class for development
    class BaseSkellyTracker:
        pass


class TrackerBackend(str, Enum):
    """Supported tracker backends."""
    RTMPOSE = "rtmpose"
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"


class SwimVisionTracker(BaseSkellyTracker):
    """
    SwimVision tracker compatible with SkellyTracker API.

    Wraps our pose estimators to provide consistent interface with
    other SkellyTracker-compatible trackers.

    Attributes:
        backend: Which pose estimator to use (rtmpose, mediapipe, yolo)
        model_variant: Model variant for the backend
        estimator: The underlying pose estimator instance
    """

    def __init__(
        self,
        backend: str = "rtmpose",
        model_variant: str = "m",
        device: str = "auto",
        min_confidence: float = 0.3,
        **kwargs
    ):
        """
        Initialize SwimVision tracker.

        Args:
            backend: Pose estimator backend ('rtmpose', 'mediapipe', or 'yolo')
            model_variant: Model variant (e.g., 't', 's', 'm', 'l' for RTMPose)
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
            min_confidence: Minimum confidence threshold for detections
            **kwargs: Additional arguments passed to underlying estimator
        """
        super().__init__()

        self.backend = TrackerBackend(backend.lower())
        self.model_variant = model_variant
        self.device = device
        self.min_confidence = min_confidence

        # Initialize the appropriate estimator
        self.estimator = self._create_estimator(**kwargs)

        logger.info(f"SwimVisionTracker initialized with {self.backend} backend")

    def _create_estimator(self, **kwargs):
        """Create the appropriate pose estimator based on backend."""
        if self.backend == TrackerBackend.RTMPOSE:
            from src.pose.rtmpose_estimator import RTMPoseEstimator
            return RTMPoseEstimator(
                model_size=self.model_variant,
                device=self.device,
                min_detection_confidence=self.min_confidence,
                **kwargs
            )

        elif self.backend == TrackerBackend.MEDIAPIPE:
            from src.pose.mediapipe_estimator import MediaPipeEstimator
            # Map variant to complexity (0=lite, 1=full, 2=heavy)
            complexity_map = {'lite': 0, 'full': 1, 'heavy': 2, 't': 0, 'm': 1, 'l': 2}
            complexity = complexity_map.get(self.model_variant.lower(), 1)
            return MediaPipeEstimator(
                model_complexity=complexity,
                min_detection_confidence=self.min_confidence,
                device=self.device,
                **kwargs
            )

        elif self.backend == TrackerBackend.YOLO:
            from src.pose.yolo_estimator import YOLOEstimator
            return YOLOEstimator(
                model_size=self.model_variant,
                device=self.device,
                min_detection_confidence=self.min_confidence,
                **kwargs
            )

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def track_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Track pose in a single frame.

        Args:
            frame: Input image (BGR format, HxWx3)

        Returns:
            Dictionary with tracking results in SkellyTracker format:
            {
                'keypoints': np.ndarray,  # Nx3 (x, y, confidence)
                'keypoint_names': List[str],
                'bbox': List[float],  # [x1, y1, x2, y2]
                'tracking_id': int,
                'frame_number': int,
                '3d_data': Optional[np.ndarray]  # If available
            }
        """
        try:
            # Run pose estimation
            pose_data, _ = self.estimator.estimate_pose(frame, return_image=False)

            if pose_data is None:
                return None

            # Convert to SkellyTracker format
            result = {
                'keypoints': pose_data['keypoints'],
                'keypoint_names': pose_data['keypoint_names'],
                'bbox': pose_data['bbox'],
                'tracking_id': pose_data.get('person_id', 0),
                'confidence': np.mean(pose_data['keypoints'][:, 2]),  # Average confidence
            }

            # Add 3D data if available (e.g., from MediaPipe)
            if '3d_data' in pose_data.get('metadata', {}):
                result['3d_data'] = pose_data['metadata']['3d_data']
            elif 'world_landmarks' in pose_data.get('metadata', {}):
                result['3d_data'] = pose_data['metadata']['world_landmarks']

            return result

        except Exception as e:
            logger.error(f"Error tracking frame: {e}")
            return None

    def track_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Track poses in a video file.

        Args:
            video_path: Path to input video
            output_path: Optional path to save results
            show_progress: Whether to show progress bar

        Returns:
            List of tracking results for each frame
        """
        import cv2
        from tqdm import tqdm

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []

        pbar = tqdm(total=total_frames, desc="Tracking", disable=not show_progress)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.track_frame(frame)
            if result is not None:
                result['frame_number'] = frame_idx

            results.append(result)
            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        # Save results if requested
        if output_path:
            self._save_results(results, output_path)

        return results

    def get_3d_data(self) -> Optional[np.ndarray]:
        """
        Get 3D landmark data if supported by the backend.

        Returns:
            3D landmark positions (Nx3) or None if not available
        """
        if self.backend == TrackerBackend.MEDIAPIPE:
            # MediaPipe supports 3D world landmarks
            if hasattr(self.estimator, 'get_3d_landmarks'):
                return self.estimator.get_3d_landmarks()

        return None

    def get_keypoint_format(self) -> str:
        """Get the keypoint format used by this tracker."""
        return self.estimator.get_keypoint_format().value

    def supports_3d(self) -> bool:
        """Check if this tracker supports 3D output."""
        if hasattr(self.estimator, 'supports_3d'):
            return self.estimator.supports_3d()
        return False

    def supports_multi_person(self) -> bool:
        """Check if this tracker supports multiple people."""
        if hasattr(self.estimator, 'supports_multi_person'):
            return self.estimator.supports_multi_person()
        return False

    def _save_results(self, results: List[Dict], output_path: str):
        """Save tracking results to file."""
        import json

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            if result is None:
                serializable_results.append(None)
                continue

            serializable = {
                'frame_number': result.get('frame_number', 0),
                'tracking_id': result.get('tracking_id', 0),
                'confidence': float(result.get('confidence', 0)),
                'keypoints': result['keypoints'].tolist() if result.get('keypoints') is not None else None,
                'keypoint_names': result.get('keypoint_names', []),
                'bbox': result.get('bbox', []),
            }

            if '3d_data' in result:
                serializable['3d_data'] = result['3d_data'].tolist()

            serializable_results.append(serializable)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved tracking results to {output_path}")

    def close(self):
        """Cleanup resources."""
        if hasattr(self.estimator, 'close'):
            self.estimator.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def create_tracker(
    backend: str = "rtmpose",
    **kwargs
) -> SwimVisionTracker:
    """
    Convenience function to create a SwimVision tracker.

    Args:
        backend: Pose estimator backend
        **kwargs: Additional arguments passed to tracker

    Returns:
        Initialized SwimVisionTracker instance

    Example:
        >>> tracker = create_tracker("rtmpose", model_variant="m")
        >>> results = tracker.track_video("swimming.mp4")
    """
    return SwimVisionTracker(backend=backend, **kwargs)
