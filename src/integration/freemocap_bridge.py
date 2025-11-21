"""
FreeMoCap Integration Bridge
=============================

Connects SwimVision's pose estimation pipeline with FreeMoCap's multi-camera
motion capture system.

FreeMoCap provides:
- Multi-camera synchronization and calibration
- 3D triangulation from multiple views
- Export to standard formats (BVH, FBX, C3D)
- GUI for camera setup and calibration

This bridge allows SwimVision to:
1. Use FreeMoCap's camera calibration for multi-camera setups
2. Export data in FreeMoCap-compatible formats
3. Leverage FreeMoCap's 3D triangulation
4. Compare single-camera 3D (temporal lifters) with multi-camera 3D

Usage:
    from src.integration.freemocap_bridge import FreeMoCapBridge

    # Create bridge
    bridge = FreeMoCapBridge(
        camera_count=4,
        calibration_path="calibration.toml"
    )

    # Process multi-camera frames
    pose_3d = bridge.triangulate_frames(frames_dict)

    # Export to FreeMoCap format
    bridge.export_to_freemocap(tracking_data, "output/")
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

# Conditional imports
try:
    import freemocap
    FREEMOCAP_AVAILABLE = True
except ImportError:
    FREEMOCAP_AVAILABLE = False
    logger.warning("FreeMoCap not available. Install with: pip install freemocap")

try:
    from aniposelib.cameras import CameraGroup
    ANIPOSE_AVAILABLE = True
except ImportError:
    ANIPOSE_AVAILABLE = False
    logger.warning("Aniposelib not available (required for triangulation)")


class FreeMoCapBridge:
    """
    Bridge between SwimVision and FreeMoCap ecosystems.

    Enables multi-camera motion capture using FreeMoCap's calibration
    and triangulation capabilities with SwimVision's pose estimators.

    Attributes:
        camera_count: Number of cameras in the setup
        calibration: Camera calibration data
        camera_group: Aniposelib camera group for triangulation
    """

    def __init__(
        self,
        camera_count: int = 1,
        calibration_path: Optional[str] = None,
        use_charuco_calibration: bool = False,
    ):
        """
        Initialize FreeMoCap bridge.

        Args:
            camera_count: Number of cameras (1 for single-camera)
            calibration_path: Path to calibration file (TOML format)
            use_charuco_calibration: Whether to use CharuCo board calibration
        """
        if not FREEMOCAP_AVAILABLE:
            logger.warning(
                "FreeMoCap not installed. Multi-camera features will be limited."
            )

        self.camera_count = camera_count
        self.calibration_path = calibration_path
        self.use_charuco_calibration = use_charuco_calibration

        # Load calibration if provided
        self.calibration = None
        self.camera_group = None

        if calibration_path and Path(calibration_path).exists():
            self.load_calibration(calibration_path)

        logger.info(
            f"FreeMoCapBridge initialized with {camera_count} cameras"
        )

    def load_calibration(self, calibration_path: str):
        """
        Load camera calibration from file.

        Args:
            calibration_path: Path to calibration file
        """
        calibration_path = Path(calibration_path)

        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

        try:
            if calibration_path.suffix == '.toml':
                self._load_toml_calibration(calibration_path)
            elif calibration_path.suffix == '.json':
                self._load_json_calibration(calibration_path)
            else:
                raise ValueError(f"Unsupported calibration format: {calibration_path.suffix}")

            logger.info(f"Loaded calibration from {calibration_path}")

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            raise

    def _load_toml_calibration(self, path: Path):
        """Load TOML calibration (FreeMoCap format)."""
        import toml

        with open(path, 'r') as f:
            self.calibration = toml.load(f)

        # Create camera group for triangulation
        if ANIPOSE_AVAILABLE:
            self._create_camera_group_from_calibration()

    def _load_json_calibration(self, path: Path):
        """Load JSON calibration."""
        with open(path, 'r') as f:
            self.calibration = json.load(f)

        # Create camera group for triangulation
        if ANIPOSE_AVAILABLE:
            self._create_camera_group_from_calibration()

    def _create_camera_group_from_calibration(self):
        """Create Aniposelib CameraGroup from calibration data."""
        if not ANIPOSE_AVAILABLE:
            logger.warning("Aniposelib not available for triangulation")
            return

        # Extract camera parameters from calibration
        # This depends on the calibration format
        try:
            # Assuming calibration has camera matrices and distortion coefficients
            camera_matrices = []
            dist_coeffs = []
            rvecs = []
            tvecs = []

            for i in range(self.camera_count):
                cam_key = f"camera_{i}"
                if cam_key in self.calibration:
                    cam_data = self.calibration[cam_key]
                    camera_matrices.append(np.array(cam_data['matrix']))
                    dist_coeffs.append(np.array(cam_data['distortion']))
                    rvecs.append(np.array(cam_data.get('rvec', [0, 0, 0])))
                    tvecs.append(np.array(cam_data.get('tvec', [0, 0, 0])))

            self.camera_group = CameraGroup(
                camera_matrices=camera_matrices,
                dist_coeffs=dist_coeffs,
                rvecs=rvecs,
                tvecs=tvecs
            )

            logger.info(f"Created camera group with {len(camera_matrices)} cameras")

        except Exception as e:
            logger.error(f"Failed to create camera group: {e}")

    def triangulate_2d_to_3d(
        self,
        points_2d_dict: Dict[int, np.ndarray],
        min_cameras: int = 2
    ) -> Optional[np.ndarray]:
        """
        Triangulate 2D points from multiple cameras to 3D.

        Args:
            points_2d_dict: Dictionary mapping camera_id -> 2D points (Nx2)
            min_cameras: Minimum number of cameras required for triangulation

        Returns:
            3D points (Nx3) or None if triangulation fails
        """
        if not ANIPOSE_AVAILABLE:
            logger.error("Aniposelib required for triangulation")
            return None

        if self.camera_group is None:
            logger.error("No camera calibration loaded")
            return None

        if len(points_2d_dict) < min_cameras:
            logger.warning(f"Insufficient cameras ({len(points_2d_dict)} < {min_cameras})")
            return None

        try:
            # Convert dict to list ordered by camera index
            points_2d_list = []
            for cam_id in sorted(points_2d_dict.keys()):
                points_2d_list.append(points_2d_dict[cam_id])

            # Stack points for triangulation (shape: n_cameras x n_points x 2)
            points_2d_array = np.array(points_2d_list)

            # Triangulate
            points_3d = self.camera_group.triangulate(
                points_2d_array,
                progress=False
            )

            return points_3d

        except Exception as e:
            logger.error(f"Triangulation failed: {e}")
            return None

    def triangulate_frames(
        self,
        frames_2d: Dict[int, Dict[str, Any]],
        keypoint_indices: Optional[List[int]] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Triangulate pose data from multiple camera views.

        Args:
            frames_2d: Dict mapping camera_id -> pose_data for current frame
            keypoint_indices: Optional list of keypoint indices to triangulate

        Returns:
            Dictionary with:
            - 'points_3d': 3D joint positions (Nx3)
            - 'keypoint_names': List of keypoint names
            - 'confidence': Per-point confidence scores
        """
        # Extract 2D points from each camera
        points_2d_dict = {}
        confidences = {}
        keypoint_names = None

        for cam_id, pose_data in frames_2d.items():
            if pose_data is None:
                continue

            keypoints = pose_data['keypoints']  # Nx3 (x, y, conf)

            if keypoint_indices is not None:
                keypoints = keypoints[keypoint_indices]

            points_2d_dict[cam_id] = keypoints[:, :2]  # Extract x, y
            confidences[cam_id] = keypoints[:, 2]  # Extract confidence

            if keypoint_names is None:
                if keypoint_indices is not None:
                    keypoint_names = [pose_data['keypoint_names'][i] for i in keypoint_indices]
                else:
                    keypoint_names = pose_data['keypoint_names']

        # Triangulate
        points_3d = self.triangulate_2d_to_3d(points_2d_dict)

        if points_3d is None:
            return None

        # Average confidence across cameras
        conf_array = np.array([confidences[cam_id] for cam_id in sorted(confidences.keys())])
        avg_confidence = np.mean(conf_array, axis=0)

        return {
            'points_3d': points_3d,
            'keypoint_names': keypoint_names,
            'confidence': avg_confidence,
            'n_cameras': len(points_2d_dict)
        }

    def export_to_freemocap(
        self,
        tracking_data: List[Dict[str, Any]],
        output_dir: str,
        session_name: str = "swimvision_session"
    ):
        """
        Export tracking data in FreeMoCap-compatible format.

        Args:
            tracking_data: List of per-frame tracking results
            output_dir: Output directory path
            session_name: Name for this recording session
        """
        output_path = Path(output_dir) / session_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save 2D data
        self._save_2d_data(tracking_data, output_path)

        # Save 3D data if available
        self._save_3d_data(tracking_data, output_path)

        # Save metadata
        self._save_metadata(tracking_data, output_path)

        logger.info(f"Exported data to FreeMoCap format: {output_path}")

    def _save_2d_data(self, tracking_data: List[Dict], output_path: Path):
        """Save 2D tracking data."""
        # Collect 2D points per frame
        frames_2d = []
        for frame_data in tracking_data:
            if frame_data is None:
                frames_2d.append(None)
                continue

            frames_2d.append({
                'keypoints': frame_data['keypoints'].tolist(),
                'keypoint_names': frame_data['keypoint_names'],
                'confidence': frame_data.get('confidence', 0.0),
            })

        # Save as NPZ
        output_file = output_path / "output_data_2d.npz"
        np.savez(
            output_file,
            data_2d=np.array([f['keypoints'] if f else None for f in frames_2d])
        )

        logger.info(f"Saved 2D data: {output_file}")

    def _save_3d_data(self, tracking_data: List[Dict], output_path: Path):
        """Save 3D tracking data."""
        # Collect 3D points per frame
        frames_3d = []
        for frame_data in tracking_data:
            if frame_data is None or '3d_data' not in frame_data:
                frames_3d.append(None)
                continue

            frames_3d.append(frame_data['3d_data'])

        if all(f is None for f in frames_3d):
            logger.info("No 3D data to save")
            return

        # Save as NPZ
        output_file = output_path / "output_data_3d.npz"
        np.savez(
            output_file,
            data_3d=np.array(frames_3d)
        )

        logger.info(f"Saved 3D data: {output_file}")

    def _save_metadata(self, tracking_data: List[Dict], output_path: Path):
        """Save session metadata."""
        metadata = {
            'session_name': output_path.name,
            'n_frames': len(tracking_data),
            'n_cameras': self.camera_count,
            'has_calibration': self.calibration is not None,
            'keypoint_format': tracking_data[0]['keypoint_names'] if tracking_data else None,
        }

        output_file = output_path / "metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata: {output_file}")

    def export_to_bvh(
        self,
        tracking_data: List[Dict[str, Any]],
        output_path: str,
        skeleton_type: str = "mixamo"
    ):
        """
        Export tracking data to BVH format for animation software.

        Args:
            tracking_data: List of per-frame tracking results
            output_path: Output BVH file path
            skeleton_type: Skeleton hierarchy type
        """
        # TODO: Implement BVH export
        logger.warning("BVH export not yet implemented")

    def export_to_c3d(
        self,
        tracking_data: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Export tracking data to C3D format for biomechanics analysis.

        Args:
            tracking_data: List of per-frame tracking results
            output_path: Output C3D file path
        """
        # TODO: Implement C3D export
        logger.warning("C3D export not yet implemented")

    def close(self):
        """Cleanup resources."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def create_multicamera_setup(
    camera_count: int,
    calibration_path: Optional[str] = None
) -> FreeMoCapBridge:
    """
    Convenience function to create a multi-camera setup.

    Args:
        camera_count: Number of cameras
        calibration_path: Path to calibration file

    Returns:
        Initialized FreeMoCapBridge

    Example:
        >>> bridge = create_multicamera_setup(4, "pool_calib.toml")
        >>> # Process frames from 4 cameras
        >>> pose_3d = bridge.triangulate_frames(frames_dict)
    """
    return FreeMoCapBridge(
        camera_count=camera_count,
        calibration_path=calibration_path
    )
