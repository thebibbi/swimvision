"""Multi-camera fusion framework for 3D pose estimation.

Combines pose estimates from multiple synchronized cameras to:
- Improve accuracy through triangulation
- Handle occlusions (underwater/above water views)
- Reconstruct full 3D body pose
- Validate predictions using epipolar geometry
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.pose.base_estimator import KeypointFormat


@dataclass
class CameraParameters:
    """Camera calibration parameters."""

    camera_id: str  # Unique camera identifier
    intrinsic_matrix: np.ndarray  # 3x3 camera intrinsic matrix (K)
    distortion_coeffs: np.ndarray  # Distortion coefficients
    rotation_matrix: np.ndarray  # 3x3 rotation matrix (world to camera)
    translation_vector: np.ndarray  # 3x1 translation vector
    resolution: tuple[int, int]  # (width, height)
    position: str = "overhead"  # Camera position description


@dataclass
class MultiCameraDetection:
    """Detection from a single camera."""

    camera_id: str  # Source camera ID
    keypoints_2d: np.ndarray  # 2D keypoints (Nx3)
    keypoint_names: list[str]  # Keypoint names
    confidence: float  # Overall detection confidence
    timestamp: float  # Detection timestamp
    metadata: dict  # Additional metadata


@dataclass
class Fused3DPose:
    """Fused 3D pose from multiple cameras."""

    keypoints_3d: np.ndarray  # 3D keypoints (Nx4: x, y, z, confidence)
    keypoint_names: list[str]  # Keypoint names
    contributing_cameras: list[str]  # Cameras that contributed
    reprojection_errors: dict[str, float]  # Reprojection error per camera
    triangulation_quality: np.ndarray  # Quality score per keypoint
    format: KeypointFormat  # Keypoint format
    metadata: dict  # Additional metadata


class MultiCameraFusion:
    """Fuse pose estimates from multiple calibrated cameras."""

    def __init__(
        self,
        cameras: list[CameraParameters],
        min_cameras: int = 2,
        max_reprojection_error: float = 10.0,
        confidence_threshold: float = 0.3,
    ):
        """Initialize multi-camera fusion.

        Args:
            cameras: List of calibrated camera parameters.
            min_cameras: Minimum cameras required for triangulation.
            max_reprojection_error: Maximum allowed reprojection error (pixels).
            confidence_threshold: Minimum confidence for 2D keypoints.
        """
        self.cameras = {cam.camera_id: cam for cam in cameras}
        self.min_cameras = min_cameras
        self.max_reprojection_error = max_reprojection_error
        self.confidence_threshold = confidence_threshold

        # Precompute projection matrices
        self.projection_matrices = {}
        for cam_id, cam in self.cameras.items():
            self.projection_matrices[cam_id] = self._compute_projection_matrix(cam)

    def fuse_detections(
        self,
        detections: list[MultiCameraDetection],
    ) -> Fused3DPose | None:
        """Fuse 2D detections from multiple cameras into 3D pose.

        Args:
            detections: List of 2D detections from different cameras.

        Returns:
            Fused 3D pose or None if fusion failed.
        """
        if len(detections) < self.min_cameras:
            return None

        # Group detections by keypoint
        num_keypoints = len(detections[0].keypoints_2d)
        keypoint_names = detections[0].keypoint_names

        keypoints_3d = np.zeros((num_keypoints, 4))  # x, y, z, confidence
        triangulation_quality = np.zeros(num_keypoints)
        reprojection_errors = {det.camera_id: [] for det in detections}

        # Triangulate each keypoint
        for kp_idx in range(num_keypoints):
            # Collect 2D observations for this keypoint
            observations = []
            camera_ids = []

            for detection in detections:
                if kp_idx < len(detection.keypoints_2d):
                    kp_2d = detection.keypoints_2d[kp_idx]

                    if kp_2d[2] > self.confidence_threshold:
                        observations.append(kp_2d[:2])
                        camera_ids.append(detection.camera_id)

            # Triangulate if we have enough observations
            if len(observations) >= self.min_cameras:
                point_3d, quality, reproj_errors = self._triangulate_point(
                    observations,
                    camera_ids,
                )

                if point_3d is not None:
                    # Calculate confidence from quality and 2D confidences
                    avg_2d_conf = np.mean(
                        [
                            detection.keypoints_2d[kp_idx][2]
                            for detection in detections
                            if kp_idx < len(detection.keypoints_2d)
                            and detection.keypoints_2d[kp_idx][2] > self.confidence_threshold
                        ]
                    )

                    confidence = quality * avg_2d_conf

                    keypoints_3d[kp_idx] = [point_3d[0], point_3d[1], point_3d[2], confidence]
                    triangulation_quality[kp_idx] = quality

                    # Store reprojection errors
                    for cam_id, error in zip(camera_ids, reproj_errors, strict=False):
                        reprojection_errors[cam_id].append(error)

        # Calculate average reprojection errors
        avg_reproj_errors = {
            cam_id: np.mean(errors) if len(errors) > 0 else 0.0
            for cam_id, errors in reprojection_errors.items()
        }

        return Fused3DPose(
            keypoints_3d=keypoints_3d,
            keypoint_names=keypoint_names,
            contributing_cameras=[det.camera_id for det in detections],
            reprojection_errors=avg_reproj_errors,
            triangulation_quality=triangulation_quality,
            format=KeypointFormat.COCO_17,  # Assuming COCO format
            metadata={
                "num_cameras": len(detections),
                "min_cameras_used": self.min_cameras,
                "avg_quality": float(np.mean(triangulation_quality)),
            },
        )

    def _triangulate_point(
        self,
        observations: list[np.ndarray],
        camera_ids: list[str],
    ) -> tuple[np.ndarray | None, float, list[float]]:
        """Triangulate 3D point from multiple 2D observations.

        Args:
            observations: List of 2D points (x, y) from different cameras.
            camera_ids: Corresponding camera IDs.

        Returns:
            Tuple of (3D point, quality score, reprojection errors).
        """
        if len(observations) < 2:
            return None, 0.0, []

        # Build linear system for triangulation (DLT)
        A = []
        projection_matrices = []

        for obs, cam_id in zip(observations, camera_ids, strict=False):
            P = self.projection_matrices[cam_id]
            projection_matrices.append(P)

            x, y = obs

            # Add constraints from this observation
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])

        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        point_3d_homogeneous = Vt[-1, :]

        # Convert from homogeneous coordinates
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

        # Calculate reprojection errors
        reproj_errors = []
        for obs, P in zip(observations, projection_matrices, strict=False):
            # Project 3D point back to 2D
            point_2d_proj = self._project_point(point_3d, P)

            # Calculate error
            error = np.linalg.norm(obs - point_2d_proj)
            reproj_errors.append(float(error))

        # Calculate quality score (inverse of average reprojection error)
        avg_error = np.mean(reproj_errors)
        if avg_error > self.max_reprojection_error:
            return None, 0.0, reproj_errors

        quality = 1.0 / (1.0 + avg_error)

        return point_3d, quality, reproj_errors

    def _project_point(
        self,
        point_3d: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> np.ndarray:
        """Project 3D point to 2D using projection matrix.

        Args:
            point_3d: 3D point (x, y, z).
            projection_matrix: 3x4 projection matrix.

        Returns:
            2D point (x, y).
        """
        point_3d_hom = np.append(point_3d, 1.0)
        point_2d_hom = projection_matrix @ point_3d_hom

        # Normalize by w
        if point_2d_hom[2] != 0:
            point_2d = point_2d_hom[:2] / point_2d_hom[2]
        else:
            point_2d = point_2d_hom[:2]

        return point_2d

    def _compute_projection_matrix(
        self,
        camera: CameraParameters,
    ) -> np.ndarray:
        """Compute 3x4 projection matrix from camera parameters.

        Args:
            camera: Camera parameters.

        Returns:
            3x4 projection matrix P = K [R | t].
        """
        # Extrinsic matrix [R | t]
        extrinsic = np.hstack([camera.rotation_matrix, camera.translation_vector.reshape(3, 1)])

        # Projection matrix
        P = camera.intrinsic_matrix @ extrinsic

        return P

    def validate_with_epipolar_geometry(
        self,
        detection1: MultiCameraDetection,
        detection2: MultiCameraDetection,
    ) -> dict[str, float]:
        """Validate detections using epipolar geometry.

        Args:
            detection1: Detection from camera 1.
            detection2: Detection from camera 2.

        Returns:
            Dictionary with validation scores per keypoint.
        """
        cam1 = self.cameras[detection1.camera_id]
        cam2 = self.cameras[detection2.camera_id]

        # Compute fundamental matrix
        F = self._compute_fundamental_matrix(cam1, cam2)

        validation_scores = {}

        # Check each keypoint pair
        for kp_idx, kp_name in enumerate(detection1.keypoint_names):
            if kp_idx >= len(detection2.keypoints_2d):
                continue

            kp1 = detection1.keypoints_2d[kp_idx]
            kp2 = detection2.keypoints_2d[kp_idx]

            if kp1[2] > self.confidence_threshold and kp2[2] > self.confidence_threshold:
                # Calculate epipolar constraint error
                # x2^T F x1 should be close to 0
                x1 = np.array([kp1[0], kp1[1], 1.0])
                x2 = np.array([kp2[0], kp2[1], 1.0])

                error = abs(x2.T @ F @ x1)

                # Convert to score (0-1, higher is better)
                score = np.exp(-error / 10.0)  # Decay with error
                validation_scores[kp_name] = float(score)

        return validation_scores

    def _compute_fundamental_matrix(
        self,
        cam1: CameraParameters,
        cam2: CameraParameters,
    ) -> np.ndarray:
        """Compute fundamental matrix between two cameras.

        Args:
            cam1: First camera parameters.
            cam2: Second camera parameters.

        Returns:
            3x3 fundamental matrix.
        """
        # Relative pose between cameras
        R_rel = cam2.rotation_matrix @ cam1.rotation_matrix.T
        t_rel = cam2.translation_vector - R_rel @ cam1.translation_vector

        # Essential matrix E = [t]_x R
        t_skew = self._skew_symmetric(t_rel.flatten())
        E = t_skew @ R_rel

        # Fundamental matrix F = K2^-T E K1^-1
        K1_inv = np.linalg.inv(cam1.intrinsic_matrix)
        K2_inv_T = np.linalg.inv(cam2.intrinsic_matrix).T

        F = K2_inv_T @ E @ K1_inv

        return F

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector.

        Args:
            v: 3D vector.

        Returns:
            3x3 skew-symmetric matrix.
        """
        return np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )

    def synchronize_detections(
        self,
        all_detections: list[list[MultiCameraDetection]],
        time_threshold: float = 0.033,  # ~1 frame at 30fps
    ) -> list[list[MultiCameraDetection]]:
        """Synchronize detections from multiple cameras by timestamp.

        Args:
            all_detections: List of detection sequences (one per camera).
            time_threshold: Maximum time difference for synchronization.

        Returns:
            List of synchronized detection groups.
        """
        if len(all_detections) == 0:
            return []

        synchronized_groups = []

        # Get all unique timestamps
        all_timestamps = set()
        for camera_detections in all_detections:
            for detection in camera_detections:
                all_timestamps.add(detection.timestamp)

        all_timestamps = sorted(all_timestamps)

        # For each timestamp, find matching detections from all cameras
        for timestamp in all_timestamps:
            group = []

            for camera_detections in all_detections:
                # Find detection closest in time
                closest_detection = None
                min_time_diff = float("inf")

                for detection in camera_detections:
                    time_diff = abs(detection.timestamp - timestamp)

                    if time_diff < min_time_diff and time_diff <= time_threshold:
                        min_time_diff = time_diff
                        closest_detection = detection

                if closest_detection is not None:
                    group.append(closest_detection)

            # Only add groups with sufficient cameras
            if len(group) >= self.min_cameras:
                synchronized_groups.append(group)

        return synchronized_groups

    def get_camera_info(self, camera_id: str) -> CameraParameters | None:
        """Get camera parameters by ID.

        Args:
            camera_id: Camera identifier.

        Returns:
            Camera parameters or None.
        """
        return self.cameras.get(camera_id)

    def add_camera(self, camera: CameraParameters):
        """Add a new camera to the fusion system.

        Args:
            camera: Camera parameters to add.
        """
        self.cameras[camera.camera_id] = camera
        self.projection_matrices[camera.camera_id] = self._compute_projection_matrix(camera)

    def remove_camera(self, camera_id: str):
        """Remove a camera from the fusion system.

        Args:
            camera_id: Camera identifier to remove.
        """
        if camera_id in self.cameras:
            del self.cameras[camera_id]
            del self.projection_matrices[camera_id]


def create_default_pool_cameras() -> list[CameraParameters]:
    """Create default camera setup for swimming pool analysis.

    Returns:
        List of camera parameters for typical pool setup.
    """
    cameras = []

    # Overhead camera (above pool, looking down)
    K_overhead = np.array(
        [
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    R_overhead = np.eye(3, dtype=np.float32)  # Looking straight down
    t_overhead = np.array([0, 0, 5], dtype=np.float32)  # 5 meters above pool

    cameras.append(
        CameraParameters(
            camera_id="overhead",
            intrinsic_matrix=K_overhead,
            distortion_coeffs=np.zeros(5, dtype=np.float32),
            rotation_matrix=R_overhead,
            translation_vector=t_overhead,
            resolution=(1920, 1080),
            position="overhead",
        )
    )

    # Side camera (side of pool, at water level)
    K_side = np.array(
        [
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Rotated 90 degrees to look across pool
    R_side = cv2.Rodrigues(np.array([0, np.pi / 2, 0], dtype=np.float32))[0]
    t_side = np.array([10, 0, 0], dtype=np.float32)  # 10 meters to side

    cameras.append(
        CameraParameters(
            camera_id="side",
            intrinsic_matrix=K_side,
            distortion_coeffs=np.zeros(5, dtype=np.float32),
            rotation_matrix=R_side,
            translation_vector=t_side,
            resolution=(1920, 1080),
            position="side",
        )
    )

    # Underwater camera (below water, looking up)
    K_underwater = np.array(
        [
            [800, 0, 640],  # Different focal length due to water refraction
            [0, 800, 360],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Looking up from below
    R_underwater = cv2.Rodrigues(np.array([np.pi, 0, 0], dtype=np.float32))[0]
    t_underwater = np.array([0, 0, -2], dtype=np.float32)  # 2 meters below surface

    cameras.append(
        CameraParameters(
            camera_id="underwater",
            intrinsic_matrix=K_underwater,
            distortion_coeffs=np.zeros(5, dtype=np.float32),
            rotation_matrix=R_underwater,
            translation_vector=t_underwater,
            resolution=(1280, 720),
            position="underwater",
        )
    )

    return cameras
