"""SMPL/SMPL-X 3D body model wrapper for pose estimation.

SMPL (Skinned Multi-Person Linear) is a parametric 3D body model.
SMPL-X extends SMPL with expressive hands and face.

Excellent for 3D underwater pose reconstruction and biomechanical analysis.

Install:
- pip install smplx
- Download SMPL/SMPL-X models from https://smpl-x.is.tue.mpg.de/
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import pickle

from src.pose.base_estimator import (
    BasePoseEstimator,
    KeypointFormat,
    PoseModel,
)

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False


class SMPLEstimator(BasePoseEstimator):
    """SMPL/SMPL-X wrapper for 3D body model estimation."""

    def __init__(
        self,
        model_type: str = "smplx",  # 'smpl', 'smplh', 'smplx'
        model_path: Optional[str] = None,
        gender: str = "neutral",  # 'male', 'female', 'neutral'
        device: str = "cpu",
        confidence: float = 0.5,
        use_pca: bool = True,
        num_pca_comps: int = 12,
        ext: str = "npz",
    ):
        """Initialize SMPL/SMPL-X estimator.

        Args:
            model_type: Model type ('smpl', 'smplh', 'smplx').
            model_path: Path to SMPL model files.
            gender: Body model gender.
            device: Device to run on.
            confidence: Confidence threshold.
            use_pca: Use PCA for hand pose.
            num_pca_comps: Number of PCA components for hands.
            ext: Model file extension ('pkl' or 'npz').
        """
        super().__init__(model_type, device, confidence)

        if not SMPLX_AVAILABLE:
            raise ImportError(
                "SMPL-X not installed. Install with: pip install smplx\n"
                "Download models from: https://smpl-x.is.tue.mpg.de/"
            )

        self.model_type = model_type
        self.model_path = model_path or self._get_default_model_path()
        self.gender = gender
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.ext = ext

        self.load_model()

    def load_model(self):
        """Load SMPL/SMPL-X model."""
        model_params = {
            'model_path': str(self.model_path),
            'gender': self.gender,
            'ext': self.ext,
        }

        if self.model_type == 'smplx':
            model_params['use_pca'] = self.use_pca
            model_params['num_pca_comps'] = self.num_pca_comps
            model_params['use_face_contour'] = True
            self.model = smplx.create(**model_params, model_type='smplx')
        elif self.model_type == 'smplh':
            model_params['use_pca'] = self.use_pca
            model_params['num_pca_comps'] = self.num_pca_comps
            self.model = smplx.create(**model_params, model_type='smplh')
        else:  # smpl
            self.model = smplx.create(**model_params, model_type='smpl')

        self.model = self.model.to(self.device)
        self.model.eval()

    def estimate_pose(
        self,
        image: np.ndarray,
        return_image: bool = True,
        body_pose: Optional[np.ndarray] = None,
        global_orient: Optional[np.ndarray] = None,
        transl: Optional[np.ndarray] = None,
        betas: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Estimate 3D body model from parameters.

        Args:
            image: Input image (for visualization).
            return_image: Whether to return annotated image.
            body_pose: Body pose parameters (default: zero pose).
            global_orient: Global orientation (default: facing forward).
            transl: Translation (default: centered).
            betas: Shape parameters (default: mean shape).

        Returns:
            Tuple of (pose_data, annotated_image).

        Note:
            This method requires pose parameters as input. For pose estimation
            from images, use this with an optimization-based approach or
            regression network (e.g., SPIN, HMR, VIBE).
        """
        # Set default parameters if not provided
        batch_size = 1

        if body_pose is None:
            # Default to zero pose (T-pose)
            num_body_joints = 21 if self.model_type == 'smpl' else 21
            body_pose = torch.zeros((batch_size, num_body_joints * 3)).to(self.device)
        else:
            body_pose = torch.from_numpy(body_pose).float().to(self.device)
            if len(body_pose.shape) == 1:
                body_pose = body_pose.unsqueeze(0)

        if global_orient is None:
            global_orient = torch.zeros((batch_size, 3)).to(self.device)
        else:
            global_orient = torch.from_numpy(global_orient).float().to(self.device)
            if len(global_orient.shape) == 1:
                global_orient = global_orient.unsqueeze(0)

        if transl is None:
            transl = torch.zeros((batch_size, 3)).to(self.device)
        else:
            transl = torch.from_numpy(transl).float().to(self.device)
            if len(transl.shape) == 1:
                transl = transl.unsqueeze(0)

        if betas is None:
            # Default to mean shape
            num_betas = 10
            betas = torch.zeros((batch_size, num_betas)).to(self.device)
        else:
            betas = torch.from_numpy(betas).float().to(self.device)
            if len(betas.shape) == 1:
                betas = betas.unsqueeze(0)

        # Forward pass through SMPL model
        with torch.no_grad():
            output = self.model(
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                betas=betas,
            )

        # Extract outputs
        vertices = output.vertices[0].cpu().numpy()  # 3D mesh vertices
        joints = output.joints[0].cpu().numpy()      # 3D joint locations

        # Format output
        pose_data = self._format_output(
            vertices=vertices,
            joints=joints,
            body_pose=body_pose[0].cpu().numpy(),
            global_orient=global_orient[0].cpu().numpy(),
            transl=transl[0].cpu().numpy(),
            betas=betas[0].cpu().numpy(),
            image_shape=image.shape if image is not None else None,
        )

        # Annotate image if requested
        annotated_image = None
        if return_image and image is not None:
            annotated_image = self._render_mesh(image, vertices, joints)

        return pose_data, annotated_image

    def _format_output(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray,
        image_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Dict:
        """Format SMPL output to standard format.

        Args:
            vertices: 3D mesh vertices.
            joints: 3D joint locations.
            body_pose: Body pose parameters.
            global_orient: Global orientation.
            transl: Translation.
            betas: Shape parameters.
            image_shape: Image shape for projection.

        Returns:
            Formatted pose data.
        """
        # SMPL/SMPL-X joint names
        if self.model_type == 'smpl':
            joint_names = [
                'pelvis', 'left_hip', 'right_hip', 'spine1',
                'left_knee', 'right_knee', 'spine2',
                'left_ankle', 'right_ankle', 'spine3',
                'left_foot', 'right_foot', 'neck',
                'left_collar', 'right_collar', 'head',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hand', 'right_hand',
            ]
        else:  # smplx has more joints
            joint_names = [
                'pelvis', 'left_hip', 'right_hip', 'spine1',
                'left_knee', 'right_knee', 'spine2',
                'left_ankle', 'right_ankle', 'spine3',
                'left_foot', 'right_foot', 'neck',
                'left_collar', 'right_collar', 'head',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
            ] + [f'joint_{i}' for i in range(len(joints) - 22)]

        # Project 3D joints to 2D if image shape provided
        keypoints_2d = None
        if image_shape is not None:
            keypoints_2d = self._project_to_2d(joints, image_shape)

        return {
            'keypoints': keypoints_2d if keypoints_2d is not None else joints[:, :2],
            'keypoint_names': joint_names[:len(joints)],
            'bbox': self._calculate_bbox_from_joints(joints) if keypoints_2d is not None else None,
            'person_id': 0,
            'format': self.get_keypoint_format(),
            'metadata': {
                'model': self.model_type,
                'gender': self.gender,
                'vertices': vertices,       # 3D mesh
                'joints_3d': joints,        # 3D joints
                'body_pose': body_pose,     # Pose parameters
                'global_orient': global_orient,
                'transl': transl,
                'betas': betas,             # Shape parameters
                'num_vertices': len(vertices),
                'num_joints': len(joints),
            },
        }

    def _project_to_2d(
        self,
        joints_3d: np.ndarray,
        image_shape: Tuple[int, int, int],
        focal_length: float = 5000.0,
    ) -> np.ndarray:
        """Project 3D joints to 2D image coordinates.

        Args:
            joints_3d: 3D joint locations (Nx3).
            image_shape: Image shape (height, width, channels).
            focal_length: Camera focal length.

        Returns:
            2D keypoints with confidence (Nx3).
        """
        height, width, _ = image_shape

        # Simple perspective projection
        # Assume camera at origin, looking down -Z axis
        camera_center = np.array([width / 2, height / 2])

        # Project to 2D
        joints_2d = np.zeros((len(joints_3d), 3))

        for i, joint_3d in enumerate(joints_3d):
            x, y, z = joint_3d

            # Perspective projection
            if z != 0:
                u = focal_length * (x / z) + camera_center[0]
                v = focal_length * (y / z) + camera_center[1]
            else:
                u, v = camera_center

            # Confidence based on depth (closer = higher confidence)
            confidence = 1.0 / (1.0 + abs(z) / 10.0)

            joints_2d[i] = [u, v, confidence]

        return joints_2d

    def _calculate_bbox_from_joints(self, joints: np.ndarray) -> List[float]:
        """Calculate bounding box from 3D joints.

        Args:
            joints: 3D joint locations.

        Returns:
            Bounding box [x1, y1, x2, y2].
        """
        x_coords = joints[:, 0]
        y_coords = joints[:, 1]

        x1, y1 = np.min(x_coords), np.min(y_coords)
        x2, y2 = np.max(x_coords), np.max(y_coords)

        # Add padding
        width, height = x2 - x1, y2 - y1
        padding_x, padding_y = width * 0.1, height * 0.1

        return [
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            x2 + padding_x,
            y2 + padding_y,
        ]

    def _render_mesh(
        self,
        image: np.ndarray,
        vertices: np.ndarray,
        joints: np.ndarray,
    ) -> np.ndarray:
        """Render 3D mesh overlay on image.

        Args:
            image: Input image.
            vertices: 3D mesh vertices.
            joints: 3D joint locations.

        Returns:
            Image with mesh overlay.
        """
        annotated = image.copy()

        # Project joints to 2D
        joints_2d = self._project_to_2d(joints, image.shape)

        # Draw joints
        for joint_2d in joints_2d:
            x, y, conf = joint_2d
            if conf > self.confidence:
                cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Draw skeleton connections (simplified)
        skeleton = [
            (0, 1), (0, 2), (0, 3),  # Pelvis to hips and spine
            (1, 4), (2, 5), (3, 6),  # Hips to knees, spine
            (4, 7), (5, 8), (6, 9),  # Knees to ankles, spine
            (9, 12), (12, 15),       # Spine to neck to head
            (9, 13), (9, 14),        # Spine to collars
            (13, 16), (14, 17),      # Collars to shoulders
            (16, 18), (17, 19),      # Shoulders to elbows
            (18, 20), (19, 21),      # Elbows to wrists
        ]

        for start_idx, end_idx in skeleton:
            if start_idx < len(joints_2d) and end_idx < len(joints_2d):
                start = joints_2d[start_idx]
                end = joints_2d[end_idx]

                if start[2] > self.confidence and end[2] > self.confidence:
                    pt1 = (int(start[0]), int(start[1]))
                    pt2 = (int(end[0]), int(end[1]))
                    cv2.line(annotated, pt1, pt2, (255, 0, 0), 2)

        return annotated

    def get_keypoint_format(self) -> KeypointFormat:
        """Get keypoint format."""
        if self.model_type == 'smpl':
            return KeypointFormat.SMPL_24
        else:
            return KeypointFormat.SMPL_X_127

    def supports_3d(self) -> bool:
        """SMPL/SMPL-X provides full 3D body model."""
        return True

    def supports_multi_person(self) -> bool:
        """SMPL/SMPL-X is single-person (per instance)."""
        return False

    def get_3d_mesh(self, pose_data: Dict) -> Optional[np.ndarray]:
        """Get 3D mesh vertices from pose data.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            3D mesh vertices (Nx3) or None.
        """
        return pose_data.get('metadata', {}).get('vertices')

    def get_3d_joints(self, pose_data: Dict) -> Optional[np.ndarray]:
        """Get 3D joint locations from pose data.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            3D joints (Nx3) or None.
        """
        return pose_data.get('metadata', {}).get('joints_3d')

    def get_shape_parameters(self, pose_data: Dict) -> Optional[np.ndarray]:
        """Get body shape parameters.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            Shape parameters (betas) or None.
        """
        return pose_data.get('metadata', {}).get('betas')

    def get_pose_parameters(self, pose_data: Dict) -> Dict[str, np.ndarray]:
        """Get all pose parameters.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            Dictionary with pose parameters.
        """
        metadata = pose_data.get('metadata', {})
        return {
            'body_pose': metadata.get('body_pose'),
            'global_orient': metadata.get('global_orient'),
            'transl': metadata.get('transl'),
        }

    def _get_default_model_path(self) -> str:
        """Get default SMPL model path.

        Returns:
            Path to models folder.
        """
        # Try common installation locations
        possible_paths = [
            Path("models/smplx"),
            Path.home() / "smplx" / "models",
            Path("/usr/local/share/smplx/models"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Return default path
        return "models/smplx"

    def save_mesh(self, pose_data: Dict, output_path: str, format: str = "obj"):
        """Save 3D mesh to file.

        Args:
            pose_data: Pose data with mesh.
            output_path: Output file path.
            format: Mesh format ('obj', 'ply').
        """
        vertices = self.get_3d_mesh(pose_data)
        if vertices is None:
            raise ValueError("No mesh data available")

        faces = self.model.faces

        if format == "obj":
            self._save_obj(vertices, faces, output_path)
        elif format == "ply":
            self._save_ply(vertices, faces, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_obj(self, vertices: np.ndarray, faces: np.ndarray, path: str):
        """Save mesh as OBJ file."""
        with open(path, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (1-indexed)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def _save_ply(self, vertices: np.ndarray, faces: np.ndarray, path: str):
        """Save mesh as PLY file."""
        with open(path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Vertices
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
