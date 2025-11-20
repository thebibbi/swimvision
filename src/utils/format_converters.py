"""Format conversion utilities for pose keypoints.

Handles conversions between different keypoint formats:
- COCO-17 ↔ COCO-133 (wholebody)
- COCO-17 ↔ SMPL-24
- SMPL-24 ↔ OpenSim markers
- MediaPipe-33 ↔ COCO-17
- And more...
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum

from src.pose.base_estimator import KeypointFormat


class KeypointConverter:
    """Convert between different keypoint formats."""

    # COCO-17 keypoint names
    COCO17_NAMES = [
        'nose',
        'left_eye', 'right_eye',
        'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
    ]

    # SMPL-24 joint names
    SMPL24_NAMES = [
        'pelvis',           # 0
        'left_hip',         # 1
        'right_hip',        # 2
        'spine1',           # 3
        'left_knee',        # 4
        'right_knee',       # 5
        'spine2',           # 6
        'left_ankle',       # 7
        'right_ankle',      # 8
        'spine3',           # 9
        'left_foot',        # 10
        'right_foot',       # 11
        'neck',             # 12
        'left_collar',      # 13
        'right_collar',     # 14
        'head',             # 15
        'left_shoulder',    # 16
        'right_shoulder',   # 17
        'left_elbow',       # 18
        'right_elbow',      # 19
        'left_wrist',       # 20
        'right_wrist',      # 21
        'left_hand',        # 22
        'right_hand',       # 23
    ]

    @staticmethod
    def coco17_to_smpl24(coco_keypoints: np.ndarray) -> np.ndarray:
        """Convert COCO-17 keypoints to SMPL-24 format.

        Args:
            coco_keypoints: COCO-17 keypoints (17, 3) [x, y, confidence]

        Returns:
            SMPL-24 keypoints (24, 3)
        """
        assert coco_keypoints.shape[0] == 17, "Expected 17 COCO keypoints"

        smpl_keypoints = np.zeros((24, 3))

        # Direct mappings (indices based on COCO-17 and SMPL-24 order)
        # COCO: 5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow,
        #       9=l_wrist, 10=r_wrist, 11=l_hip, 12=r_hip,
        #       13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle

        smpl_keypoints[16] = coco_keypoints[5]   # left_shoulder
        smpl_keypoints[17] = coco_keypoints[6]   # right_shoulder
        smpl_keypoints[18] = coco_keypoints[7]   # left_elbow
        smpl_keypoints[19] = coco_keypoints[8]   # right_elbow
        smpl_keypoints[20] = coco_keypoints[9]   # left_wrist
        smpl_keypoints[21] = coco_keypoints[10]  # right_wrist
        smpl_keypoints[1] = coco_keypoints[11]   # left_hip
        smpl_keypoints[2] = coco_keypoints[12]   # right_hip
        smpl_keypoints[4] = coco_keypoints[13]   # left_knee
        smpl_keypoints[5] = coco_keypoints[14]   # right_knee
        smpl_keypoints[7] = coco_keypoints[15]   # left_ankle
        smpl_keypoints[8] = coco_keypoints[16]   # right_ankle

        # Interpolated/estimated joints
        # Pelvis: midpoint of hips
        smpl_keypoints[0] = (coco_keypoints[11] + coco_keypoints[12]) / 2

        # Spine joints (estimate based on body proportions)
        hip_to_shoulder = (
            (coco_keypoints[5] + coco_keypoints[6]) / 2 - smpl_keypoints[0]
        )
        smpl_keypoints[3] = smpl_keypoints[0] + hip_to_shoulder * 0.3  # spine1
        smpl_keypoints[6] = smpl_keypoints[0] + hip_to_shoulder * 0.6  # spine2
        smpl_keypoints[9] = smpl_keypoints[0] + hip_to_shoulder * 0.9  # spine3

        # Neck: midpoint between shoulders
        smpl_keypoints[12] = (coco_keypoints[5] + coco_keypoints[6]) / 2

        # Head: use nose as proxy
        smpl_keypoints[15] = coco_keypoints[0]

        # Collars (estimate from shoulders)
        shoulder_center = smpl_keypoints[12]
        smpl_keypoints[13] = shoulder_center + (coco_keypoints[5] - shoulder_center) * 0.3  # l_collar
        smpl_keypoints[14] = shoulder_center + (coco_keypoints[6] - shoulder_center) * 0.3  # r_collar

        # Hands (extend from wrists)
        wrist_to_hand_offset = np.array([0, -0.08, 0])  # ~8cm down
        smpl_keypoints[22] = coco_keypoints[9] + wrist_to_hand_offset   # l_hand
        smpl_keypoints[23] = coco_keypoints[10] + wrist_to_hand_offset  # r_hand

        # Feet (extend from ankles)
        ankle_to_foot_offset = np.array([0, -0.05, 0])  # ~5cm down
        smpl_keypoints[10] = coco_keypoints[15] + ankle_to_foot_offset  # l_foot
        smpl_keypoints[11] = coco_keypoints[16] + ankle_to_foot_offset  # r_foot

        # Copy confidence scores (use minimum of contributing keypoints for interpolated)
        for i in [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]:
            # Direct mappings keep confidence
            pass

        # For interpolated joints, use average confidence of contributing joints
        smpl_keypoints[0, 2] = np.mean([coco_keypoints[11, 2], coco_keypoints[12, 2]])  # pelvis
        smpl_keypoints[3, 2] = smpl_keypoints[0, 2]  # spine1
        smpl_keypoints[6, 2] = smpl_keypoints[0, 2]  # spine2
        smpl_keypoints[9, 2] = smpl_keypoints[0, 2]  # spine3
        smpl_keypoints[12, 2] = np.mean([coco_keypoints[5, 2], coco_keypoints[6, 2]])  # neck
        smpl_keypoints[15, 2] = coco_keypoints[0, 2]  # head
        smpl_keypoints[13, 2] = coco_keypoints[5, 2]  # l_collar
        smpl_keypoints[14, 2] = coco_keypoints[6, 2]  # r_collar
        smpl_keypoints[22, 2] = coco_keypoints[9, 2]  # l_hand
        smpl_keypoints[23, 2] = coco_keypoints[10, 2]  # r_hand
        smpl_keypoints[10, 2] = coco_keypoints[15, 2]  # l_foot
        smpl_keypoints[11, 2] = coco_keypoints[16, 2]  # r_foot

        return smpl_keypoints

    @staticmethod
    def smpl24_to_coco17(smpl_keypoints: np.ndarray) -> np.ndarray:
        """Convert SMPL-24 keypoints to COCO-17 format.

        Args:
            smpl_keypoints: SMPL-24 keypoints (24, 3)

        Returns:
            COCO-17 keypoints (17, 3)
        """
        assert smpl_keypoints.shape[0] == 24, "Expected 24 SMPL keypoints"

        coco_keypoints = np.zeros((17, 3))

        # Direct reverse mappings
        coco_keypoints[0] = smpl_keypoints[15]   # nose (from head)
        coco_keypoints[5] = smpl_keypoints[16]   # left_shoulder
        coco_keypoints[6] = smpl_keypoints[17]   # right_shoulder
        coco_keypoints[7] = smpl_keypoints[18]   # left_elbow
        coco_keypoints[8] = smpl_keypoints[19]   # right_elbow
        coco_keypoints[9] = smpl_keypoints[20]   # left_wrist
        coco_keypoints[10] = smpl_keypoints[21]  # right_wrist
        coco_keypoints[11] = smpl_keypoints[1]   # left_hip
        coco_keypoints[12] = smpl_keypoints[2]   # right_hip
        coco_keypoints[13] = smpl_keypoints[4]   # left_knee
        coco_keypoints[14] = smpl_keypoints[5]   # right_knee
        coco_keypoints[15] = smpl_keypoints[7]   # left_ankle
        coco_keypoints[16] = smpl_keypoints[8]   # right_ankle

        # Estimate eyes and ears from head position
        # (SMPL doesn't have these, so we approximate)
        head = smpl_keypoints[15]
        neck = smpl_keypoints[12]
        head_dir = head[:2] - neck[:2]
        if np.linalg.norm(head_dir) > 0:
            head_dir = head_dir / np.linalg.norm(head_dir)
        else:
            head_dir = np.array([0, 1])

        # Eyes (offset from head)
        eye_offset = 0.03  # 3cm
        perp = np.array([-head_dir[1], head_dir[0]])  # Perpendicular

        coco_keypoints[1, :2] = head[:2] + perp * eye_offset  # left_eye
        coco_keypoints[2, :2] = head[:2] - perp * eye_offset  # right_eye
        coco_keypoints[1, 2] = head[2] * 0.8  # Lower confidence for estimated
        coco_keypoints[2, 2] = head[2] * 0.8

        # Ears (offset further)
        ear_offset = 0.05  # 5cm
        coco_keypoints[3, :2] = head[:2] + perp * ear_offset  # left_ear
        coco_keypoints[4, :2] = head[:2] - perp * ear_offset  # right_ear
        coco_keypoints[3, 2] = head[2] * 0.7
        coco_keypoints[4, 2] = head[2] * 0.7

        return coco_keypoints

    @staticmethod
    def smpl24_to_opensim_markers(smpl_keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert SMPL-24 keypoints to OpenSim marker positions.

        Args:
            smpl_keypoints: SMPL-24 keypoints (24, 3)

        Returns:
            Dictionary of marker_name -> position (3,)
        """
        markers = {}

        # Direct mappings
        markers['pelvis'] = smpl_keypoints[0, :3]
        markers['l_hip'] = smpl_keypoints[1, :3]
        markers['r_hip'] = smpl_keypoints[2, :3]
        markers['l_knee'] = smpl_keypoints[4, :3]
        markers['r_knee'] = smpl_keypoints[5, :3]
        markers['l_ankle'] = smpl_keypoints[7, :3]
        markers['r_ankle'] = smpl_keypoints[8, :3]
        markers['l_foot'] = smpl_keypoints[10, :3]
        markers['r_foot'] = smpl_keypoints[11, :3]
        markers['neck'] = smpl_keypoints[12, :3]
        markers['head'] = smpl_keypoints[15, :3]
        markers['l_shoulder'] = smpl_keypoints[16, :3]
        markers['r_shoulder'] = smpl_keypoints[17, :3]
        markers['l_elbow'] = smpl_keypoints[18, :3]
        markers['r_elbow'] = smpl_keypoints[19, :3]
        markers['l_wrist'] = smpl_keypoints[20, :3]
        markers['r_wrist'] = smpl_keypoints[21, :3]
        markers['l_hand'] = smpl_keypoints[22, :3]
        markers['r_hand'] = smpl_keypoints[23, :3]

        # Additional OpenSim markers (estimated)
        # Spine markers
        markers['spine'] = smpl_keypoints[3, :3]  # spine1
        markers['thorax'] = smpl_keypoints[6, :3]  # spine2
        markers['chest'] = smpl_keypoints[9, :3]  # spine3

        # ASIS/PSIS (anterior/posterior superior iliac spine)
        pelvis = smpl_keypoints[0, :3]
        l_hip = smpl_keypoints[1, :3]
        r_hip = smpl_keypoints[2, :3]

        hip_width = np.linalg.norm(l_hip - r_hip)
        hip_center = (l_hip + r_hip) / 2

        # Estimate ASIS (front of pelvis)
        markers['l_asis'] = l_hip + np.array([0, 0, 0.05])
        markers['r_asis'] = r_hip + np.array([0, 0, 0.05])

        # Estimate PSIS (back of pelvis)
        markers['l_psis'] = l_hip + np.array([0, 0, -0.05])
        markers['r_psis'] = r_hip + np.array([0, 0, -0.05])

        # Acromion (top of shoulder)
        markers['l_acromion'] = smpl_keypoints[16, :3] + np.array([0, 0.02, 0])
        markers['r_acromion'] = smpl_keypoints[17, :3] + np.array([0, 0.02, 0])

        # Heel markers
        markers['l_heel'] = smpl_keypoints[10, :3] + np.array([0, -0.03, -0.05])
        markers['r_heel'] = smpl_keypoints[11, :3] + np.array([0, -0.03, -0.05])

        # Toe markers
        markers['l_toe'] = smpl_keypoints[10, :3] + np.array([0, -0.03, 0.1])
        markers['r_toe'] = smpl_keypoints[11, :3] + np.array([0, -0.03, 0.1])

        return markers

    @staticmethod
    def mediapipe33_to_coco17(mp_keypoints: np.ndarray) -> np.ndarray:
        """Convert MediaPipe-33 landmarks to COCO-17 format.

        Args:
            mp_keypoints: MediaPipe-33 landmarks (33, 3)

        Returns:
            COCO-17 keypoints (17, 3)
        """
        assert mp_keypoints.shape[0] == 33, "Expected 33 MediaPipe landmarks"

        # MediaPipe landmark indices (from mediapipe_estimator.py mapping)
        MP_TO_COCO17 = {
            0: 0,    # nose
            2: 1,    # left_eye (left_eye_inner)
            5: 2,    # right_eye (right_eye_inner)
            7: 3,    # left_ear
            8: 4,    # right_ear
            11: 5,   # left_shoulder
            12: 6,   # right_shoulder
            13: 7,   # left_elbow
            14: 8,   # right_elbow
            15: 9,   # left_wrist
            16: 10,  # right_wrist
            23: 11,  # left_hip
            24: 12,  # right_hip
            25: 13,  # left_knee
            26: 14,  # right_knee
            27: 15,  # left_ankle
            28: 16,  # right_ankle
        }

        coco_keypoints = np.zeros((17, 3))

        for coco_idx, mp_idx in MP_TO_COCO17.items():
            coco_keypoints[coco_idx] = mp_keypoints[mp_idx]

        return coco_keypoints

    @staticmethod
    def normalize_keypoints_to_bbox(
        keypoints: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Normalize keypoints to bounding box coordinates [0, 1].

        Args:
            keypoints: Keypoints (N, 2 or 3)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Normalized keypoints (N, 2 or 3)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        normalized = keypoints.copy()
        normalized[:, 0] = (keypoints[:, 0] - x1) / width
        normalized[:, 1] = (keypoints[:, 1] - y1) / height

        return normalized

    @staticmethod
    def denormalize_keypoints_from_bbox(
        normalized_keypoints: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Denormalize keypoints from bounding box coordinates.

        Args:
            normalized_keypoints: Normalized keypoints (N, 2 or 3) in [0, 1]
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Absolute keypoints (N, 2 or 3)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        denormalized = normalized_keypoints.copy()
        denormalized[:, 0] = normalized_keypoints[:, 0] * width + x1
        denormalized[:, 1] = normalized_keypoints[:, 1] * height + y1

        return denormalized

    @staticmethod
    def convert_format(
        keypoints: np.ndarray,
        source_format: KeypointFormat,
        target_format: KeypointFormat
    ) -> np.ndarray:
        """Convert keypoints between formats (routing function).

        Args:
            keypoints: Source keypoints
            source_format: Source format
            target_format: Target format

        Returns:
            Converted keypoints

        Raises:
            ValueError: If conversion not supported
        """
        if source_format == target_format:
            return keypoints.copy()

        # COCO-17 conversions
        if source_format == KeypointFormat.COCO_17:
            if target_format == KeypointFormat.SMPL_24:
                return KeypointConverter.coco17_to_smpl24(keypoints)
            # Add more as needed

        # SMPL-24 conversions
        elif source_format == KeypointFormat.SMPL_24:
            if target_format == KeypointFormat.COCO_17:
                return KeypointConverter.smpl24_to_coco17(keypoints)
            # Add more as needed

        # MediaPipe conversions
        elif source_format == KeypointFormat.MEDIAPIPE_33:
            if target_format == KeypointFormat.COCO_17:
                return KeypointConverter.mediapipe33_to_coco17(keypoints)
            # Chain conversions if needed
            elif target_format == KeypointFormat.SMPL_24:
                coco17 = KeypointConverter.mediapipe33_to_coco17(keypoints)
                return KeypointConverter.coco17_to_smpl24(coco17)

        raise ValueError(
            f"Conversion from {source_format.value} to {target_format.value} not implemented"
        )


def test_conversions():
    """Test format conversions."""
    print("Testing format conversions...")

    # Test COCO-17 → SMPL-24
    coco17 = np.random.rand(17, 3)
    coco17[:, 2] = 0.9  # High confidence

    smpl24 = KeypointConverter.coco17_to_smpl24(coco17)
    assert smpl24.shape == (24, 3), "SMPL-24 shape incorrect"
    print("✅ COCO-17 → SMPL-24")

    # Test SMPL-24 → COCO-17
    coco17_back = KeypointConverter.smpl24_to_coco17(smpl24)
    assert coco17_back.shape == (17, 3), "COCO-17 shape incorrect"
    print("✅ SMPL-24 → COCO-17")

    # Test SMPL-24 → OpenSim markers
    markers = KeypointConverter.smpl24_to_opensim_markers(smpl24)
    assert 'pelvis' in markers, "Missing pelvis marker"
    assert 'l_shoulder' in markers, "Missing left shoulder marker"
    print(f"✅ SMPL-24 → OpenSim markers ({len(markers)} markers)")

    # Test MediaPipe-33 → COCO-17
    mp33 = np.random.rand(33, 3)
    mp33[:, 2] = 0.8

    coco17_from_mp = KeypointConverter.mediapipe33_to_coco17(mp33)
    assert coco17_from_mp.shape == (17, 3), "COCO-17 from MP shape incorrect"
    print("✅ MediaPipe-33 → COCO-17")

    # Test normalization
    bbox = (100, 100, 300, 400)
    normalized = KeypointConverter.normalize_keypoints_to_bbox(coco17, bbox)
    assert np.all(normalized[:, :2] >= 0) and np.all(normalized[:, :2] <= 1), "Normalization failed"
    print("✅ Normalization to bbox")

    # Test denormalization
    denormalized = KeypointConverter.denormalize_keypoints_from_bbox(normalized, bbox)
    assert np.allclose(denormalized[:, :2], coco17[:, :2]), "Denormalization failed"
    print("✅ Denormalization from bbox")

    print("\n✅ All format conversion tests passed!")


if __name__ == "__main__":
    test_conversions()
