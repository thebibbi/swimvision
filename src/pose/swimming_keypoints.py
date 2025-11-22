"""Swimming-specific keypoint analysis."""

import numpy as np

from src.utils.geometry import calculate_angle, calculate_body_roll


class SwimmingKeypoints:
    """Extract swimming-specific biomechanical measurements from pose data."""

    # Swimming-critical joint groups
    ARM_JOINTS = ["shoulder", "elbow", "wrist"]
    LEG_JOINTS = ["hip", "knee", "ankle"]
    TORSO_JOINTS = ["shoulder", "hip"]

    def __init__(self, min_confidence: float = 0.3):
        """Initialize swimming keypoints analyzer.

        Args:
            min_confidence: Minimum keypoint confidence to use.
        """
        self.min_confidence = min_confidence

    def get_body_angles(self, pose_data: dict) -> dict[str, float | None]:
        """Calculate all swimming-relevant body angles.

        Args:
            pose_data: Pose data from YOLO estimator.

        Returns:
            Dictionary of angle measurements (in degrees).
        """
        angles = {
            # Arms
            "left_elbow_angle": self.calculate_elbow_angle(pose_data, "left"),
            "right_elbow_angle": self.calculate_elbow_angle(pose_data, "right"),
            "left_shoulder_angle": self.calculate_shoulder_angle(pose_data, "left"),
            "right_shoulder_angle": self.calculate_shoulder_angle(pose_data, "right"),
            # Legs
            "left_knee_angle": self.calculate_knee_angle(pose_data, "left"),
            "right_knee_angle": self.calculate_knee_angle(pose_data, "right"),
            "left_hip_angle": self.calculate_hip_angle(pose_data, "left"),
            "right_hip_angle": self.calculate_hip_angle(pose_data, "right"),
            # Torso
            "body_roll": self.calculate_body_roll_angle(pose_data),
        }

        return angles

    def calculate_elbow_angle(self, pose_data: dict, side: str) -> float | None:
        """Calculate elbow angle (shoulder-elbow-wrist).

        Args:
            pose_data: Pose data dictionary.
            side: 'left' or 'right'.

        Returns:
            Elbow angle in degrees (0-180) or None if keypoints missing.
        """
        shoulder = self._get_keypoint(pose_data, f"{side}_shoulder")
        elbow = self._get_keypoint(pose_data, f"{side}_elbow")
        wrist = self._get_keypoint(pose_data, f"{side}_wrist")

        if shoulder is None or elbow is None or wrist is None:
            return None

        return calculate_angle(shoulder, elbow, wrist)

    def calculate_shoulder_angle(self, pose_data: dict, side: str) -> float | None:
        """Calculate shoulder angle (hip-shoulder-elbow).

        Args:
            pose_data: Pose data dictionary.
            side: 'left' or 'right'.

        Returns:
            Shoulder angle in degrees or None.
        """
        hip = self._get_keypoint(pose_data, f"{side}_hip")
        shoulder = self._get_keypoint(pose_data, f"{side}_shoulder")
        elbow = self._get_keypoint(pose_data, f"{side}_elbow")

        if hip is None or shoulder is None or elbow is None:
            return None

        return calculate_angle(hip, shoulder, elbow)

    def calculate_knee_angle(self, pose_data: dict, side: str) -> float | None:
        """Calculate knee angle (hip-knee-ankle).

        Args:
            pose_data: Pose data dictionary.
            side: 'left' or 'right'.

        Returns:
            Knee angle in degrees or None.
        """
        hip = self._get_keypoint(pose_data, f"{side}_hip")
        knee = self._get_keypoint(pose_data, f"{side}_knee")
        ankle = self._get_keypoint(pose_data, f"{side}_ankle")

        if hip is None or knee is None or ankle is None:
            return None

        return calculate_angle(hip, knee, ankle)

    def calculate_hip_angle(self, pose_data: dict, side: str) -> float | None:
        """Calculate hip angle (shoulder-hip-knee).

        Args:
            pose_data: Pose data dictionary.
            side: 'left' or 'right'.

        Returns:
            Hip angle in degrees or None.
        """
        shoulder = self._get_keypoint(pose_data, f"{side}_shoulder")
        hip = self._get_keypoint(pose_data, f"{side}_hip")
        knee = self._get_keypoint(pose_data, f"{side}_knee")

        if shoulder is None or hip is None or knee is None:
            return None

        return calculate_angle(shoulder, hip, knee)

    def calculate_body_roll_angle(self, pose_data: dict) -> float | None:
        """Calculate body roll angle from shoulders and hips.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            Body roll angle in degrees (-90 to 90) or None.
        """
        left_shoulder = self._get_keypoint(pose_data, "left_shoulder")
        right_shoulder = self._get_keypoint(pose_data, "right_shoulder")
        left_hip = self._get_keypoint(pose_data, "left_hip")
        right_hip = self._get_keypoint(pose_data, "right_hip")

        if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
            return None

        return calculate_body_roll(left_shoulder, right_shoulder, left_hip, right_hip)

    def get_hand_path(self, pose_sequence: list[dict], side: str) -> list[tuple[float, float]]:
        """Extract hand trajectory from pose sequence.

        Args:
            pose_sequence: List of pose data dictionaries over time.
            side: 'left' or 'right'.

        Returns:
            List of (x, y) hand positions.
        """
        hand_path = []
        for pose_data in pose_sequence:
            wrist = self._get_keypoint(pose_data, f"{side}_wrist")
            if wrist is not None:
                hand_path.append(wrist)

        return hand_path

    def get_arm_extension(self, pose_data: dict, side: str) -> float | None:
        """Calculate arm extension (0 = fully bent, 180 = fully straight).

        Args:
            pose_data: Pose data dictionary.
            side: 'left' or 'right'.

        Returns:
            Elbow angle (same as calculate_elbow_angle).
        """
        return self.calculate_elbow_angle(pose_data, side)

    def check_symmetric_position(self, pose_data: dict, tolerance: float = 15.0) -> dict[str, bool]:
        """Check if left and right sides are symmetric.

        Args:
            pose_data: Pose data dictionary.
            tolerance: Angle tolerance in degrees for symmetry.

        Returns:
            Dictionary indicating symmetry for different body parts.
        """
        symmetry = {}

        # Check arm symmetry
        left_elbow = self.calculate_elbow_angle(pose_data, "left")
        right_elbow = self.calculate_elbow_angle(pose_data, "right")
        if left_elbow is not None and right_elbow is not None:
            symmetry["elbows"] = abs(left_elbow - right_elbow) <= tolerance
        else:
            symmetry["elbows"] = None

        # Check leg symmetry
        left_knee = self.calculate_knee_angle(pose_data, "left")
        right_knee = self.calculate_knee_angle(pose_data, "right")
        if left_knee is not None and right_knee is not None:
            symmetry["knees"] = abs(left_knee - right_knee) <= tolerance
        else:
            symmetry["knees"] = None

        return symmetry

    def validate_pose(self, pose_data: dict) -> tuple[bool, list[str]]:
        """Validate that essential keypoints are present for swimming analysis.

        Args:
            pose_data: Pose data dictionary.

        Returns:
            Tuple of (is_valid, missing_keypoints).
        """
        required_keypoints = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
        ]

        missing = []
        for kpt_name in required_keypoints:
            if self._get_keypoint(pose_data, kpt_name) is None:
                missing.append(kpt_name)

        is_valid = len(missing) == 0
        return is_valid, missing

    def _get_keypoint(self, pose_data: dict, keypoint_name: str) -> tuple[float, float] | None:
        """Get keypoint coordinates if confidence is above threshold.

        Args:
            pose_data: Pose data dictionary.
            keypoint_name: Name of keypoint.

        Returns:
            Tuple of (x, y) or None if not available/low confidence.
        """
        if pose_data is None or "keypoints" not in pose_data:
            return None

        keypoints = pose_data["keypoints"]

        # Handle both numpy array format (from YOLO) and dict format
        if isinstance(keypoints, np.ndarray):
            # YOLO format: numpy array (17, 3) with [x, y, conf]
            # Need to map keypoint_name to index
            if "keypoint_names" not in pose_data:
                return None

            keypoint_names = pose_data["keypoint_names"]
            if keypoint_name not in keypoint_names:
                return None

            idx = keypoint_names.index(keypoint_name)
            if idx >= len(keypoints):
                return None

            kpt = keypoints[idx]  # [x, y, conf]
            x, y, conf = kpt[0], kpt[1], kpt[2]

            # Check confidence
            if conf < self.min_confidence:
                return None

            return (float(x), float(y))
        else:
            # Dictionary format (from other estimators)
            kpt = keypoints.get(keypoint_name)
            if kpt is None:
                return None

            # Check confidence
            if kpt["confidence"] < self.min_confidence:
                return None

            return (kpt["x"], kpt["y"])
