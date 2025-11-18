"""Pose overlay visualization for swimming analysis."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class PoseOverlay:
    """Draw pose skeleton and annotations on video frames."""

    # COCO skeleton connections (pairs of keypoint indices)
    SKELETON_CONNECTIONS = [
        # Face
        (0, 1),  # nose -> left_eye
        (0, 2),  # nose -> right_eye
        (1, 3),  # left_eye -> left_ear
        (2, 4),  # right_eye -> right_ear
        # Torso
        (5, 6),  # left_shoulder -> right_shoulder
        (5, 11),  # left_shoulder -> left_hip
        (6, 12),  # right_shoulder -> right_hip
        (11, 12),  # left_hip -> right_hip
        # Left arm
        (5, 7),  # left_shoulder -> left_elbow
        (7, 9),  # left_elbow -> left_wrist
        # Right arm
        (6, 8),  # right_shoulder -> right_elbow
        (8, 10),  # right_elbow -> right_wrist
        # Left leg
        (11, 13),  # left_hip -> left_knee
        (13, 15),  # left_knee -> left_ankle
        # Right leg
        (12, 14),  # right_hip -> right_knee
        (14, 16),  # right_knee -> right_ankle
    ]

    # Keypoint name to index mapping
    KEYPOINT_INDICES = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(
        self,
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),
        skeleton_color_left: Tuple[int, int, int] = (255, 0, 0),
        skeleton_color_right: Tuple[int, int, int] = (0, 0, 255),
        keypoint_radius: int = 5,
        skeleton_thickness: int = 2,
        min_confidence: float = 0.3,
    ):
        """Initialize pose overlay visualizer.

        Args:
            keypoint_color: Color for keypoints (B, G, R).
            skeleton_color_left: Color for left side skeleton (B, G, R).
            skeleton_color_right: Color for right side skeleton (B, G, R).
            keypoint_radius: Radius of keypoint circles.
            skeleton_thickness: Thickness of skeleton lines.
            min_confidence: Minimum confidence to draw keypoint.
        """
        self.keypoint_color = keypoint_color
        self.skeleton_color_left = skeleton_color_left
        self.skeleton_color_right = skeleton_color_right
        self.keypoint_radius = keypoint_radius
        self.skeleton_thickness = skeleton_thickness
        self.min_confidence = min_confidence

    def draw_skeleton(
        self, frame: np.ndarray, pose_data: Dict, show_confidence: bool = False
    ) -> np.ndarray:
        """Draw pose skeleton on frame.

        Args:
            frame: Input frame (BGR format).
            pose_data: Pose data from YOLO estimator.
            show_confidence: Whether to show confidence scores.

        Returns:
            Frame with skeleton drawn.
        """
        if pose_data is None or "keypoints" not in pose_data:
            return frame

        frame_copy = frame.copy()
        keypoints = pose_data["keypoints"]

        # Draw skeleton connections
        for connection in self.SKELETON_CONNECTIONS:
            idx1, idx2 = connection
            kpt1_name = list(self.KEYPOINT_INDICES.keys())[idx1]
            kpt2_name = list(self.KEYPOINT_INDICES.keys())[idx2]

            kpt1 = keypoints.get(kpt1_name)
            kpt2 = keypoints.get(kpt2_name)

            if kpt1 and kpt2:
                if (
                    kpt1["confidence"] >= self.min_confidence
                    and kpt2["confidence"] >= self.min_confidence
                ):
                    pt1 = (int(kpt1["x"]), int(kpt1["y"]))
                    pt2 = (int(kpt2["x"]), int(kpt2["y"]))

                    # Choose color based on left/right side
                    if "left" in kpt1_name or "left" in kpt2_name:
                        color = self.skeleton_color_left
                    elif "right" in kpt1_name or "right" in kpt2_name:
                        color = self.skeleton_color_right
                    else:
                        color = (255, 255, 255)  # White for center line

                    cv2.line(frame_copy, pt1, pt2, color, self.skeleton_thickness)

        # Draw keypoints
        for kpt_name, kpt in keypoints.items():
            if kpt["confidence"] >= self.min_confidence:
                pt = (int(kpt["x"]), int(kpt["y"]))

                # Draw keypoint circle
                cv2.circle(frame_copy, pt, self.keypoint_radius, self.keypoint_color, -1)
                cv2.circle(frame_copy, pt, self.keypoint_radius + 1, (0, 0, 0), 1)

                # Draw confidence if requested
                if show_confidence:
                    conf_text = f"{kpt['confidence']:.2f}"
                    cv2.putText(
                        frame_copy,
                        conf_text,
                        (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                    )

        return frame_copy

    def draw_angles(
        self, frame: np.ndarray, pose_data: Dict, angles: Dict[str, Optional[float]]
    ) -> np.ndarray:
        """Draw angle measurements on frame.

        Args:
            frame: Input frame.
            pose_data: Pose data dictionary.
            angles: Angle measurements from SwimmingKeypoints.

        Returns:
            Frame with angles drawn.
        """
        if pose_data is None or "keypoints" not in pose_data:
            return frame

        frame_copy = frame.copy()
        keypoints = pose_data["keypoints"]

        # Define angle display positions (keypoint name, angle key, offset)
        angle_displays = [
            ("left_elbow", "left_elbow_angle", (-30, -10)),
            ("right_elbow", "right_elbow_angle", (10, -10)),
            ("left_knee", "left_knee_angle", (-30, -10)),
            ("right_knee", "right_knee_angle", (10, -10)),
        ]

        for kpt_name, angle_key, offset in angle_displays:
            angle = angles.get(angle_key)
            kpt = keypoints.get(kpt_name)

            if angle is not None and kpt and kpt["confidence"] >= self.min_confidence:
                pt = (int(kpt["x"]) + offset[0], int(kpt["y"]) + offset[1])
                text = f"{angle:.1f}°"

                # Draw background rectangle
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame_copy,
                    (pt[0] - 2, pt[1] - text_h - 2),
                    (pt[0] + text_w + 2, pt[1] + 2),
                    (0, 0, 0),
                    -1,
                )

                # Draw text
                cv2.putText(
                    frame_copy,
                    text,
                    pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

        return frame_copy

    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (255, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw hand/body trajectory path on frame.

        Args:
            frame: Input frame.
            trajectory: List of (x, y) positions.
            color: Trajectory color (B, G, R).
            thickness: Line thickness.

        Returns:
            Frame with trajectory drawn.
        """
        if len(trajectory) < 2:
            return frame

        frame_copy = frame.copy()

        for i in range(len(trajectory) - 1):
            pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
            pt2 = (int(trajectory[i + 1][0]), int(trajectory[i + 1][1]))
            cv2.line(frame_copy, pt1, pt2, color, thickness)

        return frame_copy

    def draw_bbox(
        self, frame: np.ndarray, bbox: Dict, label: str = "Swimmer"
    ) -> np.ndarray:
        """Draw bounding box around detected person.

        Args:
            frame: Input frame.
            bbox: Bounding box dictionary with x1, y1, x2, y2, confidence.
            label: Label text.

        Returns:
            Frame with bbox drawn.
        """
        if bbox is None:
            return frame

        frame_copy = frame.copy()

        pt1 = (int(bbox["x1"]), int(bbox["y1"]))
        pt2 = (int(bbox["x2"]), int(bbox["y2"]))

        # Draw rectangle
        cv2.rectangle(frame_copy, pt1, pt2, (0, 255, 0), 2)

        # Draw label
        label_text = f"{label} {bbox['confidence']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(
            frame_copy,
            (pt1[0], pt1[1] - text_h - 10),
            (pt1[0] + text_w, pt1[1]),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            frame_copy,
            label_text,
            (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        return frame_copy

    def draw_fps(
        self, frame: np.ndarray, fps: float, position: str = "top_left"
    ) -> np.ndarray:
        """Draw FPS counter on frame.

        Args:
            frame: Input frame.
            fps: Current FPS.
            position: Position ('top_left', 'top_right', 'bottom_left', 'bottom_right').

        Returns:
            Frame with FPS drawn.
        """
        frame_copy = frame.copy()
        h, w = frame.shape[:2]

        text = f"FPS: {fps:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Determine position
        if position == "top_left":
            pt = (10, 30)
        elif position == "top_right":
            pt = (w - text_w - 10, 30)
        elif position == "bottom_left":
            pt = (10, h - 10)
        else:  # bottom_right
            pt = (w - text_w - 10, h - 10)

        # Draw background
        cv2.rectangle(
            frame_copy,
            (pt[0] - 5, pt[1] - text_h - 5),
            (pt[0] + text_w + 5, pt[1] + 5),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(frame_copy, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame_copy
