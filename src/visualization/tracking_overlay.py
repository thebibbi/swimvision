"""Visualization overlay for hand tracking with occlusion handling."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.tracking.occlusion_detector import OcclusionState
from src.tracking.hand_tracker import TrackingResult


class TrackingOverlay:
    """Visualize hand tracking results with occlusion indicators."""

    def __init__(
        self,
        observed_color: Tuple[int, int, int] = (0, 255, 0),      # Green
        predicted_color: Tuple[int, int, int] = (255, 165, 0),   # Orange
        occluded_color: Tuple[int, int, int] = (255, 0, 0),      # Red
        trail_length: int = 30,
        trail_thickness: int = 2,
    ):
        """Initialize tracking overlay.

        Args:
            observed_color: Color for observed positions (BGR).
            predicted_color: Color for predicted positions (BGR).
            occluded_color: Color for fully occluded markers (BGR).
            trail_length: Number of frames to show in trajectory trail.
            trail_thickness: Thickness of trail lines.
        """
        self.observed_color = observed_color
        self.predicted_color = predicted_color
        self.occluded_color = occluded_color
        self.trail_length = trail_length
        self.trail_thickness = trail_thickness

    def draw_tracking_result(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        label: str = "Hand",
        show_confidence: bool = True,
        show_velocity: bool = False,
    ) -> np.ndarray:
        """Draw tracking result on frame.

        Args:
            frame: Video frame.
            result: Tracking result to visualize.
            label: Label for this tracking point.
            show_confidence: Show confidence value.
            show_velocity: Show velocity vector.

        Returns:
            Frame with overlay.
        """
        frame = frame.copy()

        # Choose color based on occlusion state
        if result.is_predicted:
            if result.occlusion_state == OcclusionState.FULLY_OCCLUDED:
                color = self.occluded_color
                marker = '?'
            else:
                color = self.predicted_color
                marker = '~'
        else:
            color = self.observed_color
            marker = '✓'

        # Draw position marker
        pos = result.position.astype(int)
        cv2.circle(frame, tuple(pos), radius=8, color=color, thickness=-1)
        cv2.circle(frame, tuple(pos), radius=10, color=(255, 255, 255), thickness=2)

        # Draw state marker
        cv2.putText(
            frame,
            marker,
            (pos[0] + 15, pos[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Draw label and confidence
        if show_confidence:
            text = f"{label} ({result.confidence:.2f})"
        else:
            text = label

        cv2.putText(
            frame,
            text,
            (pos[0] + 15, pos[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # Draw velocity vector if requested
        if show_velocity and result.velocity is not None:
            vel = result.velocity * 5  # Scale for visibility
            end_pos = (pos + vel.astype(int))
            cv2.arrowedLine(
                frame,
                tuple(pos),
                tuple(end_pos),
                color,
                2,
                tipLength=0.3,
            )

        return frame

    def draw_trajectory_trail(
        self,
        frame: np.ndarray,
        trajectory: List[np.ndarray],
        occlusion_states: Optional[List[OcclusionState]] = None,
    ) -> np.ndarray:
        """Draw trajectory trail with color-coded occlusion states.

        Args:
            frame: Video frame.
            trajectory: List of positions.
            occlusion_states: List of occlusion states (optional).

        Returns:
            Frame with trail overlay.
        """
        frame = frame.copy()

        if len(trajectory) < 2:
            return frame

        # Get recent trajectory
        trail = trajectory[-self.trail_length:]
        states = (
            occlusion_states[-self.trail_length:]
            if occlusion_states and len(occlusion_states) >= len(trail)
            else None
        )

        # Draw trail segments
        for i in range(len(trail) - 1):
            pt1 = trail[i].astype(int)
            pt2 = trail[i + 1].astype(int)

            # Choose color based on occlusion state
            if states:
                if states[i] == OcclusionState.FULLY_OCCLUDED:
                    color = self.occluded_color
                elif states[i] == OcclusionState.PARTIALLY_OCCLUDED:
                    color = self.predicted_color
                else:
                    color = self.observed_color
            else:
                color = self.observed_color

            # Fade older segments
            alpha = (i + 1) / len(trail)
            faded_color = tuple(int(c * alpha) for c in color)

            cv2.line(
                frame,
                tuple(pt1),
                tuple(pt2),
                faded_color,
                self.trail_thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_occlusion_indicator(
        self,
        frame: np.ndarray,
        occlusion_state: OcclusionState,
        position: Tuple[int, int] = (10, 30),
    ) -> np.ndarray:
        """Draw occlusion state indicator.

        Args:
            frame: Video frame.
            occlusion_state: Current occlusion state.
            position: Position to draw indicator (x, y).

        Returns:
            Frame with indicator.
        """
        frame = frame.copy()

        # State text and color
        state_info = {
            OcclusionState.VISIBLE: ("VISIBLE", self.observed_color),
            OcclusionState.PARTIALLY_OCCLUDED: ("PARTIAL OCCLUSION", self.predicted_color),
            OcclusionState.FULLY_OCCLUDED: ("UNDERWATER", self.occluded_color),
            OcclusionState.TRANSITIONING: ("TRANSITIONING", (255, 255, 0)),
        }

        text, color = state_info.get(occlusion_state, ("UNKNOWN", (128, 128, 128)))

        # Draw background box
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        box_coords = (
            (position[0] - 5, position[1] - text_size[1] - 5),
            (position[0] + text_size[0] + 5, position[1] + 5),
        )
        cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)
        cv2.rectangle(frame, box_coords[0], box_coords[1], color, 2)

        # Draw text
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        return frame

    def draw_comparison_overlay(
        self,
        frame: np.ndarray,
        observed: Optional[np.ndarray],
        predicted: np.ndarray,
        label: str = "Hand",
    ) -> np.ndarray:
        """Draw comparison between observed and predicted positions.

        Args:
            frame: Video frame.
            observed: Observed position (or None if occluded).
            predicted: Predicted position.
            label: Label for this comparison.

        Returns:
            Frame with comparison overlay.
        """
        frame = frame.copy()

        # Draw predicted position
        pred_pos = predicted.astype(int)
        cv2.circle(frame, tuple(pred_pos), 8, self.predicted_color, -1)
        cv2.circle(frame, tuple(pred_pos), 10, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"{label} (Pred)",
            (pred_pos[0] + 15, pred_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.predicted_color,
            1,
        )

        # Draw observed position if available
        if observed is not None:
            obs_pos = observed.astype(int)
            cv2.circle(frame, tuple(obs_pos), 8, self.observed_color, -1)
            cv2.circle(frame, tuple(obs_pos), 10, (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"{label} (Obs)",
                (obs_pos[0] + 15, obs_pos[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.observed_color,
                1,
            )

            # Draw line between observed and predicted
            cv2.line(
                frame,
                tuple(obs_pos),
                tuple(pred_pos),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Calculate and show error
            error = np.linalg.norm(observed - predicted)
            cv2.putText(
                frame,
                f"Error: {error:.1f}px",
                ((obs_pos[0] + pred_pos[0]) // 2, (obs_pos[1] + pred_pos[1]) // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return frame

    def create_legend(
        self,
        width: int = 300,
        height: int = 150,
    ) -> np.ndarray:
        """Create a legend explaining visualization colors.

        Args:
            width: Legend width.
            height: Legend height.

        Returns:
            Legend image.
        """
        legend = np.zeros((height, width, 3), dtype=np.uint8)

        # Title
        cv2.putText(
            legend,
            "Tracking Legend",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Observed
        cv2.circle(legend, (30, 55), 8, self.observed_color, -1)
        cv2.putText(
            legend,
            "Observed (Visible)",
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Predicted
        cv2.circle(legend, (30, 85), 8, self.predicted_color, -1)
        cv2.putText(
            legend,
            "Predicted (Partial)",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Occluded
        cv2.circle(legend, (30, 115), 8, self.occluded_color, -1)
        cv2.putText(
            legend,
            "Occluded (Underwater)",
            (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return legend
