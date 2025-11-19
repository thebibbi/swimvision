"""Water surface detection and analysis for swimming videos.

Detects water surface boundary, entry/exit events, and integrates
with occlusion detection for improved underwater tracking.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2


class WaterState(Enum):
    """State of body part relative to water."""
    ABOVE_WATER = "above_water"
    AT_SURFACE = "at_surface"
    UNDERWATER = "underwater"
    UNKNOWN = "unknown"


@dataclass
class WaterSurfaceInfo:
    """Information about detected water surface."""
    surface_line: np.ndarray                    # Fitted line (y = mx + b)
    surface_points: np.ndarray                  # Detected surface points
    confidence: float                           # Detection confidence
    water_level: float                          # Average y-coordinate of surface
    surface_normal: np.ndarray                  # Normal vector to surface
    metadata: Dict                              # Additional metadata


@dataclass
class EntryExitEvent:
    """Water entry/exit event."""
    frame_number: int                           # Frame where event occurred
    event_type: str                             # 'entry' or 'exit'
    body_part: str                              # Which body part
    position: Tuple[float, float]               # (x, y) position
    velocity: Optional[Tuple[float, float]]     # Velocity at entry/exit
    splash_intensity: float                     # Estimated splash intensity


class WaterSurfaceDetector:
    """Detect and track water surface in swimming videos."""

    def __init__(
        self,
        pool_type: str = "indoor",  # 'indoor', 'outdoor'
        detection_method: str = "edge",  # 'edge', 'color', 'optical_flow', 'hybrid'
        confidence_threshold: float = 0.7,
        surface_smoothing: int = 5,
    ):
        """Initialize water surface detector.

        Args:
            pool_type: Type of pool (affects detection parameters).
            detection_method: Method for surface detection.
            confidence_threshold: Minimum confidence for valid detection.
            surface_smoothing: Temporal smoothing window size.
        """
        self.pool_type = pool_type
        self.detection_method = detection_method
        self.confidence_threshold = confidence_threshold
        self.surface_smoothing = surface_smoothing

        # History for temporal smoothing
        self.surface_history = []
        self.water_level_history = []

        # Entry/exit event tracking
        self.tracked_body_parts = {}
        self.entry_exit_events = []

    def detect_surface(
        self,
        frame: np.ndarray,
        previous_surface: Optional[WaterSurfaceInfo] = None,
    ) -> Optional[WaterSurfaceInfo]:
        """Detect water surface in frame.

        Args:
            frame: Input frame (BGR).
            previous_surface: Previous surface detection for temporal consistency.

        Returns:
            Water surface information or None.
        """
        if self.detection_method == "edge":
            surface_info = self._detect_surface_edges(frame, previous_surface)
        elif self.detection_method == "color":
            surface_info = self._detect_surface_color(frame, previous_surface)
        elif self.detection_method == "optical_flow":
            surface_info = self._detect_surface_flow(frame, previous_surface)
        else:  # hybrid
            surface_info = self._detect_surface_hybrid(frame, previous_surface)

        # Apply temporal smoothing
        if surface_info is not None:
            surface_info = self._smooth_surface(surface_info)

        return surface_info

    def _detect_surface_edges(
        self,
        frame: np.ndarray,
        previous_surface: Optional[WaterSurfaceInfo] = None,
    ) -> Optional[WaterSurfaceInfo]:
        """Detect water surface using edge detection.

        Args:
            frame: Input frame.
            previous_surface: Previous surface for temporal consistency.

        Returns:
            Surface information or None.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Focus on horizontal edges (water surface is typically horizontal)
        kernel_horizontal = np.array([[1, 1, 1],
                                      [0, 0, 0],
                                      [-1, -1, -1]])
        horizontal_edges = cv2.filter2D(edges, -1, kernel_horizontal)

        # Find strong horizontal lines using Hough transform
        lines = cv2.HoughLinesP(
            horizontal_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=50,
        )

        if lines is None or len(lines) == 0:
            return None

        # Filter to near-horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))

            # Accept lines within 20 degrees of horizontal
            if angle < np.pi/9 or angle > 8*np.pi/9:
                horizontal_lines.append(line[0])

        if len(horizontal_lines) == 0:
            return None

        # Cluster lines by y-coordinate to find the water surface
        y_coords = np.array([np.mean([line[1], line[3]]) for line in horizontal_lines])

        # Use median y-coordinate as water level
        water_level = float(np.median(y_coords))

        # Collect points near the water level
        surface_points = []
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            avg_y = (y1 + y2) / 2

            if abs(avg_y - water_level) < 20:  # Within 20 pixels
                surface_points.extend([[x1, y1], [x2, y2]])

        if len(surface_points) < 2:
            return None

        surface_points = np.array(surface_points)

        # Fit line to surface points using RANSAC
        if len(surface_points) >= 2:
            # Fit line: y = mx + b
            xs = surface_points[:, 0]
            ys = surface_points[:, 1]

            # Use polyfit for robustness
            coeffs = np.polyfit(xs, ys, deg=1)
            m, b = coeffs

            surface_line = np.array([m, b])

            # Calculate surface normal (perpendicular to fitted line)
            surface_normal = np.array([-m, 1.0])
            surface_normal = surface_normal / np.linalg.norm(surface_normal)

            # Calculate confidence based on number of inliers
            predicted_ys = m * xs + b
            residuals = np.abs(ys - predicted_ys)
            inliers = np.sum(residuals < 5)
            confidence = min(1.0, inliers / max(len(surface_points) * 0.5, 1))

            return WaterSurfaceInfo(
                surface_line=surface_line,
                surface_points=surface_points,
                confidence=confidence,
                water_level=water_level,
                surface_normal=surface_normal,
                metadata={
                    'method': 'edge',
                    'num_points': len(surface_points),
                    'num_lines': len(horizontal_lines),
                },
            )

        return None

    def _detect_surface_color(
        self,
        frame: np.ndarray,
        previous_surface: Optional[WaterSurfaceInfo] = None,
    ) -> Optional[WaterSurfaceInfo]:
        """Detect water surface using color segmentation.

        Args:
            frame: Input frame.
            previous_surface: Previous surface.

        Returns:
            Surface information or None.
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define water color range (blue/cyan for pool water)
        if self.pool_type == "indoor":
            # Indoor pools tend to be lighter blue
            lower_water = np.array([80, 50, 50])
            upper_water = np.array([130, 255, 255])
        else:
            # Outdoor pools may vary more
            lower_water = np.array([70, 30, 30])
            upper_water = np.array([140, 255, 255])

        # Create mask for water
        water_mask = cv2.inRange(hsv, lower_water, upper_water)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        # Find the top edge of the water mask (surface)
        surface_points = []
        height, width = water_mask.shape

        for x in range(0, width, 10):  # Sample every 10 pixels
            column = water_mask[:, x]
            water_pixels = np.where(column > 0)[0]

            if len(water_pixels) > 0:
                # Top of water in this column
                y = water_pixels[0]
                surface_points.append([x, y])

        if len(surface_points) < 10:
            return None

        surface_points = np.array(surface_points)

        # Fit line
        xs = surface_points[:, 0]
        ys = surface_points[:, 1]

        coeffs = np.polyfit(xs, ys, deg=1)
        m, b = coeffs

        surface_line = np.array([m, b])
        water_level = float(np.mean(ys))
        surface_normal = np.array([-m, 1.0])
        surface_normal = surface_normal / np.linalg.norm(surface_normal)

        # Calculate confidence
        predicted_ys = m * xs + b
        residuals = np.abs(ys - predicted_ys)
        confidence = 1.0 - min(1.0, np.mean(residuals) / 50.0)

        return WaterSurfaceInfo(
            surface_line=surface_line,
            surface_points=surface_points,
            confidence=confidence,
            water_level=water_level,
            surface_normal=surface_normal,
            metadata={
                'method': 'color',
                'num_points': len(surface_points),
            },
        )

    def _detect_surface_flow(
        self,
        frame: np.ndarray,
        previous_surface: Optional[WaterSurfaceInfo] = None,
    ) -> Optional[WaterSurfaceInfo]:
        """Detect water surface using optical flow (detects motion discontinuity).

        Args:
            frame: Input frame.
            previous_surface: Previous surface.

        Returns:
            Surface information or None.
        """
        # This would require maintaining previous frame
        # Placeholder - returns previous surface if available
        return previous_surface

    def _detect_surface_hybrid(
        self,
        frame: np.ndarray,
        previous_surface: Optional[WaterSurfaceInfo] = None,
    ) -> Optional[WaterSurfaceInfo]:
        """Detect water surface using hybrid approach.

        Args:
            frame: Input frame.
            previous_surface: Previous surface.

        Returns:
            Surface information or None.
        """
        # Combine edge and color detection
        edge_surface = self._detect_surface_edges(frame, previous_surface)
        color_surface = self._detect_surface_color(frame, previous_surface)

        # If both succeed, weighted average
        if edge_surface and color_surface:
            # Weight by confidence
            w1 = edge_surface.confidence
            w2 = color_surface.confidence
            total_weight = w1 + w2

            if total_weight > 0:
                # Weighted average of water levels
                water_level = (w1 * edge_surface.water_level + w2 * color_surface.water_level) / total_weight

                # Weighted average of line parameters
                line_m = (w1 * edge_surface.surface_line[0] + w2 * color_surface.surface_line[0]) / total_weight
                line_b = (w1 * edge_surface.surface_line[1] + w2 * color_surface.surface_line[1]) / total_weight

                surface_line = np.array([line_m, line_b])

                # Combine surface points
                combined_points = np.vstack([edge_surface.surface_points, color_surface.surface_points])

                # Average confidence
                confidence = (w1 + w2) / 2

                surface_normal = np.array([-line_m, 1.0])
                surface_normal = surface_normal / np.linalg.norm(surface_normal)

                return WaterSurfaceInfo(
                    surface_line=surface_line,
                    surface_points=combined_points,
                    confidence=confidence,
                    water_level=water_level,
                    surface_normal=surface_normal,
                    metadata={
                        'method': 'hybrid',
                        'edge_confidence': edge_surface.confidence,
                        'color_confidence': color_surface.confidence,
                    },
                )

        # Fall back to whichever succeeded
        if edge_surface:
            return edge_surface
        if color_surface:
            return color_surface

        return None

    def _smooth_surface(self, surface_info: WaterSurfaceInfo) -> WaterSurfaceInfo:
        """Apply temporal smoothing to surface detection.

        Args:
            surface_info: Current surface info.

        Returns:
            Smoothed surface info.
        """
        # Add to history
        self.surface_history.append(surface_info)
        self.water_level_history.append(surface_info.water_level)

        # Limit history size
        if len(self.surface_history) > self.surface_smoothing:
            self.surface_history.pop(0)
            self.water_level_history.pop(0)

        # Calculate smoothed water level
        smoothed_water_level = float(np.mean(self.water_level_history))

        # Calculate smoothed line parameters
        smoothed_m = float(np.mean([s.surface_line[0] for s in self.surface_history]))
        smoothed_b = float(np.mean([s.surface_line[1] for s in self.surface_history]))

        smoothed_line = np.array([smoothed_m, smoothed_b])
        smoothed_normal = np.array([-smoothed_m, 1.0])
        smoothed_normal = smoothed_normal / np.linalg.norm(smoothed_normal)

        return WaterSurfaceInfo(
            surface_line=smoothed_line,
            surface_points=surface_info.surface_points,
            confidence=surface_info.confidence,
            water_level=smoothed_water_level,
            surface_normal=smoothed_normal,
            metadata={
                **surface_info.metadata,
                'smoothed': True,
                'history_size': len(self.surface_history),
            },
        )

    def get_water_state(
        self,
        point: Tuple[float, float],
        surface_info: WaterSurfaceInfo,
        threshold_pixels: float = 10.0,
    ) -> WaterState:
        """Determine if a point is above, at, or below water surface.

        Args:
            point: (x, y) point to check.
            surface_info: Water surface information.
            threshold_pixels: Threshold for "at surface" state.

        Returns:
            Water state for the point.
        """
        x, y = point
        m, b = surface_info.surface_line

        # Calculate expected y on surface at this x
        surface_y = m * x + b

        # Calculate distance to surface
        distance = y - surface_y

        if abs(distance) < threshold_pixels:
            return WaterState.AT_SURFACE
        elif distance < 0:
            # Point is above surface (y is smaller in image coordinates)
            return WaterState.ABOVE_WATER
        else:
            # Point is below surface
            return WaterState.UNDERWATER

    def detect_entry_exit_events(
        self,
        keypoints: np.ndarray,
        keypoint_names: List[str],
        surface_info: WaterSurfaceInfo,
        frame_number: int,
        velocities: Optional[np.ndarray] = None,
    ) -> List[EntryExitEvent]:
        """Detect water entry/exit events for keypoints.

        Args:
            keypoints: Current keypoints (Nx3).
            keypoint_names: Keypoint names.
            surface_info: Water surface info.
            frame_number: Current frame number.
            velocities: Keypoint velocities (Nx2), if available.

        Returns:
            List of detected entry/exit events.
        """
        events = []

        for i, (kp, name) in enumerate(zip(keypoints, keypoint_names)):
            if kp[2] < 0.3:  # Low confidence, skip
                continue

            point = (kp[0], kp[1])
            current_state = self.get_water_state(point, surface_info)

            # Check if we have previous state for this body part
            if name in self.tracked_body_parts:
                previous_state = self.tracked_body_parts[name]

                # Detect state transitions
                if previous_state != current_state:
                    event_type = None

                    if previous_state == WaterState.ABOVE_WATER and current_state in [WaterState.AT_SURFACE, WaterState.UNDERWATER]:
                        event_type = "entry"
                    elif previous_state == WaterState.UNDERWATER and current_state in [WaterState.AT_SURFACE, WaterState.ABOVE_WATER]:
                        event_type = "exit"

                    if event_type:
                        # Get velocity if available
                        velocity = None
                        if velocities is not None and i < len(velocities):
                            velocity = (float(velocities[i, 0]), float(velocities[i, 1]))

                        # Estimate splash intensity from velocity
                        splash_intensity = 0.0
                        if velocity:
                            splash_intensity = min(1.0, np.linalg.norm(velocity) / 100.0)

                        event = EntryExitEvent(
                            frame_number=frame_number,
                            event_type=event_type,
                            body_part=name,
                            position=point,
                            velocity=velocity,
                            splash_intensity=splash_intensity,
                        )

                        events.append(event)
                        self.entry_exit_events.append(event)

            # Update tracked state
            self.tracked_body_parts[name] = current_state

        return events

    def visualize_surface(
        self,
        frame: np.ndarray,
        surface_info: WaterSurfaceInfo,
        draw_points: bool = True,
        draw_line: bool = True,
    ) -> np.ndarray:
        """Visualize detected water surface on frame.

        Args:
            frame: Input frame.
            surface_info: Surface information.
            draw_points: Whether to draw detected points.
            draw_line: Whether to draw fitted line.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        # Draw surface points
        if draw_points:
            for point in surface_info.surface_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(annotated, (x, y), 2, (0, 255, 255), -1)

        # Draw fitted line
        if draw_line:
            m, b = surface_info.surface_line
            height, width = frame.shape[:2]

            x1, x2 = 0, width
            y1 = int(m * x1 + b)
            y2 = int(m * x2 + b)

            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw confidence
            text = f"Water Surface (conf: {surface_info.confidence:.2f})"
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated

    def reset(self):
        """Reset detector state."""
        self.surface_history.clear()
        self.water_level_history.clear()
        self.tracked_body_parts.clear()
        self.entry_exit_events.clear()
