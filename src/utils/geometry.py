"""Geometry utilities for pose analysis."""

import math
from typing import List, Optional, Tuple

import numpy as np


def calculate_angle(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    point3: Tuple[float, float],
) -> float:
    """Calculate angle formed by three points (in degrees).

    The angle is calculated at point2 (vertex), formed by vectors:
    point2->point1 and point2->point3.

    Args:
        point1: First point (x, y).
        point2: Vertex point (x, y).
        point3: Third point (x, y).

    Returns:
        Angle in degrees (0-180).

    Example:
        >>> calculate_angle((0, 1), (0, 0), (1, 0))
        90.0
    """
    # Create vectors
    v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def calculate_angle_3d(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    point3: Tuple[float, float, float],
) -> float:
    """Calculate 3D angle formed by three points (in degrees).

    Args:
        point1: First point (x, y, z).
        point2: Vertex point (x, y, z).
        point3: Third point (x, y, z).

    Returns:
        Angle in degrees (0-180).
    """
    v1 = np.array([point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]])
    v2 = np.array([point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def euclidean_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two 2D points.

    Args:
        point1: First point (x, y).
        point2: Second point (x, y).

    Returns:
        Distance between points.
    """
    return float(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def euclidean_distance_3d(
    point1: Tuple[float, float, float], point2: Tuple[float, float, float]
) -> float:
    """Calculate Euclidean distance between two 3D points.

    Args:
        point1: First point (x, y, z).
        point2: Second point (x, y, z).

    Returns:
        Distance between points.
    """
    return float(
        np.sqrt(
            (point1[0] - point2[0]) ** 2
            + (point1[1] - point2[1]) ** 2
            + (point1[2] - point2[2]) ** 2
        )
    )


def midpoint(point1: Tuple[float, float], point2: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate midpoint between two 2D points.

    Args:
        point1: First point (x, y).
        point2: Second point (x, y).

    Returns:
        Midpoint (x, y).
    """
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        vector: Input vector.

    Returns:
        Normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector
    return vector / norm


def calculate_velocity(
    positions: List[Tuple[float, float]], fps: float
) -> List[float]:
    """Calculate velocity from position sequence.

    Args:
        positions: List of (x, y) positions over time.
        fps: Frames per second.

    Returns:
        List of velocities (pixels/second).
    """
    if len(positions) < 2:
        return [0.0]

    velocities = []
    dt = 1.0 / fps

    for i in range(len(positions) - 1):
        dist = euclidean_distance(positions[i], positions[i + 1])
        vel = dist / dt
        velocities.append(vel)

    # Add last velocity (duplicate of previous)
    velocities.append(velocities[-1] if velocities else 0.0)

    return velocities


def calculate_acceleration(velocities: List[float], fps: float) -> List[float]:
    """Calculate acceleration from velocity sequence.

    Args:
        velocities: List of velocities over time.
        fps: Frames per second.

    Returns:
        List of accelerations (pixels/second²).
    """
    if len(velocities) < 2:
        return [0.0]

    accelerations = []
    dt = 1.0 / fps

    for i in range(len(velocities) - 1):
        acc = (velocities[i + 1] - velocities[i]) / dt
        accelerations.append(acc)

    # Add last acceleration
    accelerations.append(accelerations[-1] if accelerations else 0.0)

    return accelerations


def project_point_to_line(
    point: Tuple[float, float],
    line_point1: Tuple[float, float],
    line_point2: Tuple[float, float],
) -> Tuple[float, float]:
    """Project a point onto a line defined by two points.

    Args:
        point: Point to project (x, y).
        line_point1: First point on line (x, y).
        line_point2: Second point on line (x, y).

    Returns:
        Projected point (x, y).
    """
    # Line vector
    line_vec = np.array([line_point2[0] - line_point1[0], line_point2[1] - line_point1[1]])
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-8:
        return line_point1

    # Vector from line_point1 to point
    point_vec = np.array([point[0] - line_point1[0], point[1] - line_point1[1]])

    # Project
    t = np.dot(point_vec, line_vec) / line_len_sq
    projection = np.array([line_point1[0], line_point1[1]]) + t * line_vec

    return (float(projection[0]), float(projection[1]))


def perpendicular_distance_to_line(
    point: Tuple[float, float],
    line_point1: Tuple[float, float],
    line_point2: Tuple[float, float],
) -> float:
    """Calculate perpendicular distance from point to line.

    Args:
        point: Point (x, y).
        line_point1: First point on line (x, y).
        line_point2: Second point on line (x, y).

    Returns:
        Perpendicular distance.
    """
    projection = project_point_to_line(point, line_point1, line_point2)
    return euclidean_distance(point, projection)


def calculate_body_roll(
    left_shoulder: Tuple[float, float],
    right_shoulder: Tuple[float, float],
    left_hip: Tuple[float, float],
    right_hip: Tuple[float, float],
) -> float:
    """Calculate body roll angle from shoulder and hip positions.

    Body roll is the rotation of the body around the longitudinal axis.

    Args:
        left_shoulder: Left shoulder position (x, y).
        right_shoulder: Right shoulder position (x, y).
        left_hip: Left hip position (x, y).
        right_hip: Right hip position (x, y).

    Returns:
        Body roll angle in degrees (-90 to 90).
        Positive = rolled to the left, Negative = rolled to the right.
    """
    # Calculate shoulder line angle
    shoulder_vec = np.array(
        [right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]
    )
    shoulder_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])

    # Calculate hip line angle
    hip_vec = np.array([right_hip[0] - left_hip[0], right_hip[1] - left_hip[1]])
    hip_angle = np.arctan2(hip_vec[1], hip_vec[0])

    # Average the two angles
    avg_angle = (shoulder_angle + hip_angle) / 2

    # Convert to degrees
    roll_deg = np.degrees(avg_angle)

    # Normalize to -90 to 90
    if roll_deg > 90:
        roll_deg -= 180
    elif roll_deg < -90:
        roll_deg += 180

    return float(roll_deg)


def calculate_trajectory_length(trajectory: List[Tuple[float, float]]) -> float:
    """Calculate total path length of a trajectory.

    Args:
        trajectory: List of (x, y) positions.

    Returns:
        Total path length.
    """
    if len(trajectory) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(trajectory) - 1):
        total_length += euclidean_distance(trajectory[i], trajectory[i + 1])

    return total_length


def smooth_trajectory(
    trajectory: List[Tuple[float, float]], window_size: int = 5
) -> List[Tuple[float, float]]:
    """Smooth a trajectory using moving average.

    Args:
        trajectory: List of (x, y) positions.
        window_size: Size of moving average window (must be odd).

    Returns:
        Smoothed trajectory.
    """
    if len(trajectory) < window_size:
        return trajectory

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2
    smoothed = []

    for i in range(len(trajectory)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(trajectory), i + half_window + 1)

        window_points = trajectory[start_idx:end_idx]
        avg_x = sum(p[0] for p in window_points) / len(window_points)
        avg_y = sum(p[1] for p in window_points) / len(window_points)

        smoothed.append((avg_x, avg_y))

    return smoothed
