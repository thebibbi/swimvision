"""Unit tests for geometry utilities."""

from src.utils.geometry import (
    calculate_angle,
    calculate_body_roll,
    calculate_trajectory_length,
    calculate_velocity,
    euclidean_distance,
    midpoint,
    smooth_trajectory,
)


class TestAngleCalculation:
    """Test angle calculation functions."""

    def test_right_angle(self):
        """Test 90-degree angle calculation."""
        point1 = (0, 1)
        point2 = (0, 0)
        point3 = (1, 0)

        angle = calculate_angle(point1, point2, point3)
        assert abs(angle - 90.0) < 0.01

    def test_straight_line(self):
        """Test 180-degree angle (straight line)."""
        point1 = (0, 0)
        point2 = (1, 0)
        point3 = (2, 0)

        angle = calculate_angle(point1, point2, point3)
        assert abs(angle - 180.0) < 0.01

    def test_acute_angle(self):
        """Test 45-degree angle."""
        point1 = (0, 1)
        point2 = (0, 0)
        point3 = (1, 1)

        angle = calculate_angle(point1, point2, point3)
        assert abs(angle - 45.0) < 0.01


class TestDistance:
    """Test distance calculation functions."""

    def test_euclidean_distance_horizontal(self):
        """Test horizontal distance."""
        point1 = (0, 0)
        point2 = (3, 0)

        dist = euclidean_distance(point1, point2)
        assert abs(dist - 3.0) < 0.01

    def test_euclidean_distance_diagonal(self):
        """Test diagonal distance (3-4-5 triangle)."""
        point1 = (0, 0)
        point2 = (3, 4)

        dist = euclidean_distance(point1, point2)
        assert abs(dist - 5.0) < 0.01

    def test_euclidean_distance_same_point(self):
        """Test distance between same point."""
        point = (5, 5)

        dist = euclidean_distance(point, point)
        assert dist == 0.0


class TestMidpoint:
    """Test midpoint calculation."""

    def test_midpoint_horizontal(self):
        """Test midpoint of horizontal line."""
        point1 = (0, 0)
        point2 = (4, 0)

        mid = midpoint(point1, point2)
        assert mid == (2.0, 0.0)

    def test_midpoint_diagonal(self):
        """Test midpoint of diagonal line."""
        point1 = (0, 0)
        point2 = (2, 4)

        mid = midpoint(point1, point2)
        assert mid == (1.0, 2.0)


class TestVelocity:
    """Test velocity calculation."""

    def test_constant_velocity(self):
        """Test velocity from constant motion."""
        positions = [(0, 0), (1, 0), (2, 0), (3, 0)]
        fps = 30.0

        velocities = calculate_velocity(positions, fps)

        # All velocities should be equal (30 pixels/sec)
        for vel in velocities[:-1]:
            assert abs(vel - 30.0) < 0.01

    def test_zero_velocity(self):
        """Test velocity when stationary."""
        positions = [(5, 5), (5, 5), (5, 5)]
        fps = 30.0

        velocities = calculate_velocity(positions, fps)

        for vel in velocities:
            assert vel == 0.0


class TestBodyRoll:
    """Test body roll calculation."""

    def test_level_body(self):
        """Test body roll when level (0 degrees)."""
        left_shoulder = (0, 10)
        right_shoulder = (10, 10)
        left_hip = (0, 20)
        right_hip = (10, 20)

        roll = calculate_body_roll(left_shoulder, right_shoulder, left_hip, right_hip)
        assert abs(roll) < 0.01  # Should be close to 0

    def test_rolled_body(self):
        """Test body roll when tilted."""
        # Rolled 45 degrees to the left
        left_shoulder = (0, 0)
        right_shoulder = (10, 10)
        left_hip = (0, 10)
        right_hip = (10, 20)

        roll = calculate_body_roll(left_shoulder, right_shoulder, left_hip, right_hip)
        # Should be around 45 degrees
        assert 40 < roll < 50


class TestTrajectory:
    """Test trajectory analysis functions."""

    def test_trajectory_length_straight(self):
        """Test trajectory length for straight line."""
        trajectory = [(0, 0), (1, 0), (2, 0), (3, 0)]

        length = calculate_trajectory_length(trajectory)
        assert abs(length - 3.0) < 0.01

    def test_trajectory_length_empty(self):
        """Test trajectory length for empty path."""
        trajectory = []

        length = calculate_trajectory_length(trajectory)
        assert length == 0.0

    def test_smooth_trajectory(self):
        """Test trajectory smoothing."""
        # Create noisy trajectory
        trajectory = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]

        smoothed = smooth_trajectory(trajectory, window_size=3)

        # Smoothed trajectory should have same length
        assert len(smoothed) == len(trajectory)

        # Smoothed values should be between min and max
        for i, (x, y) in enumerate(smoothed):
            assert 0 <= x <= 4
            assert 0 <= y <= 1
