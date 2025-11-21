"""Smoothing and filtering utilities for pose sequences.

This module provides various smoothing techniques to reduce noise in pose estimation data:
- Kalman filtering for state estimation
- Savitzky-Golay filtering for polynomial smoothing
- Moving average filtering for simple smoothing
- Outlier detection and removal
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import zscore


class KalmanFilter1D:
    """Simple 1D Kalman filter for trajectory smoothing.

    Useful for smoothing individual coordinate sequences (x or y positions).

    Attributes:
        process_variance: Process noise variance (Q).
        measurement_variance: Measurement noise variance (R).
        state: Current state estimate.
        covariance: Current error covariance.
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-1,
        initial_value: float = 0.0,
        initial_covariance: float = 1.0,
    ):
        """Initialize Kalman filter.

        Args:
            process_variance: Process noise variance (Q). Lower = trust model more.
            measurement_variance: Measurement noise variance (R). Lower = trust measurements more.
            initial_value: Initial state estimate.
            initial_covariance: Initial error covariance.
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = initial_value
        self.covariance = initial_covariance

    def update(self, measurement: float) -> float:
        """Update filter with new measurement.

        Args:
            measurement: New observed value.

        Returns:
            Filtered state estimate.
        """
        # Prediction step
        predicted_state = self.state
        predicted_covariance = self.covariance + self.process_variance

        # Update step
        kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_variance)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.covariance = (1 - kalman_gain) * predicted_covariance

        return self.state

    def reset(self, value: float | None = None):
        """Reset filter state.

        Args:
            value: New initial value. If None, uses first value passed to update.
        """
        if value is not None:
            self.state = value
        self.covariance = 1.0


class KalmanFilter2D:
    """2D Kalman filter for trajectory smoothing with velocity estimation.

    Tracks position and velocity in 2D space. Useful for smoothing hand trajectories.

    State vector: [x, y, vx, vy]
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-1,
        dt: float = 1.0,
    ):
        """Initialize 2D Kalman filter.

        Args:
            process_variance: Process noise variance.
            measurement_variance: Measurement noise variance.
            dt: Time step between measurements.
        """
        self.dt = dt

        # State: [x, y, vx, vy]
        self.state = np.zeros(4)

        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Measurement matrix (we only observe position)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Process noise covariance
        self.Q = np.eye(4) * process_variance

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_variance

        # Error covariance
        self.P = np.eye(4)

        # Track initialization
        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement.

        Args:
            measurement: 2D position [x, y].

        Returns:
            Filtered position estimate [x, y].
        """
        measurement = np.array(measurement).flatten()

        if not self.initialized:
            # Initialize state with first measurement
            self.state[0:2] = measurement
            self.initialized = True
            return measurement

        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update step
        y = measurement - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.state[0:2]

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate.

        Returns:
            Velocity [vx, vy].
        """
        return self.state[2:4]

    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False


def smooth_trajectory_kalman(
    trajectory: np.ndarray,
    process_variance: float = 1e-5,
    measurement_variance: float = 1e-1,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a 2D trajectory using Kalman filtering.

    Args:
        trajectory: Nx2 array of [x, y] positions.
        process_variance: Process noise variance.
        measurement_variance: Measurement noise variance.
        dt: Time step between measurements.

    Returns:
        Tuple of (smoothed_trajectory, velocities).
        - smoothed_trajectory: Nx2 array of smoothed positions.
        - velocities: Nx2 array of velocity estimates.
    """
    if len(trajectory) == 0:
        return np.array([]), np.array([])

    kf = KalmanFilter2D(process_variance, measurement_variance, dt)

    smoothed = np.zeros_like(trajectory)
    velocities = np.zeros_like(trajectory)

    for i, point in enumerate(trajectory):
        smoothed[i] = kf.update(point)
        velocities[i] = kf.get_velocity()

    return smoothed, velocities


def smooth_signal_savgol(
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    axis: int = 0,
) -> np.ndarray:
    """Smooth a signal using Savitzky-Golay filter.

    Good for preserving features like peaks while reducing noise.

    Args:
        signal: Input signal (1D or 2D array).
        window_length: Length of filter window (must be odd).
        polyorder: Order of polynomial to fit.
        axis: Axis along which to filter.

    Returns:
        Smoothed signal.
    """
    if len(signal) < window_length:
        # If signal too short, return as-is
        return signal

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure window_length <= signal length
    window_length = min(window_length, len(signal))

    # Ensure polyorder < window_length
    polyorder = min(polyorder, window_length - 1)

    return savgol_filter(signal, window_length, polyorder, axis=axis)


def smooth_trajectory_savgol(
    trajectory: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Smooth a 2D trajectory using Savitzky-Golay filter.

    Args:
        trajectory: Nx2 array of [x, y] positions.
        window_length: Length of filter window (must be odd).
        polyorder: Order of polynomial to fit.

    Returns:
        Smoothed trajectory (Nx2 array).
    """
    if len(trajectory) < 3:
        return trajectory

    # Smooth x and y separately
    smoothed = np.zeros_like(trajectory)
    smoothed[:, 0] = smooth_signal_savgol(trajectory[:, 0], window_length, polyorder)
    smoothed[:, 1] = smooth_signal_savgol(trajectory[:, 1], window_length, polyorder)

    return smoothed


def moving_average(
    signal: np.ndarray,
    window_size: int = 5,
    mode: str = "valid",
) -> np.ndarray:
    """Apply moving average filter to signal.

    Args:
        signal: Input signal (1D array).
        window_size: Size of averaging window.
        mode: Convolution mode ('valid', 'same', 'full').
            - 'valid': Output length = len(signal) - window_size + 1
            - 'same': Output length = len(signal)
            - 'full': Output length = len(signal) + window_size - 1

    Returns:
        Smoothed signal.
    """
    if len(signal) < window_size:
        return signal

    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode=mode)


def smooth_trajectory_ma(
    trajectory: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Smooth a 2D trajectory using moving average.

    Args:
        trajectory: Nx2 array of [x, y] positions.
        window_size: Size of averaging window.

    Returns:
        Smoothed trajectory (same length as input).
    """
    if len(trajectory) < window_size:
        return trajectory

    smoothed = np.zeros_like(trajectory)
    smoothed[:, 0] = moving_average(trajectory[:, 0], window_size, mode="same")
    smoothed[:, 1] = moving_average(trajectory[:, 1], window_size, mode="same")

    return smoothed


def detect_outliers_zscore(
    signal: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """Detect outliers using Z-score method.

    Args:
        signal: Input signal (1D array).
        threshold: Z-score threshold for outlier detection.

    Returns:
        Boolean array where True indicates outlier.
    """
    if len(signal) < 3:
        return np.zeros(len(signal), dtype=bool)

    z_scores = np.abs(zscore(signal, nan_policy="omit"))
    return z_scores > threshold


def detect_outliers_iqr(
    signal: np.ndarray,
    factor: float = 1.5,
) -> np.ndarray:
    """Detect outliers using IQR (Interquartile Range) method.

    Args:
        signal: Input signal (1D array).
        factor: IQR multiplier for outlier threshold.

    Returns:
        Boolean array where True indicates outlier.
    """
    if len(signal) < 3:
        return np.zeros(len(signal), dtype=bool)

    q1 = np.percentile(signal, 25)
    q3 = np.percentile(signal, 75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    return (signal < lower_bound) | (signal > upper_bound)


def remove_outliers(
    signal: np.ndarray,
    method: str = "zscore",
    threshold: float = 3.0,
    interpolate: bool = True,
) -> np.ndarray:
    """Remove outliers from signal.

    Args:
        signal: Input signal (1D array).
        method: Outlier detection method ('zscore' or 'iqr').
        threshold: Threshold for outlier detection.
        interpolate: If True, interpolate over outliers. If False, replace with NaN.

    Returns:
        Signal with outliers removed/interpolated.
    """
    if method == "zscore":
        outliers = detect_outliers_zscore(signal, threshold)
    elif method == "iqr":
        outliers = detect_outliers_iqr(signal, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    signal_clean = signal.copy()

    if interpolate:
        # Interpolate over outliers
        valid_indices = np.where(~outliers)[0]
        if len(valid_indices) > 1:
            signal_clean[outliers] = np.interp(
                np.where(outliers)[0], valid_indices, signal[valid_indices]
            )
    else:
        # Replace with NaN
        signal_clean[outliers] = np.nan

    return signal_clean


def smooth_pose_sequence(
    pose_sequence: list[dict],
    keypoint_name: str,
    method: str = "kalman",
    **kwargs,
) -> list[tuple[float, float]]:
    """Smooth a specific keypoint trajectory across pose sequence.

    Args:
        pose_sequence: List of pose dictionaries from pose estimator.
        keypoint_name: Name of keypoint to smooth (e.g., 'left_wrist').
        method: Smoothing method ('kalman', 'savgol', 'ma').
        **kwargs: Additional arguments for smoothing method.

    Returns:
        List of smoothed (x, y) positions.
    """
    # Extract trajectory
    trajectory = []
    for pose in pose_sequence:
        keypoints = pose.get("keypoints", [])
        keypoint_names = pose.get("keypoint_names", [])

        if keypoint_name in keypoint_names:
            idx = keypoint_names.index(keypoint_name)
            x, y = keypoints[idx][:2]
            trajectory.append([x, y])
        else:
            # If keypoint missing, use previous value or skip
            if len(trajectory) > 0:
                trajectory.append(trajectory[-1])
            else:
                trajectory.append([0.0, 0.0])

    trajectory = np.array(trajectory)

    if len(trajectory) == 0:
        return []

    # Apply smoothing
    if method == "kalman":
        smoothed, _ = smooth_trajectory_kalman(trajectory, **kwargs)
    elif method == "savgol":
        smoothed = smooth_trajectory_savgol(trajectory, **kwargs)
    elif method == "ma":
        smoothed = smooth_trajectory_ma(trajectory, **kwargs)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return [(x, y) for x, y in smoothed]


def calculate_velocity(
    trajectory: np.ndarray,
    dt: float = 1.0,
    smooth: bool = True,
) -> np.ndarray:
    """Calculate velocity from trajectory.

    Args:
        trajectory: Nx2 array of [x, y] positions.
        dt: Time step between measurements.
        smooth: Whether to smooth velocity estimates.

    Returns:
        (N-1)x2 array of velocities [vx, vy].
    """
    if len(trajectory) < 2:
        return np.array([])

    # Compute finite differences
    velocities = np.diff(trajectory, axis=0) / dt

    if smooth and len(velocities) > 5:
        # Smooth velocities with moving average
        velocities = smooth_trajectory_ma(velocities, window_size=3)

    return velocities


def calculate_acceleration(
    trajectory: np.ndarray,
    dt: float = 1.0,
    smooth: bool = True,
) -> np.ndarray:
    """Calculate acceleration from trajectory.

    Args:
        trajectory: Nx2 array of [x, y] positions.
        dt: Time step between measurements.
        smooth: Whether to smooth acceleration estimates.

    Returns:
        (N-2)x2 array of accelerations [ax, ay].
    """
    velocities = calculate_velocity(trajectory, dt, smooth=smooth)

    if len(velocities) < 2:
        return np.array([])

    # Compute finite differences of velocity
    accelerations = np.diff(velocities, axis=0) / dt

    if smooth and len(accelerations) > 5:
        # Smooth accelerations with moving average
        accelerations = smooth_trajectory_ma(accelerations, window_size=3)

    return accelerations


def calculate_speed(
    velocity: np.ndarray,
) -> np.ndarray:
    """Calculate speed (magnitude of velocity).

    Args:
        velocity: Nx2 array of velocities [vx, vy].

    Returns:
        N-length array of speeds.
    """
    return np.linalg.norm(velocity, axis=1)
