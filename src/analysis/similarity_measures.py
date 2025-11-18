"""Additional similarity measures for time series comparison."""

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import correlate


class SoftDTW:
    """Soft Dynamic Time Warping for differentiable alignment."""

    def __init__(self, gamma: float = 1.0):
        """Initialize Soft-DTW.

        Args:
            gamma: Smoothing parameter (smaller = closer to hard DTW).
        """
        self.gamma = gamma

    def compute_distance(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> float:
        """Compute Soft-DTW distance.

        Args:
            sequence1: First time series (n_frames, n_features).
            sequence2: Second time series (m_frames, n_features).

        Returns:
            Soft-DTW distance.
        """
        if sequence1.ndim == 1:
            sequence1 = sequence1.reshape(-1, 1)
        if sequence2.ndim == 1:
            sequence2 = sequence2.reshape(-1, 1)

        n, m = len(sequence1), len(sequence2)

        # Compute pairwise distance matrix
        dist_matrix = self._pairwise_distances(sequence1, sequence2)

        # Initialize DP matrix
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0

        # Fill DP matrix with soft-min
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i - 1, j - 1]
                dp[i, j] = cost + self._soft_min(
                    dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]
                )

        return float(dp[n, m])

    def _soft_min(self, a: float, b: float, c: float) -> float:
        """Compute soft minimum using log-sum-exp trick.

        Args:
            a, b, c: Values to compute soft-min.

        Returns:
            Soft minimum value.
        """
        # Soft-min: -gamma * log(sum(exp(-values/gamma)))
        values = np.array([a, b, c])
        max_val = np.max(values)

        # Numerical stability
        exp_sum = np.sum(np.exp(-(values - max_val) / self.gamma))
        soft_min = -self.gamma * np.log(exp_sum) + max_val

        return float(soft_min)

    def _pairwise_distances(
        self, sequence1: np.ndarray, sequence2: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise Euclidean distances.

        Args:
            sequence1: First sequence.
            sequence2: Second sequence.

        Returns:
            Distance matrix (n, m).
        """
        n, m = len(sequence1), len(sequence2)
        dist_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                dist_matrix[i, j] = euclidean(sequence1[i], sequence2[j])

        return dist_matrix


class LCSS:
    """Longest Common Subsequence for time series."""

    def __init__(self, epsilon: float = 0.5, delta: int = 5):
        """Initialize LCSS.

        Args:
            epsilon: Distance threshold for point matching.
            delta: Time constraint for matching.
        """
        self.epsilon = epsilon
        self.delta = delta

    def compute_similarity(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
        normalize: bool = True,
    ) -> float:
        """Compute LCSS similarity.

        Args:
            sequence1: First time series.
            sequence2: Second time series.
            normalize: Whether to normalize by length.

        Returns:
            LCSS similarity (higher = more similar).
        """
        if sequence1.ndim == 1:
            sequence1 = sequence1.reshape(-1, 1)
        if sequence2.ndim == 1:
            sequence2 = sequence2.reshape(-1, 1)

        n, m = len(sequence1), len(sequence2)

        # Initialize DP matrix
        dp = np.zeros((n + 1, m + 1))

        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Check time constraint
                if abs(i - j) <= self.delta:
                    # Check distance threshold
                    dist = euclidean(sequence1[i - 1], sequence2[j - 1])
                    if dist < self.epsilon:
                        dp[i, j] = dp[i - 1, j - 1] + 1
                    else:
                        dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

        lcss_length = dp[n, m]

        # Normalize by minimum length
        if normalize:
            lcss_length = lcss_length / min(n, m)

        return float(lcss_length)

    def compute_distance(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> float:
        """Compute LCSS distance (inverse of similarity).

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            LCSS distance (0 = identical).
        """
        similarity = self.compute_similarity(sequence1, sequence2, normalize=True)
        return 1.0 - similarity


class CosineSimilarityAnalyzer:
    """Cosine similarity for angle sequences."""

    @staticmethod
    def compute_similarity(
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two sequences.

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            Cosine similarity (-1 to 1, 1 = identical direction).
        """
        # Flatten if multidimensional
        vec1 = sequence1.flatten()
        vec2 = sequence2.flatten()

        # Pad shorter sequence with zeros
        if len(vec1) < len(vec2):
            vec1 = np.pad(vec1, (0, len(vec2) - len(vec1)))
        elif len(vec2) < len(vec1):
            vec2 = np.pad(vec2, (0, len(vec1) - len(vec2)))

        # Compute cosine similarity (1 - cosine distance)
        similarity = 1.0 - cosine(vec1, vec2)

        return float(similarity)

    @staticmethod
    def compute_distance(
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> float:
        """Compute cosine distance.

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            Cosine distance (0 = identical).
        """
        similarity = CosineSimilarityAnalyzer.compute_similarity(sequence1, sequence2)
        return (1.0 - similarity) / 2.0  # Normalize to [0, 1]


class CrossCorrelationAnalyzer:
    """Cross-correlation for phase alignment detection."""

    @staticmethod
    def compute_correlation(
        sequence1: np.ndarray,
        sequence2: np.ndarray,
        mode: str = "valid",
    ) -> np.ndarray:
        """Compute cross-correlation between sequences.

        Args:
            sequence1: First time series.
            sequence2: Second time series.
            mode: Correlation mode ('valid', 'same', 'full').

        Returns:
            Cross-correlation array.
        """
        # Ensure 1D
        seq1 = sequence1.flatten()
        seq2 = sequence2.flatten()

        # Normalize sequences (zero mean, unit variance)
        seq1 = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-8)
        seq2 = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-8)

        # Compute cross-correlation
        correlation = correlate(seq1, seq2, mode=mode)

        return correlation

    @staticmethod
    def find_best_alignment(
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> Tuple[int, float]:
        """Find optimal time shift for alignment.

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            Tuple of (optimal_shift, max_correlation).
        """
        correlation = CrossCorrelationAnalyzer.compute_correlation(
            sequence1, sequence2, mode="full"
        )

        # Find maximum correlation
        max_idx = np.argmax(correlation)
        max_corr = correlation[max_idx]

        # Convert to shift relative to center
        center = len(correlation) // 2
        optimal_shift = max_idx - center

        return int(optimal_shift), float(max_corr)

    @staticmethod
    def compute_max_correlation(
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> float:
        """Compute maximum cross-correlation value.

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            Maximum correlation coefficient (0-1).
        """
        _, max_corr = CrossCorrelationAnalyzer.find_best_alignment(
            sequence1, sequence2
        )

        # Normalize to [0, 1]
        return (max_corr + 1.0) / 2.0


class EuclideanDistanceAnalyzer:
    """Point-to-point Euclidean distance (baseline)."""

    @staticmethod
    def compute_distance(
        sequence1: np.ndarray,
        sequence2: np.ndarray,
        normalize: bool = True,
    ) -> float:
        """Compute point-to-point Euclidean distance.

        Args:
            sequence1: First time series.
            sequence2: Second time series.
            normalize: Whether to normalize by length.

        Returns:
            Euclidean distance.
        """
        # Pad shorter sequence
        len1, len2 = len(sequence1), len(sequence2)
        max_len = max(len1, len2)

        if len1 < max_len:
            padding = np.zeros((max_len - len1, *sequence1.shape[1:]))
            sequence1 = np.vstack([sequence1, padding])
        if len2 < max_len:
            padding = np.zeros((max_len - len2, *sequence2.shape[1:]))
            sequence2 = np.vstack([sequence2, padding])

        # Compute Euclidean distance
        distance = np.linalg.norm(sequence1 - sequence2)

        if normalize:
            distance = distance / max_len

        return float(distance)
