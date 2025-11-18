"""Dynamic Time Warping (DTW) analysis for stroke comparison."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_ndim

from src.utils.config import load_analysis_config


class DTWAnalyzer:
    """Dynamic Time Warping analyzer for comparing swimming strokes."""

    def __init__(
        self,
        window: Optional[int] = None,
        use_c: bool = True,
    ):
        """Initialize DTW analyzer.

        Args:
            window: Sakoe-Chiba band width (None = no constraint).
            use_c: Use C-optimized DTW implementation for speed.
        """
        # Load configuration
        config = load_analysis_config()
        dtw_config = config.get("dtw", {})

        self.window = window if window is not None else dtw_config.get("window_size", 10)
        self.use_c = use_c

    def compute_distance(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
        normalize: bool = True,
    ) -> float:
        """Compute DTW distance between two sequences.

        Args:
            sequence1: First time series (shape: [n_frames, n_features]).
            sequence2: Second time series (shape: [m_frames, n_features]).
            normalize: Whether to normalize by path length.

        Returns:
            DTW distance.
        """
        # Ensure 2D arrays
        if sequence1.ndim == 1:
            sequence1 = sequence1.reshape(-1, 1)
        if sequence2.ndim == 1:
            sequence2 = sequence2.reshape(-1, 1)

        # Use multidimensional DTW if more than 1 feature
        if sequence1.shape[1] > 1:
            distance = dtw_ndim.distance(
                sequence1,
                sequence2,
                window=self.window,
                use_c=self.use_c,
            )
        else:
            # Use 1D DTW (faster)
            distance = dtw.distance(
                sequence1.flatten(),
                sequence2.flatten(),
                window=self.window,
                use_c=self.use_c,
            )

        # Normalize by path length
        if normalize:
            max_len = max(len(sequence1), len(sequence2))
            distance = distance / max_len

        return float(distance)

    def compute_distance_fast(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> float:
        """Compute DTW distance using fast approximation.

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            Approximate DTW distance.
        """
        if sequence1.ndim == 1:
            sequence1 = sequence1.reshape(-1, 1)
        if sequence2.ndim == 1:
            sequence2 = sequence2.reshape(-1, 1)

        if sequence1.shape[1] > 1:
            distance = dtw_ndim.distance_fast(
                sequence1,
                sequence2,
                window=self.window,
                use_c=self.use_c,
            )
        else:
            distance = dtw.distance_fast(
                sequence1.flatten(),
                sequence2.flatten(),
                window=self.window,
                use_c=self.use_c,
            )

        return float(distance)

    def get_warping_path(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Get the optimal warping path between two sequences.

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            List of (index1, index2) tuples representing the warping path.
        """
        if sequence1.ndim == 1:
            sequence1 = sequence1.reshape(-1, 1)
        if sequence2.ndim == 1:
            sequence2 = sequence2.reshape(-1, 1)

        if sequence1.shape[1] > 1:
            path = dtw_ndim.warping_path(
                sequence1,
                sequence2,
                window=self.window,
            )
        else:
            path = dtw.warping_path(
                sequence1.flatten(),
                sequence2.flatten(),
                window=self.window,
            )

        return path

    def compare_strokes(
        self,
        stroke1: Dict[str, np.ndarray],
        stroke2: Dict[str, np.ndarray],
        features: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compare two swimming strokes using DTW.

        Args:
            stroke1: First stroke data (dict of feature arrays).
            stroke2: Second stroke data (dict of feature arrays).
            features: List of features to compare (None = all common features).

        Returns:
            Dictionary of DTW distances for each feature.
        """
        if features is None:
            # Use all common features
            features = list(set(stroke1.keys()) & set(stroke2.keys()))

        distances = {}
        for feature in features:
            if feature in stroke1 and feature in stroke2:
                distances[feature] = self.compute_distance(
                    stroke1[feature],
                    stroke2[feature],
                    normalize=True,
                )

        return distances

    def compute_barycenter(
        self,
        sequences: List[np.ndarray],
        max_iter: int = 10,
    ) -> np.ndarray:
        """Compute DTW barycenter averaging (DBA) for a set of sequences.

        This creates an "average" sequence that represents the typical pattern.

        Args:
            sequences: List of time series to average.
            max_iter: Maximum number of DBA iterations.

        Returns:
            Barycenter sequence.
        """
        # Ensure all sequences are 2D
        sequences = [
            seq.reshape(-1, 1) if seq.ndim == 1 else seq for seq in sequences
        ]

        # Use the first sequence as initial barycenter
        barycenter = sequences[0].copy()

        for iteration in range(max_iter):
            # Align all sequences to current barycenter
            associations = [[] for _ in range(len(barycenter))]

            for seq in sequences:
                path = self.get_warping_path(barycenter, seq)

                for idx_bary, idx_seq in path:
                    if idx_bary < len(barycenter):
                        associations[idx_bary].append(seq[idx_seq])

            # Update barycenter by averaging aligned points
            new_barycenter = np.zeros_like(barycenter)
            for i, assoc in enumerate(associations):
                if len(assoc) > 0:
                    new_barycenter[i] = np.mean(assoc, axis=0)
                else:
                    new_barycenter[i] = barycenter[i]

            # Check convergence
            diff = np.linalg.norm(new_barycenter - barycenter)
            barycenter = new_barycenter

            if diff < 1e-4:
                break

        return barycenter

    def compute_similarity_score(
        self,
        distance: float,
        max_distance: float = 10.0,
    ) -> float:
        """Convert DTW distance to similarity score (0-100).

        Args:
            distance: DTW distance.
            max_distance: Maximum expected distance (for normalization).

        Returns:
            Similarity score (0 = completely different, 100 = identical).
        """
        # Clip distance to max
        distance = min(distance, max_distance)

        # Convert to similarity score (inverse)
        similarity = (1.0 - distance / max_distance) * 100.0

        return max(0.0, min(100.0, similarity))

    def align_sequences(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align two sequences using DTW warping path.

        Args:
            sequence1: First time series.
            sequence2: Second time series.

        Returns:
            Tuple of aligned (sequence1, sequence2) with same length.
        """
        path = self.get_warping_path(sequence1, sequence2)

        aligned1 = []
        aligned2 = []

        for idx1, idx2 in path:
            if idx1 < len(sequence1) and idx2 < len(sequence2):
                aligned1.append(sequence1[idx1])
                aligned2.append(sequence2[idx2])

        return np.array(aligned1), np.array(aligned2)
