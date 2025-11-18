"""Unit tests for DTW analyzer."""

import numpy as np
import pytest

from src.analysis.dtw_analyzer import DTWAnalyzer


class TestDTWAnalyzer:
    """Test DTW analysis functionality."""

    def test_compute_distance_identical_sequences(self):
        """Test DTW distance for identical sequences."""
        analyzer = DTWAnalyzer()

        seq = np.array([[1, 2], [3, 4], [5, 6]])
        distance = analyzer.compute_distance(seq, seq, normalize=True)

        assert distance == 0.0

    def test_compute_distance_different_sequences(self):
        """Test DTW distance for different sequences."""
        analyzer = DTWAnalyzer()

        seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        seq2 = np.array([[2, 3], [4, 5], [6, 7]])

        distance = analyzer.compute_distance(seq1, seq2, normalize=True)

        assert distance > 0.0

    def test_compute_distance_1d_sequences(self):
        """Test DTW with 1D sequences."""
        analyzer = DTWAnalyzer()

        seq1 = np.array([1, 2, 3, 4, 5])
        seq2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        distance = analyzer.compute_distance(seq1, seq2, normalize=True)

        assert distance < 1.0  # Should be small for similar sequences

    def test_compute_distance_different_lengths(self):
        """Test DTW handles different length sequences."""
        analyzer = DTWAnalyzer()

        seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        seq2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        distance = analyzer.compute_distance(seq1, seq2, normalize=True)

        assert distance >= 0.0

    def test_warping_path(self):
        """Test warping path computation."""
        analyzer = DTWAnalyzer()

        seq1 = np.array([1, 2, 3])
        seq2 = np.array([1, 2, 3, 4])

        path = analyzer.get_warping_path(seq1, seq2)

        assert len(path) > 0
        assert all(isinstance(p, tuple) for p in path)

    def test_compute_barycenter(self):
        """Test DTW barycenter averaging."""
        analyzer = DTWAnalyzer()

        sequences = [
            np.array([[1, 1], [2, 2], [3, 3]]),
            np.array([[1.1, 1.1], [2.1, 2.1], [3.1, 3.1]]),
            np.array([[0.9, 0.9], [1.9, 1.9], [2.9, 2.9]]),
        ]

        barycenter = analyzer.compute_barycenter(sequences, max_iter=5)

        assert barycenter.shape[0] > 0
        assert barycenter.shape[1] == 2

    def test_similarity_score_conversion(self):
        """Test distance to similarity score conversion."""
        analyzer = DTWAnalyzer()

        # Zero distance = 100% similarity
        assert analyzer.compute_similarity_score(0.0, max_distance=10.0) == 100.0

        # Max distance = 0% similarity
        assert analyzer.compute_similarity_score(10.0, max_distance=10.0) == 0.0

        # Half distance = 50% similarity
        score = analyzer.compute_similarity_score(5.0, max_distance=10.0)
        assert 49.0 < score < 51.0

    def test_align_sequences(self):
        """Test sequence alignment."""
        analyzer = DTWAnalyzer()

        seq1 = np.array([[1, 1], [2, 2], [3, 3]])
        seq2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

        aligned1, aligned2 = analyzer.align_sequences(seq1, seq2)

        # Aligned sequences should have same length
        assert len(aligned1) == len(aligned2)
