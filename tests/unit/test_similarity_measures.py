"""Unit tests for similarity measures."""

import numpy as np

from src.analysis.similarity_measures import (
    LCSS,
    CosineSimilarityAnalyzer,
    CrossCorrelationAnalyzer,
    EuclideanDistanceAnalyzer,
    SoftDTW,
)


class TestSoftDTW:
    """Test Soft-DTW implementation."""

    def test_identical_sequences(self):
        """Test Soft-DTW for identical sequences."""
        soft_dtw = SoftDTW(gamma=1.0)

        seq = np.array([[1, 2], [3, 4], [5, 6]])
        distance = soft_dtw.compute_distance(seq, seq)

        assert distance >= 0.0  # Should be close to 0 but not exactly due to smoothing

    def test_different_sequences(self):
        """Test Soft-DTW for different sequences."""
        soft_dtw = SoftDTW(gamma=1.0)

        seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        seq2 = np.array([[10, 20], [30, 40], [50, 60]])

        distance = soft_dtw.compute_distance(seq1, seq2)

        assert distance > 0.0


class TestLCSS:
    """Test LCSS implementation."""

    def test_identical_sequences(self):
        """Test LCSS for identical sequences."""
        lcss = LCSS(epsilon=0.5, delta=5)

        seq = np.array([[1, 2], [3, 4], [5, 6]])
        similarity = lcss.compute_similarity(seq, seq, normalize=True)

        assert similarity == 1.0  # Perfect match

    def test_similar_sequences(self):
        """Test LCSS for similar sequences."""
        lcss = LCSS(epsilon=1.0, delta=5)

        seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        seq2 = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])

        similarity = lcss.compute_similarity(seq1, seq2, normalize=True)

        assert 0.0 <= similarity <= 1.0

    def test_distance_conversion(self):
        """Test LCSS distance is inverse of similarity."""
        lcss = LCSS(epsilon=0.5, delta=5)

        seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        seq2 = np.array([[2, 3], [4, 5], [6, 7]])

        similarity = lcss.compute_similarity(seq1, seq2, normalize=True)
        distance = lcss.compute_distance(seq1, seq2)

        assert abs(similarity + distance - 1.0) < 0.01


class TestCosineSimilarity:
    """Test cosine similarity."""

    def test_identical_sequences(self):
        """Test cosine similarity for identical sequences."""
        seq = np.array([1, 2, 3, 4, 5])

        similarity = CosineSimilarityAnalyzer.compute_similarity(seq, seq)

        assert abs(similarity - 1.0) < 0.01

    def test_opposite_sequences(self):
        """Test cosine similarity for opposite sequences."""
        seq1 = np.array([1, 2, 3, 4, 5])
        seq2 = np.array([-1, -2, -3, -4, -5])

        similarity = CosineSimilarityAnalyzer.compute_similarity(seq1, seq2)

        assert abs(similarity + 1.0) < 0.01  # Should be close to -1

    def test_orthogonal_sequences(self):
        """Test cosine similarity for orthogonal sequences."""
        seq1 = np.array([1, 0])
        seq2 = np.array([0, 1])

        similarity = CosineSimilarityAnalyzer.compute_similarity(seq1, seq2)

        assert abs(similarity) < 0.1  # Should be close to 0


class TestCrossCorrelation:
    """Test cross-correlation analysis."""

    def test_identical_sequences(self):
        """Test cross-correlation for identical sequences."""
        seq = np.array([1, 2, 3, 4, 5])

        shift, corr = CrossCorrelationAnalyzer.find_best_alignment(seq, seq)

        assert shift == 0  # Should have zero shift
        assert corr > 0.9  # High correlation

    def test_shifted_sequences(self):
        """Test cross-correlation detects shifts."""
        seq1 = np.array([0, 0, 1, 2, 3, 4, 5, 0, 0])
        seq2 = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0])

        shift, corr = CrossCorrelationAnalyzer.find_best_alignment(seq1, seq2)

        assert shift != 0  # Should detect shift

    def test_max_correlation(self):
        """Test maximum correlation computation."""
        seq1 = np.array([1, 2, 3, 4, 5])
        seq2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        max_corr = CrossCorrelationAnalyzer.compute_max_correlation(seq1, seq2)

        assert 0.0 <= max_corr <= 1.0
        assert max_corr > 0.9  # Should be high for similar sequences


class TestEuclideanDistance:
    """Test Euclidean distance."""

    def test_identical_sequences(self):
        """Test Euclidean distance for identical sequences."""
        seq = np.array([[1, 2], [3, 4], [5, 6]])

        distance = EuclideanDistanceAnalyzer.compute_distance(seq, seq, normalize=True)

        assert distance == 0.0

    def test_different_sequences(self):
        """Test Euclidean distance for different sequences."""
        seq1 = np.array([[1, 2], [3, 4], [5, 6]])
        seq2 = np.array([[2, 3], [4, 5], [6, 7]])

        distance = EuclideanDistanceAnalyzer.compute_distance(seq1, seq2, normalize=True)

        assert distance > 0.0

    def test_different_lengths(self):
        """Test Euclidean distance handles different lengths."""
        seq1 = np.array([[1, 2], [3, 4]])
        seq2 = np.array([[1, 2], [3, 4], [5, 6]])

        distance = EuclideanDistanceAnalyzer.compute_distance(seq1, seq2, normalize=True)

        assert distance >= 0.0
