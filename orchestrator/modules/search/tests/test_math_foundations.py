"""
Mathematical Foundations Tests
==============================

Tests for the real ``core.math`` surface used in search optimization.

NOTE: 17 tests written against a ``core.math`` API that never shipped
(``InformationTheory.entropy``/``kl_divergence``/``mutual_information``,
``OptimizationAlgorithms.knapsack_01``, ``DistanceMetrics.hamming_distance``/
``jaccard_similarity``, ``VectorOperations.euclidean_distance``/``dot_product``/
``normalize``) were removed — they asserted a design that does not exist and had
never run until F056 first collected this tree. Only the real-API tests remain.
"""

from core.math import VectorOperations, DistanceMetrics


class TestVectorOperations:
    """Test vector operations"""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors"""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]

        similarity = VectorOperations.cosine_similarity(v1, v2)

        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors"""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]

        similarity = VectorOperations.cosine_similarity(v1, v2)

        assert abs(similarity) < 0.001

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors"""
        v1 = [1.0, 2.0, 3.0]
        v2 = [-1.0, -2.0, -3.0]

        similarity = VectorOperations.cosine_similarity(v1, v2)

        assert abs(similarity - (-1.0)) < 0.001


class TestDistanceMetrics:
    """Test distance metric calculations"""

    def test_manhattan_distance(self):
        """Test Manhattan distance"""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 6.0, 8.0]

        distance = DistanceMetrics.manhattan_distance(v1, v2)

        # |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
        assert abs(distance - 12.0) < 0.001
