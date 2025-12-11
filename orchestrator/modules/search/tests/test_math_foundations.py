"""
Mathematical Foundations Tests
==============================

Tests for core mathematical algorithms used in search optimization.
Verifies information theory, vector operations, and optimization correctness.
"""

import pytest
import numpy as np
from typing import List

from core.math import (
    InformationTheory,
    VectorOperations,
    OptimizationAlgorithms,
    DistanceMetrics
)


class TestInformationTheory:
    """Test information theory calculations"""
    
    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution"""
        # Uniform distribution should have maximum entropy
        probs = [0.25, 0.25, 0.25, 0.25]
        entropy = InformationTheory.entropy(probs)
        
        # log2(4) = 2.0
        assert abs(entropy - 2.0) < 0.001
    
    def test_entropy_certain_distribution(self):
        """Test entropy of certain distribution"""
        # Certain outcome should have zero entropy
        probs = [1.0, 0.0, 0.0, 0.0]
        entropy = InformationTheory.entropy(probs)
        
        assert entropy == 0.0
    
    def test_entropy_skewed_distribution(self):
        """Test entropy of skewed distribution"""
        probs = [0.7, 0.2, 0.05, 0.05]
        entropy = InformationTheory.entropy(probs)
        
        # Should be between 0 and log2(4)
        assert 0 < entropy < 2.0
    
    def test_kl_divergence_identical(self):
        """Test KL divergence of identical distributions"""
        p = [0.25, 0.25, 0.25, 0.25]
        q = [0.25, 0.25, 0.25, 0.25]
        
        kl = InformationTheory.kl_divergence(p, q)
        
        # KL(P||P) = 0
        assert abs(kl) < 0.001
    
    def test_kl_divergence_different(self):
        """Test KL divergence of different distributions"""
        p = [0.5, 0.3, 0.2]
        q = [0.3, 0.4, 0.3]
        
        kl = InformationTheory.kl_divergence(p, q)
        
        # Should be positive
        assert kl > 0
    
    def test_mutual_information_independent(self):
        """Test mutual information of independent variables"""
        # For independent variables, MI should be close to 0
        x = [0, 0, 1, 1, 0, 0, 1, 1]
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        
        mi = InformationTheory.mutual_information(x, y)
        
        # Should be close to 0 for independent variables
        assert abs(mi) < 0.1
    
    def test_mutual_information_identical(self):
        """Test mutual information of identical variables"""
        # MI(X, X) = H(X)
        x = [0, 0, 1, 1, 0, 1, 0, 1]
        
        mi = InformationTheory.mutual_information(x, x)
        
        # Should equal entropy of x
        probs = [0.5, 0.5]  # 4 zeros, 4 ones
        expected_entropy = InformationTheory.entropy(probs)
        
        assert abs(mi - expected_entropy) < 0.1


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
    
    def test_euclidean_distance_identical(self):
        """Test Euclidean distance of identical vectors"""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        
        distance = VectorOperations.euclidean_distance(v1, v2)
        
        assert abs(distance) < 0.001
    
    def test_euclidean_distance_known(self):
        """Test Euclidean distance with known result"""
        v1 = [0.0, 0.0, 0.0]
        v2 = [3.0, 4.0, 0.0]
        
        distance = VectorOperations.euclidean_distance(v1, v2)
        
        # 3-4-5 triangle
        assert abs(distance - 5.0) < 0.001
    
    def test_dot_product(self):
        """Test dot product calculation"""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        
        dot = VectorOperations.dot_product(v1, v2)
        
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert abs(dot - 32.0) < 0.001
    
    def test_normalize_vector(self):
        """Test vector normalization"""
        v = [3.0, 4.0, 0.0]
        
        normalized = VectorOperations.normalize(v)
        
        # Should have unit length
        length = np.linalg.norm(normalized)
        assert abs(length - 1.0) < 0.001
        
        # Direction should be preserved
        assert abs(normalized[0] - 0.6) < 0.001  # 3/5
        assert abs(normalized[1] - 0.8) < 0.001  # 4/5


class TestOptimizationAlgorithms:
    """Test optimization algorithms"""
    
    def test_knapsack_simple(self):
        """Test knapsack with simple case"""
        values = [60, 100, 120]
        weights = [10, 20, 30]
        capacity = 50
        
        selected_indices = OptimizationAlgorithms.knapsack_01(values, weights, capacity)
        
        # Should select items 1 and 2 (100 + 120 = 220 value, 20 + 30 = 50 weight)
        assert 1 in selected_indices
        assert 2 in selected_indices
        assert 0 not in selected_indices
    
    def test_knapsack_all_fit(self):
        """Test knapsack when all items fit"""
        values = [10, 20, 30]
        weights = [5, 10, 15]
        capacity = 100
        
        selected_indices = OptimizationAlgorithms.knapsack_01(values, weights, capacity)
        
        # All items should be selected
        assert len(selected_indices) == 3
    
    def test_knapsack_none_fit(self):
        """Test knapsack when no items fit"""
        values = [10, 20, 30]
        weights = [50, 60, 70]
        capacity = 10
        
        selected_indices = OptimizationAlgorithms.knapsack_01(values, weights, capacity)
        
        # No items should be selected
        assert len(selected_indices) == 0
    
    def test_knapsack_with_max_items(self):
        """Test knapsack with maximum items constraint"""
        values = [10, 20, 30, 40, 50]
        weights = [5, 10, 15, 20, 25]
        capacity = 100
        max_items = 3
        
        selected_indices = OptimizationAlgorithms.knapsack_01(
            values, weights, capacity, max_items=max_items
        )
        
        # Should select at most 3 items
        assert len(selected_indices) <= max_items
        
        # Should select highest value items that fit
        assert 4 in selected_indices  # 50 value
        assert 3 in selected_indices  # 40 value
        assert 2 in selected_indices  # 30 value


class TestDistanceMetrics:
    """Test distance metric calculations"""
    
    def test_manhattan_distance(self):
        """Test Manhattan distance"""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 6.0, 8.0]
        
        distance = DistanceMetrics.manhattan_distance(v1, v2)
        
        # |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
        assert abs(distance - 12.0) < 0.001
    
    def test_hamming_distance(self):
        """Test Hamming distance"""
        v1 = [1, 0, 1, 1, 0]
        v2 = [1, 1, 1, 0, 0]
        
        distance = DistanceMetrics.hamming_distance(v1, v2)
        
        # Differ in positions 1 and 3
        assert distance == 2
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity"""
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        
        similarity = DistanceMetrics.jaccard_similarity(set1, set2)
        
        # Intersection: {3, 4} = 2
        # Union: {1, 2, 3, 4, 5, 6} = 6
        # Jaccard = 2/6 = 0.333...
        assert abs(similarity - 0.333) < 0.01
