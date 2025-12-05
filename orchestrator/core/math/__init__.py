"""
Shared Mathematical Foundations
===============================

Common mathematical algorithms used across all modules:
- Information Theory (entropy, mutual information)
- Vector Operations (cosine similarity, normalization)
- Statistical Analysis (trends, anomalies, correlations)
- Graph Theory (dependencies, topological sort)
- Probability Theory (confidence intervals, Bayesian updates)
- Distance Metrics (cosine, euclidean, manhattan)
- Optimization Algorithms (gradient descent, simulated annealing)

Usage:
    from core.math import InformationTheory, VectorOperations, GraphTheory
    
    entropy = InformationTheory.calculate_entropy(text)
    similarity = VectorOperations.cosine_similarity(v1, v2)
"""

from .information_theory import InformationTheory
from .vector_operations import VectorOperations
from .statistical_analysis import StatisticalAnalysis, TrendAnalysis
from .graph_theory import GraphTheory
from .probability_theory import ProbabilityTheory, ConfidenceInterval
from .distance_metrics import DistanceMetrics
from .optimization_algorithms import OptimizationAlgorithms

__all__ = [
    "InformationTheory",
    "VectorOperations",
    "StatisticalAnalysis",
    "TrendAnalysis",
    "GraphTheory",
    "ProbabilityTheory",
    "ConfidenceInterval",
    "DistanceMetrics",
    "OptimizationAlgorithms",
]

