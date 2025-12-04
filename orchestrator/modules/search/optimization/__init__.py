"""
Search Optimization Module
==========================

Mathematical optimization algorithms for context selection and search.

Includes:
- Context Optimizer (knapsack, MMR, entropy-based selection)
- Information Theory (Shannon entropy, mutual information)
- Vector Operations (cosine similarity, normalization)
- Statistical Analysis (trends, anomalies)
- Graph Theory (dependency analysis)
- Probability Theory (confidence intervals)
- Distance Metrics (cosine, euclidean, manhattan)
- Optimization Algorithms (gradient descent, simulated annealing)
"""

from .context_optimizer import (
    ContextOptimizer,
    ContextItem,
    Example,
    OptimizedContext,
    AtomicPrompt,
    EnhancedPrompt,
    create_context_optimizer,
    optimize_prompt_context,
)

# Re-export mathematical foundations from shared.math for backward compatibility
from shared.math import (
    InformationTheory,
    VectorOperations,
    OptimizationAlgorithms,
    DistanceMetrics,
    GraphTheory,
    ProbabilityTheory,
    ConfidenceInterval,
    StatisticalAnalysis,
    TrendAnalysis,
)

__all__ = [
    # Context Optimizer
    "ContextOptimizer",
    "ContextItem",
    "Example",
    "OptimizedContext",
    "AtomicPrompt",
    "EnhancedPrompt",
    "create_context_optimizer",
    "optimize_prompt_context",
    # Mathematical Foundations
    "InformationTheory",
    "VectorOperations",
    "OptimizationAlgorithms",
    "DistanceMetrics",
    "GraphTheory",
    "ProbabilityTheory",
    "ConfidenceInterval",
    "StatisticalAnalysis",
    "TrendAnalysis",
]
