"""
Search Module - Optimization & Vector Backends
==============================================

Foundation module providing context optimization, mathematical foundations,
and the pluggable vector-store backends (``modules.search.vector_store``).

PRD-197 S1: the F079 zombie layer (``SearchService`` / ``EnhancedVectorStore``
/ ``ContextRetrievalEngine``) is deleted — it was a parallel retrieval stack
the live path never imported, whose "cosine" ranking used the L2 operator and
whose namesake table was dropped in PRD-135. The live document plane is
``S3VectorsBackend`` via ``modules.rag.service``; local/OSS documents go
through the pgvector-local backend (PRD-197 S5). Do not resurrect.

Usage:
    from modules.search import ContextOptimizer, ContextItem
    from modules.search.vector_store import get_vector_store
"""

# Optimization
from .optimization.context_optimizer import (
    ContextOptimizer,
    ContextItem,
    Example,
    OptimizedContext,
    AtomicPrompt,
    EnhancedPrompt,
    create_context_optimizer,
    optimize_prompt_context,
)

# Mathematical foundations (from core.math)
from core.math import (
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
    # Optimization
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
