"""
Search Module - Core Search Engine
==================================

Foundation module providing vector search, embeddings, retrieval, and optimization.
Other modules (RAG, Knowledge, NL-to-SQL, CodeGraph) depend on this.

Usage:
    from modules.search import SearchService, SearchConfig
    from modules.search import ContextOptimizer, ContextItem
    from modules.search import EnhancedVectorStore
    from modules.search import ContextRetrievalEngine, ContextType, RetrievalStrategy
"""

# Service layer
from .service import SearchService, SearchResult, create_search_service
from .config import SearchConfig

# Vector store
from .vector_store.store import (
    EnhancedVectorStore,
    VectorDocument,
    SearchFilter,
    SearchResult as VectorSearchResult,
    SearchMode,
    RankingStrategy,
)

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

# Mathematical foundations (from shared.math)
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

# Retrieval
from .retrieval.context_retrieval_engine import (
    ContextRetrievalEngine,
    ContextType,
    RetrievalStrategy,
    ContextQuery,
    ContextPiece,
    RetrievalResult,
)

__all__ = [
    # Service
    "SearchService",
    "SearchResult",
    "SearchConfig",
    "create_search_service",
    
    # Vector Store
    "EnhancedVectorStore",
    "VectorDocument",
    "SearchFilter",
    "VectorSearchResult",
    "SearchMode",
    "RankingStrategy",
    
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
    
    # Retrieval
    "ContextRetrievalEngine",
    "ContextType",
    "RetrievalStrategy",
    "ContextQuery",
    "ContextPiece",
    "RetrievalResult",
]
