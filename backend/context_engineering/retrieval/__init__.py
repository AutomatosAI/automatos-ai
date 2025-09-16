
"""
Context Engineering Retrieval Module
====================================

Advanced context retrieval system with multiple retrieval strategies,
intelligent ranking, and multi-modal content processing.
"""

from .vector_store_enhanced import (
    EnhancedVectorStore,
    VectorDocument, 
    SearchResult,
    SearchMode,
    RankingStrategy,
    SearchFilter
)

from .context_retrieval_engine import (
    ContextRetrievalEngine,
    ContextQuery,
    ContextPiece,
    RetrievalResult,
    ContextType,
    RetrievalStrategy
)

from .query_processor import (
    QueryProcessor,
    ProcessedQuery,
    QueryIntent,
    QueryComplexity,
    QueryEntity,
    QueryConcept
)

from .multimodal_processor import (
    MultiModalProcessor,
    ProcessedContent,
    ContentModality,
    CodeLanguage,
    CodeAnalysis,
    StructuredDataAnalysis
)

__all__ = [
    # Vector Store
    'EnhancedVectorStore',
    'VectorDocument',
    'SearchResult', 
    'SearchMode',
    'RankingStrategy',
    'SearchFilter',
    
    # Context Retrieval Engine
    'ContextRetrievalEngine',
    'ContextQuery',
    'ContextPiece',
    'RetrievalResult',
    'ContextType',
    'RetrievalStrategy',
    
    # Query Processor
    'QueryProcessor',
    'ProcessedQuery',
    'QueryIntent',
    'QueryComplexity',
    'QueryEntity',
    'QueryConcept',
    
    # Multi-Modal Processor
    'MultiModalProcessor',
    'ProcessedContent',
    'ContentModality',
    'CodeLanguage',
    'CodeAnalysis',
    'StructuredDataAnalysis'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Automatos AI Context Engineering Team'
