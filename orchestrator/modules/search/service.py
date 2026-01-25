"""
Search Service
==============

Unified search service that orchestrates vector store, retrieval, and optimization.
This is the main entry point for search operations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import SearchConfig
from .vector_store.store import EnhancedVectorStore, VectorDocument, SearchMode, RankingStrategy
from .optimization.context_optimizer import ContextOptimizer, ContextItem, OptimizedContext
from .retrieval.context_retrieval_engine import (
    ContextRetrievalEngine, ContextQuery, RetrievalResult, RetrievalStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result"""
    chunks: List[Dict[str, Any]]
    total_tokens: int
    sources: List[str]
    metrics: Dict[str, float]


class SearchService:
    """
    Unified Search Service
    
    Provides a single interface for all search operations:
    - Vector similarity search
    - Context retrieval
    - Context optimization
    
    Usage:
        from modules.search import SearchService
        
        service = SearchService()
        result = await service.search("How do agents work?", max_results=5)
    """
    
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig.from_env()
        
        # Lazy initialization
        self._vector_store: Optional[EnhancedVectorStore] = None
        self._optimizer: Optional[ContextOptimizer] = None
        self._retrieval_engine: Optional[ContextRetrievalEngine] = None
    
    @property
    def vector_store(self) -> EnhancedVectorStore:
        """Lazy initialization of vector store"""
        if self._vector_store is None:
            self._vector_store = EnhancedVectorStore(
                database_url=self.config.database_url,
                embedding_dimension=self.config.embedding_dimension,
                similarity_function=self.config.similarity_function,
                table_name=self.config.vector_table_name
            )
        return self._vector_store
    
    @property
    def optimizer(self) -> ContextOptimizer:
        """Lazy initialization of optimizer"""
        if self._optimizer is None:
            self._optimizer = ContextOptimizer()
        return self._optimizer
    
    async def search(
        self,
        query: str,
        max_results: int = None,
        max_tokens: int = None,
        diversity: float = None,
        search_mode: SearchMode = SearchMode.HYBRID,
        optimize: bool = True
    ) -> SearchResult:
        """
        Perform a search with optional optimization.
        
        Args:
            query: Search query text
            max_results: Maximum number of results
            max_tokens: Maximum token budget for results
            diversity: Diversity factor (0=relevance, 1=diversity)
            search_mode: Vector search mode
            optimize: Whether to apply context optimization
            
        Returns:
            SearchResult with chunks, tokens, sources, and metrics
        """
        max_results = max_results or self.config.default_max_results
        max_tokens = max_tokens or 4000
        diversity = diversity or self.config.mmr_lambda
        
        try:
            # Get embedding for query
            query_embedding = await self.optimizer.embedding_manager.generate_embedding(query)
            
            # Perform vector search
            await self.vector_store.initialize()
            
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                mode=search_mode,
                ranking_strategy=RankingStrategy.RELEVANCE,
                limit=max_results * 2,  # Get more for optimization
                query_text=query
            )
            
            if not search_results:
                return SearchResult(
                    chunks=[],
                    total_tokens=0,
                    sources=[],
                    metrics={'diversity_score': 0.0, 'information_gain': 0.0}
                )
            
            # Convert to ContextItems for optimization
            context_items = [
                ContextItem(
                    content=r.document.content,
                    source=r.document.source,
                    context_type=r.document.document_type or "general",
                    relevance_score=r.final_score,
                    embedding=r.document.embedding,
                    metadata=r.document.metadata
                )
                for r in search_results
            ]
            
            if optimize:
                # Apply context optimization
                objective = "maximize_diversity" if diversity > 0.6 else (
                    "maximize_information" if diversity < 0.3 else "balanced"
                )
                
                optimized = await self.optimizer.optimize_context(
                    available_context=context_items,
                    max_tokens=max_tokens,
                    objective=objective
                )
                
                return SearchResult(
                    chunks=[
                        {
                            'content': c.content,
                            'source': c.source,
                            'type': c.context_type,
                            'relevance': c.relevance_score,
                            'tokens': c.token_count
                        }
                        for c in optimized.contexts
                    ],
                    total_tokens=optimized.total_tokens,
                    sources=list(set(c.source for c in optimized.contexts)),
                    metrics={
                        'diversity_score': optimized.diversity_score,
                        'information_gain': optimized.information_gain,
                        'entropy_reduction': optimized.entropy_reduction,
                        'token_utilization': optimized.optimization_metadata.get('token_utilization', 0)
                    }
                )
            else:
                # Return unoptimized results
                chunks = [
                    {
                        'content': c.content,
                        'source': c.source,
                        'type': c.context_type,
                        'relevance': c.relevance_score,
                        'tokens': c.token_count
                    }
                    for c in context_items[:max_results]
                ]
                
                return SearchResult(
                    chunks=chunks,
                    total_tokens=sum(c.token_count for c in context_items[:max_results]),
                    sources=list(set(c.source for c in context_items[:max_results])),
                    metrics={'diversity_score': 0.0, 'information_gain': 0.0}
                )
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResult(
                chunks=[],
                total_tokens=0,
                sources=[],
                metrics={'error': str(e)}
            )
    
    async def close(self):
        """Close connections"""
        if self._vector_store:
            await self._vector_store.close()


# Factory function for easy usage
def create_search_service(config: SearchConfig = None) -> SearchService:
    """Create a search service instance"""
    return SearchService(config)

