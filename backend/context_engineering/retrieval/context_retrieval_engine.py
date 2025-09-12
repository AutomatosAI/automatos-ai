
"""
Context Retrieval Engine
========================

Advanced context retrieval system with intelligent ranking, filtering,
and multi-modal context processing capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

# Import components
from .vector_store_enhanced import (
    EnhancedVectorStore, VectorDocument, SearchResult, 
    SearchMode, RankingStrategy, SearchFilter
)
from ..chunking.semantic_chunker import SemanticChunker, SemanticChunk, ChunkingStrategy
from ..mathematical_foundations.information_theory import InformationTheory
from ..mathematical_foundations.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context to retrieve"""
    BACKGROUND = "background"  # General background information
    SPECIFIC = "specific"      # Specific to the query
    HISTORICAL = "historical"  # Previous conversation history
    RELATED = "related"        # Related but not directly relevant
    FACTUAL = "factual"        # Factual information
    PROCEDURAL = "procedural"  # How-to information

class RetrievalStrategy(Enum):
    """Context retrieval strategies"""
    SIMILARITY_BASED = "similarity_based"
    MULTI_HOP = "multi_hop"           # Follow relationships
    HIERARCHICAL = "hierarchical"     # Consider document structure
    TEMPORAL = "temporal"             # Time-aware retrieval
    ADAPTIVE = "adaptive"             # Learn from feedback

@dataclass
class ContextQuery:
    """A context retrieval query"""
    text: str
    embedding: Optional[List[float]] = None
    context_types: List[ContextType] = None
    max_results: int = 10
    time_range: Optional[Tuple[datetime, datetime]] = None
    required_sources: Optional[List[str]] = None
    exclude_sources: Optional[List[str]] = None
    min_relevance: float = 0.5
    include_metadata: bool = True

@dataclass
class ContextPiece:
    """A piece of retrieved context"""
    content: str
    source: str
    context_type: ContextType
    relevance_score: float
    confidence_score: float
    metadata: Dict[str, Any]
    chunk_id: str
    relationships: List[str]
    timestamp: datetime

@dataclass
class RetrievalResult:
    """Complete context retrieval result"""
    query: ContextQuery
    contexts: List[ContextPiece]
    total_found: int
    execution_time_ms: float
    strategy_used: RetrievalStrategy
    quality_score: float  # Overall quality of retrieval
    diversity_score: float  # Diversity of sources/types

class ContextRetrievalEngine:
    """Advanced context retrieval engine"""
    
    def __init__(
        self,
        vector_store: EnhancedVectorStore,
        chunker: SemanticChunker,
        default_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY_BASED
    ):
        self.vector_store = vector_store
        self.chunker = chunker
        self.default_strategy = default_strategy
        
        # Mathematical components
        self.info_theory = InformationTheory()
        self.stats = StatisticalAnalysis()
        
        # Caching for performance
        self.query_cache = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Learning components (simple for now)
        self.feedback_history = []
        self.successful_queries = {}
        
    async def retrieve_context(
        self,
        query: ContextQuery,
        strategy: Optional[RetrievalStrategy] = None
    ) -> RetrievalResult:
        """Main context retrieval method"""
        
        start_time = datetime.now()
        strategy = strategy or self.default_strategy
        
        # Check cache first
        cache_key = self._generate_cache_key(query)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Returning cached result for query: {query.text[:50]}...")
            return cached_result
        
        try:
            # Execute retrieval strategy
            if strategy == RetrievalStrategy.SIMILARITY_BASED:
                contexts = await self._similarity_based_retrieval(query)
            elif strategy == RetrievalStrategy.MULTI_HOP:
                contexts = await self._multi_hop_retrieval(query)
            elif strategy == RetrievalStrategy.HIERARCHICAL:
                contexts = await self._hierarchical_retrieval(query)
            elif strategy == RetrievalStrategy.TEMPORAL:
                contexts = await self._temporal_retrieval(query)
            elif strategy == RetrievalStrategy.ADAPTIVE:
                contexts = await self._adaptive_retrieval(query)
            else:
                contexts = await self._similarity_based_retrieval(query)
            
            # Post-process results
            processed_contexts = await self._post_process_contexts(contexts, query)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(processed_contexts, query)
            diversity_score = self._calculate_diversity_score(processed_contexts)
            
            # Create result
            result = RetrievalResult(
                query=query,
                contexts=processed_contexts,
                total_found=len(processed_contexts),
                execution_time_ms=execution_time,
                strategy_used=strategy,
                quality_score=quality_score,
                diversity_score=diversity_score
            )
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            logger.info(f"Retrieved {len(processed_contexts)} contexts in {execution_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error in context retrieval: {e}")
            # Return empty result on error
            return RetrievalResult(
                query=query,
                contexts=[],
                total_found=0,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                strategy_used=strategy,
                quality_score=0.0,
                diversity_score=0.0
            )
    
    async def _similarity_based_retrieval(self, query: ContextQuery) -> List[ContextPiece]:
        """Basic similarity-based context retrieval"""
        
        if not query.embedding:
            logger.warning("No embedding provided for similarity-based retrieval")
            return []
        
        # Create search filter
        search_filter = SearchFilter(
            date_range=query.time_range,
            sources=query.required_sources
        )
        
        # Exclude sources if specified
        if query.exclude_sources:
            # This would need to be implemented in the vector store
            pass
        
        # Perform vector search
        search_results = await self.vector_store.search(
            query_embedding=query.embedding,
            mode=SearchMode.HYBRID,
            ranking_strategy=RankingStrategy.RELEVANCE,
            limit=query.max_results * 2,  # Get more for filtering
            search_filter=search_filter,
            query_text=query.text
        )
        
        # Convert to context pieces
        contexts = []
        for result in search_results:
            if result.final_score < query.min_relevance:
                continue
                
            # Determine context type
            context_type = self._determine_context_type(result.document, query)
            
            # Skip if not in requested types
            if query.context_types and context_type not in query.context_types:
                continue
            
            context_piece = ContextPiece(
                content=result.document.content,
                source=result.document.source,
                context_type=context_type,
                relevance_score=result.final_score,
                confidence_score=result.similarity_score,
                metadata=result.document.metadata if query.include_metadata else {},
                chunk_id=result.document.id,
                relationships=[],  # Will be populated later
                timestamp=result.document.timestamp
            )
            
            contexts.append(context_piece)
            
            if len(contexts) >= query.max_results:
                break
        
        return contexts
    
    async def _multi_hop_retrieval(self, query: ContextQuery) -> List[ContextPiece]:
        """Multi-hop retrieval following document relationships"""
        
        # Start with similarity-based retrieval
        initial_contexts = await self._similarity_based_retrieval(query)
        
        if not initial_contexts:
            return []
        
        # Get related documents through relationships
        all_contexts = initial_contexts.copy()
        
        # For each initial result, find related documents
        for context in initial_contexts[:3]:  # Only expand top 3 results
            related_docs = await self._get_related_documents(context.chunk_id)
            
            for related_doc in related_docs:
                # Check if we already have this document
                if any(c.chunk_id == related_doc.id for c in all_contexts):
                    continue
                
                # Calculate relevance to original query
                relevance = self._calculate_relevance_to_query(related_doc, query)
                
                if relevance >= query.min_relevance * 0.7:  # Lower threshold for related docs
                    context_type = self._determine_context_type(related_doc, query)
                    
                    related_context = ContextPiece(
                        content=related_doc.content,
                        source=related_doc.source,
                        context_type=context_type,
                        relevance_score=relevance,
                        confidence_score=relevance * 0.8,  # Lower confidence for related
                        metadata=related_doc.metadata if query.include_metadata else {},
                        chunk_id=related_doc.id,
                        relationships=[context.chunk_id],
                        timestamp=related_doc.timestamp
                    )
                    
                    all_contexts.append(related_context)
        
        # Sort by relevance and limit
        all_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_contexts[:query.max_results]
    
    async def _hierarchical_retrieval(self, query: ContextQuery) -> List[ContextPiece]:
        """Hierarchical retrieval considering document structure"""
        
        # Start with base retrieval
        base_contexts = await self._similarity_based_retrieval(query)
        
        # Group by source document and expand hierarchically
        source_groups = {}
        for context in base_contexts:
            source = context.source
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(context)
        
        hierarchical_contexts = []
        
        for source, contexts in source_groups.items():
            # For each source, try to get surrounding context
            for context in contexts:
                # Add the original context
                hierarchical_contexts.append(context)
                
                # Try to get parent/child chunks if available
                surrounding_chunks = await self._get_surrounding_chunks(context.chunk_id)
                
                for chunk in surrounding_chunks:
                    if len(hierarchical_contexts) >= query.max_results:
                        break
                    
                    chunk_relevance = self._calculate_relevance_to_query(chunk, query)
                    
                    if chunk_relevance >= query.min_relevance * 0.6:
                        context_type = self._determine_context_type(chunk, query)
                        
                        hierarchical_context = ContextPiece(
                            content=chunk.content,
                            source=chunk.source,
                            context_type=context_type,
                            relevance_score=chunk_relevance,
                            confidence_score=chunk_relevance * 0.9,
                            metadata=chunk.metadata if query.include_metadata else {},
                            chunk_id=chunk.id,
                            relationships=[context.chunk_id],
                            timestamp=chunk.timestamp
                        )
                        
                        hierarchical_contexts.append(hierarchical_context)
        
        # Remove duplicates and sort
        unique_contexts = {c.chunk_id: c for c in hierarchical_contexts}.values()
        sorted_contexts = sorted(unique_contexts, key=lambda x: x.relevance_score, reverse=True)
        
        return list(sorted_contexts)[:query.max_results]
    
    async def _temporal_retrieval(self, query: ContextQuery) -> List[ContextPiece]:
        """Time-aware context retrieval"""
        
        # Adjust retrieval based on time factors
        now = datetime.now()
        
        # If no time range specified, use recency weighting
        if not query.time_range:
            # Create time-weighted search
            contexts = await self._similarity_based_retrieval(query)
            
            # Apply temporal weighting
            for context in contexts:
                age_days = (now - context.timestamp).days
                
                # Recency boost (more recent = higher relevance)
                temporal_factor = 1.0 / (1.0 + age_days / 30.0)  # 30-day half-life
                context.relevance_score *= (0.7 + 0.3 * temporal_factor)
                
                # Update context type based on age
                if age_days > 365:
                    context.context_type = ContextType.HISTORICAL
            
            # Re-sort by updated relevance
            contexts.sort(key=lambda x: x.relevance_score, reverse=True)
            return contexts
        
        else:
            # Use provided time range
            return await self._similarity_based_retrieval(query)
    
    async def _adaptive_retrieval(self, query: ContextQuery) -> List[ContextPiece]:
        """Adaptive retrieval that learns from past interactions"""
        
        # Check if we have successful patterns for similar queries
        similar_successful = self._find_similar_successful_queries(query.text)
        
        if similar_successful:
            # Use the most successful strategy
            best_strategy = max(similar_successful.keys(), 
                              key=lambda k: similar_successful[k]['success_rate'])
            
            logger.info(f"Using adaptive strategy: {best_strategy}")
            
            # Map to actual retrieval strategies
            strategy_map = {
                'similarity': RetrievalStrategy.SIMILARITY_BASED,
                'multi_hop': RetrievalStrategy.MULTI_HOP,
                'hierarchical': RetrievalStrategy.HIERARCHICAL,
                'temporal': RetrievalStrategy.TEMPORAL
            }
            
            adaptive_strategy = strategy_map.get(best_strategy, RetrievalStrategy.SIMILARITY_BASED)
            return await self.retrieve_context(query, adaptive_strategy)
        
        else:
            # Fall back to similarity-based for new query types
            return await self._similarity_based_retrieval(query)
    
    async def _post_process_contexts(
        self, 
        contexts: List[ContextPiece], 
        query: ContextQuery
    ) -> List[ContextPiece]:
        """Post-process retrieved contexts"""
        
        # Remove duplicates
        unique_contexts = {c.chunk_id: c for c in contexts}.values()
        contexts = list(unique_contexts)
        
        # Ensure diversity
        contexts = self._ensure_diversity(contexts, query)
        
        # Final ranking adjustment
        contexts = self._adjust_final_ranking(contexts, query)
        
        return contexts[:query.max_results]
    
    # Helper methods
    
    async def _get_related_documents(self, chunk_id: str) -> List[VectorDocument]:
        """Get documents related to a specific chunk"""
        # This would query the document_relationships table
        # For now, return empty list
        return []
    
    async def _get_surrounding_chunks(self, chunk_id: str) -> List[VectorDocument]:
        """Get chunks surrounding a specific chunk (parent/child/sibling)"""
        # This would implement hierarchical chunk relationships
        # For now, return empty list
        return []
    
    def _determine_context_type(self, document: VectorDocument, query: ContextQuery) -> ContextType:
        """Determine the type of context this document represents"""
        
        # Simple heuristic-based classification
        content_lower = document.content.lower()
        query_lower = query.text.lower()
        
        # Check for procedural content
        if any(word in content_lower for word in ['how to', 'step', 'procedure', 'method']):
            return ContextType.PROCEDURAL
        
        # Check for factual content
        if any(word in content_lower for word in ['fact', 'data', 'statistic', 'research']):
            return ContextType.FACTUAL
        
        # Check for historical content
        age_days = (datetime.now() - document.timestamp).days
        if age_days > 30:
            return ContextType.HISTORICAL
        
        # Check for specific vs background
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words.intersection(content_words))
        
        if overlap > 2:
            return ContextType.SPECIFIC
        else:
            return ContextType.BACKGROUND
    
    def _calculate_relevance_to_query(self, document: VectorDocument, query: ContextQuery) -> float:
        """Calculate relevance of a document to a query"""
        
        # Simple keyword-based relevance for now
        # In production, this would use embeddings
        query_words = set(query.text.lower().split())
        content_words = set(document.content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _ensure_diversity(self, contexts: List[ContextPiece], query: ContextQuery) -> List[ContextPiece]:
        """Ensure diversity in retrieved contexts"""
        
        if len(contexts) <= 3:
            return contexts
        
        diverse_contexts = [contexts[0]]  # Always include the top result
        
        for context in contexts[1:]:
            # Check diversity with existing contexts
            is_diverse = True
            
            for existing in diverse_contexts:
                # Check source diversity
                if context.source == existing.source and len(diverse_contexts) > 2:
                    is_diverse = False
                    break
                
                # Check content similarity (simple)
                content_similarity = self._simple_content_similarity(
                    context.content, existing.content
                )
                if content_similarity > 0.8:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_contexts.append(context)
            
            if len(diverse_contexts) >= query.max_results:
                break
        
        return diverse_contexts
    
    def _adjust_final_ranking(self, contexts: List[ContextPiece], query: ContextQuery) -> List[ContextPiece]:
        """Make final adjustments to context ranking"""
        
        # Boost specific context types if requested
        if query.context_types:
            for context in contexts:
                if context.context_type in query.context_types:
                    context.relevance_score *= 1.2
        
        # Re-sort
        contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        return contexts
    
    def _simple_content_similarity(self, content1: str, content2: str) -> float:
        """Simple content similarity calculation"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_quality_score(self, contexts: List[ContextPiece], query: ContextQuery) -> float:
        """Calculate overall quality score for retrieved contexts"""
        
        if not contexts:
            return 0.0
        
        # Average relevance score
        avg_relevance = sum(c.relevance_score for c in contexts) / len(contexts)
        
        # Coverage score (how well we covered the query)
        coverage = min(1.0, len(contexts) / query.max_results)
        
        # Confidence score
        avg_confidence = sum(c.confidence_score for c in contexts) / len(contexts)
        
        return (avg_relevance * 0.4 + coverage * 0.3 + avg_confidence * 0.3)
    
    def _calculate_diversity_score(self, contexts: List[ContextPiece]) -> float:
        """Calculate diversity score for retrieved contexts"""
        
        if len(contexts) <= 1:
            return 1.0
        
        # Source diversity
        unique_sources = len(set(c.source for c in contexts))
        source_diversity = unique_sources / len(contexts)
        
        # Type diversity
        unique_types = len(set(c.context_type for c in contexts))
        type_diversity = unique_types / min(len(contexts), len(ContextType))
        
        return (source_diversity * 0.6 + type_diversity * 0.4)
    
    # Caching methods
    
    def _generate_cache_key(self, query: ContextQuery) -> str:
        """Generate cache key for a query"""
        key_parts = [
            query.text,
            str(query.max_results),
            str(query.min_relevance),
            str(sorted(query.context_types) if query.context_types else ""),
            str(query.required_sources or ""),
            str(query.exclude_sources or "")
        ]
        return "|".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[RetrievalResult]:
        """Get cached result if available and not expired"""
        if cache_key not in self.query_cache:
            return None
        
        cached_data = self.query_cache[cache_key]
        if datetime.now() - cached_data['timestamp'] > self.cache_ttl:
            del self.query_cache[cache_key]
            return None
        
        return cached_data['result']
    
    def _cache_result(self, cache_key: str, result: RetrievalResult) -> None:
        """Cache a retrieval result"""
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # Simple cache size management
        if len(self.query_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.query_cache.keys(),
                key=lambda k: self.query_cache[k]['timestamp']
            )[:20]
            
            for key in oldest_keys:
                del self.query_cache[key]
    
    def _find_similar_successful_queries(self, query_text: str) -> Dict[str, Dict]:
        """Find similar queries that were successful in the past"""
        # Simple implementation for now
        # In production, this would use semantic similarity
        similar_queries = {}
        
        for successful_query, data in self.successful_queries.items():
            if self._simple_content_similarity(query_text, successful_query) > 0.6:
                similar_queries[data['strategy']] = data
        
        return similar_queries
    
    def record_feedback(self, query: ContextQuery, result: RetrievalResult, feedback_score: float):
        """Record user feedback for learning"""
        
        self.feedback_history.append({
            'query': query.text,
            'strategy': result.strategy_used.value,
            'quality_score': result.quality_score,
            'feedback_score': feedback_score,
            'timestamp': datetime.now()
        })
        
        # Update successful queries
        if feedback_score > 0.7:  # Positive feedback threshold
            query_key = query.text
            strategy_key = result.strategy_used.value
            
            if query_key not in self.successful_queries:
                self.successful_queries[query_key] = {
                    'strategy': strategy_key,
                    'success_count': 1,
                    'total_count': 1,
                    'success_rate': 1.0
                }
            else:
                data = self.successful_queries[query_key]
                data['total_count'] += 1
                if feedback_score > 0.7:
                    data['success_count'] += 1
                data['success_rate'] = data['success_count'] / data['total_count']
