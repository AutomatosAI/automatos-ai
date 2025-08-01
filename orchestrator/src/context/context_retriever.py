
"""
Context Retrieval System
========================

Advanced context retrieval with relevance scoring and multi-source synthesis.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re

from .vector_store import PgVectorStore, VectorSearchResult
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

@dataclass
class ContextResult:
    """Result from context retrieval"""
    content: str
    source: str
    relevance_score: float
    context_type: str
    metadata: Dict[str, Any]
    chunk_id: str

@dataclass
class RetrievalConfig:
    """Configuration for context retrieval"""
    max_results: int = 10
    similarity_threshold: float = 0.3
    context_window_size: int = 3  # Number of surrounding chunks to include
    enable_reranking: bool = True
    diversity_threshold: float = 0.8  # Avoid too similar results
    temporal_decay: float = 0.1  # Decay factor for older content

class ContextRetriever:
    """Advanced context retrieval with multi-strategy approach"""
    
    def __init__(self, vector_store: PgVectorStore, embedding_generator: EmbeddingGenerator,
                 config: RetrievalConfig = None):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or RetrievalConfig()
        
        # Context type priorities
        self.context_priorities = {
            'code': 1.0,
            'documentation': 0.9,
            'api': 0.95,
            'tutorial': 0.8,
            'example': 0.85,
            'error': 0.7,
            'config': 0.6
        }
    
    async def retrieve_context(self, query: str, task_type: str = None,
                             filters: Dict[str, Any] = None) -> List[ContextResult]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Search query
            task_type: Type of task (for context prioritization)
            filters: Additional filters
            
        Returns:
            List of relevant context results
        """
        try:
            # Generate query embedding
            query_embeddings = await self.embedding_generator.generate_embeddings([query])
            if not query_embeddings:
                logger.error("Failed to generate query embedding")
                return []
            
            query_embedding = query_embeddings[0]['embedding']
            
            # Multi-strategy retrieval
            results = []
            
            # 1. Vector similarity search
            vector_results = await self._vector_similarity_search(query_embedding, filters)
            results.extend(vector_results)
            
            # 2. Hybrid search (if supported)
            hybrid_results = await self._hybrid_search(query_embedding, query, filters)
            results.extend(hybrid_results)
            
            # 3. Context-aware search based on task type
            if task_type:
                context_results = await self._context_aware_search(
                    query_embedding, task_type, filters
                )
                results.extend(context_results)
            
            # 4. Historical pattern matching
            pattern_results = await self._pattern_based_search(query_embedding, task_type)
            results.extend(pattern_results)
            
            # Deduplicate and rerank
            final_results = await self._deduplicate_and_rerank(results, query)
            
            # Apply context window expansion
            expanded_results = await self._expand_context_windows(final_results)
            
            logger.info(f"Retrieved {len(expanded_results)} context results for query: {query[:50]}...")
            return expanded_results[:self.config.max_results]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    async def _vector_similarity_search(self, query_embedding: List[float],
                                      filters: Dict[str, Any] = None) -> List[ContextResult]:
        """Perform vector similarity search"""
        
        search_results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            limit=self.config.max_results * 2,  # Get more for reranking
            similarity_threshold=self.config.similarity_threshold,
            filters=filters
        )
        
        context_results = []
        for result in search_results:
            context_result = ContextResult(
                content=result.content,
                source=result.source_file,
                relevance_score=result.similarity_score,
                context_type=result.content_type,
                metadata=result.metadata,
                chunk_id=result.id
            )
            context_results.append(context_result)
        
        return context_results
    
    async def _hybrid_search(self, query_embedding: List[float], query_text: str,
                           filters: Dict[str, Any] = None) -> List[ContextResult]:
        """Perform hybrid vector + text search"""
        
        try:
            if hasattr(self.vector_store, 'hybrid_search'):
                search_results = await self.vector_store.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query_text,
                    limit=self.config.max_results
                )
                
                context_results = []
                for result in search_results:
                    context_result = ContextResult(
                        content=result.content,
                        source=result.source_file,
                        relevance_score=result.similarity_score,
                        context_type=result.content_type,
                        metadata=result.metadata,
                        chunk_id=result.id
                    )
                    context_results.append(context_result)
                
                return context_results
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {str(e)}")
        
        return []
    
    async def _context_aware_search(self, query_embedding: List[float], task_type: str,
                                  filters: Dict[str, Any] = None) -> List[ContextResult]:
        """Search with task-type specific context awareness"""
        
        # Define task-type specific filters and boosts
        task_filters = filters or {}
        
        if task_type == 'code_generation':
            task_filters['content_type'] = 'code'
        elif task_type == 'documentation':
            task_filters['content_type'] = 'markdown'
        elif task_type == 'api_development':
            # Look for API-related content
            pass
        
        search_results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            limit=self.config.max_results,
            similarity_threshold=self.config.similarity_threshold * 0.8,  # Lower threshold for context-aware
            filters=task_filters
        )
        
        context_results = []
        for result in search_results:
            # Apply context-type priority boost
            priority_boost = self.context_priorities.get(result.content_type, 0.5)
            adjusted_score = result.similarity_score * priority_boost
            
            context_result = ContextResult(
                content=result.content,
                source=result.source_file,
                relevance_score=adjusted_score,
                context_type=result.content_type,
                metadata=result.metadata,
                chunk_id=result.id
            )
            context_results.append(context_result)
        
        return context_results
    
    async def _pattern_based_search(self, query_embedding: List[float],
                                  task_type: str = None) -> List[ContextResult]:
        """Search based on historical patterns"""
        
        try:
            if hasattr(self.vector_store, 'find_similar_patterns'):
                patterns = await self.vector_store.find_similar_patterns(
                    query_embedding=query_embedding,
                    pattern_type=task_type,
                    limit=5
                )
                
                # Convert patterns to context results
                context_results = []
                for pattern in patterns:
                    if pattern['similarity'] > 0.5:  # Only high-similarity patterns
                        context_result = ContextResult(
                            content=f"Pattern: {pattern['pattern_name']}",
                            source="historical_patterns",
                            relevance_score=pattern['similarity'] * 0.8,  # Slightly lower weight
                            context_type="pattern",
                            metadata=pattern['metadata'],
                            chunk_id=f"pattern_{pattern['pattern_name']}"
                        )
                        context_results.append(context_result)
                
                return context_results
        except Exception as e:
            logger.warning(f"Pattern-based search failed: {str(e)}")
        
        return []
    
    async def _deduplicate_and_rerank(self, results: List[ContextResult],
                                    query: str) -> List[ContextResult]:
        """Remove duplicates and rerank results"""
        
        if not results:
            return []
        
        # Remove exact duplicates
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Remove similar results (diversity filtering)
        if self.config.enable_reranking:
            diverse_results = await self._apply_diversity_filter(unique_results)
        else:
            diverse_results = unique_results
        
        # Rerank based on multiple factors
        reranked_results = await self._rerank_results(diverse_results, query)
        
        return reranked_results
    
    async def _apply_diversity_filter(self, results: List[ContextResult]) -> List[ContextResult]:
        """Apply diversity filtering to avoid too similar results"""
        
        if len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include the top result
        
        for result in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            
            for selected in diverse_results:
                # Simple content similarity check
                similarity = self._compute_text_similarity(result.content, selected.content)
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity (Jaccard similarity)"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _rerank_results(self, results: List[ContextResult], query: str) -> List[ContextResult]:
        """Rerank results based on multiple factors"""
        
        # Factors for reranking:
        # 1. Original relevance score
        # 2. Content type priority
        # 3. Source reliability
        # 4. Recency (if applicable)
        # 5. Content length appropriateness
        
        for result in results:
            score_factors = []
            
            # Original relevance (weight: 0.4)
            score_factors.append(result.relevance_score * 0.4)
            
            # Content type priority (weight: 0.2)
            type_priority = self.context_priorities.get(result.context_type, 0.5)
            score_factors.append(type_priority * 0.2)
            
            # Source reliability (weight: 0.1)
            source_score = self._compute_source_reliability(result.source)
            score_factors.append(source_score * 0.1)
            
            # Content length appropriateness (weight: 0.1)
            length_score = self._compute_length_score(result.content)
            score_factors.append(length_score * 0.1)
            
            # Query-specific relevance (weight: 0.2)
            query_relevance = self._compute_query_relevance(result.content, query)
            score_factors.append(query_relevance * 0.2)
            
            # Update relevance score
            result.relevance_score = sum(score_factors)
        
        # Sort by updated relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _compute_source_reliability(self, source: str) -> float:
        """Compute source reliability score"""
        
        # Simple heuristics for source reliability
        if 'official' in source.lower() or 'docs' in source.lower():
            return 1.0
        elif 'example' in source.lower() or 'tutorial' in source.lower():
            return 0.8
        elif 'test' in source.lower():
            return 0.6
        else:
            return 0.7
    
    def _compute_length_score(self, content: str) -> float:
        """Compute content length appropriateness score"""
        
        length = len(content)
        
        # Prefer medium-length content (not too short, not too long)
        if 100 <= length <= 1000:
            return 1.0
        elif 50 <= length < 100 or 1000 < length <= 2000:
            return 0.8
        elif length < 50:
            return 0.5
        else:
            return 0.6
    
    def _compute_query_relevance(self, content: str, query: str) -> float:
        """Compute query-specific relevance"""
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.5
        
        # Count query word matches
        matches = query_words.intersection(content_words)
        relevance = len(matches) / len(query_words)
        
        return min(relevance, 1.0)
    
    async def _expand_context_windows(self, results: List[ContextResult]) -> List[ContextResult]:
        """Expand context by including surrounding chunks"""
        
        if self.config.context_window_size <= 1:
            return results
        
        expanded_results = []
        
        for result in results:
            try:
                # Get surrounding chunks
                surrounding_chunks = await self._get_surrounding_chunks(
                    result.source, result.chunk_id, self.config.context_window_size
                )
                
                if surrounding_chunks:
                    # Combine content with surrounding context
                    combined_content = self._combine_chunks(result.content, surrounding_chunks)
                    
                    expanded_result = ContextResult(
                        content=combined_content,
                        source=result.source,
                        relevance_score=result.relevance_score,
                        context_type=result.context_type,
                        metadata=result.metadata,
                        chunk_id=result.chunk_id
                    )
                    expanded_results.append(expanded_result)
                else:
                    expanded_results.append(result)
                    
            except Exception as e:
                logger.warning(f"Failed to expand context for {result.chunk_id}: {str(e)}")
                expanded_results.append(result)
        
        return expanded_results
    
    async def _get_surrounding_chunks(self, source_file: str, chunk_id: str,
                                    window_size: int) -> List[Dict[str, Any]]:
        """Get surrounding chunks for context expansion"""
        
        try:
            # Extract chunk index from chunk_id
            chunk_index = self._extract_chunk_index(chunk_id)
            if chunk_index is None:
                return []
            
            # Get surrounding chunks
            async with self.vector_store.pool.acquire() as conn:
                query = f"""
                    SELECT content, chunk_index, metadata
                    FROM {self.vector_store.table_name}
                    WHERE source_file = $1 
                    AND chunk_index BETWEEN $2 AND $3
                    AND id != $4
                    ORDER BY chunk_index;
                """
                
                start_index = max(0, chunk_index - window_size)
                end_index = chunk_index + window_size
                
                rows = await conn.fetch(query, source_file, start_index, end_index, chunk_id)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting surrounding chunks: {str(e)}")
            return []
    
    def _extract_chunk_index(self, chunk_id: str) -> Optional[int]:
        """Extract chunk index from chunk ID"""
        
        try:
            # Assuming chunk_id format: "filename_index_hash"
            parts = chunk_id.split('_')
            if len(parts) >= 2:
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _combine_chunks(self, main_content: str, surrounding_chunks: List[Dict[str, Any]]) -> str:
        """Combine main content with surrounding chunks"""
        
        if not surrounding_chunks:
            return main_content
        
        # Sort by chunk index
        surrounding_chunks.sort(key=lambda x: x['chunk_index'])
        
        # Find the position of main content
        main_chunk_index = None
        for chunk in surrounding_chunks:
            if chunk['content'] == main_content:
                main_chunk_index = chunk['chunk_index']
                break
        
        # Combine chunks with separators
        combined_parts = []
        
        for chunk in surrounding_chunks:
            if chunk['chunk_index'] < main_chunk_index:
                combined_parts.append(f"[CONTEXT BEFORE]\n{chunk['content']}")
            elif chunk['chunk_index'] > main_chunk_index:
                combined_parts.append(f"[CONTEXT AFTER]\n{chunk['content']}")
        
        # Add main content in the middle
        if main_chunk_index is not None:
            before_parts = [p for p in combined_parts if '[CONTEXT BEFORE]' in p]
            after_parts = [p for p in combined_parts if '[CONTEXT AFTER]' in p]
            
            result_parts = before_parts + [f"[MAIN CONTENT]\n{main_content}"] + after_parts
        else:
            result_parts = [main_content] + combined_parts
        
        return '\n\n'.join(result_parts)

class SmartContextRetriever(ContextRetriever):
    """Enhanced context retriever with learning capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_stats = {}
        self.success_patterns = {}
    
    async def retrieve_context_with_learning(self, query: str, task_type: str = None,
                                           user_feedback: Dict[str, Any] = None) -> List[ContextResult]:
        """Retrieve context with learning from user feedback"""
        
        # Standard retrieval
        results = await self.retrieve_context(query, task_type)
        
        # Learn from feedback if provided
        if user_feedback:
            await self._learn_from_feedback(query, task_type, results, user_feedback)
        
        # Apply learned patterns
        enhanced_results = await self._apply_learned_patterns(results, query, task_type)
        
        return enhanced_results
    
    async def _learn_from_feedback(self, query: str, task_type: str,
                                 results: List[ContextResult], feedback: Dict[str, Any]):
        """Learn from user feedback to improve future retrievals"""
        
        # Record successful patterns
        if feedback.get('success', False):
            pattern_key = f"{task_type}_{hash(query) % 1000}"
            
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = {
                    'query_patterns': [],
                    'successful_sources': [],
                    'successful_types': [],
                    'usage_count': 0
                }
            
            pattern = self.success_patterns[pattern_key]
            pattern['query_patterns'].append(query)
            pattern['successful_sources'].extend([r.source for r in results])
            pattern['successful_types'].extend([r.context_type for r in results])
            pattern['usage_count'] += 1
        
        # Record usage statistics
        for result in results:
            source_key = result.source
            if source_key not in self.usage_stats:
                self.usage_stats[source_key] = {'used': 0, 'successful': 0}
            
            self.usage_stats[source_key]['used'] += 1
            if feedback.get('success', False):
                self.usage_stats[source_key]['successful'] += 1
    
    async def _apply_learned_patterns(self, results: List[ContextResult],
                                    query: str, task_type: str) -> List[ContextResult]:
        """Apply learned patterns to enhance results"""
        
        # Boost results from historically successful sources
        for result in results:
            if result.source in self.usage_stats:
                stats = self.usage_stats[result.source]
                if stats['used'] > 0:
                    success_rate = stats['successful'] / stats['used']
                    result.relevance_score *= (1 + success_rate * 0.2)  # Up to 20% boost
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
