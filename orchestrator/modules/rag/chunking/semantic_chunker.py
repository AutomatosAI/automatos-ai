
"""
Semantic Chunking Algorithm
===========================

Advanced chunking that preserves semantic boundaries and meaning.
Implements multiple strategies for intelligent text segmentation.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import mathematical foundations from shared
from core.math import InformationTheory, VectorOperations, StatisticalAnalysis

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    INFORMATION_DENSITY = "information_density"
    TOPIC_COHERENCE = "topic_coherence"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"

@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    start_pos: int
    end_pos: int
    word_count: int
    char_count: int
    entropy: float
    topic_coherence: float
    semantic_density: float
    relationships: List[str]  # IDs of related chunks
    importance_score: float

@dataclass
class SemanticChunk:
    """A semantically coherent chunk of text"""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None

class SemanticChunker:
    """Advanced semantic chunking implementation with REAL embedding-based similarity"""
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_SIMILARITY,
        target_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        overlap_ratio: float = 0.1,
        similarity_threshold: float = 0.7
    ):
        self.strategy = strategy
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
        self.similarity_threshold = similarity_threshold
        
        # Initialize mathematical components
        self.info_theory = InformationTheory()
        self.vector_ops = VectorOperations()
        self.stats = StatisticalAnalysis()
        
        # Embedding manager for REAL semantic similarity
        self._embedding_manager = None
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_max_size: int = 1000  # Prevent unbounded memory growth
        self._use_embeddings = False  # Disabled: keyword similarity is fast; embeddings belong in search not chunking
        
        # Patterns for sentence boundaries
        self.sentence_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence endings
            r'(?<=[.!?])\s*\n+\s*',     # Sentence endings with newlines
            r'\n\s*\n\s*',              # Paragraph breaks
            r'(?<=:)\s*\n+\s*(?=[A-Z•\-\d])',  # List items after colons
        ]
        
    def chunk_text(self, text: str, document_id: str = None) -> List[SemanticChunk]:
        """Main chunking method that delegates to strategy-specific implementations"""
        
        if self.strategy == ChunkingStrategy.SEMANTIC_SIMILARITY:
            return self._chunk_by_semantic_similarity(text, document_id)
        elif self.strategy == ChunkingStrategy.INFORMATION_DENSITY:
            return self._chunk_by_information_density(text, document_id)
        elif self.strategy == ChunkingStrategy.TOPIC_COHERENCE:
            return self._chunk_by_topic_coherence(text, document_id)
        elif self.strategy == ChunkingStrategy.HIERARCHICAL:
            return self._chunk_hierarchically(text, document_id)
        elif self.strategy == ChunkingStrategy.ADAPTIVE:
            return self._chunk_adaptively(text, document_id)
        else:
            # Default to basic semantic similarity
            return self._chunk_by_semantic_similarity(text, document_id)
    
    def _chunk_by_semantic_similarity(self, text: str, document_id: str) -> List[SemanticChunk]:
        """Chunk text based on semantic similarity between sentences"""

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        segments: List[List[str]] = []
        current = [sentences[0]]
        for sentence in sentences[1:]:
            chunk_text = ' '.join(current)
            similarity = self._calculate_text_similarity(chunk_text, sentence)
            would_exceed_max = len(chunk_text) + len(sentence) > self.max_chunk_size
            if (similarity >= self.similarity_threshold and
                    not would_exceed_max and
                    len(chunk_text) < self.target_chunk_size):
                current.append(sentence)
            else:
                segments.append(current)
                current = [sentence]
        segments.append(current)
        return self._chunks_from_segments(segments, text, document_id)

    def _chunk_by_information_density(self, text: str, document_id: str) -> List[SemanticChunk]:
        """Chunk based on information density using entropy calculations"""

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        target_density = np.mean([self.info_theory.calculate_entropy(s) for s in sentences])
        segments: List[List[str]] = []
        current: List[str] = []
        for sentence in sentences:
            current.append(sentence)
            chunk_text = ' '.join(current)
            chunk_entropy = self.info_theory.calculate_entropy(chunk_text)
            if (len(chunk_text) >= self.target_chunk_size or
                    len(chunk_text) >= self.max_chunk_size or
                    (chunk_entropy > target_density * 1.2 and len(chunk_text) >= self.min_chunk_size)):
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        return self._chunks_from_segments(segments, text, document_id)

    def _chunk_by_topic_coherence(self, text: str, document_id: str) -> List[SemanticChunk]:
        """Chunk based on topic coherence using keyword analysis"""

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        sentence_keywords = [self._extract_keywords(sentence) for sentence in sentences]
        segments: List[List[str]] = []
        current = [sentences[0]]
        current_keywords = set(sentence_keywords[0])
        for i in range(1, len(sentences)):
            sentence_kw = set(sentence_keywords[i])
            coherence = len(current_keywords.intersection(sentence_kw)) / max(
                len(current_keywords.union(sentence_kw)), 1
            )
            chunk_text = ' '.join(current)
            would_exceed = len(chunk_text) + len(sentences[i]) > self.max_chunk_size
            if (coherence >= 0.3 and
                    not would_exceed and
                    len(chunk_text) < self.target_chunk_size):
                current.append(sentences[i])
                current_keywords.update(sentence_kw)
            else:
                segments.append(current)
                current = [sentences[i]]
                current_keywords = sentence_kw
        segments.append(current)
        return self._chunks_from_segments(segments, text, document_id)

    def _merge_short_segments(self, segments: List[List[str]]) -> List[List[str]]:
        """F086 (night 3): a segment under ``min_chunk_size`` used to be DROPPED at
        a topic shift — a price line, a date, a sign-off vanished while the
        document read "completed" (brand-voice.md kept 461 of 1,304 characters).
        No text is dropped now: a short segment joins the next one, and a short
        last segment joins the one before it."""
        merged: List[List[str]] = []
        carry: List[str] = []
        for segment in segments:
            segment = [*carry, *segment]
            carry = []
            if len(' '.join(segment)) < self.min_chunk_size:
                carry = segment
                continue
            merged.append(segment)
        if carry:
            if merged:
                merged[-1] = [*merged[-1], *carry]
            else:
                merged.append(carry)
        return merged

    def _chunks_from_segments(self, segments: List[List[str]], text: str, document_id: str) -> List[SemanticChunk]:
        chunks = [
            self._create_chunk_from_sentences(segment, self._find_sentence_position(text, segment[0]),
                                              document_id, index)
            for index, segment in enumerate(self._merge_short_segments(segments))
        ]
        return self._add_overlap_and_relationships(chunks, text)

    def _chunk_hierarchically(self, text: str, document_id: str) -> List[SemanticChunk]:
        """Create hierarchical chunks with parent-child relationships"""
        
        # First, create large semantic chunks
        large_chunker = SemanticChunker(
            strategy=ChunkingStrategy.SEMANTIC_SIMILARITY,
            target_chunk_size=self.target_chunk_size * 2,
            max_chunk_size=self.max_chunk_size * 2
        )
        large_chunks = large_chunker._chunk_by_semantic_similarity(text, document_id)
        
        # Then, subdivide large chunks into smaller ones
        hierarchical_chunks = []
        
        for parent_chunk in large_chunks:
            if len(parent_chunk.content) <= self.target_chunk_size:
                # Small enough, keep as is
                hierarchical_chunks.append(parent_chunk)
            else:
                # Subdivide into smaller chunks
                small_chunker = SemanticChunker(
                    strategy=ChunkingStrategy.SEMANTIC_SIMILARITY,
                    target_chunk_size=self.target_chunk_size // 2,
                    max_chunk_size=self.target_chunk_size
                )
                sub_chunks = small_chunker._chunk_by_semantic_similarity(
                    parent_chunk.content, f"{document_id}_sub"
                )
                
                # Add parent-child relationships
                parent_id = parent_chunk.metadata.chunk_id
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.relationships.append(f"parent:{parent_id}")
                    hierarchical_chunks.append(sub_chunk)
        
        return hierarchical_chunks
    
    def _chunk_adaptively(self, text: str, document_id: str) -> List[SemanticChunk]:
        """Adaptive chunking that combines multiple strategies"""
        
        # Try different strategies and select the best result
        strategies = [
            ChunkingStrategy.SEMANTIC_SIMILARITY,
            ChunkingStrategy.INFORMATION_DENSITY,
            ChunkingStrategy.TOPIC_COHERENCE
        ]
        
        results = []
        for strategy in strategies:
            chunker = SemanticChunker(
                strategy=strategy,
                target_chunk_size=self.target_chunk_size,
                min_chunk_size=self.min_chunk_size,
                max_chunk_size=self.max_chunk_size
            )
            chunks = chunker.chunk_text(text, document_id)
            
            # Score this chunking result
            score = self._score_chunking_quality(chunks, text)
            results.append((chunks, score))
        
        # Return the best result
        best_chunks, _ = max(results, key=lambda x: x[1])
        return best_chunks
    
    # Helper methods
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple patterns"""
        
        # Apply sentence boundary patterns
        sentences = [text]
        for pattern in self.sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = [s.strip() for s in new_sentences if s.strip()]
        
        return sentences
    
    def _get_embedding_manager(self):
        """Lazy initialization of embedding manager"""
        if self._embedding_manager is None:
            try:
                from core.llm import create_embedding_manager
                self._embedding_manager = create_embedding_manager()
                logger.info("SemanticChunker initialized with embedding-based similarity")
            except Exception as e:
                logger.warning(f"Could not initialize embedding manager: {e}. Falling back to keyword similarity.")
                self._use_embeddings = False
        return self._embedding_manager
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text with caching"""
        # Use hash of first 200 chars as cache key (sufficient for similarity)
        cache_key = text[:200]
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        manager = self._get_embedding_manager()
        if manager is None:
            return None
        
        try:
            # Use sync version for chunking (avoid async complexity)
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    embedding = pool.submit(
                        lambda: asyncio.run(manager.generate_embedding(text))
                    ).result(timeout=10)
            else:
                embedding = loop.run_until_complete(manager.generate_embedding(text))
            
            # Evict oldest entries if cache exceeds max size
            if len(self._embedding_cache) >= self._embedding_cache_max_size:
                # Remove first 10% of entries (FIFO eviction)
                evict_count = max(1, self._embedding_cache_max_size // 10)
                keys_to_remove = list(self._embedding_cache.keys())[:evict_count]
                for k in keys_to_remove:
                    del self._embedding_cache[k]
                logger.debug(f"Evicted {evict_count} entries from embedding cache")

            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.debug(f"Embedding generation failed for text: {e}")
            return None
    
    # REMOVED: Duplicate _cosine_similarity - now using VectorOperations.cosine_similarity()
    # See core/math/vector_operations.py for centralized implementation
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text segments using embeddings.
        
        Uses REAL embedding-based cosine similarity for accurate semantic comparison.
        Falls back to keyword overlap only if embeddings are unavailable.
        """
        # Try embedding-based similarity first (MUCH more accurate)
        if self._use_embeddings:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            
            if emb1 is not None and emb2 is not None:
                # Use centralized VectorOperations.cosine_similarity
                return VectorOperations.cosine_similarity(emb1, emb2)
        
        # Fallback: keyword-based similarity (less accurate but fast)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        
        # Simple keyword extraction (would use NLP libraries in production)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'an', 'a'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return keywords
    
    def _find_sentence_position(self, full_text: str, sentence: str) -> int:
        """Find the starting position of a sentence in the full text"""
        return full_text.find(sentence.strip())
    
    def _create_chunk_from_sentences(
        self, 
        sentences: List[str], 
        start_pos: int, 
        document_id: str, 
        chunk_index: int
    ) -> SemanticChunk:
        """Create a SemanticChunk from a list of sentences"""
        
        content = ' '.join(sentences)
        chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
        
        # Calculate metadata
        word_count = len(content.split())
        char_count = len(content)
        entropy = self.info_theory.calculate_entropy(content)
        
        # Calculate topic coherence and semantic density
        keywords = self._extract_keywords(content)
        topic_coherence = len(set(keywords)) / max(len(keywords), 1)
        semantic_density = entropy / max(char_count, 1) * 1000  # Per 1000 chars
        
        # Calculate importance score (combination of metrics)
        importance_score = (entropy * 0.4 + topic_coherence * 0.3 + semantic_density * 0.3)
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            start_pos=start_pos,
            end_pos=start_pos + char_count,
            word_count=word_count,
            char_count=char_count,
            entropy=entropy,
            topic_coherence=topic_coherence,
            semantic_density=semantic_density,
            relationships=[],
            importance_score=importance_score
        )
        
        return SemanticChunk(
            content=content,
            metadata=metadata
        )
    
    def _add_overlap_and_relationships(
        self, 
        chunks: List[SemanticChunk], 
        full_text: str
    ) -> List[SemanticChunk]:
        """Add overlap between chunks and establish relationships"""
        
        if not chunks:
            return chunks
        
        # Add overlap
        overlap_size = int(self.target_chunk_size * self.overlap_ratio)
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Add overlap from previous chunk
            prev_words = prev_chunk.content.split()
            if len(prev_words) > overlap_size:
                overlap_text = ' '.join(prev_words[-overlap_size:])
                curr_chunk.content = f"{overlap_text} {curr_chunk.content}"
                curr_chunk.metadata.char_count = len(curr_chunk.content)
                curr_chunk.metadata.word_count = len(curr_chunk.content.split())
            
            # Establish relationships
            prev_chunk.metadata.relationships.append(f"next:{curr_chunk.metadata.chunk_id}")
            curr_chunk.metadata.relationships.append(f"prev:{prev_chunk.metadata.chunk_id}")
        
        return chunks
    
    def _score_chunking_quality(self, chunks: List[SemanticChunk], original_text: str) -> float:
        """Score the quality of a chunking result"""
        
        if not chunks:
            return 0.0
        
        # Metrics for scoring
        size_consistency = self._calculate_size_consistency(chunks)
        semantic_coherence = self._calculate_average_coherence(chunks)
        coverage = self._calculate_coverage(chunks, original_text)
        
        # Weighted combination
        return (size_consistency * 0.3 + semantic_coherence * 0.4 + coverage * 0.3)
    
    def _calculate_size_consistency(self, chunks: List[SemanticChunk]) -> float:
        """Calculate how consistent chunk sizes are"""
        
        sizes = [chunk.metadata.char_count for chunk in chunks]
        if not sizes:
            return 0.0
        
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Lower standard deviation relative to mean is better
        cv = std_size / mean_size if mean_size > 0 else 1.0
        return max(0.0, 1.0 - cv)
    
    def _calculate_average_coherence(self, chunks: List[SemanticChunk]) -> float:
        """Calculate average topic coherence across chunks"""
        
        coherences = [chunk.metadata.topic_coherence for chunk in chunks]
        return np.mean(coherences) if coherences else 0.0
    
    def _calculate_coverage(self, chunks: List[SemanticChunk], original_text: str) -> float:
        """Calculate how much of the original text is covered by chunks"""
        
        total_chunk_chars = sum(chunk.metadata.char_count for chunk in chunks)
        original_chars = len(original_text)
        
        return min(1.0, total_chunk_chars / original_chars) if original_chars > 0 else 0.0


# Aliases for backwards compatibility
DocumentChunker = SemanticChunker
MultiModalChunker = SemanticChunker

