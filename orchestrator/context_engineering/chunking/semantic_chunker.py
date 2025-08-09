
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

# Import mathematical foundations
from ..mathematical_foundations.information_theory import InformationTheory
from ..mathematical_foundations.vector_operations import VectorOperations
from ..mathematical_foundations.statistical_analysis import StatisticalAnalysis

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
    """Advanced semantic chunking implementation"""
    
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
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_start = 0
        
        for i in range(1, len(sentences)):
            # Calculate semantic similarity with current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            similarity = self._calculate_text_similarity(chunk_text, sentences[i])
            
            # Check if we should continue current chunk or start new one
            current_size = len(chunk_text)
            would_exceed_max = current_size + len(sentences[i]) > self.max_chunk_size
            
            if (similarity >= self.similarity_threshold and 
                not would_exceed_max and 
                current_size < self.target_chunk_size):
                # Continue current chunk
                current_chunk_sentences.append(sentences[i])
            else:
                # Finalize current chunk if it meets minimum size
                if current_size >= self.min_chunk_size:
                    chunk = self._create_chunk_from_sentences(
                        current_chunk_sentences, current_start, document_id, len(chunks)
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_start = self._find_sentence_position(text, sentences[i])
        
        # Handle final chunk
        if current_chunk_sentences:
            final_chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, current_start, document_id, len(chunks)
            )
            chunks.append(final_chunk)
        
        return self._add_overlap_and_relationships(chunks, text)
    
    def _chunk_by_information_density(self, text: str, document_id: str) -> List[SemanticChunk]:
        """Chunk based on information density using entropy calculations"""
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Calculate entropy for each sentence
        sentence_entropies = [
            self.info_theory.calculate_entropy(sentence) 
            for sentence in sentences
        ]
        
        # Find optimal chunk boundaries based on entropy patterns
        chunks = []
        current_sentences = []
        current_start = 0
        target_density = np.mean(sentence_entropies)
        
        for i, (sentence, entropy) in enumerate(zip(sentences, sentence_entropies)):
            current_sentences.append(sentence)
            
            # Calculate current chunk density
            chunk_text = ' '.join(current_sentences)
            chunk_entropy = self.info_theory.calculate_entropy(chunk_text)
            
            # Decide whether to finalize chunk
            should_finalize = (
                len(chunk_text) >= self.target_chunk_size or
                len(chunk_text) >= self.max_chunk_size or
                (chunk_entropy > target_density * 1.2 and len(chunk_text) >= self.min_chunk_size) or
                i == len(sentences) - 1
            )
            
            if should_finalize and len(chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk_from_sentences(
                    current_sentences, current_start, document_id, len(chunks)
                )
                chunks.append(chunk)
                
                current_sentences = []
                if i < len(sentences) - 1:
                    current_start = self._find_sentence_position(text, sentences[i + 1])
        
        return self._add_overlap_and_relationships(chunks, text)
    
    def _chunk_by_topic_coherence(self, text: str, document_id: str) -> List[SemanticChunk]:
        """Chunk based on topic coherence using keyword analysis"""
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Extract keywords from each sentence
        sentence_keywords = [self._extract_keywords(sentence) for sentence in sentences]
        
        chunks = []
        current_sentences = [sentences[0]]
        current_keywords = set(sentence_keywords[0])
        current_start = 0
        
        for i in range(1, len(sentences)):
            sentence_kw = set(sentence_keywords[i])
            
            # Calculate topic coherence (keyword overlap)
            coherence = len(current_keywords.intersection(sentence_kw)) / max(
                len(current_keywords.union(sentence_kw)), 1
            )
            
            chunk_text = ' '.join(current_sentences)
            would_exceed = len(chunk_text) + len(sentences[i]) > self.max_chunk_size
            
            if (coherence >= 0.3 and 
                not would_exceed and 
                len(chunk_text) < self.target_chunk_size):
                # Continue current chunk
                current_sentences.append(sentences[i])
                current_keywords.update(sentence_kw)
            else:
                # Finalize current chunk
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk_from_sentences(
                        current_sentences, current_start, document_id, len(chunks)
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_sentences = [sentences[i]]
                current_keywords = sentence_kw
                current_start = self._find_sentence_position(text, sentences[i])
        
        # Handle final chunk
        if current_sentences:
            final_chunk = self._create_chunk_from_sentences(
                current_sentences, current_start, document_id, len(chunks)
            )
            chunks.append(final_chunk)
        
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
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments"""
        
        # Simple keyword-based similarity for now
        # In production, this would use embeddings
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
