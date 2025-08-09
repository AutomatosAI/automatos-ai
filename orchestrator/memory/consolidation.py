
"""
Memory Consolidation System
===========================

Intelligent memory consolidation with summarization, prioritization, and efficiency optimization.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic text processing")

from .memory_types import MemoryItem, MemoryType, MemoryLevel

logger = logging.getLogger(__name__)

class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation"""
    SUMMARIZATION = "summarization"
    CLUSTERING = "clustering"
    PRIORITIZATION = "prioritization"
    COMPRESSION = "compression"
    HYBRID = "hybrid"

@dataclass
class ConsolidationMetrics:
    """Metrics for memory consolidation performance"""
    items_consolidated: int = 0
    compression_ratio: float = 0.0
    information_retention: float = 0.0
    processing_time: float = 0.0
    storage_saved: int = 0

class MemoryConsolidator:
    """
    Advanced memory consolidation system with multiple strategies
    """
    
    def __init__(
        self,
        consolidation_threshold: float = 0.8,
        max_summary_length: int = 500,
        similarity_threshold: float = 0.7,
        importance_decay: float = 0.1
    ):
        self.consolidation_threshold = consolidation_threshold
        self.max_summary_length = max_summary_length
        self.similarity_threshold = similarity_threshold
        self.importance_decay = importance_decay
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stemmer = None
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
            }
        
        # Consolidation metrics
        self.metrics = ConsolidationMetrics()
        
        # Cache for consolidation results
        self.consolidation_cache: Dict[str, Any] = {}
        
        logger.info("Memory consolidator initialized")
    
    async def consolidate_memories(
        self,
        memories: List[MemoryItem],
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID,
        target_level: MemoryLevel = MemoryLevel.LONG_TERM
    ) -> List[MemoryItem]:
        """Consolidate a list of memories using specified strategy"""
        
        start_time = time.time()
        original_count = len(memories)
        original_size = self._calculate_memory_size(memories)
        
        logger.info(f"Starting consolidation of {original_count} memories using {strategy.value}")
        
        consolidated_memories = []
        
        if strategy == ConsolidationStrategy.SUMMARIZATION:
            consolidated_memories = await self._consolidate_by_summarization(memories)
        elif strategy == ConsolidationStrategy.CLUSTERING:
            consolidated_memories = await self._consolidate_by_clustering(memories)
        elif strategy == ConsolidationStrategy.PRIORITIZATION:
            consolidated_memories = await self._consolidate_by_prioritization(memories)
        elif strategy == ConsolidationStrategy.COMPRESSION:
            consolidated_memories = await self._consolidate_by_compression(memories)
        elif strategy == ConsolidationStrategy.HYBRID:
            consolidated_memories = await self._consolidate_hybrid(memories)
        else:
            consolidated_memories = memories  # No consolidation
        
        # Update metrics
        processing_time = time.time() - start_time
        final_size = self._calculate_memory_size(consolidated_memories)
        
        self.metrics.items_consolidated = original_count - len(consolidated_memories)
        self.metrics.compression_ratio = final_size / original_size if original_size > 0 else 1.0
        self.metrics.processing_time = processing_time
        self.metrics.storage_saved = original_size - final_size
        self.metrics.information_retention = await self._calculate_information_retention(
            memories, consolidated_memories
        )
        
        logger.info(
            f"Consolidation completed: {original_count} -> {len(consolidated_memories)} items, "
            f"compression: {self.metrics.compression_ratio:.2f}, "
            f"retention: {self.metrics.information_retention:.2f}"
        )
        
        return consolidated_memories
    
    async def _consolidate_by_summarization(
        self,
        memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """Consolidate memories by creating summaries"""
        
        if not memories:
            return []
        
        # Group memories by type and time period
        grouped_memories = self._group_memories_by_similarity(memories)
        consolidated = []
        
        for group in grouped_memories:
            if len(group) <= 1:
                consolidated.extend(group)
                continue
            
            # Create summary for the group
            summary_item = await self._create_summary(group)
            if summary_item:
                consolidated.append(summary_item)
            else:
                # If summarization fails, keep most important item
                group.sort(key=lambda x: x.importance, reverse=True)
                consolidated.append(group[0])
        
        return consolidated
    
    async def _consolidate_by_clustering(
        self,
        memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """Consolidate memories by clustering similar items"""
        
        if len(memories) <= 3:
            return memories
        
        # Simple clustering based on content similarity
        clusters = []
        used_indices = set()
        
        for i, memory in enumerate(memories):
            if i in used_indices:
                continue
            
            cluster = [memory]
            used_indices.add(i)
            
            # Find similar memories
            for j, other_memory in enumerate(memories):
                if j in used_indices:
                    continue
                
                similarity = await self._calculate_content_similarity(memory, other_memory)
                if similarity > self.similarity_threshold:
                    cluster.append(other_memory)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        # Consolidate each cluster
        consolidated = []
        for cluster in clusters:
            if len(cluster) == 1:
                consolidated.extend(cluster)
            else:
                # Create representative item for cluster
                representative = await self._create_cluster_representative(cluster)
                consolidated.append(representative)
        
        return consolidated
    
    async def _consolidate_by_prioritization(
        self,
        memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """Consolidate by keeping only high-priority items"""
        
        if not memories:
            return []
        
        # Calculate priority scores
        prioritized_memories = []
        
        for memory in memories:
            priority_score = self._calculate_priority_score(memory)
            prioritized_memories.append((memory, priority_score))
        
        # Sort by priority and keep top items
        prioritized_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 70% of items or at least 1 item
        keep_count = max(1, int(len(prioritized_memories) * 0.7))
        consolidated = [memory for memory, _ in prioritized_memories[:keep_count]]
        
        return consolidated
    
    async def _consolidate_by_compression(
        self,
        memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """Consolidate by compressing individual memory content"""
        
        compressed_memories = []
        
        for memory in memories:
            compressed_memory = await self._compress_memory_content(memory)
            compressed_memories.append(compressed_memory)
        
        return compressed_memories
    
    async def _consolidate_hybrid(
        self,
        memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """Hybrid consolidation using multiple strategies"""
        
        if not memories:
            return []
        
        # Step 1: Compress individual items
        compressed = await self._consolidate_by_compression(memories)
        
        # Step 2: Cluster similar items
        clustered = await self._consolidate_by_clustering(compressed)
        
        # Step 3: Apply prioritization
        prioritized = await self._consolidate_by_prioritization(clustered)
        
        return prioritized
    
    def _group_memories_by_similarity(
        self,
        memories: List[MemoryItem],
        time_window_hours: int = 2
    ) -> List[List[MemoryItem]]:
        """Group memories by content similarity and temporal proximity"""
        
        if not memories:
            return []
        
        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda x: x.creation_time)
        groups = []
        
        for memory in sorted_memories:
            # Find existing group or create new one
            placed = False
            
            for group in groups:
                if self._should_group_together(memory, group[0], time_window_hours):
                    group.append(memory)
                    placed = True
                    break
            
            if not placed:
                groups.append([memory])
        
        return groups
    
    def _should_group_together(
        self,
        memory1: MemoryItem,
        memory2: MemoryItem,
        time_window_hours: int
    ) -> bool:
        """Check if two memories should be grouped together"""
        
        # Check temporal proximity
        time_diff = abs((memory1.creation_time - memory2.creation_time).total_seconds())
        if time_diff > time_window_hours * 3600:
            return False
        
        # Check memory type compatibility
        if memory1.memory_type != memory2.memory_type:
            return False
        
        # Check content similarity (simplified)
        content1 = json.dumps(memory1.content, default=str).lower()
        content2 = json.dumps(memory2.content, default=str).lower()
        
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.3  # Lower threshold for grouping
    
    async def _create_summary(
        self,
        memories: List[MemoryItem]
    ) -> Optional[MemoryItem]:
        """Create a summary memory item from a group of memories"""
        
        if not memories:
            return None
        
        try:
            # Extract key information
            all_content = []
            combined_importance = 0
            combined_tags = set()
            latest_access = max(memory.last_access for memory in memories)
            
            for memory in memories:
                content_text = self._extract_text_content(memory.content)
                all_content.append(content_text)
                combined_importance += memory.importance
                combined_tags.update(memory.tags)
            
            # Create summary text
            full_text = " ".join(all_content)
            summary_text = await self._summarize_text(full_text)
            
            # Create summary memory item
            summary_memory = MemoryItem(
                id=f"summary_{int(time.time())}_{len(memories)}",
                content={
                    "summary": summary_text,
                    "original_count": len(memories),
                    "memory_ids": [m.id for m in memories],
                    "consolidated": True
                },
                memory_type=memories[0].memory_type,  # Use first memory's type
                importance=combined_importance / len(memories),  # Average importance
                access_count=sum(m.access_count for m in memories),
                last_access=latest_access,
                tags=list(combined_tags)
            )
            
            return summary_memory
            
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            return None
    
    async def _create_cluster_representative(
        self,
        cluster: List[MemoryItem]
    ) -> MemoryItem:
        """Create a representative item for a cluster of memories"""
        
        if not cluster:
            return None
        
        # Find the most important item as base
        cluster.sort(key=lambda x: x.importance, reverse=True)
        representative = cluster[0]
        
        # Merge information from other items
        combined_tags = set(representative.tags)
        combined_access_count = representative.access_count
        latest_access = representative.last_access
        
        additional_info = []
        
        for memory in cluster[1:]:
            combined_tags.update(memory.tags)
            combined_access_count += memory.access_count
            
            if memory.last_access > latest_access:
                latest_access = memory.last_access
            
            # Extract key info from other memories
            content_text = self._extract_text_content(memory.content)
            if len(content_text) > 50:  # Only add substantial content
                additional_info.append(content_text[:100])
        
        # Update representative content
        representative.content["cluster_info"] = {
            "cluster_size": len(cluster),
            "member_ids": [m.id for m in cluster[1:]],
            "additional_content": additional_info[:3]  # Limit additional content
        }
        
        representative.tags = list(combined_tags)
        representative.access_count = combined_access_count
        representative.last_access = latest_access
        representative.consolidation_score += 0.2  # Boost consolidation score
        
        return representative
    
    def _calculate_priority_score(self, memory: MemoryItem) -> float:
        """Calculate priority score for memory item"""
        
        # Base importance
        priority = memory.importance
        
        # Access frequency boost
        access_boost = min(memory.access_count * 0.1, 0.5)
        
        # Recency boost
        hours_since_access = (datetime.utcnow() - memory.last_access).total_seconds() / 3600
        recency_boost = 1.0 / (1.0 + hours_since_access * 0.1)
        
        # Content richness boost
        content_size = len(json.dumps(memory.content, default=str))
        content_boost = min(content_size / 1000, 0.3)  # Up to 0.3 boost for rich content
        
        return priority + access_boost + recency_boost + content_boost
    
    async def _compress_memory_content(self, memory: MemoryItem) -> MemoryItem:
        """Compress content of individual memory item"""
        
        compressed_content = memory.content.copy()
        original_size = len(json.dumps(compressed_content, default=str))
        
        # Compress text fields
        for key, value in memory.content.items():
            if isinstance(value, str) and len(value) > 200:
                compressed_value = await self._compress_text(value)
                compressed_content[key] = compressed_value
            elif key == "description" and isinstance(value, str) and len(value) > 100:
                # Special handling for descriptions
                compressed_content[key] = value[:100] + "..." if len(value) > 100 else value
        
        # Add compression metadata
        final_size = len(json.dumps(compressed_content, default=str))
        compression_ratio = final_size / original_size if original_size > 0 else 1.0
        
        compressed_content["_compression"] = {
            "original_size": original_size,
            "compressed_size": final_size,
            "ratio": compression_ratio,
            "compressed_at": datetime.utcnow().isoformat()
        }
        
        # Create new memory item with compressed content
        compressed_memory = MemoryItem(
            id=memory.id,
            content=compressed_content,
            memory_type=memory.memory_type,
            importance=memory.importance,
            access_count=memory.access_count,
            creation_time=memory.creation_time,
            last_access=memory.last_access,
            consolidation_score=memory.consolidation_score + 0.1,
            tags=memory.tags
        )
        
        return compressed_memory
    
    async def _summarize_text(self, text: str) -> str:
        """Create summary of text content"""
        
        if len(text) <= self.max_summary_length:
            return text
        
        if NLTK_AVAILABLE:
            return self._summarize_with_nltk(text)
        else:
            return self._summarize_simple(text)
    
    def _summarize_with_nltk(self, text: str) -> str:
        """Summarize text using NLTK"""
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= 3:
                return text
            
            # Score sentences based on word frequency
            words = word_tokenize(text.lower())
            words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
            
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            # Score sentences
            sentence_scores = []
            for sentence in sentences:
                sentence_words = word_tokenize(sentence.lower())
                score = 0
                word_count = 0
                
                for word in sentence_words:
                    if word not in self.stop_words:
                        stemmed_word = self.stemmer.stem(word)
                        score += word_freq[stemmed_word]
                        word_count += 1
                
                if word_count > 0:
                    sentence_scores.append((sentence, score / word_count))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            summary_sentences = [sentence for sentence, _ in sentence_scores[:3]]
            
            # Reorder to maintain original sequence
            summary = []
            for sentence in sentences:
                if sentence in summary_sentences:
                    summary.append(sentence)
            
            summary_text = " ".join(summary)
            
            # Truncate if still too long
            if len(summary_text) > self.max_summary_length:
                summary_text = summary_text[:self.max_summary_length] + "..."
            
            return summary_text
            
        except Exception as e:
            logger.warning(f"NLTK summarization failed: {e}")
            return self._summarize_simple(text)
    
    def _summarize_simple(self, text: str) -> str:
        """Simple text summarization without NLTK"""
        
        # Split into sentences (simple approach)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return text[:self.max_summary_length] + ("..." if len(text) > self.max_summary_length else "")
        
        # Take first and last sentences, plus middle if space allows
        summary_parts = [sentences[0]]
        
        if len(sentences) > 2:
            summary_parts.append(sentences[-1])
        
        if len(sentences) > 3:
            middle_idx = len(sentences) // 2
            summary_parts.insert(1, sentences[middle_idx])
        
        summary = ". ".join(summary_parts)
        
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + "..."
        
        return summary
    
    async def _compress_text(self, text: str) -> str:
        """Compress text content while preserving meaning"""
        
        if len(text) <= 200:
            return text
        
        # Remove extra whitespace
        compressed = " ".join(text.split())
        
        # Remove common filler words if text is still too long
        if len(compressed) > 300:
            filler_words = ['very', 'really', 'quite', 'rather', 'pretty', 'fairly', 'extremely']
            words = compressed.split()
            words = [word for word in words if word.lower() not in filler_words]
            compressed = " ".join(words)
        
        # Truncate with ellipsis if still too long
        if len(compressed) > 250:
            compressed = compressed[:247] + "..."
        
        return compressed
    
    async def _calculate_content_similarity(
        self,
        memory1: MemoryItem,
        memory2: MemoryItem
    ) -> float:
        """Calculate content similarity between two memories"""
        
        content1 = self._extract_text_content(memory1.content)
        content2 = self._extract_text_content(memory2.content)
        
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_text_content(self, content: Dict[str, Any]) -> str:
        """Extract text content from structured data"""
        
        if isinstance(content, str):
            return content
        
        text_parts = []
        
        for field in ["description", "content", "text", "message", "summary", "title"]:
            if field in content and isinstance(content[field], str):
                text_parts.append(content[field])
        
        return " ".join(text_parts) if text_parts else json.dumps(content, default=str)
    
    def _calculate_memory_size(self, memories: List[MemoryItem]) -> int:
        """Calculate total size of memory items in bytes"""
        total_size = 0
        
        for memory in memories:
            memory_json = json.dumps(asdict(memory), default=str)
            total_size += len(memory_json.encode('utf-8'))
        
        return total_size
    
    async def _calculate_information_retention(
        self,
        original: List[MemoryItem],
        consolidated: List[MemoryItem]
    ) -> float:
        """Calculate how much information was retained after consolidation"""
        
        if not original:
            return 1.0
        
        original_importance = sum(memory.importance for memory in original)
        consolidated_importance = sum(memory.importance for memory in consolidated)
        
        return consolidated_importance / original_importance if original_importance > 0 else 0.0
    
    async def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get comprehensive consolidation statistics"""
        
        return {
            "metrics": {
                "items_consolidated": self.metrics.items_consolidated,
                "compression_ratio": self.metrics.compression_ratio,
                "information_retention": self.metrics.information_retention,
                "processing_time": self.metrics.processing_time,
                "storage_saved_bytes": self.metrics.storage_saved
            },
            "configuration": {
                "consolidation_threshold": self.consolidation_threshold,
                "max_summary_length": self.max_summary_length,
                "similarity_threshold": self.similarity_threshold,
                "importance_decay": self.importance_decay
            },
            "capabilities": {
                "nltk_available": NLTK_AVAILABLE,
                "summarization": True,
                "clustering": True,
                "compression": True
            },
            "cache_stats": {
                "cache_size": len(self.consolidation_cache),
                "cache_hits": 0  # Could be implemented
            }
        }
    
    def clear_cache(self):
        """Clear consolidation cache"""
        self.consolidation_cache.clear()
        logger.info("Consolidation cache cleared")
