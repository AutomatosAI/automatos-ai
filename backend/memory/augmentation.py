
"""
Memory Augmentation System
==========================

External memory augmentation using vector stores, knowledge graphs, and semantic search.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from .memory_types import MemoryItem, MemoryType

logger = logging.getLogger(__name__)

# Optional imports for advanced features
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, vector store augmentation disabled")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available, using basic text similarity")

class AugmentationStrategy(Enum):
    """Strategies for memory augmentation"""
    VECTOR_SIMILARITY = "vector_similarity"
    SEMANTIC_SEARCH = "semantic_search" 
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HYBRID = "hybrid"

@dataclass
class AugmentedMemory:
    """Augmented memory item with external context"""
    original_memory: MemoryItem
    augmented_content: Dict[str, Any]
    similarity_score: float
    augmentation_source: str
    augmentation_metadata: Dict[str, Any]
    timestamp: datetime

class VectorStoreAugmenter:
    """
    Vector-based memory augmentation using FAISS and sentence transformers
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        max_external_items: int = 10000,
        similarity_threshold: float = 0.7
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_external_items = max_external_items
        
        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(model_name)
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
                self.embeddings_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize SentenceTransformer: {e}")
                self.embeddings_enabled = False
                self.encoder = None
                self.embedding_dim = 384  # Default dimension
        else:
            self.embeddings_enabled = False
            self.encoder = None
            self.embedding_dim = 384
        
        # Initialize FAISS index if available
        if FAISS_AVAILABLE and self.embeddings_enabled:
            try:
                if index_type == "flat":
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                elif index_type == "ivf":
                    quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
                else:
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.vector_search_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS index: {e}")
                self.vector_search_enabled = False
                self.index = None
        else:
            self.vector_search_enabled = False
            self.index = None
        
        # Store external knowledge
        self.external_memories: Dict[int, Dict[str, Any]] = {}
        self.memory_embeddings: Dict[int, np.ndarray] = {}
        self.next_id = 0
        
        logger.info(f"Initialized VectorStoreAugmenter: embeddings={self.embeddings_enabled}, vector_search={self.vector_search_enabled}")
    
    async def add_external_knowledge(
        self,
        content: Dict[str, Any],
        source: str = "external",
        metadata: Dict[str, Any] = None
    ) -> int:
        """Add external knowledge to the augmentation store"""
        
        if len(self.external_memories) >= self.max_external_items:
            await self._manage_external_capacity()
        
        # Generate embedding if possible
        text_content = self._extract_text_content(content)
        
        if self.embeddings_enabled:
            embedding = self.encoder.encode([text_content])[0]
            
            # Store in FAISS index if available
            if self.vector_search_enabled:
                embedding = embedding.astype('float32').reshape(1, -1)
                self.index.add(embedding)
                self.memory_embeddings[memory_id] = embedding.flatten()
        else:
            embedding = None
        
        # Store memory and metadata
        memory_id = self.next_id
        self.external_memories[memory_id] = {
            "content": content,
            "source": source,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "access_count": 0
        }
        
        self.next_id += 1
        logger.debug(f"Added external knowledge item {memory_id}")
        
        return memory_id
    
    async def augment_memory(
        self,
        memory_item: MemoryItem,
        strategy: AugmentationStrategy = AugmentationStrategy.VECTOR_SIMILARITY,
        max_augmentations: int = 5
    ) -> List[AugmentedMemory]:
        """Augment memory item with external knowledge"""
        
        if strategy == AugmentationStrategy.VECTOR_SIMILARITY:
            return await self._augment_with_vectors(memory_item, max_augmentations)
        elif strategy == AugmentationStrategy.SEMANTIC_SEARCH:
            return await self._augment_with_semantic_search(memory_item, max_augmentations)
        elif strategy == AugmentationStrategy.HYBRID:
            return await self._augment_hybrid(memory_item, max_augmentations)
        else:
            return await self._augment_with_vectors(memory_item, max_augmentations)
    
    async def _augment_with_vectors(
        self,
        memory_item: MemoryItem,
        max_augmentations: int
    ) -> List[AugmentedMemory]:
        """Augment using vector similarity search"""
        
        # Fall back to semantic search if vector search is not available
        if not self.vector_search_enabled or not self.index or self.index.ntotal == 0:
            return await self._augment_with_semantic_search(memory_item, max_augmentations)
        
        # Generate query embedding
        query_text = self._extract_text_content(memory_item.content)
        if not self.embeddings_enabled:
            return await self._augment_with_semantic_search(memory_item, max_augmentations)
            
        query_embedding = self.encoder.encode([query_text])[0]
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search similar items
        k = min(max_augmentations * 2, self.index.ntotal)  # Get more than needed
        distances, indices = self.index.search(query_embedding, k)
        
        augmented_memories = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.external_memories):
                continue
                
            # Convert distance to similarity (FAISS L2 distance)
            similarity = 1.0 / (1.0 + distance)
            
            if similarity < self.similarity_threshold:
                continue
            
            external_memory = self.external_memories[idx]
            external_memory["access_count"] += 1
            
            augmented_memory = AugmentedMemory(
                original_memory=memory_item,
                augmented_content=external_memory["content"],
                similarity_score=similarity,
                augmentation_source=external_memory["source"],
                augmentation_metadata={
                    "method": "vector_similarity",
                    "distance": float(distance),
                    "external_id": idx,
                    **external_memory["metadata"]
                },
                timestamp=datetime.utcnow()
            )
            
            augmented_memories.append(augmented_memory)
            
            if len(augmented_memories) >= max_augmentations:
                break
        
        logger.debug(f"Generated {len(augmented_memories)} vector augmentations for {memory_item.id}")
        return augmented_memories
    
    async def _augment_with_semantic_search(
        self,
        memory_item: MemoryItem,
        max_augmentations: int
    ) -> List[AugmentedMemory]:
        """Augment using semantic search with keyword extraction"""
        
        # Extract keywords from memory content
        query_text = self._extract_text_content(memory_item.content)
        keywords = self._extract_keywords(query_text)
        
        augmented_memories = []
        
        # Search external memories for keyword matches
        for ext_id, ext_memory in self.external_memories.items():
            ext_text = self._extract_text_content(ext_memory["content"])
            
            # Calculate semantic similarity
            semantic_score = self._calculate_semantic_similarity(query_text, ext_text, keywords)
            
            if semantic_score > self.similarity_threshold:
                ext_memory["access_count"] += 1
                
                augmented_memory = AugmentedMemory(
                    original_memory=memory_item,
                    augmented_content=ext_memory["content"],
                    similarity_score=semantic_score,
                    augmentation_source=ext_memory["source"],
                    augmentation_metadata={
                        "method": "semantic_search",
                        "keywords_matched": keywords,
                        "external_id": ext_id,
                        **ext_memory["metadata"]
                    },
                    timestamp=datetime.utcnow()
                )
                
                augmented_memories.append((augmented_memory, semantic_score))
        
        # Sort by similarity and return top results
        augmented_memories.sort(key=lambda x: x[1], reverse=True)
        result = [aug_mem for aug_mem, _ in augmented_memories[:max_augmentations]]
        
        logger.debug(f"Generated {len(result)} semantic augmentations for {memory_item.id}")
        return result
    
    async def _augment_hybrid(
        self,
        memory_item: MemoryItem,
        max_augmentations: int
    ) -> List[AugmentedMemory]:
        """Hybrid augmentation combining vector and semantic approaches"""
        
        # Get results from both methods
        vector_results = await self._augment_with_vectors(memory_item, max_augmentations)
        semantic_results = await self._augment_with_semantic_search(memory_item, max_augmentations)
        
        # Combine and deduplicate
        all_results = {}
        
        for aug_mem in vector_results:
            key = aug_mem.augmentation_metadata.get("external_id", "unknown")
            all_results[key] = aug_mem
        
        for aug_mem in semantic_results:
            key = aug_mem.augmentation_metadata.get("external_id", "unknown")
            if key in all_results:
                # Combine scores
                existing = all_results[key]
                combined_score = (existing.similarity_score + aug_mem.similarity_score) / 2
                existing.similarity_score = combined_score
                existing.augmentation_metadata["method"] = "hybrid"
            else:
                all_results[key] = aug_mem
        
        # Sort by combined score and return top results
        final_results = sorted(
            all_results.values(),
            key=lambda x: x.similarity_score,
            reverse=True
        )
        
        return final_results[:max_augmentations]
    
    def _extract_text_content(self, content: Dict[str, Any]) -> str:
        """Extract text content from structured data"""
        if isinstance(content, str):
            return content
        
        text_parts = []
        
        # Common text fields
        for field in ["description", "content", "text", "message", "summary", "title", "name"]:
            if field in content and content[field]:
                text_parts.append(str(content[field]))
        
        # If no standard fields, convert entire content
        if not text_parts:
            text_parts.append(json.dumps(content, default=str))
        
        return " ".join(text_parts)
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Simple keyword extraction (can be enhanced with NLP)"""
        words = text.lower().split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get most frequent words
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _calculate_semantic_similarity(
        self,
        text1: str, 
        text2: str, 
        keywords: List[str]
    ) -> float:
        """Calculate semantic similarity between texts"""
        
        # Keyword-based similarity
        text2_lower = text2.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in text2_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0
        
        # Simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return keyword_score
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        overlap_score = len(intersection) / len(union)
        
        # Combine scores
        return (keyword_score * 0.6) + (overlap_score * 0.4)
    
    async def _manage_external_capacity(self):
        """Manage external memory capacity by removing least accessed items"""
        
        if len(self.external_memories) < self.max_external_items:
            return
        
        # Sort by access count and creation time
        sorted_items = sorted(
            self.external_memories.items(),
            key=lambda x: (x[1]["access_count"], x[1]["created_at"])
        )
        
        # Remove 10% of least accessed items
        remove_count = max(1, len(sorted_items) // 10)
        
        for i in range(remove_count):
            ext_id, _ = sorted_items[i]
            
            # Remove from all data structures
            if ext_id in self.external_memories:
                del self.external_memories[ext_id]
            if ext_id in self.memory_embeddings:
                del self.memory_embeddings[ext_id]
        
        # Rebuild FAISS index
        await self._rebuild_index()
        
        logger.info(f"Removed {remove_count} external memory items")
    
    async def _rebuild_index(self):
        """Rebuild FAISS index after removing items"""
        # Create new index
        if hasattr(self.index, 'nlist'):  # IVF index
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            new_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add remaining embeddings
        if self.memory_embeddings:
            embeddings = np.array(list(self.memory_embeddings.values())).astype('float32')
            new_index.add(embeddings)
            
            if hasattr(new_index, 'train') and not new_index.is_trained:
                new_index.train(embeddings)
        
        self.index = new_index
        logger.debug("FAISS index rebuilt")
    
    async def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get comprehensive augmentation statistics"""
        
        if not self.external_memories:
            return {
                "total_external_items": 0,
                "index_size": 0,
                "avg_access_count": 0,
                "sources": {},
                "embedding_stats": {}
            }
        
        # Calculate statistics
        access_counts = [mem["access_count"] for mem in self.external_memories.values()]
        sources = {}
        
        for mem in self.external_memories.values():
            source = mem["source"]
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_external_items": len(self.external_memories),
            "index_size": self.index.ntotal,
            "avg_access_count": np.mean(access_counts),
            "max_access_count": max(access_counts),
            "sources": sources,
            "embedding_stats": {
                "dimension": self.embedding_dim,
                "model": self.model_name,
                "similarity_threshold": self.similarity_threshold
            },
            "capacity": {
                "current": len(self.external_memories),
                "max": self.max_external_items,
                "utilization": len(self.external_memories) / self.max_external_items * 100
            }
        }
    
    async def save_external_knowledge(self, file_path: str):
        """Save external knowledge to file"""
        state = {
            "external_memories": self.external_memories,
            "config": {
                "model_name": self.model_name,
                "similarity_threshold": self.similarity_threshold,
                "max_external_items": self.max_external_items,
                "embedding_dim": self.embedding_dim
            },
            "stats": {
                "total_items": len(self.external_memories),
                "next_id": self.next_id
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save state
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save FAISS index
        index_path = str(Path(file_path).with_suffix('.faiss'))
        faiss.write_index(self.index, index_path)
        
        logger.info(f"External knowledge saved to {file_path}")
    
    async def load_external_knowledge(self, file_path: str):
        """Load external knowledge from file"""
        try:
            # Load state
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.external_memories = {
                int(k): v for k, v in state["external_memories"].items()
            }
            self.next_id = state["stats"]["next_id"]
            
            # Load FAISS index
            index_path = str(Path(file_path).with_suffix('.faiss'))
            if Path(index_path).exists():
                self.index = faiss.read_index(index_path)
            
            # Regenerate embeddings if needed
            if not hasattr(self, 'memory_embeddings'):
                self.memory_embeddings = {}
                
                for ext_id, ext_memory in self.external_memories.items():
                    text_content = self._extract_text_content(ext_memory["content"])
                    embedding = self.encoder.encode([text_content])[0]
                    self.memory_embeddings[ext_id] = embedding
            
            logger.info(f"External knowledge loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load external knowledge: {e}")
            raise
    
    def clear_external_knowledge(self):
        """Clear all external knowledge"""
        self.external_memories.clear()
        self.memory_embeddings.clear()
        
        # Reset FAISS index
        if hasattr(self.index, 'nlist'):
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.next_id = 0
        logger.info("External knowledge cleared")
