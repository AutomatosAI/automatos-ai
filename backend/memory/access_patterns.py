
"""
Memory Access Pattern Optimizer
===============================

Optimizes memory access patterns using locality principles and predictive algorithms.
"""

import asyncio
import logging
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .memory_types import MemoryItem, MemoryLevel, HierarchicalMemoryManager

logger = logging.getLogger(__name__)

class AccessPattern(Enum):
    """Different memory access patterns"""
    SEQUENTIAL = "sequential"        # Linear access pattern
    RANDOM = "random"               # Random access pattern  
    LOCALITY_BASED = "locality"     # Spatial/temporal locality
    FREQUENCY_BASED = "frequency"   # Access frequency based
    RECENCY_BASED = "recency"      # Recently accessed items
    SEMANTIC_BASED = "semantic"     # Content similarity based

@dataclass
class AccessMetrics:
    """Metrics for memory access patterns"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_response_time: float = 0.0
    cache_utilization: float = 0.0
    locality_factor: float = 0.0
    prediction_accuracy: float = 0.0

class MemoryAccessOptimizer:
    """
    Optimizes memory access patterns and implements intelligent caching
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        prediction_window: int = 100,
        locality_threshold: float = 0.7
    ):
        self.cache_size = cache_size
        self.prediction_window = prediction_window
        self.locality_threshold = locality_threshold
        
        # LRU Cache for frequently accessed items
        self.access_cache: OrderedDict[str, MemoryItem] = OrderedDict()
        
        # Access pattern tracking
        self.access_history: List[Tuple[str, datetime]] = []
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.access_sequences: List[str] = []
        
        # Pattern recognition
        self.pattern_predictors = {
            AccessPattern.SEQUENTIAL: self._predict_sequential,
            AccessPattern.LOCALITY_BASED: self._predict_locality,
            AccessPattern.FREQUENCY_BASED: self._predict_frequency,
            AccessPattern.RECENCY_BASED: self._predict_recency
        }
        
        # Performance metrics
        self.metrics = AccessMetrics()
        self._total_accesses = 0
        self._cache_hits = 0
        self._response_times = []
    
    async def optimize_access(
        self,
        memory_manager: HierarchicalMemoryManager,
        query: str,
        session_id: str,
        access_pattern: AccessPattern = AccessPattern.LOCALITY_BASED
    ) -> List[MemoryItem]:
        """Optimize memory access using specified pattern"""
        start_time = time.time()
        
        # Check cache first
        cached_result = await self._check_cache(query, session_id)
        if cached_result:
            self._record_cache_hit()
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            return cached_result
        
        # Predict likely items to access
        predicted_items = await self._predict_access_pattern(
            memory_manager, query, session_id, access_pattern
        )
        
        # Retrieve items with optimization
        optimized_items = await self._optimized_retrieval(
            memory_manager, predicted_items, query, session_id
        )
        
        # Cache the results
        await self._cache_result(query, session_id, optimized_items)
        
        # Record access pattern
        self._record_access(session_id, query)
        
        response_time = time.time() - start_time
        self._update_metrics(response_time, False)
        
        return optimized_items
    
    async def _check_cache(self, query: str, session_id: str) -> Optional[List[MemoryItem]]:
        """Check if result is in access cache"""
        cache_key = f"{session_id}:{hash(query)}"
        
        if cache_key in self.access_cache:
            # Move to end (LRU)
            item = self.access_cache.pop(cache_key)
            self.access_cache[cache_key] = item
            
            logger.debug(f"Cache hit for query: {query[:50]}")
            return [item] if isinstance(item, MemoryItem) else item
        
        return None
    
    async def _cache_result(self, query: str, session_id: str, items: List[MemoryItem]):
        """Cache access result"""
        cache_key = f"{session_id}:{hash(query)}"
        
        # Manage cache size
        if len(self.access_cache) >= self.cache_size:
            # Remove oldest item
            self.access_cache.popitem(last=False)
        
        self.access_cache[cache_key] = items
    
    async def _predict_access_pattern(
        self,
        memory_manager: HierarchicalMemoryManager,
        query: str,
        session_id: str,
        pattern: AccessPattern
    ) -> List[str]:
        """Predict likely memory items to be accessed"""
        predictor = self.pattern_predictors.get(pattern, self._predict_locality)
        return await predictor(memory_manager, query, session_id)
    
    async def _predict_sequential(
        self,
        memory_manager: HierarchicalMemoryManager,
        query: str,
        session_id: str
    ) -> List[str]:
        """Predict sequential access pattern"""
        # Look for sequential patterns in access history
        recent_accesses = [item_id for item_id, _ in self.access_history[-10:]]
        
        if len(recent_accesses) > 1:
            # Find items that were accessed after the recent ones
            predicted_ids = []
            for level in MemoryLevel:
                store = memory_manager.memory_stores[level]
                store_ids = list(store.keys())
                
                # Look for sequential patterns
                for recent_id in recent_accesses:
                    try:
                        idx = store_ids.index(recent_id)
                        if idx < len(store_ids) - 1:
                            predicted_ids.append(store_ids[idx + 1])
                    except ValueError:
                        continue
            
            return predicted_ids[:10]
        
        return []
    
    async def _predict_locality(
        self,
        memory_manager: HierarchicalMemoryManager,
        query: str,
        session_id: str
    ) -> List[str]:
        """Predict based on temporal and spatial locality"""
        recent_time = datetime.utcnow() - timedelta(minutes=30)
        locality_candidates = []
        
        # Temporal locality - recently accessed items
        recent_items = [
            item_id for item_id, access_time in self.access_history
            if access_time > recent_time
        ]
        
        # Spatial locality - items with similar content
        for level in MemoryLevel:
            store = memory_manager.memory_stores[level]
            
            for item_id, item in store.items():
                if item_id.startswith(session_id):  # Same session
                    # Simple content similarity check
                    content_text = str(item.content).lower()
                    query_lower = query.lower()
                    
                    similarity = self._calculate_text_similarity(content_text, query_lower)
                    if similarity > self.locality_threshold:
                        locality_candidates.append((item_id, similarity))
        
        # Combine and rank candidates
        combined_candidates = set(recent_items)
        combined_candidates.update(item_id for item_id, _ in locality_candidates)
        
        # Sort by access frequency and recency
        ranked_candidates = []
        for item_id in combined_candidates:
            frequency = self.access_counts.get(item_id, 0)
            recency_score = 1.0
            
            # Calculate recency score
            for access_id, access_time in reversed(self.access_history):
                if access_id == item_id:
                    time_diff = (datetime.utcnow() - access_time).total_seconds()
                    recency_score = 1.0 / (1.0 + time_diff / 3600)  # Decay over hours
                    break
            
            combined_score = frequency * 0.6 + recency_score * 0.4
            ranked_candidates.append((item_id, combined_score))
        
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in ranked_candidates[:15]]
    
    async def _predict_frequency(
        self,
        memory_manager: HierarchicalMemoryManager,
        query: str,
        session_id: str
    ) -> List[str]:
        """Predict based on access frequency"""
        # Sort by access frequency
        frequent_items = sorted(
            self.access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [item_id for item_id, _ in frequent_items[:10]]
    
    async def _predict_recency(
        self,
        memory_manager: HierarchicalMemoryManager,
        query: str,
        session_id: str
    ) -> List[str]:
        """Predict based on recency"""
        # Return most recently accessed items
        recent_items = [
            item_id for item_id, _ in reversed(self.access_history[-20:])
        ]
        
        return recent_items[:10]
    
    async def _optimized_retrieval(
        self,
        memory_manager: HierarchicalMemoryManager,
        predicted_items: List[str],
        query: str,
        session_id: str
    ) -> List[MemoryItem]:
        """Perform optimized retrieval based on predictions"""
        retrieved_items = []
        
        # First, try to get predicted items directly
        for level in MemoryLevel:
            store = memory_manager.memory_stores[level]
            
            for item_id in predicted_items:
                if item_id in store:
                    item = store[item_id]
                    relevance = await memory_manager._calculate_relevance(item, query)
                    
                    if relevance > 0.3:
                        retrieved_items.append((item, relevance))
        
        # If we don't have enough, fall back to regular retrieval
        if len(retrieved_items) < 5:
            fallback_items = await memory_manager.retrieve_memory(
                session_id, query, max_items=10
            )
            
            for item in fallback_items:
                if item.id not in [existing[0].id for existing in retrieved_items]:
                    relevance = await memory_manager._calculate_relevance(item, query)
                    retrieved_items.append((item, relevance))
        
        # Sort by relevance and return top items
        retrieved_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in retrieved_items[:10]]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (can be enhanced with embeddings)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _record_access(self, session_id: str, query: str):
        """Record memory access for pattern learning"""
        access_time = datetime.utcnow()
        item_id = f"{session_id}:{hash(query)}"
        
        self.access_history.append((item_id, access_time))
        self.access_counts[item_id] += 1
        self.access_sequences.append(item_id)
        
        # Keep history bounded
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-5000:]
        
        if len(self.access_sequences) > 1000:
            self.access_sequences = self.access_sequences[-500:]
    
    def _record_cache_hit(self):
        """Record cache hit for metrics"""
        self._cache_hits += 1
        self._total_accesses += 1
    
    def _update_metrics(self, response_time: float, cache_hit: bool):
        """Update performance metrics"""
        self._total_accesses += 1
        self._response_times.append(response_time)
        
        # Keep response times bounded
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-500:]
        
        # Calculate metrics
        self.metrics.hit_rate = self._cache_hits / self._total_accesses if self._total_accesses > 0 else 0
        self.metrics.miss_rate = 1.0 - self.metrics.hit_rate
        self.metrics.avg_response_time = np.mean(self._response_times)
        self.metrics.cache_utilization = len(self.access_cache) / self.cache_size * 100
        
        # Calculate locality factor
        if len(self.access_sequences) > 10:
            locality_hits = 0
            for i in range(1, min(len(self.access_sequences), 100)):
                if self.access_sequences[i] in self.access_sequences[max(0, i-5):i]:
                    locality_hits += 1
            self.metrics.locality_factor = locality_hits / 99 if len(self.access_sequences) > 10 else 0
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            "metrics": {
                "hit_rate": self.metrics.hit_rate,
                "miss_rate": self.metrics.miss_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "cache_utilization": self.metrics.cache_utilization,
                "locality_factor": self.metrics.locality_factor
            },
            "cache_info": {
                "size": len(self.access_cache),
                "capacity": self.cache_size,
                "utilization": len(self.access_cache) / self.cache_size * 100
            },
            "access_patterns": {
                "total_accesses": self._total_accesses,
                "unique_items": len(self.access_counts),
                "avg_access_frequency": np.mean(list(self.access_counts.values())) if self.access_counts else 0
            },
            "performance": {
                "cache_hits": self._cache_hits,
                "total_accesses": self._total_accesses,
                "response_times": {
                    "mean": np.mean(self._response_times) if self._response_times else 0,
                    "std": np.std(self._response_times) if self._response_times else 0,
                    "min": min(self._response_times) if self._response_times else 0,
                    "max": max(self._response_times) if self._response_times else 0
                }
            }
        }
    
    def clear_cache(self):
        """Clear access cache"""
        self.access_cache.clear()
        logger.info("Access cache cleared")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = AccessMetrics()
        self._total_accesses = 0
        self._cache_hits = 0
        self._response_times = []
        self.access_history = []
        self.access_counts.clear()
        self.access_sequences = []
        logger.info("Access metrics reset")
