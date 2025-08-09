
"""
Hierarchical Memory Types Implementation
=======================================

Implements multi-level memory system with retention curves and intelligent promotion/demotion.
Based on cognitive science principles and optimized for AI systems.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryLevel(Enum):
    """Memory hierarchy levels based on cognitive science"""
    IMMEDIATE = "immediate"        # 0.1-2 seconds, very limited capacity
    WORKING = "working"           # 2-30 seconds, limited capacity  
    SHORT_TERM = "short_term"     # 30 seconds - 1 hour, moderate capacity
    LONG_TERM = "long_term"       # 1 hour - weeks, large capacity
    ARCHIVAL = "archival"         # Permanent storage, unlimited capacity

class MemoryType(Enum):
    """Types of memory content"""
    SEMANTIC = "semantic"         # Facts and knowledge
    EPISODIC = "episodic"        # Experiences and events
    PROCEDURAL = "procedural"    # Skills and procedures
    WORKING_DATA = "working_data" # Active processing data
    CONTEXTUAL = "contextual"    # Context information

@dataclass
class MemoryItem:
    """Individual memory item with metadata"""
    id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    importance: float = 0.5
    access_count: int = 0
    creation_time: datetime = None
    last_access: datetime = None
    decay_factor: float = 0.1
    consolidation_score: float = 0.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = datetime.utcnow()
        if self.last_access is None:
            self.last_access = self.creation_time
        if self.tags is None:
            self.tags = []
    
    def calculate_retention(self) -> float:
        """Calculate current retention based on Ebbinghaus forgetting curve"""
        time_elapsed = (datetime.utcnow() - self.last_access).total_seconds() / 3600  # hours
        retention = np.exp(-self.decay_factor * time_elapsed)
        # Boost retention based on importance and access frequency
        boost = 1 + (self.importance * 0.5) + (min(self.access_count, 10) * 0.1)
        return min(retention * boost, 1.0)
    
    def update_access(self):
        """Update access statistics"""
        self.last_access = datetime.utcnow()
        self.access_count += 1

class HierarchicalMemoryManager:
    """
    Advanced hierarchical memory system with cognitive-inspired retention
    """
    
    def __init__(
        self,
        immediate_capacity: int = 7,      # Miller's magic number ±2
        working_capacity: int = 100,
        short_term_capacity: int = 1000,
        long_term_capacity: int = 100000,
        archival_capacity: int = 1000000,
        base_decay_rate: float = 0.1
    ):
        self.capacities = {
            MemoryLevel.IMMEDIATE: immediate_capacity,
            MemoryLevel.WORKING: working_capacity,
            MemoryLevel.SHORT_TERM: short_term_capacity,
            MemoryLevel.LONG_TERM: long_term_capacity,
            MemoryLevel.ARCHIVAL: archival_capacity
        }
        
        self.memory_stores: Dict[MemoryLevel, Dict[str, MemoryItem]] = {
            level: {} for level in MemoryLevel
        }
        
        self.base_decay_rate = base_decay_rate
        self.promotion_thresholds = {
            MemoryLevel.IMMEDIATE: 0.8,
            MemoryLevel.WORKING: 0.7,
            MemoryLevel.SHORT_TERM: 0.6,
            MemoryLevel.LONG_TERM: 0.5
        }
        
        # Background maintenance task (will be started when needed)
        self._maintenance_task = None
        # Don't start maintenance automatically to avoid event loop issues
    
    async def store_memory(
        self, 
        session_id: str,
        content: Dict[str, Any], 
        memory_type: MemoryType = MemoryType.WORKING_DATA,
        importance: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """Store new memory item in appropriate level"""
        memory_id = f"{session_id}_{int(time.time() * 1000)}"
        
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or []
        )
        
        # Start in immediate memory
        level = MemoryLevel.IMMEDIATE
        await self._store_at_level(level, memory_item)
        
        logger.info(f"Stored memory {memory_id} at {level.value} level")
        return memory_id
    
    async def _store_at_level(self, level: MemoryLevel, item: MemoryItem):
        """Store item at specific memory level with capacity management"""
        store = self.memory_stores[level]
        capacity = self.capacities[level]
        
        # Check capacity and make room if needed
        if len(store) >= capacity:
            await self._manage_capacity(level)
        
        store[item.id] = item
    
    async def _manage_capacity(self, level: MemoryLevel):
        """Manage capacity by promoting important items and forgetting others"""
        store = self.memory_stores[level]
        
        if level == MemoryLevel.ARCHIVAL:
            # Archival is unlimited, but we can compress old items
            return
        
        # Calculate retention scores for all items
        items_with_scores = []
        for item in store.values():
            retention = item.calculate_retention()
            items_with_scores.append((item, retention))
        
        # Sort by retention score
        items_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Promote top items to next level
        promotion_count = min(len(items_with_scores) // 4, 10)
        next_level = self._get_next_level(level)
        
        for item, score in items_with_scores[:promotion_count]:
            if score > self.promotion_thresholds.get(level, 0.5) and next_level:
                await self._promote_item(item, level, next_level)
        
        # Remove items with lowest retention
        removal_count = len(store) - self.capacities[level] + 1
        for item, _ in items_with_scores[-removal_count:]:
            del store[item.id]
            logger.debug(f"Removed memory item {item.id} from {level.value}")
    
    async def _promote_item(self, item: MemoryItem, from_level: MemoryLevel, to_level: MemoryLevel):
        """Promote memory item to higher level"""
        # Remove from current level
        del self.memory_stores[from_level][item.id]
        
        # Consolidate content for higher level
        if to_level in [MemoryLevel.LONG_TERM, MemoryLevel.ARCHIVAL]:
            item = await self._consolidate_item(item)
        
        # Store at new level
        await self._store_at_level(to_level, item)
        logger.debug(f"Promoted memory {item.id} from {from_level.value} to {to_level.value}")
    
    async def _consolidate_item(self, item: MemoryItem) -> MemoryItem:
        """Consolidate memory item for long-term storage"""
        # Summarize content while preserving key information
        content = item.content.copy()
        
        # Simple summarization (can be enhanced with AI)
        if "description" in content:
            desc = content["description"]
            if len(desc) > 500:
                # Simple truncation with key phrase preservation
                content["description"] = desc[:200] + "..." + desc[-100:]
                content["summarized"] = True
        
        # Increase consolidation score
        item.consolidation_score += 0.1
        item.content = content
        
        return item
    
    def _get_next_level(self, current: MemoryLevel) -> Optional[MemoryLevel]:
        """Get next memory level in hierarchy"""
        level_order = [
            MemoryLevel.IMMEDIATE,
            MemoryLevel.WORKING,
            MemoryLevel.SHORT_TERM, 
            MemoryLevel.LONG_TERM,
            MemoryLevel.ARCHIVAL
        ]
        
        try:
            current_idx = level_order.index(current)
            if current_idx < len(level_order) - 1:
                return level_order[current_idx + 1]
        except ValueError:
            pass
        
        return None
    
    async def retrieve_memory(
        self, 
        session_id: str,
        query: str = None,
        memory_type: MemoryType = None,
        max_items: int = 10,
        min_importance: float = 0.0
    ) -> List[MemoryItem]:
        """Retrieve relevant memories across all levels"""
        relevant_items = []
        
        # Search across all levels
        for level in MemoryLevel:
            store = self.memory_stores[level]
            
            for item in store.values():
                # Filter by session if specified
                if session_id and not item.id.startswith(session_id):
                    continue
                
                # Filter by memory type
                if memory_type and item.memory_type != memory_type:
                    continue
                
                # Filter by importance
                if item.importance < min_importance:
                    continue
                
                # Calculate relevance score
                relevance = await self._calculate_relevance(item, query)
                
                if relevance > 0.3:  # Minimum relevance threshold
                    item.update_access()  # Update access stats
                    relevant_items.append((item, relevance))
        
        # Sort by relevance and return top items
        relevant_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in relevant_items[:max_items]]
    
    async def _calculate_relevance(self, item: MemoryItem, query: str = None) -> float:
        """Calculate relevance score for memory item"""
        base_score = item.importance * item.calculate_retention()
        
        if query:
            # Simple text similarity (can be enhanced with embeddings)
            content_text = json.dumps(item.content, default=str).lower()
            query_lower = query.lower()
            
            # Basic keyword matching
            query_words = query_lower.split()
            matches = sum(1 for word in query_words if word in content_text)
            text_similarity = matches / len(query_words) if query_words else 0
            
            return base_score * (0.5 + text_similarity * 0.5)
        
        return base_score
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            "levels": {},
            "total_items": 0,
            "memory_types": {},
            "retention_distribution": []
        }
        
        for level in MemoryLevel:
            store = self.memory_stores[level]
            level_stats = {
                "count": len(store),
                "capacity": self.capacities[level],
                "utilization": len(store) / self.capacities[level] * 100,
                "avg_importance": 0,
                "avg_retention": 0
            }
            
            if store:
                importance_values = [item.importance for item in store.values()]
                retention_values = [item.calculate_retention() for item in store.values()]
                level_stats["avg_importance"] = np.mean(importance_values)
                level_stats["avg_retention"] = np.mean(retention_values)
            
            stats["levels"][level.value] = level_stats
            stats["total_items"] += len(store)
        
        # Memory type distribution
        for level in MemoryLevel:
            for item in self.memory_stores[level].values():
                mem_type = item.memory_type.value
                if mem_type not in stats["memory_types"]:
                    stats["memory_types"][mem_type] = 0
                stats["memory_types"][mem_type] += 1
        
        return stats
    
    def start_maintenance(self):
        """Start background maintenance task"""
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def _maintenance_loop(self):
        """Background maintenance for memory consolidation and cleanup"""
        while True:
            try:
                # Run maintenance every 5 minutes
                await asyncio.sleep(300)
                
                # Trigger capacity management for all levels
                for level in MemoryLevel:
                    if len(self.memory_stores[level]) > self.capacities[level] * 0.8:
                        await self._manage_capacity(level)
                
                logger.debug("Memory maintenance completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory maintenance: {e}")
    
    def stop_maintenance(self):
        """Stop background maintenance task"""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            self._maintenance_task = None
    
    async def save_state(self, file_path: str):
        """Save memory state to file"""
        state = {
            "memory_stores": {},
            "capacities": {k.value: v for k, v in self.capacities.items()},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Convert memory stores to serializable format
        for level, store in self.memory_stores.items():
            state["memory_stores"][level.value] = {
                item_id: asdict(item) for item_id, item in store.items()
            }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Memory state saved to {file_path}")
    
    async def load_state(self, file_path: str):
        """Load memory state from file"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore memory stores
            for level_str, store_data in state["memory_stores"].items():
                level = MemoryLevel(level_str)
                self.memory_stores[level] = {}
                
                for item_id, item_data in store_data.items():
                    # Reconstruct MemoryItem
                    item_data["memory_type"] = MemoryType(item_data["memory_type"])
                    item_data["creation_time"] = datetime.fromisoformat(item_data["creation_time"])
                    item_data["last_access"] = datetime.fromisoformat(item_data["last_access"])
                    
                    memory_item = MemoryItem(**item_data)
                    self.memory_stores[level][item_id] = memory_item
            
            logger.info(f"Memory state loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load memory state: {e}")
            raise
