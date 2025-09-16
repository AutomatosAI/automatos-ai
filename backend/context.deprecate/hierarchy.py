
"""
Hierarchical Memory Management Module
Multi-level memory system with intelligent promotion/demotion and optimization
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import numpy as np
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MemoryLevel(Enum):
    """Memory hierarchy levels"""
    IMMEDIATE = "immediate"
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    ARCHIVAL = "archival"

@dataclass
class MemoryItem:
    """Memory item with metadata"""
    content: Dict[str, Any]
    importance: float
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    decay_rate: float = 0.1
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate size after initialization"""
        self.size_bytes = len(json.dumps(self.content, default=str).encode('utf-8'))
    
    def calculate_retention_score(self) -> float:
        """Calculate retention score based on importance and access pattern"""
        try:
            # Time decay factor
            time_diff = (datetime.utcnow() - self.last_accessed).total_seconds()
            decay_factor = np.exp(-self.decay_rate * time_diff / 3600)  # hourly decay
            
            # Access frequency factor
            access_factor = min(self.access_count / 10, 1.0)  # normalize to [0,1]
            
            # Combined score
            retention_score = self.importance * decay_factor * (1 + access_factor)
            return max(0.0, min(1.0, retention_score))
        
        except Exception as e:
            logger.error(f"Error calculating retention score: {e}")
            return self.importance

@dataclass
class MemoryLevelConfig:
    """Configuration for each memory level"""
    capacity: int
    retention_threshold: float
    promotion_threshold: float
    demotion_threshold: float
    compression_enabled: bool = False

class HierarchicalMemoryManager:
    """Advanced hierarchical memory management system"""
    
    def __init__(self):
        # Memory storage
        self.memory_hierarchy: Dict[str, Dict[MemoryLevel, deque]] = defaultdict(
            lambda: {
                MemoryLevel.IMMEDIATE: deque(maxlen=50),
                MemoryLevel.WORKING: deque(maxlen=500),
                MemoryLevel.SHORT_TERM: deque(maxlen=5000),
                MemoryLevel.LONG_TERM: deque(maxlen=50000),
                MemoryLevel.ARCHIVAL: deque(maxlen=100000)
            }
        )
        
        # Level configurations
        self.level_configs = {
            MemoryLevel.IMMEDIATE: MemoryLevelConfig(50, 0.9, 0.8, 0.3),
            MemoryLevel.WORKING: MemoryLevelConfig(500, 0.8, 0.7, 0.4),
            MemoryLevel.SHORT_TERM: MemoryLevelConfig(5000, 0.7, 0.6, 0.5),
            MemoryLevel.LONG_TERM: MemoryLevelConfig(50000, 0.6, 0.5, 0.6, True),
            MemoryLevel.ARCHIVAL: MemoryLevelConfig(100000, 0.5, 0.0, 0.7, True)
        }
        
        # Access patterns and statistics
        self.access_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.promotion_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized HierarchicalMemoryManager")
    
    async def store_memory(self, session_id: str, content: Dict[str, Any], 
                          importance: float = 0.5, level: MemoryLevel = MemoryLevel.IMMEDIATE) -> str:
        """Store memory item in specified level"""
        try:
            # Create memory item
            memory_item = MemoryItem(
                content=content,
                importance=importance
            )
            
            # Add to hierarchy
            if session_id not in self.memory_hierarchy:
                self.memory_hierarchy[session_id] = {
                    MemoryLevel.IMMEDIATE: deque(maxlen=50),
                    MemoryLevel.WORKING: deque(maxlen=500),
                    MemoryLevel.SHORT_TERM: deque(maxlen=5000),
                    MemoryLevel.LONG_TERM: deque(maxlen=50000),
                    MemoryLevel.ARCHIVAL: deque(maxlen=100000)
                }
            
            hierarchy = self.memory_hierarchy[session_id]
            hierarchy[level].append(memory_item)
            
            # Check for promotion/demotion
            await self._manage_memory_levels(session_id, level)
            
            # Generate item ID
            item_id = f"{session_id}_{level.value}_{len(hierarchy[level])}"
            
            logger.debug(f"Stored memory item {item_id} in {level.value} level")
            return item_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise
    
    async def retrieve_memory(self, session_id: str, query: str, 
                            max_items: int = 10, levels: Optional[List[MemoryLevel]] = None) -> List[Dict[str, Any]]:
        """Retrieve memory items matching query from specified levels"""
        try:
            if session_id not in self.memory_hierarchy:
                return []
            
            if levels is None:
                levels = list(MemoryLevel)
            
            hierarchy = self.memory_hierarchy[session_id]
            results = []
            
            for level in levels:
                level_results = await self._search_level(hierarchy[level], query, max_items)
                for item, score in level_results:
                    # Update access statistics
                    item.access_count += 1
                    item.last_accessed = datetime.utcnow()
                    self.access_patterns[session_id][level.value] += 1
                    
                    results.append({
                        "content": item.content,
                        "level": level.value,
                        "importance": item.importance,
                        "relevance_score": score,
                        "access_count": item.access_count,
                        "age_hours": (datetime.utcnow() - item.created_at).total_seconds() / 3600
                    })
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:max_items]
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return []
    
    async def _search_level(self, level_storage: deque, query: str, max_items: int) -> List[Tuple[MemoryItem, float]]:
        """Search for relevant items in a specific memory level"""
        try:
            query_lower = query.lower()
            results = []
            
            for item in level_storage:
                # Simple text matching (can be enhanced with embeddings)
                content_str = json.dumps(item.content, default=str).lower()
                
                # Calculate relevance score
                relevance = 0.0
                
                # Exact match bonus
                if query_lower in content_str:
                    relevance += 0.8
                
                # Keyword matching
                query_words = set(query_lower.split())
                content_words = set(content_str.split())
                common_words = query_words.intersection(content_words)
                if common_words:
                    relevance += 0.6 * (len(common_words) / len(query_words))
                
                # Importance factor
                relevance *= item.importance
                
                # Retention score factor
                relevance *= item.calculate_retention_score()
                
                if relevance > 0.1:  # Minimum threshold
                    results.append((item, relevance))
            
            # Sort by relevance
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_items]
            
        except Exception as e:
            logger.error(f"Error searching level: {e}")
            return []
    
    async def _manage_memory_levels(self, session_id: str, current_level: MemoryLevel):
        """Manage memory promotion, demotion, and cleanup"""
        try:
            hierarchy = self.memory_hierarchy[session_id]
            
            # Check for promotions
            await self._check_promotions(session_id, hierarchy, current_level)
            
            # Check for demotions
            await self._check_demotions(session_id, hierarchy, current_level)
            
            # Cleanup expired items
            await self._cleanup_expired_items(session_id, hierarchy)
            
        except Exception as e:
            logger.error(f"Error managing memory levels: {e}")
    
    async def _check_promotions(self, session_id: str, hierarchy: Dict[MemoryLevel, deque], current_level: MemoryLevel):
        """Check and perform memory promotions"""
        try:
            current_storage = hierarchy[current_level]
            config = self.level_configs[current_level]
            
            # Find items eligible for promotion
            promotion_candidates = []
            
            for item in current_storage:
                retention_score = item.calculate_retention_score()
                if retention_score >= config.promotion_threshold and item.access_count >= 3:
                    promotion_candidates.append(item)
            
            # Promote eligible items
            if promotion_candidates and current_level != MemoryLevel.ARCHIVAL:
                next_level = self._get_next_level(current_level)
                if next_level:
                    for item in promotion_candidates[:10]:  # Limit promotions
                        # Remove from current level
                        try:
                            current_storage.remove(item)
                            # Add to next level
                            hierarchy[next_level].append(item)
                            
                            # Log promotion
                            self.promotion_history.append({
                                "session_id": session_id,
                                "item_id": str(id(item)),
                                "from_level": current_level.value,
                                "to_level": next_level.value,
                                "reason": "high_retention_score",
                                "timestamp": datetime.utcnow().isoformat(),
                                "retention_score": item.calculate_retention_score()
                            })
                            
                            logger.debug(f"Promoted item from {current_level.value} to {next_level.value}")
                            
                        except ValueError:
                            # Item may have been already removed
                            continue
            
        except Exception as e:
            logger.error(f"Error checking promotions: {e}")
    
    async def _check_demotions(self, session_id: str, hierarchy: Dict[MemoryLevel, deque], current_level: MemoryLevel):
        """Check and perform memory demotions"""
        try:
            current_storage = hierarchy[current_level]
            config = self.level_configs[current_level]
            
            # Find items eligible for demotion
            demotion_candidates = []
            
            for item in current_storage:
                retention_score = item.calculate_retention_score()
                if retention_score <= config.demotion_threshold:
                    demotion_candidates.append((item, retention_score))
            
            # Sort by lowest retention score first
            demotion_candidates.sort(key=lambda x: x[1])
            
            # Demote items with lowest scores
            if demotion_candidates and current_level != MemoryLevel.IMMEDIATE:
                prev_level = self._get_prev_level(current_level)
                if prev_level:
                    for item, score in demotion_candidates[:5]:  # Limit demotions
                        try:
                            current_storage.remove(item)
                            hierarchy[prev_level].append(item)
                            
                            logger.debug(f"Demoted item from {current_level.value} to {prev_level.value}")
                            
                        except ValueError:
                            continue
            
        except Exception as e:
            logger.error(f"Error checking demotions: {e}")
    
    async def _cleanup_expired_items(self, session_id: str, hierarchy: Dict[MemoryLevel, deque]):
        """Clean up expired and low-value items"""
        try:
            current_time = datetime.utcnow()
            
            for level, storage in hierarchy.items():
                config = self.level_configs[level]
                expired_items = []
                
                for item in storage:
                    # Check retention threshold
                    retention_score = item.calculate_retention_score()
                    
                    # Check age limits based on level
                    age_hours = (current_time - item.created_at).total_seconds() / 3600
                    max_age = self._get_max_age_for_level(level)
                    
                    if retention_score < config.retention_threshold or age_hours > max_age:
                        expired_items.append(item)
                
                # Remove expired items
                for item in expired_items:
                    try:
                        storage.remove(item)
                        logger.debug(f"Cleaned up expired item from {level.value}")
                    except ValueError:
                        continue
            
        except Exception as e:
            logger.error(f"Error cleaning up expired items: {e}")
    
    def _get_next_level(self, current_level: MemoryLevel) -> Optional[MemoryLevel]:
        """Get next memory level for promotion"""
        level_order = [
            MemoryLevel.IMMEDIATE,
            MemoryLevel.WORKING,
            MemoryLevel.SHORT_TERM,
            MemoryLevel.LONG_TERM,
            MemoryLevel.ARCHIVAL
        ]
        
        try:
            current_idx = level_order.index(current_level)
            if current_idx < len(level_order) - 1:
                return level_order[current_idx + 1]
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _get_prev_level(self, current_level: MemoryLevel) -> Optional[MemoryLevel]:
        """Get previous memory level for demotion"""
        level_order = [
            MemoryLevel.IMMEDIATE,
            MemoryLevel.WORKING,
            MemoryLevel.SHORT_TERM,
            MemoryLevel.LONG_TERM,
            MemoryLevel.ARCHIVAL
        ]
        
        try:
            current_idx = level_order.index(current_level)
            if current_idx > 0:
                return level_order[current_idx - 1]
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _get_max_age_for_level(self, level: MemoryLevel) -> int:
        """Get maximum age in hours for each level"""
        age_limits = {
            MemoryLevel.IMMEDIATE: 1,      # 1 hour
            MemoryLevel.WORKING: 24,       # 1 day
            MemoryLevel.SHORT_TERM: 168,   # 1 week
            MemoryLevel.LONG_TERM: 8760,   # 1 year
            MemoryLevel.ARCHIVAL: 87600    # 10 years
        }
        return age_limits.get(level, 24)
    
    async def get_memory_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive memory statistics for session"""
        try:
            if session_id not in self.memory_hierarchy:
                return {"error": "Session not found"}
            
            hierarchy = self.memory_hierarchy[session_id]
            stats = {
                "session_id": session_id,
                "levels": {},
                "total_items": 0,
                "total_size_mb": 0.0,
                "access_patterns": dict(self.access_patterns[session_id]),
                "recent_promotions": [p for p in self.promotion_history[-10:] if p["session_id"] == session_id]
            }
            
            # Calculate level statistics
            for level, storage in hierarchy.items():
                level_stats = {
                    "item_count": len(storage),
                    "capacity": self.level_configs[level].capacity,
                    "utilization": len(storage) / self.level_configs[level].capacity * 100,
                    "avg_importance": 0.0,
                    "avg_access_count": 0.0,
                    "size_mb": 0.0
                }
                
                if storage:
                    importances = [item.importance for item in storage]
                    access_counts = [item.access_count for item in storage]
                    sizes = [item.size_bytes for item in storage]
                    
                    level_stats["avg_importance"] = sum(importances) / len(importances)
                    level_stats["avg_access_count"] = sum(access_counts) / len(access_counts)
                    level_stats["size_mb"] = sum(sizes) / (1024 * 1024)
                
                stats["levels"][level.value] = level_stats
                stats["total_items"] += level_stats["item_count"]
                stats["total_size_mb"] += level_stats["size_mb"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {"error": str(e)}
    
    async def optimize_memory(self, session_id: str) -> Dict[str, Any]:
        """Optimize memory hierarchy for session"""
        try:
            if session_id not in self.memory_hierarchy:
                return {"error": "Session not found"}
            
            hierarchy = self.memory_hierarchy[session_id]
            optimization_results = {
                "session_id": session_id,
                "actions_taken": [],
                "items_promoted": 0,
                "items_demoted": 0,
                "items_removed": 0,
                "space_saved_mb": 0.0
            }
            
            initial_stats = await self.get_memory_statistics(session_id)
            
            # Perform optimization for each level
            for level in MemoryLevel:
                await self._manage_memory_levels(session_id, level)
            
            final_stats = await self.get_memory_statistics(session_id)
            
            # Calculate optimization impact
            optimization_results["space_saved_mb"] = initial_stats["total_size_mb"] - final_stats["total_size_mb"]
            optimization_results["final_utilization"] = {
                level: stats["utilization"] for level, stats in final_stats["levels"].items()
            }
            
            logger.info(f"Optimized memory for session {session_id}: saved {optimization_results['space_saved_mb']:.2f}MB")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            return {"error": str(e)}
