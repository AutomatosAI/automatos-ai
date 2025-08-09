
"""
Advanced Memory Manager
=======================

Unified manager that integrates hierarchical memory, access optimization, 
augmentation, and consolidation for the Automatos AI system.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .memory_types import (
    HierarchicalMemoryManager, MemoryItem, MemoryType, MemoryLevel
)
from .access_patterns import (
    MemoryAccessOptimizer, AccessPattern, AccessMetrics
)
from .augmentation import (
    VectorStoreAugmenter, AugmentationStrategy, AugmentedMemory
)
from .consolidation import (
    MemoryConsolidator, ConsolidationStrategy, ConsolidationMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class MemorySystemStats:
    """Comprehensive memory system statistics"""
    total_memories: int = 0
    memory_levels: Dict[str, int] = None
    access_metrics: AccessMetrics = None
    consolidation_metrics: ConsolidationMetrics = None
    augmentation_stats: Dict[str, Any] = None
    system_performance: Dict[str, float] = None
    
    def __post_init__(self):
        if self.memory_levels is None:
            self.memory_levels = {}
        if self.system_performance is None:
            self.system_performance = {}

class AdvancedMemoryManager:
    """
    Advanced memory management system integrating all memory components
    """
    
    def __init__(
        self,
        # Hierarchical memory settings
        immediate_capacity: int = 7,
        working_capacity: int = 100,
        short_term_capacity: int = 1000,
        long_term_capacity: int = 100000,
        archival_capacity: int = 1000000,
        
        # Access optimization settings
        cache_size: int = 1000,
        locality_threshold: float = 0.7,
        
        # Augmentation settings
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        max_external_items: int = 10000,
        
        # Consolidation settings
        consolidation_threshold: float = 0.8,
        max_summary_length: int = 500,
        
        # System settings
        auto_consolidation: bool = True,
        consolidation_interval_hours: int = 6,
        backup_enabled: bool = True,
        backup_interval_hours: int = 24
    ):
        # Initialize core components
        self.hierarchical_manager = HierarchicalMemoryManager(
            immediate_capacity=immediate_capacity,
            working_capacity=working_capacity,
            short_term_capacity=short_term_capacity,
            long_term_capacity=long_term_capacity,
            archival_capacity=archival_capacity
        )
        
        self.access_optimizer = MemoryAccessOptimizer(
            cache_size=cache_size,
            locality_threshold=locality_threshold
        )
        
        self.augmenter = VectorStoreAugmenter(
            model_name=embedding_model,
            similarity_threshold=similarity_threshold,
            max_external_items=max_external_items
        )
        
        self.consolidator = MemoryConsolidator(
            consolidation_threshold=consolidation_threshold,
            max_summary_length=max_summary_length,
            similarity_threshold=similarity_threshold
        )
        
        # System configuration
        self.auto_consolidation = auto_consolidation
        self.consolidation_interval_hours = consolidation_interval_hours
        self.backup_enabled = backup_enabled
        self.backup_interval_hours = backup_interval_hours
        
        # Background tasks
        self._consolidation_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None
        self._last_consolidation = datetime.utcnow()
        self._last_backup = datetime.utcnow()
        
        # System statistics
        self._operation_counts = {
            'store': 0,
            'retrieve': 0,
            'augment': 0,
            'consolidate': 0
        }
        self._response_times = []
        
        # Background services (will be started when needed)
        # Don't start automatically to avoid event loop issues
        
        logger.info("Advanced Memory Manager initialized with all components")
    
    async def store_memory(
        self,
        session_id: str,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.WORKING_DATA,
        importance: float = 0.5,
        tags: List[str] = None,
        auto_augment: bool = True
    ) -> str:
        """Store new memory with optional augmentation"""
        
        start_time = time.time()
        
        try:
            # Store in hierarchical memory
            memory_id = await self.hierarchical_manager.store_memory(
                session_id=session_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags
            )
            
            # Auto-augment if enabled and content is substantial
            if auto_augment and self._should_augment_memory(content):
                await self._auto_augment_memory(memory_id, session_id)
            
            self._operation_counts['store'] += 1
            self._record_response_time('store', time.time() - start_time)
            
            logger.debug(f"Memory {memory_id} stored successfully")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def retrieve_memory(
        self,
        session_id: str,
        query: str = None,
        memory_type: MemoryType = None,
        max_items: int = 10,
        min_importance: float = 0.0,
        access_pattern: AccessPattern = AccessPattern.LOCALITY_BASED,
        include_augmented: bool = True
    ) -> List[Union[MemoryItem, AugmentedMemory]]:
        """Retrieve memories with optimized access and optional augmentation"""
        
        start_time = time.time()
        
        try:
            # Use access optimizer for retrieval
            memories = await self.access_optimizer.optimize_access(
                memory_manager=self.hierarchical_manager,
                query=query or "",
                session_id=session_id,
                access_pattern=access_pattern
            )
            
            # Filter by criteria
            filtered_memories = []
            for memory in memories:
                if memory_type and memory.memory_type != memory_type:
                    continue
                if memory.importance < min_importance:
                    continue
                filtered_memories.append(memory)
            
            # Limit results
            filtered_memories = filtered_memories[:max_items]
            
            # Add augmented memories if requested
            result_memories = []
            if include_augmented and filtered_memories:
                for memory in filtered_memories:
                    result_memories.append(memory)
                    
                    # Add augmented versions for high-importance memories
                    if memory.importance > 0.7:
                        augmented = await self.augmenter.augment_memory(
                            memory, max_augmentations=2
                        )
                        result_memories.extend(augmented[:1])  # Add top augmentation
            else:
                result_memories = filtered_memories
            
            self._operation_counts['retrieve'] += 1
            self._record_response_time('retrieve', time.time() - start_time)
            
            logger.debug(f"Retrieved {len(result_memories)} memories for session {session_id}")
            return result_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def augment_memory_item(
        self,
        memory_id: str,
        strategy: AugmentationStrategy = AugmentationStrategy.HYBRID,
        max_augmentations: int = 5
    ) -> List[AugmentedMemory]:
        """Explicitly augment a specific memory item"""
        
        start_time = time.time()
        
        try:
            # Find the memory item
            memory_item = await self._find_memory_item(memory_id)
            if not memory_item:
                logger.warning(f"Memory item {memory_id} not found for augmentation")
                return []
            
            # Perform augmentation
            augmented_memories = await self.augmenter.augment_memory(
                memory_item=memory_item,
                strategy=strategy,
                max_augmentations=max_augmentations
            )
            
            self._operation_counts['augment'] += 1
            self._record_response_time('augment', time.time() - start_time)
            
            logger.debug(f"Generated {len(augmented_memories)} augmentations for {memory_id}")
            return augmented_memories
            
        except Exception as e:
            logger.error(f"Failed to augment memory {memory_id}: {e}")
            return []
    
    async def consolidate_memories(
        self,
        session_id: str = None,
        memory_level: MemoryLevel = None,
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID,
        force: bool = False
    ) -> Dict[str, Any]:
        """Consolidate memories with specified criteria"""
        
        start_time = time.time()
        
        try:
            # Get memories to consolidate
            memories_to_consolidate = []
            
            if memory_level:
                # Consolidate specific level
                store = self.hierarchical_manager.memory_stores[memory_level]
                for memory in store.values():
                    if not session_id or memory.id.startswith(session_id):
                        memories_to_consolidate.append(memory)
            else:
                # Consolidate across all levels for session
                for level in MemoryLevel:
                    store = self.hierarchical_manager.memory_stores[level]
                    for memory in store.values():
                        if not session_id or memory.id.startswith(session_id):
                            memories_to_consolidate.append(memory)
            
            if not memories_to_consolidate:
                return {"consolidated": 0, "message": "No memories found to consolidate"}
            
            # Check if consolidation is needed
            if not force and len(memories_to_consolidate) < 10:
                return {"consolidated": 0, "message": "Not enough memories to warrant consolidation"}
            
            # Perform consolidation
            original_count = len(memories_to_consolidate)
            consolidated_memories = await self.consolidator.consolidate_memories(
                memories=memories_to_consolidate,
                strategy=strategy,
                target_level=memory_level or MemoryLevel.LONG_TERM
            )
            
            # Update memory stores
            await self._update_memory_stores_after_consolidation(
                original_memories=memories_to_consolidate,
                consolidated_memories=consolidated_memories,
                target_level=memory_level or MemoryLevel.LONG_TERM
            )
            
            self._operation_counts['consolidate'] += 1
            self._record_response_time('consolidate', time.time() - start_time)
            
            result = {
                "original_count": original_count,
                "consolidated_count": len(consolidated_memories),
                "items_removed": original_count - len(consolidated_memories),
                "compression_ratio": self.consolidator.metrics.compression_ratio,
                "information_retention": self.consolidator.metrics.information_retention,
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Consolidation completed: {original_count} -> {len(consolidated_memories)} memories")
            return result
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            return {"error": str(e), "consolidated": 0}
    
    async def add_external_knowledge(
        self,
        content: Dict[str, Any],
        source: str = "external",
        metadata: Dict[str, Any] = None
    ) -> int:
        """Add external knowledge for augmentation"""
        
        try:
            knowledge_id = await self.augmenter.add_external_knowledge(
                content=content,
                source=source,
                metadata=metadata
            )
            
            logger.debug(f"Added external knowledge {knowledge_id}")
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Failed to add external knowledge: {e}")
            raise
    
    async def get_comprehensive_stats(self) -> MemorySystemStats:
        """Get comprehensive system statistics"""
        
        try:
            # Get stats from all components
            hierarchical_stats = await self.hierarchical_manager.get_memory_stats()
            access_stats = await self.access_optimizer.get_optimization_stats()
            augmentation_stats = await self.augmenter.get_augmentation_stats()
            consolidation_stats = await self.consolidator.get_consolidation_stats()
            
            # System performance metrics
            system_performance = {
                "total_operations": sum(self._operation_counts.values()),
                "operations_breakdown": self._operation_counts.copy(),
                "avg_response_time": sum(self._response_times) / len(self._response_times) if self._response_times else 0,
                "uptime_hours": (datetime.utcnow() - self._last_backup).total_seconds() / 3600,
                "memory_efficiency": await self._calculate_memory_efficiency()
            }
            
            stats = MemorySystemStats(
                total_memories=hierarchical_stats["total_items"],
                memory_levels={
                    level: data["count"] 
                    for level, data in hierarchical_stats["levels"].items()
                },
                access_metrics=AccessMetrics(
                    hit_rate=access_stats["metrics"]["hit_rate"],
                    miss_rate=access_stats["metrics"]["miss_rate"],
                    avg_response_time=access_stats["metrics"]["avg_response_time"],
                    cache_utilization=access_stats["metrics"]["cache_utilization"],
                    locality_factor=access_stats["metrics"]["locality_factor"]
                ),
                consolidation_metrics=self.consolidator.metrics,
                augmentation_stats=augmentation_stats,
                system_performance=system_performance
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return MemorySystemStats()
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize overall system performance"""
        
        try:
            optimization_results = {
                "cache_cleared": False,
                "consolidation_performed": False,
                "external_knowledge_cleaned": False,
                "performance_improvement": 0.0
            }
            
            # Clear access cache if hit rate is low
            access_stats = await self.access_optimizer.get_optimization_stats()
            if access_stats["metrics"]["hit_rate"] < 0.3:
                self.access_optimizer.clear_cache()
                optimization_results["cache_cleared"] = True
            
            # Trigger consolidation if many memories exist
            hierarchical_stats = await self.hierarchical_manager.get_memory_stats()
            if hierarchical_stats["total_items"] > 5000:
                consolidation_result = await self.consolidate_memories(force=True)
                optimization_results["consolidation_performed"] = True
                optimization_results["consolidation_result"] = consolidation_result
            
            # Clean external knowledge if capacity is high
            aug_stats = await self.augmenter.get_augmentation_stats()
            if aug_stats.get("capacity", {}).get("utilization", 0) > 90:
                await self.augmenter._manage_external_capacity()
                optimization_results["external_knowledge_cleaned"] = True
            
            logger.info("System performance optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize system performance: {e}")
            return {"error": str(e)}
    
    def start_background_services(self):
        """Start background maintenance services"""
        
        if self.auto_consolidation and not self._consolidation_task:
            self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        
        if self.backup_enabled and not self._backup_task:
            self._backup_task = asyncio.create_task(self._backup_loop())
    
    def stop_background_services(self):
        """Stop background maintenance services"""
        
        if self._consolidation_task:
            self._consolidation_task.cancel()
            self._consolidation_task = None
        
        if self._backup_task:
            self._backup_task.cancel()
            self._backup_task = None
    
    async def save_system_state(self, directory: str = "memory_backups"):
        """Save complete system state"""
        
        try:
            backup_dir = Path(directory)
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Save hierarchical memory
            memory_file = backup_dir / f"hierarchical_memory_{timestamp}.json"
            await self.hierarchical_manager.save_state(str(memory_file))
            
            # Save external knowledge
            knowledge_file = backup_dir / f"external_knowledge_{timestamp}.json"
            await self.augmenter.save_external_knowledge(str(knowledge_file))
            
            # Save system configuration
            config_file = backup_dir / f"system_config_{timestamp}.json"
            await self._save_system_config(str(config_file))
            
            self._last_backup = datetime.utcnow()
            logger.info(f"System state saved to {backup_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            raise
    
    async def load_system_state(self, directory: str = "memory_backups", timestamp: str = None):
        """Load complete system state"""
        
        try:
            backup_dir = Path(directory)
            
            if timestamp:
                # Load specific timestamp
                memory_file = backup_dir / f"hierarchical_memory_{timestamp}.json"
                knowledge_file = backup_dir / f"external_knowledge_{timestamp}.json"
            else:
                # Load latest backup
                memory_files = list(backup_dir.glob("hierarchical_memory_*.json"))
                knowledge_files = list(backup_dir.glob("external_knowledge_*.json"))
                
                if not memory_files:
                    raise FileNotFoundError("No memory backup files found")
                
                memory_file = sorted(memory_files)[-1]  # Latest file
                knowledge_file = sorted(knowledge_files)[-1] if knowledge_files else None
            
            # Load hierarchical memory
            if memory_file.exists():
                await self.hierarchical_manager.load_state(str(memory_file))
            
            # Load external knowledge
            if knowledge_file and knowledge_file.exists():
                await self.augmenter.load_external_knowledge(str(knowledge_file))
            
            logger.info(f"System state loaded from {backup_dir}")
            
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            raise
    
    # Private helper methods
    
    async def _auto_augment_memory(self, memory_id: str, session_id: str):
        """Automatically augment newly stored memory if beneficial"""
        
        try:
            memory_item = await self._find_memory_item(memory_id)
            if not memory_item:
                return
            
            # Only augment if importance is high enough
            if memory_item.importance > 0.6:
                augmented = await self.augmenter.augment_memory(
                    memory_item, max_augmentations=2
                )
                
                if augmented:
                    logger.debug(f"Auto-augmented memory {memory_id} with {len(augmented)} items")
                
        except Exception as e:
            logger.debug(f"Auto-augmentation failed for {memory_id}: {e}")
    
    def _should_augment_memory(self, content: Dict[str, Any]) -> bool:
        """Check if memory content should be augmented"""
        
        # Simple heuristics for augmentation decision
        content_text = json.dumps(content, default=str)
        
        # Augment if content is substantial
        if len(content_text) > 200:
            return True
        
        # Augment if contains specific keywords
        augment_keywords = ['problem', 'solution', 'error', 'issue', 'implement', 'analyze']
        content_lower = content_text.lower()
        
        return any(keyword in content_lower for keyword in augment_keywords)
    
    async def _find_memory_item(self, memory_id: str) -> Optional[MemoryItem]:
        """Find memory item across all levels"""
        
        for level in MemoryLevel:
            store = self.hierarchical_manager.memory_stores[level]
            if memory_id in store:
                return store[memory_id]
        
        return None
    
    async def _update_memory_stores_after_consolidation(
        self,
        original_memories: List[MemoryItem],
        consolidated_memories: List[MemoryItem],
        target_level: MemoryLevel
    ):
        """Update memory stores after consolidation"""
        
        # Remove original memories
        for memory in original_memories:
            for level in MemoryLevel:
                store = self.hierarchical_manager.memory_stores[level]
                if memory.id in store:
                    del store[memory.id]
        
        # Add consolidated memories to target level
        target_store = self.hierarchical_manager.memory_stores[target_level]
        for memory in consolidated_memories:
            target_store[memory.id] = memory
    
    async def _calculate_memory_efficiency(self) -> float:
        """Calculate overall memory system efficiency"""
        
        try:
            hierarchical_stats = await self.hierarchical_manager.get_memory_stats()
            access_stats = await self.access_optimizer.get_optimization_stats()
            
            # Factors contributing to efficiency
            capacity_utilization = sum(
                level_data["utilization"] for level_data in hierarchical_stats["levels"].values()
            ) / len(hierarchical_stats["levels"]) / 100
            
            cache_hit_rate = access_stats["metrics"]["hit_rate"]
            
            # Simple efficiency calculation
            efficiency = (capacity_utilization * 0.4) + (cache_hit_rate * 0.6)
            return min(efficiency, 1.0)
            
        except Exception:
            return 0.5  # Default efficiency
    
    def _record_response_time(self, operation: str, response_time: float):
        """Record response time for performance tracking"""
        
        self._response_times.append(response_time)
        
        # Keep response times bounded
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-500:]
    
    async def _consolidation_loop(self):
        """Background consolidation loop"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                time_since_last = (datetime.utcnow() - self._last_consolidation).total_seconds() / 3600
                
                if time_since_last >= self.consolidation_interval_hours:
                    logger.info("Starting scheduled consolidation")
                    await self.consolidate_memories(force=False)
                    self._last_consolidation = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
    
    async def _backup_loop(self):
        """Background backup loop"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                time_since_last = (datetime.utcnow() - self._last_backup).total_seconds() / 3600
                
                if time_since_last >= self.backup_interval_hours:
                    logger.info("Starting scheduled backup")
                    await self.save_system_state()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
    
    async def _save_system_config(self, file_path: str):
        """Save system configuration"""
        
        config = {
            "hierarchical_settings": {
                "capacities": {k.value: v for k, v in self.hierarchical_manager.capacities.items()}
            },
            "access_settings": {
                "cache_size": self.access_optimizer.cache_size,
                "locality_threshold": self.access_optimizer.locality_threshold
            },
            "augmentation_settings": {
                "model_name": self.augmenter.model_name,
                "similarity_threshold": self.augmenter.similarity_threshold,
                "max_external_items": self.augmenter.max_external_items
            },
            "consolidation_settings": {
                "consolidation_threshold": self.consolidator.consolidation_threshold,
                "max_summary_length": self.consolidator.max_summary_length
            },
            "system_settings": {
                "auto_consolidation": self.auto_consolidation,
                "consolidation_interval_hours": self.consolidation_interval_hours,
                "backup_enabled": self.backup_enabled,
                "backup_interval_hours": self.backup_interval_hours
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
