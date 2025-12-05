"""
Memory Service
==============

Main service class for Memory module.
Coordinates hierarchical memory, augmentation, consolidation, and access optimization.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .types import (
    MemoryType,
    MemoryLevel,
    MemoryItem,
    HierarchicalMemoryManager
)
from .operations import (
    VectorStoreAugmenter,
    AugmentationStrategy,
    MemoryConsolidator,
    ConsolidationStrategy,
    MemoryAccessOptimizer,
    AccessPattern
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for Memory service"""
    immediate_capacity: int = 7
    working_capacity: int = 100
    short_term_capacity: int = 1000
    long_term_capacity: int = 100000
    archival_capacity: int = 1000000
    base_decay_rate: float = 0.1
    consolidation_threshold: float = 0.8
    similarity_threshold: float = 0.7


class MemoryService:
    """Main Memory service for hierarchical memory management"""
    
    def __init__(
        self,
        config: MemoryConfig = None
    ):
        self.config = config or MemoryConfig()
        
        # Initialize hierarchical memory manager
        self.memory_manager = HierarchicalMemoryManager(
            immediate_capacity=self.config.immediate_capacity,
            working_capacity=self.config.working_capacity,
            short_term_capacity=self.config.short_term_capacity,
            long_term_capacity=self.config.long_term_capacity,
            archival_capacity=self.config.archival_capacity,
            base_decay_rate=self.config.base_decay_rate
        )
        
        # Initialize augmentation
        self.augmenter = VectorStoreAugmenter(
            similarity_threshold=self.config.similarity_threshold
        )
        
        # Initialize consolidation
        self.consolidator = MemoryConsolidator(
            consolidation_threshold=self.config.consolidation_threshold,
            similarity_threshold=self.config.similarity_threshold
        )
        
        # Initialize access optimizer
        self.access_optimizer = MemoryAccessOptimizer()
        
        logger.info("MemoryService initialized")
    
    async def store(
        self,
        session_id: str,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.WORKING_DATA,
        importance: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """Store a new memory"""
        return await self.memory_manager.store_memory(
            session_id=session_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags
        )
    
    async def retrieve(
        self,
        session_id: str,
        query: str = None,
        memory_type: MemoryType = None,
        max_items: int = 10,
        min_importance: float = 0.0,
        use_optimization: bool = True
    ) -> List[MemoryItem]:
        """Retrieve relevant memories"""
        
        if use_optimization:
            return await self.access_optimizer.optimize_access(
                memory_manager=self.memory_manager,
                query=query,
                session_id=session_id
            )
        else:
            return await self.memory_manager.retrieve_memory(
                session_id=session_id,
                query=query,
                memory_type=memory_type,
                max_items=max_items,
                min_importance=min_importance
            )
    
    async def augment(
        self,
        memory_item: MemoryItem,
        strategy: AugmentationStrategy = AugmentationStrategy.HYBRID,
        max_augmentations: int = 5
    ) -> List[Dict[str, Any]]:
        """Augment a memory with external knowledge"""
        
        augmented = await self.augmenter.augment_memory(
            memory_item=memory_item,
            strategy=strategy,
            max_augmentations=max_augmentations
        )
        
        return [
            {
                'content': aug.augmented_content,
                'similarity': aug.similarity_score,
                'source': aug.augmentation_source
            }
            for aug in augmented
        ]
    
    async def consolidate(
        self,
        memories: List[MemoryItem],
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID
    ) -> List[MemoryItem]:
        """Consolidate a list of memories"""
        
        return await self.consolidator.consolidate_memories(
            memories=memories,
            strategy=strategy
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        memory_stats = await self.memory_manager.get_memory_stats()
        augmentation_stats = await self.augmenter.get_augmentation_stats()
        consolidation_stats = await self.consolidator.get_consolidation_stats()
        optimization_stats = await self.access_optimizer.get_optimization_stats()
        
        return {
            'memory': memory_stats,
            'augmentation': augmentation_stats,
            'consolidation': consolidation_stats,
            'optimization': optimization_stats
        }
    
    def start_maintenance(self):
        """Start background maintenance tasks"""
        self.memory_manager.start_maintenance()
    
    def stop_maintenance(self):
        """Stop background maintenance tasks"""
        self.memory_manager.stop_maintenance()

