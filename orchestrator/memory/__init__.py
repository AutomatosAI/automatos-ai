
"""
Memory Systems Module for Automotas AI
======================================

Advanced memory management with multi-level retention, access optimization,
external augmentation, and intelligent consolidation.
"""

from .memory_types import HierarchicalMemoryManager, MemoryType, MemoryLevel
from .access_patterns import MemoryAccessOptimizer, AccessPattern
from .augmentation import VectorStoreAugmenter, AugmentationStrategy  
from .consolidation import MemoryConsolidator, ConsolidationStrategy
from .manager import AdvancedMemoryManager

__all__ = [
    'HierarchicalMemoryManager',
    'MemoryType',
    'MemoryLevel', 
    'MemoryAccessOptimizer',
    'AccessPattern',
    'VectorStoreAugmenter',
    'AugmentationStrategy',
    'MemoryConsolidator',
    'ConsolidationStrategy',
    'AdvancedMemoryManager'
]
