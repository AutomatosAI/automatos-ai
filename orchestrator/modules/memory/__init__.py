"""
Memory Module
=============

Hierarchical memory system with episodic, semantic, procedural, and working memory.
Includes augmentation, consolidation, and access optimization.

Usage:
    from modules.memory.unified_memory_service import get_unified_memory_service

    service = get_unified_memory_service()
    await service.store_long_term(workspace_id, content)
    results = await service.search_long_term(workspace_id, query)

Sellable as: automatos-memory
"""

from .types import (
    MemoryType,
    MemoryLevel,
    MemoryItem,
    HierarchicalMemoryManager,
)
from .storage import AdvancedMemoryManager
from .operations import (
    VectorStoreAugmenter,
    AugmentationStrategy,
    AugmentedMemory,
    MemoryConsolidator,
    ConsolidationStrategy,
    ConsolidationMetrics,
    MemoryAccessOptimizer,
    AccessPattern,
    AccessMetrics,
)


def get_embedding_dimension() -> int:
    """Get embedding dimension from system settings"""
    try:
        from database.session import get_db_session
        with get_db_session() as db:
            from sqlalchemy import text
            result = db.execute(
                text("SELECT setting_value FROM system_settings WHERE setting_key = 'vector_store_dimensions'")
            ).fetchone()
            if result:
                return int(result[0])
    except Exception:
        pass
    return 2048  # Default


__all__ = [
    # Types
    "MemoryType",
    "MemoryLevel",
    "MemoryItem",
    "HierarchicalMemoryManager",
    
    # Storage
    "AdvancedMemoryManager",
    
    # Operations - Augmentation
    "VectorStoreAugmenter",
    "AugmentationStrategy",
    "AugmentedMemory",
    
    # Operations - Consolidation
    "MemoryConsolidator",
    "ConsolidationStrategy",
    "ConsolidationMetrics",
    
    # Operations - Access
    "MemoryAccessOptimizer",
    "AccessPattern",
    "AccessMetrics",
    
    # Helpers
    "get_embedding_dimension",
]
