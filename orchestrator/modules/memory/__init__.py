"""
Memory Module
=============

The unified memory stack: L1 session (Redis), L2 short-term (Postgres),
L3 durable (in-process Qdrant — PRD-187 S1), consolidated by the
contradiction-based lifecycle (PRD-159 S4).

Usage:
    from modules.memory.unified_memory_service import get_unified_memory_service

    service = get_unified_memory_service()
    await service.store_long_term(workspace_id, content)
    results = await service.search_long_term(workspace_id, query)

Sellable as: automatos-memory
"""

__all__ = ["unified_memory_service"]
