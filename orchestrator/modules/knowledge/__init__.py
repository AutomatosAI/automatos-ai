"""
Knowledge Module - Knowledge Base and Graph
===========================================

Knowledge graph construction and querying.
Depends on: modules.search

Components:
- graph/     - Graph building, traversal, entity extraction
- storage/   - PostgreSQL, Neo4j (future)

Usage:
    from modules.knowledge import KnowledgeService
    
    kb = KnowledgeService()
    entities = await kb.extract_entities(text)
    related = await kb.find_related("concept")

Sellable as: automatos-knowledge
"""

# Will export: KnowledgeService, KnowledgeGraph, EntityExtractor
__all__ = []

