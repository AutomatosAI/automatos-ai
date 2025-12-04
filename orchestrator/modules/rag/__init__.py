"""
RAG Module - Document Retrieval-Augmented Generation
====================================================

Document chunking, ingestion, and retrieval.
Depends on: modules.search

Components:
- chunking/    - Semantic chunking, strategies
- ingestion/   - Document processing, file handlers

Usage:
    from modules.rag import RAGService, SemanticChunker, get_rag_service
    
    rag = get_rag_service()
    result = await rag.retrieve("How do agents work?")

Sellable as: automatos-rag
"""

from .service import (
    RAGService,
    RAGConfig,
    RAGResult,
    get_rag_service,
    UniversalRAGService,
    get_universal_rag
)
from .chunking import (
    SemanticChunker,
    ChunkingStrategy,
    ChunkMetadata,
    SemanticChunk,
    DocumentChunker,
    MultiModalChunker
)
from .ingestion import DocumentProcessor, IngestionResult

__all__ = [
    # Main service
    "RAGService",
    "RAGConfig",
    "RAGResult",
    "get_rag_service",
    "UniversalRAGService",
    "get_universal_rag",
    
    # Chunking
    "SemanticChunker",
    "ChunkingStrategy",
    "ChunkMetadata", 
    "SemanticChunk",
    "DocumentChunker",
    "MultiModalChunker",
    
    # Ingestion
    "DocumentProcessor",
    "IngestionResult"
]
