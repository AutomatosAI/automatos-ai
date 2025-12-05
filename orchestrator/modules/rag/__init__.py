"""
RAG Module - Complete Knowledge Base System
===========================================

Full document pipeline: Upload → Parse → Chunk → Embed → Store → Search

Components:
- ingestion/       - Document upload, parsing, multimodal extraction
- chunking/        - Semantic chunking strategies
- service.py       - RAG retrieval service

Usage:
    # Upload documents
    from modules.rag import DocumentManager, DocumentStatus
    doc_manager = DocumentManager(db_config)
    doc_id = await doc_manager.upload_document("/path/to/doc.pdf")
    
    # Search/retrieve
    from modules.rag import get_rag_service
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
from .ingestion import (
    # Document Manager (main entry for uploads)
    DocumentManager,
    DocumentMetadata,
    DocumentChunk,
    DocumentStatus,
    DocumentType,
    # Processors
    DocumentProcessor,
    IngestionResult,
    IngestionPipeline,
    PipelineConfig,
    PipelineResult,
    TextHandler,
    # Multimodal
    TableProcessor,
    ImageProcessor,
    FormulaProcessor,
    MultimodalDocumentProcessor,
    ContentModality,
    TableExtraction,
    ImageExtraction,
    FormulaExtraction,
    create_multimodal_processor,
)
from .config import (
    RAGModuleConfig,
    ChunkingConfig,
    EmbeddingConfig,
    RetrievalConfig,
    IngestionConfig,
    default_config
)

__all__ = [
    # Main service (retrieval)
    "RAGService",
    "RAGConfig",
    "RAGResult",
    "get_rag_service",
    "UniversalRAGService",
    "get_universal_rag",
    
    # Document Manager (upload/storage)
    "DocumentManager",
    "DocumentMetadata",
    "DocumentChunk",
    "DocumentStatus",
    "DocumentType",
    
    # Chunking
    "SemanticChunker",
    "ChunkingStrategy",
    "ChunkMetadata", 
    "SemanticChunk",
    "DocumentChunker",
    "MultiModalChunker",
    
    # Ingestion
    "DocumentProcessor",
    "IngestionResult",
    "IngestionPipeline",
    "PipelineConfig",
    "PipelineResult",
    "TextHandler",
    
    # Multimodal
    "TableProcessor",
    "ImageProcessor",
    "FormulaProcessor",
    "MultimodalDocumentProcessor",
    "ContentModality",
    "TableExtraction",
    "ImageExtraction",
    "FormulaExtraction",
    "create_multimodal_processor",
    
    # Config
    "RAGModuleConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "IngestionConfig",
    "default_config"
]
