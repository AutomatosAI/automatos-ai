"""
Ingestion Submodule
===================

Complete document ingestion pipeline:
- Document upload and management (DocumentManager)
- File parsing (PDF, DOCX, MD, TXT)
- Multimodal extraction (tables, formulas, images)
- Chunking and embedding

Usage:
    from modules.rag.ingestion import DocumentManager, DocumentStatus
    from modules.rag.ingestion.multimodal import TableProcessor, FormulaProcessor
"""

from .processor import DocumentProcessor, IngestionResult
from .pipeline import IngestionPipeline, PipelineConfig, PipelineResult
from .handlers import TextHandler, TextContent

# Document Manager (full upload/storage system)
from .manager import (
    DocumentManager,
    DocumentProcessor as FileProcessor,  # Renamed to avoid conflict
    DocumentMetadata,
    DocumentChunk,
    DocumentStatus,
    DocumentType,
)

# Multimodal processors
from .multimodal import (
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

__all__ = [
    # Document Manager (main entry point for uploads)
    "DocumentManager",
    "DocumentMetadata",
    "DocumentChunk",
    "DocumentStatus",
    "DocumentType",
    
    # File Processor
    "FileProcessor",
    "DocumentProcessor",
    "IngestionResult",
    
    # Pipeline
    "IngestionPipeline",
    "PipelineConfig",
    "PipelineResult",
    
    # Handlers
    "TextHandler",
    "TextContent",
    
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
]
