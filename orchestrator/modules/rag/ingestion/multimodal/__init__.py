"""
Multimodal Content Processors
=============================

Processors for different content modalities:
- Tables: Extract and structure tabular data
- Images: Extract, describe, and analyze visual content
- Formulas: Parse and understand mathematical expressions
- Diagrams: Analyze flowcharts and diagrams

Usage:
    from modules.rag.ingestion.multimodal import (
        TableProcessor,
        ImageProcessor,
        FormulaProcessor,
        ContentModality,
    )
"""

from modules.rag.ingestion.multimodal.processors import (
    ContentModality,
    ProcessedContent,
    TableExtraction,
    ImageExtraction,
    FormulaExtraction,
    TableProcessor,
    ImageProcessor,
    FormulaProcessor,
    MultimodalDocumentProcessor,
    create_multimodal_processor,
)

__all__ = [
    'ContentModality',
    'ProcessedContent',
    'TableExtraction',
    'ImageExtraction',
    'FormulaExtraction',
    'TableProcessor',
    'ImageProcessor',
    'FormulaProcessor',
    'MultimodalDocumentProcessor',
    'create_multimodal_processor',
]

