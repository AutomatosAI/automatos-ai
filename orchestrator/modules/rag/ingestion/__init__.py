"""
Ingestion Submodule
===================

Document ingestion and processing pipeline.
"""

from .processor import DocumentProcessor, IngestionResult

__all__ = [
    "DocumentProcessor",
    "IngestionResult"
]

