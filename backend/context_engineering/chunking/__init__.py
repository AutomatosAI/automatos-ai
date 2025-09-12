
"""
Context Engineering Chunking Module
===================================

Advanced semantic chunking algorithms that preserve meaning and context boundaries.
"""

from .semantic_chunker import (
    SemanticChunker,
    SemanticChunk,
    ChunkMetadata,
    ChunkingStrategy
)

__all__ = [
    'SemanticChunker',
    'SemanticChunk', 
    'ChunkMetadata',
    'ChunkingStrategy'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Automatos AI Context Engineering Team'
