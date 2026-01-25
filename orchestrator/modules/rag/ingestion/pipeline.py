"""
Ingestion Pipeline
==================

Orchestrates the document ingestion process:
1. File discovery
2. Content extraction (via handlers)
3. Chunking (via modules.rag.chunking)
4. Embedding generation
5. Vector storage (via modules.search)
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .handlers import TextHandler
from .processor import DocumentProcessor, IngestionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for ingestion pipeline"""
    chunk_size: int = 500
    chunk_overlap: float = 0.1
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    batch_size: int = 10
    max_concurrent: int = 3
    skip_patterns: List[str] = None
    
    def __post_init__(self):
        if self.skip_patterns is None:
            self.skip_patterns = [
                '__pycache__',
                'node_modules',
                '.git',
                '.venv',
                'venv',
                '.env',
                '*.pyc',
                '*.pyo',
                '.DS_Store'
            ]


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    total_embeddings: int
    processing_time_seconds: float
    errors: List[Dict[str, str]]


class IngestionPipeline:
    """
    Document ingestion pipeline.
    
    Coordinates handlers, chunking, and storage.
    """
    
    def __init__(
        self,
        config: PipelineConfig = None,
        vector_store=None,
        embedding_generator=None
    ):
        self.config = config or PipelineConfig()
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
        # Initialize handlers
        self.text_handler = TextHandler()
        
        # Initialize processor
        self._init_processor()
        
        logger.info("IngestionPipeline initialized")
    
    def _init_processor(self):
        """Initialize document processor with chunker"""
        try:
            from modules.rag.chunking import SemanticChunker, ChunkingStrategy
            
            chunker = SemanticChunker(
                strategy=ChunkingStrategy.ADAPTIVE,
                target_chunk_size=self.config.chunk_size,
                min_chunk_size=self.config.min_chunk_size,
                max_chunk_size=self.config.max_chunk_size,
                overlap_ratio=self.config.chunk_overlap
            )
            
            self.processor = DocumentProcessor(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator,
                chunker=chunker
            )
            
            logger.info("✅ Using SemanticChunker from modules.rag")
        except ImportError as e:
            logger.warning(f"SemanticChunker not available: {e}")
            self.processor = DocumentProcessor(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator
            )
    
    async def ingest_file(self, file_path: str) -> IngestionResult:
        """Ingest a single file"""
        return await self.processor.ingest_file(file_path)
    
    async def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: List[str] = None
    ) -> PipelineResult:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            file_patterns: Optional list of glob patterns to match
        """
        start_time = datetime.now()
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Discover files
        files = self._discover_files(directory, recursive, file_patterns)
        logger.info(f"Discovered {len(files)} files to process")
        
        # Process files
        results = []
        errors = []
        
        # Process in batches with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(file_path: str) -> IngestionResult:
            async with semaphore:
                return await self.processor.ingest_file(file_path)
        
        for i in range(0, len(files), self.config.batch_size):
            batch = files[i:i + self.config.batch_size]
            
            batch_results = await asyncio.gather(
                *[process_with_semaphore(str(f)) for f in batch],
                return_exceptions=True
            )
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    errors.append({
                        'file': str(batch[j]),
                        'error': str(result)
                    })
                else:
                    results.append(result)
            
            logger.info(f"Processed batch {i // self.config.batch_size + 1}: {len(batch)} files")
        
        # Compile results
        processing_time = (datetime.now() - start_time).total_seconds()
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Add failed results to errors
        for r in failed:
            errors.append({
                'file': r.document_path,
                'error': r.error or 'Unknown error'
            })
        
        return PipelineResult(
            total_files=len(files),
            processed_files=len(successful),
            failed_files=len(failed) + len(errors),
            total_chunks=sum(r.chunk_count for r in successful),
            total_embeddings=sum(r.embeddings_count for r in successful),
            processing_time_seconds=processing_time,
            errors=errors
        )
    
    def _discover_files(
        self,
        directory: Path,
        recursive: bool,
        patterns: List[str] = None
    ) -> List[Path]:
        """Discover files to process"""
        files = []
        
        # Get all files
        if recursive:
            all_files = list(directory.rglob('*'))
        else:
            all_files = list(directory.glob('*'))
        
        for file_path in all_files:
            # Skip directories
            if not file_path.is_file():
                continue
            
            # Skip based on patterns
            if self._should_skip(file_path):
                continue
            
            # Filter by patterns if specified
            if patterns:
                if not any(file_path.match(p) for p in patterns):
                    continue
            
            # Check if we can handle this file
            if self.text_handler.can_handle(str(file_path)):
                files.append(file_path)
        
        return files
    
    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        path_str = str(file_path)
        
        for pattern in self.config.skip_patterns:
            if pattern.startswith('*'):
                # Glob pattern
                if file_path.match(pattern):
                    return True
            else:
                # Directory/file name
                if pattern in path_str:
                    return True
        
        return False
    
    async def ingest_text(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> IngestionResult:
        """
        Ingest raw text content directly.
        
        Useful for content from APIs, databases, etc.
        """
        metadata = metadata or {}
        
        # Use the chunker directly
        if hasattr(self.processor, 'chunker') and self.processor.chunker:
            chunks = self.processor.chunker.chunk_text(
                content,
                document_id=metadata.get('document_id', 'raw_text')
            )
        else:
            # Basic chunking fallback
            chunk_size = self.config.chunk_size
            chunks = []
            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                if chunk_content.strip():
                    chunks.append(type('Chunk', (), {
                        'content': chunk_content,
                        'metadata': type('Metadata', (), {'chunk_id': f'chunk_{i}'})()
                    })())
        
        # Generate embeddings if available
        embeddings_count = 0
        if self.embedding_generator and chunks:
            try:
                chunk_texts = [c.content for c in chunks]
                embeddings = await self.embedding_generator.generate_embeddings(texts=chunk_texts)
                embeddings_count = len(embeddings)
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")
        
        return IngestionResult(
            document_path='<raw_text>',
            chunk_count=len(chunks),
            embeddings_count=embeddings_count,
            content_type='text',
            processing_time_ms=0,
            success=True
        )

