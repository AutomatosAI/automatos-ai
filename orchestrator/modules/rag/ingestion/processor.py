"""
Document Ingestion Processor
============================

Document processing pipeline for ingesting documents into RAG system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..chunking import SemanticChunker, SemanticChunk

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion"""
    document_path: str
    chunk_count: int
    embeddings_count: int
    content_type: str
    processing_time_ms: float
    success: bool
    error: Optional[str] = None


class DocumentProcessor:
    """Complete document ingestion processor"""
    
    def __init__(
        self,
        vector_store=None,
        embedding_generator=None,
        chunker: SemanticChunker = None
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.chunker = chunker or SemanticChunker()
        
        # Supported file types
        self.supported_extensions = {
            '.txt': 'text',
            '.md': 'markdown',
            '.py': 'code',
            '.js': 'code',
            '.ts': 'code',
            '.java': 'code',
            '.cpp': 'code',
            '.c': 'code',
            '.go': 'code',
            '.rs': 'code',
            '.php': 'code',
            '.rb': 'code',
            '.json': 'json',
            '.yaml': 'config',
            '.yml': 'config',
            '.toml': 'config',
            '.ini': 'config',
            '.html': 'markup',
            '.xml': 'markup',
            '.css': 'code',
            '.sql': 'code'
        }
    
    async def ingest_directory(
        self, 
        directory_path: str, 
        recursive: bool = True
    ) -> Dict[str, Any]:
        """Ingest all supported files from a directory"""
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all supported files
        files_to_process = []
        
        if recursive:
            for ext in self.supported_extensions.keys():
                files_to_process.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in self.supported_extensions.keys():
                files_to_process.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files
        results = {
            'processed_files': 0,
            'total_chunks': 0,
            'failed_files': [],
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        for file_path in files_to_process:
            try:
                file_results = await self.ingest_file(str(file_path))
                if file_results.success:
                    results['processed_files'] += 1
                    results['total_chunks'] += file_results.chunk_count
                    logger.info(f"Processed {file_path.name}: {file_results.chunk_count} chunks")
                else:
                    results['failed_files'].append(str(file_path))
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                results['failed_files'].append(str(file_path))
        
        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    async def ingest_file(self, file_path: str) -> IngestionResult:
        """Ingest a single file"""
        
        start_time = datetime.now()
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return IngestionResult(
                document_path=file_path,
                chunk_count=0,
                embeddings_count=0,
                content_type='unknown',
                processing_time_ms=0,
                success=False,
                error=f"File does not exist: {file_path}"
            )
        
        # Determine content type
        extension = file_path_obj.suffix.lower()
        content_type = self.supported_extensions.get(extension, 'text')
        
        # Read file content
        try:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        except Exception as e:
            return IngestionResult(
                document_path=file_path,
                chunk_count=0,
                embeddings_count=0,
                content_type=content_type,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                success=False,
                error=f"Failed to read file: {str(e)}"
            )
        
        if not content.strip():
            logger.warning(f"File is empty: {file_path}")
            return IngestionResult(
                document_path=file_path,
                chunk_count=0,
                embeddings_count=0,
                content_type=content_type,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                success=True,
                error="File is empty"
            )
        
        # Chunk the document
        chunks = self.chunker.chunk_text(content, file_path)
        
        if not chunks:
            logger.warning(f"No chunks generated for: {file_path}")
            return IngestionResult(
                document_path=file_path,
                chunk_count=0,
                embeddings_count=0,
                content_type=content_type,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                success=True,
                error="No chunks generated"
            )
        
        embeddings_count = 0
        
        # Generate embeddings if generator available
        if self.embedding_generator:
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_generator.generate_embeddings(texts=chunk_texts)
            embeddings_count = len(embeddings)
            
            # Attach embeddings to chunks
            for i, embedding in enumerate(embeddings):
                if i < len(chunks):
                    chunks[i].embedding = embedding
        
        # Store in vector database if available
        if self.vector_store:
            await self._store_chunks(chunks, file_path, content_type)
        
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResult(
            document_path=file_path,
            chunk_count=len(chunks),
            embeddings_count=embeddings_count,
            content_type=content_type,
            processing_time_ms=processing_time_ms,
            success=True
        )
    
    async def _store_chunks(
        self, 
        chunks: List[SemanticChunk], 
        file_path: str, 
        content_type: str
    ) -> None:
        """Store chunks in vector store"""
        
        if not self.vector_store:
            return
            
        # Convert chunks to storage format
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                'content': chunk.content,
                'embedding': chunk.embedding,
                'metadata': {
                    'chunk_id': chunk.metadata.chunk_id,
                    'source': file_path,
                    'content_type': content_type,
                    'word_count': chunk.metadata.word_count,
                    'importance_score': chunk.metadata.importance_score
                }
            })
        
        await self.vector_store.add_embeddings(chunk_data)
    
    def _detect_programming_language(self, extension: str) -> str:
        """Detect programming language from file extension"""
        
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.css': 'css',
            '.sql': 'sql'
        }
        
        return lang_map.get(extension.lower(), 'text')

