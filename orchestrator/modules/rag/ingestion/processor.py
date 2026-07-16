"""
Document Ingestion Processor
============================

Document processing pipeline for ingesting documents into RAG system.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import pdfplumber
from docx import Document as DocxDocument

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
        embedding_generator=None,
        chunker: SemanticChunker = None
    ):
        self.embedding_generator = embedding_generator
        self.chunker = chunker or SemanticChunker()
        
        # Supported file types
        self.supported_extensions = {
            '.pdf': 'pdf',
            '.docx': 'docx',
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

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Remove null characters and other problematic characters
                        page_text = page_text.replace('\x00', '').replace('\x01', '').replace('\x02', '')
                        # Fix double characters (common PDF extraction issue)
                        page_text = re.sub(r'(.)\1+', r'\1', page_text)
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""

    def _extract_text(self, file_path: str, content_type: str) -> str:
        """Extract text based on file type"""
        if content_type == 'pdf':
            return self._extract_text_from_pdf(file_path)
        elif content_type == 'docx':
            return self._extract_text_from_docx(file_path)
        else:
            # Text-based files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    return ""

            # Clean null characters
            if text:
                text = text.replace('\x00', '').replace('\x01', '').replace('\x02', '')
            return text

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
        
        # Extract text based on file type
        try:
            content = self._extract_text(file_path, content_type)
            if not content:
                return IngestionResult(
                    document_path=file_path,
                    chunk_count=0,
                    embeddings_count=0,
                    content_type=content_type,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    success=False,
                    error="Failed to extract text from file"
                )
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
        
        # No vector-store write here (PRD-186 S1): the seam this processor
        # used to carry called a method no backend defines, with metadata that
        # never stamped workspace_id — a crash if armed, an unlabeled
        # cross-tenant chunk if it ever "worked". Vector writes happen at the
        # workspace-stamping path (S3VectorsBackend.add_documents) only.
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return IngestionResult(
            document_path=file_path,
            chunk_count=len(chunks),
            embeddings_count=embeddings_count,
            content_type=content_type,
            processing_time_ms=processing_time_ms,
            success=True
        )

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

