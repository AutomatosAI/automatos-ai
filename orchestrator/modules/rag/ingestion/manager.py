
"""
Document Management System for Context Engineering
=================================================

This module provides comprehensive document management capabilities including:
- Multi-format document processing (PDF, DOCX, MD, TXT)
- Intelligent chunking strategies
- Embedding generation and storage
- Document metadata management
- Source attribution and tracking
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import magic
import pdfplumber
from docx import Document as DocxDocument
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter
)
from langchain_core.documents import Document
# OpenAI embeddings handled by centralized manager
import psycopg2

# Use SemanticChunker from RAG module
try:
    from modules.rag.chunking.semantic_chunker import SemanticChunker, ChunkingStrategy
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False
from psycopg2.extras import RealDictCursor
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "md"
    TEXT = "txt"
    PYTHON = "py"
    JSON = "json"

@dataclass
class DocumentMetadata:
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    processed_date: Optional[datetime] = None
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    tags: List[str] = None
    description: str = ""
    created_by: str = "system"
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['upload_date'] = self.upload_date.isoformat()
        if self.processed_date:
            data['processed_date'] = self.processed_date.isoformat()
        data['status'] = self.status.value
        return data

@dataclass
class DocumentChunk:
    document_id: int
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict = None
    parent_content: Optional[str] = None  # NEW: Parent section for context expansion
    headers: Dict[str, str] = None  # NEW: Header hierarchy (h1, h2, h3)
    
    def to_dict(self) -> Dict:
        return {
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'content': self.content,
            'metadata': self.metadata or {},
            'parent_content': self.parent_content,
            'headers': self.headers or {}
        }

class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.code_splitter = PythonCodeTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def detect_file_type(self, file_path: str) -> DocumentType:
        """Detect file type using magic numbers and extension"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            extension = Path(file_path).suffix.lower()
            
            if mime_type == 'application/pdf' or extension == '.pdf':
                return DocumentType.PDF
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'] or extension == '.docx':
                return DocumentType.DOCX
            elif extension in ['.md', '.markdown']:
                return DocumentType.MARKDOWN
            elif extension == '.py':
                return DocumentType.PYTHON
            elif extension == '.json':
                return DocumentType.JSON
            else:
                return DocumentType.TEXT
        except Exception as e:
            logger.warning(f"Could not detect file type for {file_path}: {e}")
            return DocumentType.TEXT
    
    def extract_text_from_pdf(self, file_path: str) -> str:
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
                        import re
                        page_text = re.sub(r'(.)\1+', r'\1', page_text)  # Remove duplicate characters
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from any supported file type"""
        file_type = self.detect_file_type(file_path)
        
        if file_type == DocumentType.PDF:
            text = self.extract_text_from_pdf(file_path)
        elif file_type == DocumentType.DOCX:
            text = self.extract_text_from_docx(file_path)
        else:
            # For text-based files (MD, TXT, PY, JSON)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
        
        # Clean any remaining null characters from all text sources
        if text:
            text = text.replace('\x00', '').replace('\x01', '').replace('\x02', '')
        
        return text
    
    def chunk_document(self, text: str, file_type: DocumentType, metadata: Dict = None) -> List[DocumentChunk]:
        """
        Split document into chunks using EXISTING SemanticChunker.
        
        Uses context_engineering/chunking/semantic_chunker.py which has:
        - SEMANTIC_SIMILARITY strategy
        - INFORMATION_DENSITY strategy  
        - TOPIC_COHERENCE strategy
        - HIERARCHICAL strategy
        - ADAPTIVE strategy
        
        Plus mathematical foundations (entropy, information theory).
        """
        chunks = []
        
        # USE EXISTING SEMANTIC CHUNKER (NOT A DUPLICATE!)
        if SEMANTIC_CHUNKER_AVAILABLE:
            try:
                # Use ADAPTIVE strategy for best results
                semantic_chunker = SemanticChunker(
                    strategy=ChunkingStrategy.ADAPTIVE,
                    target_chunk_size=500,
                    min_chunk_size=100,
                    max_chunk_size=1500,
                    overlap_ratio=0.1,
                    similarity_threshold=0.7
                )
                
                doc_id = metadata.get('document_id') if metadata else None
                semantic_chunks = semantic_chunker.chunk_text(text, document_id=str(doc_id) if doc_id else None)
                
                for i, sc in enumerate(semantic_chunks):
                    chunk_metadata = {
                        'file_type': file_type.value,
                        'chunk_size': len(sc.content),
                        # Preserve mathematical metrics from SemanticChunker
                        'entropy': sc.metadata.entropy,
                        'topic_coherence': sc.metadata.topic_coherence,
                        'semantic_density': sc.metadata.semantic_density,
                        'importance_score': sc.metadata.importance_score,
                        **(metadata or {})
                    }
                    
                    chunks.append(DocumentChunk(
                        document_id=0,
                        chunk_index=i,
                        content=sc.content,
                        metadata=chunk_metadata,
                        parent_content=None,  # SemanticChunker handles this differently
                        headers={}
                    ))
                
                logger.info(f"SemanticChunker (ADAPTIVE) created {len(chunks)} chunks with entropy/coherence metrics")
                return chunks
                
            except Exception as e:
                logger.warning(f"SemanticChunker failed, falling back to basic chunking: {e}")
                import traceback
                traceback.print_exc()
        
        # FALLBACK: Use basic splitters for non-markdown or if SmartChunker fails
        if file_type == DocumentType.PYTHON:
            docs = self.code_splitter.create_documents([text])
        else:
            docs = self.text_splitter.create_documents([text])
        
        # POST-PROCESS: Filter and merge chunks
        valid_chunks = []
        chunk_index = 0
        
        for doc in docs:
            content = doc.page_content.strip()
            
            # FILTER OUT BAD CHUNKS: separators, empty, too short
            if not content or len(content) < 50:
                continue
            
            # Skip chunks that are just separators or code fences
            if content in ['---', '```', '```python', '```javascript', '```typescript', '```json', '```yaml', '```bash']:
                continue
            
            # Skip chunks that are mostly separators/whitespace
            without_separators = content.replace('-', '').replace('=', '').replace('_', '').replace('#', '').replace('`', '').replace('*', '').strip()
            if len(without_separators) < 30:
                continue
            
            # Skip chunks that are just markdown headers with no content
            if content.count('\n') == 0 and content.startswith('#'):
                continue
            
            # Check if chunk has meaningful content (not just formatting)
            words = without_separators.split()
            if len(words) < 5:
                continue
            
            chunk_metadata = {
                'file_type': file_type.value,
                'chunk_size': len(content),
                **(metadata or {})
            }
            
            valid_chunks.append(DocumentChunk(
                document_id=0,
                chunk_index=chunk_index,
                content=content,
                metadata=chunk_metadata
            ))
            chunk_index += 1
        
        return valid_chunks

class DocumentManager:
    """Main document management class"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.processor = DocumentProcessor()
        self._db_initialized = False
        self._init_lock = False  # Simple flag to prevent concurrent initialization
        
        # Use centralized embedding manager
        from core.llm import create_embedding_manager
        self.embedding_manager = create_embedding_manager()
        logger.info(f"DocumentManager using {self.embedding_manager.get_provider_info()['provider']} embeddings")
        
        # Don't initialize database at import time - do it lazily on first use
    
    def _ensure_database_initialized(self, max_retries: int = 5, retry_delay: float = 2.0):
        """Ensure database is initialized with retry logic"""
        if self._db_initialized:
            return
        
        if self._init_lock:
            # Wait for concurrent initialization
            for _ in range(max_retries):
                time.sleep(retry_delay)
                if self._db_initialized:
                    return
            raise RuntimeError("Database initialization timeout")
        
        self._init_lock = True
        try:
            for attempt in range(max_retries):
                try:
                    self._init_database()
                    self._db_initialized = True
                    logger.info("Database initialized successfully")
                    return
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    if "the database system is starting up" in str(e) or "connection" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    raise
            raise RuntimeError(f"Failed to initialize database after {max_retries} attempts")
        finally:
            self._init_lock = False
    
    def _init_database(self):
        """Initialize database tables if they don't exist"""
        conn = psycopg2.connect(**self.db_config)
        try:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_date TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'pending',
                    chunk_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_by VARCHAR(100) DEFAULT 'system',
                    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                    description TEXT DEFAULT '',
                    file_hash VARCHAR(64) UNIQUE,
                    workspace_id TEXT
                );
            """)
            
            # Create document_chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    workspace_id TEXT,
                    parent_content TEXT,
                    headers JSONB DEFAULT '{}'
                );
            """)
            
            # Create context_usage table for analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_usage (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    chunk_id INTEGER REFERENCES document_chunks(id),
                    query_text TEXT,
                    relevance_score FLOAT,
                    used_in_response BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);")
            # cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);")
            
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for deduplication"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def upload_document(self, file_path: str, filename: str = None, 
                            tags: List[str] = None, description: str = "",
                            created_by: str = "system") -> int:
        """Upload and process a document"""
        self._ensure_database_initialized()
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if filename is None:
                filename = os.path.basename(file_path)
            
            # NEW: Copy file to persistent storage
            storage_dir = "/var/automatos/documents"
            os.makedirs(storage_dir, exist_ok=True)
            
            persistent_path = os.path.join(storage_dir, filename)
            
            # Copy file if not already in storage
            if os.path.abspath(file_path) != os.path.abspath(persistent_path):
                import shutil
                shutil.copy2(file_path, persistent_path)
                logger.info(f"Copied document to {persistent_path}")
            
            file_size = os.path.getsize(persistent_path)
            file_hash = self._calculate_file_hash(persistent_path)
            file_type = self.processor.detect_file_type(persistent_path)
            
            # Check if document already exists
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,))
            existing = cursor.fetchone()
            if existing:
                logger.info(f"Document with hash {file_hash} already exists with ID {existing[0]}")
                cursor.close()
                conn.close()
                return existing[0]
            
            # Create document record
            metadata = DocumentMetadata(
                filename=filename,
                file_type=file_type.value,
                file_size=file_size,
                upload_date=datetime.now(),
                tags=tags or [],
                description=description,
                created_by=created_by
            )
            
            cursor.execute("""
                INSERT INTO documents (filename, file_type, file_size, upload_date, status, 
                                     metadata, created_by, tags, description, file_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                metadata.filename, metadata.file_type, metadata.file_size,
                metadata.upload_date, DocumentStatus.PROCESSING.value,
                json.dumps(metadata.to_dict()), metadata.created_by,
                metadata.tags, metadata.description, file_hash
            ))
            
            document_id = cursor.fetchone()[0]
            conn.commit()
            
            # Process document asynchronously (use persistent path)
            await self._process_document(document_id, persistent_path, file_type)
            
            cursor.close()
            conn.close()
            
            logger.info(f"Document uploaded successfully with ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            raise
    
    async def _process_document(self, document_id: int, file_path: str, file_type: DocumentType):
        """Process document: extract text, chunk, and generate embeddings"""
        self._ensure_database_initialized()
        try:
            logger.info(f"Starting document processing for document {document_id}, type: {file_type}")

            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Fetch workspace_id from document record
            cursor.execute("SELECT workspace_id FROM documents WHERE id = %s", (document_id,))
            doc_row = cursor.fetchone()
            workspace_id = doc_row[0] if doc_row else None
            logger.info(f"Processing document {document_id} for workspace {workspace_id}")

            # Extract text
            text = self.processor.extract_text_from_file(file_path)
            logger.info(f"Extracted {len(text)} characters from document {document_id}")
            
            # Multimodal processing (tables, formulas, images)
            try:
                from modules.rag.ingestion.multimodal import TableProcessor, FormulaProcessor, ImageProcessor
                from modules.knowledge import EntityExtractor, create_or_get_entity, create_entity_mention, create_entity_relationship
                
                logger.info(f"Starting multimodal processing for document {document_id}")
                
                # Create knowledge_items entry for this document (if not exists)
                cursor.execute("""
                    INSERT INTO knowledge_items (id, kb_type_id, title, content, metadata, quality_score)
                    SELECT %s, 
                           (SELECT id FROM kb_types WHERE type_name = 'document' LIMIT 1),
                           %s,
                           %s,
                           %s,
                           0.8
                    WHERE NOT EXISTS (SELECT 1 FROM knowledge_items WHERE id = %s)
                """, (
                    document_id,
                    os.path.basename(file_path),
                    text[:1000] if text else '',  # First 1000 chars as preview
                    json.dumps({'source': 'document_upload', 'file_type': str(file_type)}),
                    document_id
                ))
                conn.commit()
                logger.info(f"Created knowledge_items entry for document {document_id}")
                
                # Extract tables based on file type
                tables = []
                try:
                    if file_type == DocumentType.PDF or file_type == DocumentType.DOCX:
                        # Use camelot for PDFs
                        table_processor = TableProcessor()
                        tables = table_processor.extract_tables_from_pdf(file_path)
                    elif file_type == DocumentType.MARKDOWN or file_type == DocumentType.TEXT:
                        # Extract Markdown tables using regex
                        import re
                        
                        # Find Markdown table patterns
                        table_pattern = r'\|(.+)\|[\r\n]+\|[\s:-]+\|[\r\n]+((?:\|.+\|[\r\n]+)+)'
                        matches = re.finditer(table_pattern, text, re.MULTILINE)
                        
                        for match in matches:
                            table_text = match.group(0)
                            # Convert to simple format
                            tables.append({
                                'markdown': table_text,
                                'data': [],  # Could parse if needed
                                'csv': table_text.replace('|', ',')  # Simple CSV conversion
                            })
                        
                        logger.info(f"Found {len(tables)} markdown tables in document {document_id}")
                    
                    # Insert all extracted tables (from PDF or Markdown)
                    if tables:
                        for idx, table_data in enumerate(tables):
                            # Handle both dict and dataclass formats
                            if isinstance(table_data, dict):
                                markdown_content = table_data.get('markdown')
                                json_content = json.dumps(table_data.get('data', []))
                                csv_content = table_data.get('csv')
                            else:
                                # TableExtraction dataclass
                                markdown_content = table_data.markdown
                                json_content = json.dumps(table_data.json)
                                csv_content = table_data.csv
                            
                            # Extract table details
                            if hasattr(table_data, 'headers'):
                                headers = json.dumps(table_data.headers)
                                row_count = table_data.row_count
                                column_count = table_data.column_count
                            else:
                                headers = json.dumps([])
                                row_count = len(markdown_content.split('\n')) if markdown_content else 0
                                column_count = len(markdown_content.split('|')) if markdown_content else 0
                            
                            cursor.execute("""
                                INSERT INTO kb_tables (
                                    knowledge_item_id, headers, row_count, column_count, 
                                    markdown_representation, csv_data, json_data
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                document_id,
                                headers,
                                row_count,
                                column_count,
                                markdown_content,
                                csv_content,
                                json_content
                            ))
                        
                        conn.commit()
                        logger.info(f"Inserted {len(tables)} tables into knowledge base for document {document_id}")
                    
                except Exception as e:
                    logger.warning(f"Table extraction failed for document {document_id}: {e}")
                
                # Process formulas (for ALL file types)
                try:
                    formula_processor = FormulaProcessor()
                    formulas = formula_processor.extract_formulas_from_text(text)
                    
                    if formulas:
                        for idx, formula_data in enumerate(formulas):
                            # Handle both dict and dataclass formats
                            if isinstance(formula_data, dict):
                                latex_content = formula_data.get('latex')
                                mathml_content = formula_data.get('mathml')
                                rendered_url = formula_data.get('rendered_url')
                            else:
                                # FormulaExtraction dataclass
                                latex_content = formula_data.latex
                                mathml_content = None  # Not provided by FormulaExtraction
                                rendered_url = None    # Not provided by FormulaExtraction
                            
                            cursor.execute("""
                                INSERT INTO kb_formulas (
                                    knowledge_item_id, latex, mathml, ascii_math
                                ) VALUES (%s, %s, %s, %s)
                            """, (
                                document_id,
                                latex_content,
                                mathml_content or '',
                                formula_data.ascii_math if hasattr(formula_data, 'ascii_math') else ''
                            ))
                        
                        conn.commit()
                        logger.info(f"Inserted {len(formulas)} formulas into knowledge base for document {document_id}")
                
                except Exception as e:
                    logger.warning(f"Formula extraction failed for document {document_id}: {e}")
                
                # NEW: Entity extraction for knowledge graph
                try:
                    logger.info(f"Starting entity extraction for document {document_id}")
                    entity_extractor = EntityExtractor()
                    
                    # Extract entities from text
                    entities = await entity_extractor.extract_entities(text, use_llm=True, max_entities=30)
                    logger.info(f"Extracted {len(entities)} entities from document {document_id}")
                    
                    # Store entities and create mentions
                    entity_ids = []
                    for entity in entities:
                        entity_id = await create_or_get_entity(
                            cursor,
                            entity.entity_name,
                            entity.entity_type,
                            entity.canonical_name,
                            entity.description
                        )
                        entity_ids.append(entity_id)
                        
                        await create_entity_mention(
                            cursor,
                            document_id,
                            entity_id,
                            entity.mention_context,
                            entity.confidence,
                            entity.position,
                            "llm"
                        )
                    
                    conn.commit()
                    logger.info(f"Stored {len(entities)} entities for document {document_id}")
                    
                    # Extract relationships between entities
                    if len(entities) >= 2:
                        relationships = await entity_extractor.extract_relationships(text, entities, max_relationships=20)
                        logger.info(f"Extracted {len(relationships)} relationships from document {document_id}")
                        
                        for rel in relationships:
                            await create_entity_relationship(
                                cursor,
                                rel.from_entity,
                                rel.to_entity,
                                rel.relationship_type,
                                rel.strength,
                                document_id,
                                rel.evidence
                            )
                        
                        conn.commit()
                        logger.info(f"Stored {len(relationships)} relationships for document {document_id}")
                    
                except Exception as e:
                    logger.warning(f"Entity extraction failed for document {document_id}: {e}")
                    import traceback
                    traceback.print_exc()
                
                conn.commit()
                
            except Exception as e:
                logger.warning(f"Multimodal processing encountered errors for document {document_id}: {e}")
                # Continue processing even if multimodal extraction fails
            
            # Create chunks with file_path metadata
            chunks = self.processor.chunk_document(text, file_type, {
                'document_id': document_id,
                'source_file': os.path.basename(file_path),
                'file_path': file_path  # NEW: Add full path for downloads
            })
            
            # Generate embeddings and save chunks
            valid_chunks = []
            for chunk in chunks:
                chunk.document_id = document_id
                
                # SAFETY CHECK: Skip bad chunks (separators, empty, too short)
                content = chunk.content.strip() if chunk.content else ""
                if not content or len(content) < 20:
                    continue
                if content in ['---', '```', '```python', '```javascript', '```typescript']:
                    continue
                stripped = content.replace('-', '').replace('=', '').replace('_', '').replace('#', '').replace('`', '').strip()
                if len(stripped) < 10:
                    continue
                
                # Generate embedding
                embedding = await self._generate_embedding(chunk.content)
                chunk.embedding = embedding
                
                # Save chunk to database (with parent_content and headers for smart chunking)
                cursor.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata, parent_content, headers, workspace_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    chunk.document_id, chunk.chunk_index, chunk.content,
                    embedding, json.dumps(chunk.metadata),
                    chunk.parent_content,  # NEW: Store parent content
                    json.dumps(chunk.headers) if chunk.headers else '{}',  # NEW: Store headers
                    workspace_id  # Include workspace_id from parent document
                ))
                valid_chunks.append(chunk)
            
            # Update document status
            cursor.execute("""
                UPDATE documents 
                SET status = %s, processed_date = %s, chunk_count = %s
                WHERE id = %s
            """, (
                DocumentStatus.COMPLETED.value, datetime.now(), len(valid_chunks), document_id
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Document {document_id} processed successfully with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            
            # Update status to failed
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE documents SET status = %s WHERE id = %s
                """, (DocumentStatus.FAILED.value, document_id))
                conn.commit()
                cursor.close()
                conn.close()
            except:
                pass
            
            raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = await self.embedding_manager.generate_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def list_documents(self, status: Optional[DocumentStatus] = None, 
                      file_type: Optional[DocumentType] = None,
                      limit: int = 100, offset: int = 0) -> List[Dict]:
        """List documents with optional filtering"""
        self._ensure_database_initialized()
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM documents WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = %s"
                params.append(status.value)
            
            if file_type:
                query += " AND file_type = %s"
                params.append(file_type.value)
            
            query += " ORDER BY upload_date DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            documents = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(doc) for doc in documents]
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise
    
    def get_document(self, document_id: int) -> Optional[Dict]:
        """Get document by ID"""
        self._ensure_database_initialized()
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("SELECT * FROM documents WHERE id = %s", (document_id,))
            document = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return dict(document) if document else None
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks"""
        self._ensure_database_initialized()
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Delete chunks first (due to foreign key constraint)
            cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            
            # Delete document
            cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            
            deleted = cursor.rowcount > 0
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Document {document_id} deleted: {deleted}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents by content similarity"""
        self._ensure_database_initialized()
        try:
            # Generate query embedding
            query_embedding = asyncio.run(self._generate_embedding(query))
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Format embedding for pgvector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Search for similar chunks using real vector similarity
            cursor.execute("""
                SELECT 
                    dc.id as chunk_id,
                    dc.document_id,
                    dc.content,
                    dc.metadata,
                    d.filename,
                    d.file_type,
                    1.0 as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.status = 'completed'
                    AND dc.embedding IS NOT NULL
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, limit))
            
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def get_document_stats(self) -> Dict:
        """Get document statistics"""
        self._ensure_database_initialized()
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Total documents by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM documents
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Total chunks
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            total_chunks = cursor.fetchone()[0]
            
            # File type distribution
            cursor.execute("""
                SELECT file_type, COUNT(*) as count
                FROM documents
                GROUP BY file_type
            """)
            file_type_counts = dict(cursor.fetchall())
            
            cursor.close()
            conn.close()
            
            return {
                'total_documents': sum(status_counts.values()),
                'status_distribution': status_counts,
                'total_chunks': total_chunks,
                'file_type_distribution': file_type_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            raise

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Database configuration
    db_config = {
        'host': os.getenv('POSTGRES_HOST', '127.0.0.1'),
        'database': 'orchestrator_db',
        'user': 'postgres',
        'password': 'your_password'
    }
    
    # Initialize document manager
    doc_manager = DocumentManager(db_config, "your_openai_api_key")
    
    # Upload a document
    async def test_upload():
        doc_id = await doc_manager.upload_document(
            "/path/to/document.pdf",
            tags=["business", "process"],
            description="Business process documentation"
        )
        print(f"Uploaded document with ID: {doc_id}")
    
    # Run test
    # asyncio.run(test_upload())
