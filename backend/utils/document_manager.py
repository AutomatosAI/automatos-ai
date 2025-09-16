
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
    PythonCodeTextSplitter
)
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import psycopg2
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
    
    def to_dict(self) -> Dict:
        return {
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'content': self.content,
            'metadata': self.metadata or {}
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
            return self.extract_text_from_pdf(file_path)
        elif file_type == DocumentType.DOCX:
            return self.extract_text_from_docx(file_path)
        else:
            # For text-based files (MD, TXT, PY, JSON)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
    
    def chunk_document(self, text: str, file_type: DocumentType, metadata: Dict = None) -> List[DocumentChunk]:
        """Split document into chunks based on file type"""
        chunks = []
        
        if file_type == DocumentType.MARKDOWN:
            docs = self.markdown_splitter.create_documents([text])
        elif file_type == DocumentType.PYTHON:
            docs = self.code_splitter.create_documents([text])
        else:
            docs = self.text_splitter.create_documents([text])
        
        for i, doc in enumerate(docs):
            chunk_metadata = {
                'file_type': file_type.value,
                'chunk_size': len(doc.page_content),
                **(metadata or {})
            }
            
            chunks.append(DocumentChunk(
                document_id=0,  # Will be set when saving
                chunk_index=i,
                content=doc.page_content,
                metadata=chunk_metadata
            ))
        
        return chunks

class DocumentManager:
    """Main document management class"""
    
    def __init__(self, db_config: Dict[str, str], openai_api_key: str):
        self.db_config = db_config
        self.processor = DocumentProcessor()
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist"""
        try:
            conn = psycopg2.connect(**self.db_config)
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
                    file_hash VARCHAR(64) UNIQUE
                );
            """)
            
            # Create document_chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);")
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
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
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if filename is None:
                filename = os.path.basename(file_path)
            
            file_size = os.path.getsize(file_path)
            file_hash = self._calculate_file_hash(file_path)
            file_type = self.processor.detect_file_type(file_path)
            
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
            
            # Process document asynchronously
            await self._process_document(document_id, file_path, file_type)
            
            cursor.close()
            conn.close()
            
            logger.info(f"Document uploaded successfully with ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            raise
    
    async def _process_document(self, document_id: int, file_path: str, file_type: DocumentType):
        """Process document: extract text, chunk, and generate embeddings"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Extract text
            text = self.processor.extract_text_from_file(file_path)
            
            # Create chunks
            chunks = self.processor.chunk_document(text, file_type, {
                'document_id': document_id,
                'source_file': os.path.basename(file_path)
            })
            
            # Generate embeddings and save chunks
            for chunk in chunks:
                chunk.document_id = document_id
                
                # Generate embedding
                embedding = await self._generate_embedding(chunk.content)
                chunk.embedding = embedding
                
                # Save chunk to database
                cursor.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    chunk.document_id, chunk.chunk_index, chunk.content,
                    embedding, json.dumps(chunk.metadata)
                ))
            
            # Update document status
            cursor.execute("""
                UPDATE documents 
                SET status = %s, processed_date = %s, chunk_count = %s
                WHERE id = %s
            """, (
                DocumentStatus.COMPLETED.value, datetime.now(), len(chunks), document_id
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
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def list_documents(self, status: Optional[DocumentStatus] = None, 
                      file_type: Optional[DocumentType] = None,
                      limit: int = 100, offset: int = 0) -> List[Dict]:
        """List documents with optional filtering"""
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
        try:
            # Generate query embedding
            query_embedding = asyncio.run(self._generate_embedding(query))
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search for similar chunks
            cursor.execute("""
                SELECT 
                    dc.id as chunk_id,
                    dc.document_id,
                    dc.content,
                    dc.metadata,
                    d.filename,
                    d.file_type,
                    1 - (dc.embedding <=> %s::vector) as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.status = 'completed'
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def get_document_stats(self) -> Dict:
        """Get document statistics"""
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
        'host': 'localhost',
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
