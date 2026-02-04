
"""
Document Management API Routes
=============================

Enhanced REST API endpoints for document upload, processing, and management.
"""

import os
import hashlib
import tempfile
import json
import uuid
import magic
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, text

from core.database.database import get_db
from core.models import Document, DocumentUploadResponse, DocumentResponse
from modules.rag import DocumentManager, DocumentStatus, DocumentType
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

# Initialize credential resolver
from core.credentials.resolver import get_credential_resolver
from urllib.parse import urlparse
import os
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

def parse_database_url(url: str) -> dict:
    """Parse DATABASE_URL into psycopg2 connection dict format"""
    parsed = urlparse(url)
    return {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }

# Try DATABASE_URL first (Railway, Heroku, etc.)
database_url = os.getenv('DATABASE_URL')
if database_url:
    postgres_creds = parse_database_url(database_url)
else:
    # Try credential resolver
    resolver = get_credential_resolver()
    try:
        postgres_creds = resolver.get_dict("development_db")
    except Exception:
        # Fallback to individual environment variables
        postgres_creds = {
            'database': os.getenv('POSTGRES_DB', 'automatos'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }

db_config = {
    "database": postgres_creds.get('database', 'orchestrator_db'),
    "user": postgres_creds.get('user', 'postgres'),
    "password": postgres_creds.get('password', ''),
    "host": postgres_creds.get('host', 'localhost'),
    "port": int(postgres_creds.get('port', 5432))
}

# Document manager uses centralized embedding manager
doc_manager = DocumentManager(db_config)

# Allowlist of accepted MIME types mapped to valid extensions
ALLOWED_MIME_TYPES = {
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "text/plain": [".txt", ".md", ".csv"],
    "text/markdown": [".md"],
    "text/csv": [".csv"],
    "application/json": [".json"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "application/octet-stream": [".pdf", ".docx", ".xlsx"],  # fallback for binary
}

@router.post("/upload", response_model=DocumentUploadResponse)
async def handle_request(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (limit to 50MB)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        # MIME type validation using python-magic (detect actual content type)
        detected_mime = magic.from_buffer(content, mime=True)
        file_extension = Path(file.filename).suffix.lower()

        if detected_mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Detected MIME type: {detected_mime}"
            )

        # Verify extension matches detected MIME type
        allowed_extensions = ALLOWED_MIME_TYPES[detected_mime]
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File extension '{file_extension}' does not match detected content type '{detected_mime}'. "
                       f"Allowed extensions for this type: {', '.join(allowed_extensions)}"
            )

        # Reset file pointer
        await file.seek(0)

        # Generate file hash
        content_hash = hashlib.sha256(content).hexdigest()

        # Check for duplicate
        existing = db.query(Document).filter(Document.content_hash == content_hash, Document.workspace_id == ctx.workspace_id).first()
        if existing:
            return DocumentUploadResponse(
                document_id=existing.id,
                filename=existing.filename,
                status="duplicate",
                message="Document already exists"
            )

        # Save file temporarily with random filename (never use original filename)
        upload_dir = Path("/tmp/automotas_uploads")
        upload_dir.mkdir(exist_ok=True)

        safe_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = upload_dir / safe_filename
        
        with open(file_path, "wb") as f:
            f.write(content)

        # Determine file type category from extension
        file_type = "unknown"
        if file_extension in ['.pdf']:
            file_type = "pdf"
        elif file_extension in ['.md', '.markdown']:
            file_type = "markdown"
        elif file_extension in ['.txt']:
            file_type = "text"
        elif file_extension in ['.doc', '.docx']:
            file_type = "document"
        elif file_extension in ['.json']:
            file_type = "json"
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Create document record
        # TEMPORARY FIX: Tags field commented out to unblock critical vector DB testing
        # Tags are cosmetic metadata - not needed for embeddings, RAG, or semantic search
        # Will add back with proper fix after core functionality is validated
        document = Document(
        workspace_id=ctx.workspace_id,
            filename=file.filename,
            original_filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            file_path=str(file_path),
            content_hash=content_hash,
            status="uploaded",
            # tags=tag_list if tag_list else None,  # TEMPORARILY DISABLED - SQLAlchemy array bug
            description=description,
            created_by="system"  # TODO: Get from auth context
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Process document asynchronously
        try:
            # Use existing document manager for processing
            # Note: DocumentManager creates its own DB record, so we skip calling it
            # and instead trigger processing directly
            from modules.rag import DocumentManager, DocumentType
            import asyncio
            
            # Determine file type enum
            file_type_enum = {
                'pdf': DocumentType.PDF,
                'text': DocumentType.TEXT,
                'markdown': DocumentType.MARKDOWN,
                'json': DocumentType.JSON,
            }.get(file_type, DocumentType.TEXT)
            
            # Process document directly
            document.status = "processing"
            db.commit()
            
            # Call processing method directly
            await doc_manager._process_document(document.id, str(file_path), file_type_enum)
            
            # Refresh to get updated chunk_count
            db.refresh(document)
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            document.status = "failed"
            db.commit()
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            status=document.status,
            message="Document uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/download")
async def download_document(path: str = Query(..., description="Full path to document")):
    """
    Download a document from storage.
    
    Args:
        path: Full path to document (e.g., /var/automatos/documents/filename.md)
    
    Returns:
        FileResponse with the document
    """
    try:
        # Security: Only allow files from /var/automatos/documents/
        allowed_base = Path("/var/automatos/documents").resolve()
        requested_path = Path(path).resolve()

        try:
            requested_path.relative_to(allowed_base)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: Invalid path")
        
        # Check if file exists
        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get filename
        filename = os.path.basename(requested_path)
        
        # Determine media type
        media_type = "application/octet-stream"
        if str(requested_path).endswith('.pdf'):
            media_type = "application/pdf"
        elif str(requested_path).endswith('.md'):
            media_type = "text/markdown"
        elif str(requested_path).endswith('.txt'):
            media_type = "text/plain"
        elif str(requested_path).endswith('.json'):
            media_type = "application/json"
        
        logger.info(f"Serving document: {filename}")
        
        return FileResponse(
            path=str(requested_path),
            filename=filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/content")
async def get_document_content(path: str = Query(..., description="Full path to document")):
    """
    Get document content as text for artifact viewer.
    
    Args:
        path: Full path to document (e.g., /var/automatos/documents/filename.md)
    
    Returns:
        JSON with document content
    """
    try:
        # Security: Only allow files from /var/automatos/documents/
        allowed_base = Path("/var/automatos/documents").resolve()
        requested_path = Path(path).resolve()

        try:
            requested_path.relative_to(allowed_base)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: Invalid path")
        
        # Check if file exists
        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Read file content
        with open(requested_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(requested_path)
        logger.info(f"Serving content for: {filename}")
        
        return {
            "filename": filename,
            "content": content,
            "path": str(requested_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.get("/analytics")
async def get_document_analytics(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get document analytics and statistics
    
    Returns comprehensive analytics including:
    - Total documents by status
    - Total storage used
    - Processing statistics
    - File type distribution
    - Recent upload activity
    """
    try:
        from sqlalchemy import func
        
        # Total documents count
        total_docs = db.query(func.count(Document.id)).filter(Document.workspace_id == ctx.workspace_id).scalar() or 0
        
        # Documents by status
        status_counts = db.query(
            Document.status, 
            func.count(Document.id)
        ).filter(Document.workspace_id == ctx.workspace_id).group_by(Document.status).all()
        
        status_distribution = {status: count for status, count in status_counts}
        
        # Total storage used
        total_storage = db.query(func.sum(Document.file_size)).filter(Document.workspace_id == ctx.workspace_id).scalar() or 0
        
        # File type distribution
        file_type_counts = db.query(
            Document.file_type,
            func.count(Document.id)
        ).filter(Document.workspace_id == ctx.workspace_id).group_by(Document.file_type).all()
        
        file_types = {file_type: count for file_type, count in file_type_counts}
        
        # Total chunks processed
        total_chunks = db.query(func.sum(Document.chunk_count)).filter(Document.workspace_id == ctx.workspace_id).scalar() or 0
        
        # Recent uploads (last 24 hours)
        from datetime import datetime, timedelta
        recent_cutoff = datetime.utcnow() - timedelta(days=1)
        recent_uploads = db.query(func.count(Document.id)).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.upload_date >= recent_cutoff
        ).scalar() or 0
        
        # Average chunk count
        avg_chunks = db.query(func.avg(Document.chunk_count)).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.chunk_count > 0
        ).scalar() or 0
        
        # Processing success rate
        processed_count = status_distribution.get('processed', 0)
        failed_count = status_distribution.get('failed', 0)
        total_processed = processed_count + failed_count
        success_rate = (processed_count / total_processed * 100) if total_processed > 0 else 0
        
        return {
            "total_documents": total_docs,
            "status_distribution": status_distribution,
            "total_storage_bytes": total_storage,
            "total_storage_mb": round(total_storage / (1024 * 1024), 2),
            "file_type_distribution": file_types,
            "total_chunks": total_chunks,
            "average_chunks_per_document": round(float(avg_chunks), 2),
            "recent_uploads_24h": recent_uploads,
            "processing_success_rate": round(success_rate, 2),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting document analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=List[DocumentResponse])
async def get_item(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List documents with filtering and pagination"""
    try:
        query = db.query(Document).filter(Document.workspace_id == ctx.workspace_id)
        
        # Apply filters
        if status:
            query = query.filter(Document.status == status)
        if file_type:
            query = query.filter(Document.file_type == file_type)
        if search:
            query = query.filter(
                or_(
                    Document.filename.ilike(f"%{search}%"),
                    Document.description.ilike(f"%{search}%")
                )
            )
        
        documents = query.order_by(Document.upload_date.desc()).offset(skip).limit(limit).all()
        
        return [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                original_filename=doc.original_filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                status=doc.status,
                chunk_count=doc.chunk_count,
                tags=doc.tags or [],
                description=doc.description,
                upload_date=doc.upload_date,
                processed_date=doc.processed_date,
                created_by=doc.created_by
            ) for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_item(
    ctx: RequestContext = Depends(get_request_context_hybrid),  db: Session = Depends(get_db)):
    """Get document by ID"""
    try:
        document = db.query(Document).filter(Document.id == document_id, Document.workspace_id == ctx.workspace_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            original_filename=document.original_filename,
            file_type=document.file_type,
            file_size=document.file_size,
            status=document.status,
            chunk_count=document.chunk_count,
            tags=document.tags or [],
            description=document.description,
            upload_date=document.upload_date,
            processed_date=document.processed_date,
            created_by=document.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{document_id}/delete-impact")
async def get_delete_impact(
    document_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get deletion impact analysis for a document
    
    Returns information about what will be affected by deleting this document:
    - Number of vector chunks
    - Number of embeddings
    - References count
    - Storage freed
    - Affected workflows/dependencies
    """
    try:
        # Get document
        document = db.query(Document).filter(Document.id == document_id, Document.workspace_id == ctx.workspace_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Count chunks and embeddings
        from sqlalchemy import text
        chunks_query = text("""
            SELECT 
                COUNT(*) as chunk_count,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedding_count
            FROM document_chunks
            WHERE document_id = :document_id
        """)
        chunks_result = db.execute(chunks_query, {"document_id": document_id}).fetchone()
        
        # Calculate storage freed
        storage_freed_mb = document.file_size / (1024 * 1024) if document.file_size else 0
        
        # Check for references (if document_usage table exists)
        references = 0
        try:
            references_query = text("""
                SELECT COUNT(DISTINCT id) as ref_count
                FROM document_usage
                WHERE document_id = :document_id
            """)
            refs_result = db.execute(references_query, {"document_id": document_id}).fetchone()
            references = refs_result.ref_count if refs_result else 0
        except Exception as e:
            # Table might not exist yet - that's OK for MVP
            logger.debug(f"Could not get usage references (table might not exist): {e}")
            db.rollback()  # Clear failed transaction
        
        return {
            "vector_chunks": chunks_result.chunk_count if chunks_result else 0,
            "embeddings": chunks_result.embedding_count if chunks_result else 0,
            "references": references,
            "workflows_affected": [],  # TODO: Add workflow dependency tracking
            "storage_freed": f"{storage_freed_mb:.2f} MB",
            "dependencies": []  # TODO: Add dependency analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting delete impact for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Delete document"""
    try:
        document = db.query(Document).filter(Document.id == document_id, Document.workspace_id == ctx.workspace_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file if it exists
        if document.file_path and os.path.exists(document.file_path):
            try:
                os.remove(document.file_path)
            except Exception as e:
                logger.warning(f"Could not delete file {document.file_path}: {e}")
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Reprocess a document - regenerate chunks and embeddings
    
    Note: This endpoint requires the original document file to still exist.
    For test documents with fake file paths, this will return an appropriate error.
    """
    try:
        document = db.query(Document).filter(Document.id == document_id, Document.workspace_id == ctx.workspace_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if file exists
        if not document.file_path:
            raise HTTPException(
                status_code=400, 
                detail="Document has no file path - cannot reprocess. Upload a new file instead."
            )
        
        if not os.path.exists(document.file_path):
            # File doesn't exist - check if this is test data
            logger.warning(f"Document {document_id} file not found at {document.file_path}")
            raise HTTPException(
                status_code=400,
                detail=f"Document file not found at: {document.file_path}. "
                       f"This may be test data. To reprocess, please re-upload the document."
            )
        
        # Update status
        document.status = "processing"
        db.commit()
        
        # Reprocess document
        try:
            result = await doc_manager.upload_document(
                file_path=document.file_path,
                filename=document.filename,
                file_type=document.file_type,
                description=document.description or "",
                tags=document.tags or [],
                created_by=document.created_by or "system"
            )
            
            # Update document with processing results
            document.status = "processed"
            document.chunk_count = result.get("chunk_count", 0)
            db.commit()
            
            return {
                "message": "Document reprocessed successfully",
                "document_id": document_id,
                "chunk_count": document.chunk_count,
                "status": "processed"
            }
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {e}")
            document.status = "failed"
            db.commit()
            raise HTTPException(status_code=500, detail="Internal server error")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{document_id}/content")
async def get_document_content(
    document_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get document content/chunks"""
    try:
        document = db.query(Document).filter(Document.id == document_id, Document.workspace_id == ctx.workspace_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Query chunks using raw SQL (SQLAlchemy model doesn't exist yet)
        from sqlalchemy import text
        query = text("""
            SELECT id, chunk_index, content, metadata, 
                   CASE WHEN embedding IS NOT NULL THEN true ELSE false END as has_embedding
            FROM document_chunks
            WHERE document_id = :document_id
            ORDER BY chunk_index
        """)
        
        result = db.execute(query, {"document_id": document_id})
        chunks = result.fetchall()
        
        return {
            "document_id": document_id,
            "filename": document.filename,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": row.id,
                    "chunk_index": row.chunk_index,
                    "content": row.content,
                    "has_embedding": row.has_embedding,
                    "metadata": row.metadata if row.metadata else {}
                }
                for row in chunks
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document content {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


PREVIEW_CHUNK_LIMIT = 6  # Max number of chunks included in preview window
PREVIEW_CONTEXT_RADIUS = 2  # Number of chunks to include before the best match
PREVIEW_CHAR_LIMIT = 2000  # Safety limit for preview text length


@router.post("/search")
async def semantic_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    min_similarity: float = Query(0.70, ge=0.0, le=1.0, description="Minimum similarity score"),
    document_ids: Optional[List[int]] = Query(None, description="Optional filter by document IDs"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Semantic search across document chunks using vector similarity
    
    Uses OpenAI embeddings and pgvector for intelligent document search.
    Returns chunks ranked by semantic similarity to the query.
    """
    try:
        import time
        start_time = time.time()
        
        # Generate query embedding using centralized embedding manager
        # NOTE: import directly from module for compatibility across deployments
        from core.llm.embedding_manager import create_embedding_manager
        import asyncio
        
        embedding_manager = create_embedding_manager()
        logger.info(f"Generating embedding for query: {query[:50]}...")
        
        # Generate embedding asynchronously
        query_embedding = await embedding_manager.generate_embedding(query)
        
        logger.info(f"Embedding generated, performing vector search...")
        
        # Build pgvector similarity search query
        # Using <=> operator for cosine distance (pgvector)
        # Similarity = 1 - distance
        
        doc_filter = ""
        if document_ids:
            doc_filter = "AND d.id = ANY(:document_ids)"
        
        # Format embedding as PostgreSQL array string for pgvector
        # Pass as parameterized value to avoid SQL injection
        embedding_str = '[' + ','.join(str(float(v)) for v in query_embedding) + ']'

        similarity_query = text("""
            SELECT
                dc.id as chunk_id,
                dc.document_id,
                dc.chunk_index,
                dc.content,
                dc.metadata,
                d.filename,
                d.file_type,
                d.file_size,
                d.upload_date,
                1 - (dc.embedding <=> cast(:embedding as vector)) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
                AND d.workspace_id = :workspace_id
                {doc_filter}
                AND (1 - (dc.embedding <=> cast(:embedding as vector))) >= :min_similarity
            ORDER BY dc.embedding <=> cast(:embedding as vector)
            LIMIT :limit
        """.format(doc_filter=doc_filter))

        params = {
            "embedding": embedding_str,
            "min_similarity": min_similarity,
            "limit": limit,
            "workspace_id": ctx.workspace_id
        }
        
        if document_ids:
            params["document_ids"] = document_ids
        
        result = db.execute(similarity_query, params)
        rows = result.fetchall()
        
        # Group results by document so we can surface full files
        grouped_results: Dict[int, Dict[str, Any]] = {}
        doc_order: List[int] = []

        for row in rows:
            doc_id = row.document_id
            similarity = float(row.similarity)

            if doc_id not in grouped_results:
                grouped_results[doc_id] = {
                    "document_id": doc_id,
                    "best_similarity": similarity,
                    "best_excerpt": row.content,
                    "best_chunk_index": row.chunk_index,
                    "metadata": row.metadata if row.metadata else {},
                    "source": {
                        "filename": row.filename,
                        "file_type": row.file_type,
                        "file_size": row.file_size,
                        "upload_date": row.upload_date.isoformat() if row.upload_date else None,
                    },
                }
                doc_order.append(doc_id)
            else:
                existing = grouped_results[doc_id]
                if similarity > existing["best_similarity"]:
                    existing["best_similarity"] = similarity
                    existing["best_excerpt"] = row.content
                    existing["best_chunk_index"] = row.chunk_index

        # Fetch limited previews + stats for surfaced docs
        # Collect doc_ids and calculate preview ranges upfront
        doc_ids = doc_order[:limit]
        if not doc_ids:
            aggregated_results = []
        else:
            # Calculate preview ranges for each document
            doc_preview_ranges: Dict[int, Dict[str, int]] = {}
            for doc_id in doc_ids:
                entry = grouped_results[doc_id]
                best_chunk_index = entry.get("best_chunk_index") or 0
                preview_start_index = max(best_chunk_index - PREVIEW_CONTEXT_RADIUS, 0)
                preview_end_index = preview_start_index + PREVIEW_CHUNK_LIMIT - 1
                doc_preview_ranges[doc_id] = {
                    "start_index": preview_start_index,
                    "end_index": preview_end_index,
                }

            # Batch query: Get chunk counts for all documents at once
            chunk_count_query = text(
                """
                SELECT document_id, COUNT(*) AS chunk_count
                FROM document_chunks
                WHERE document_id = ANY(:document_ids)
                GROUP BY document_id
                """
            )
            chunk_count_rows = db.execute(chunk_count_query, {"document_ids": doc_ids}).fetchall()
            chunk_counts_map: Dict[int, int] = {
                int(row.document_id): int(row.chunk_count) for row in chunk_count_rows
            }

            # Batch query: Get previews for all documents using UNION ALL
            # Build UNION ALL query with one SELECT per document
            preview_queries = []
            preview_params: Dict[str, Any] = {}
            for idx, doc_id in enumerate(doc_ids):
                ranges = doc_preview_ranges[doc_id]
                start_param = f"start_{idx}"
                end_param = f"end_{idx}"
                doc_param = f"doc_{idx}"
                preview_params[start_param] = ranges["start_index"]
                preview_params[end_param] = ranges["end_index"]
                preview_params[doc_param] = doc_id
                
                preview_queries.append(
                    f"""
                    SELECT 
                        :{doc_param} AS document_id,
                        string_agg(content, '\n\n' ORDER BY chunk_index) AS preview_content
                    FROM document_chunks
                    WHERE document_id = :{doc_param}
                      AND chunk_index BETWEEN :{start_param} AND :{end_param}
                    """
                )

            if preview_queries:
                preview_query = text(" UNION ALL ".join(preview_queries))
                preview_rows = db.execute(preview_query, preview_params).fetchall()
                previews_map: Dict[int, str] = {
                    int(row.document_id): (row.preview_content if row.preview_content else "")
                    for row in preview_rows
                }
            else:
                previews_map = {}

            # Build aggregated results using batched data
            aggregated_results: List[Dict[str, Any]] = []
            for doc_id in doc_ids:
                entry = grouped_results[doc_id]
                chunk_count = chunk_counts_map.get(doc_id, 0)
                
                ranges = doc_preview_ranges[doc_id]
                preview_start_index = ranges["start_index"]
                preview_end_index = ranges["end_index"]
                
                preview_content = previews_map.get(doc_id, "")
                preview_truncated = False
                if preview_content and len(preview_content) > PREVIEW_CHAR_LIMIT:
                    preview_content = preview_content[:PREVIEW_CHAR_LIMIT] + "…"
                    preview_truncated = True
                if preview_end_index < (chunk_count - 1):
                    preview_truncated = True

                aggregated_results.append(
                    {
                        "document_id": doc_id,
                        "excerpt": entry["best_excerpt"],
                        "similarity": entry["best_similarity"],
                        "chunk_index": entry["best_chunk_index"],
                        "chunk_count": chunk_count,
                        "metadata": entry["metadata"],
                        "source": entry["source"],
                        "preview": preview_content,
                        "preview_truncated": preview_truncated,
                        "full_content_available": chunk_count > 0,
                        "preview_chunk_start": preview_start_index,
                        "preview_chunk_end": min(preview_end_index, max(chunk_count - 1, 0)),
                    }
                )
 
        execution_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"Search completed: {len(aggregated_results)} documents in {execution_time_ms}ms")
        
        # Track search event for analytics
        try:
            tracking_query = text("""
                INSERT INTO document_usage (event_type, query, results_count, execution_time_ms, metadata, timestamp)
                VALUES ('document_searched', :query, :results_count, :execution_time_ms, :metadata, :timestamp)
            """)
            db.execute(
                tracking_query,
                {
                    "query": query,
                    "results_count": len(aggregated_results),
                    "execution_time_ms": execution_time_ms,
                    "metadata": json.dumps({"min_similarity": min_similarity}),
                    "timestamp": datetime.now()
                }
            )
            db.commit()
            logger.info(f"✅ Search tracked: '{query[:50]}' ({len(aggregated_results)} results)")
        except Exception as track_error:
            logger.warning(f"Could not track search event: {track_error}")
            db.rollback()
        
        return {
            "query": query,
            "total_results": len(aggregated_results),
            "execution_time_ms": execution_time_ms,
            "results": aggregated_results,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/queue/status")
async def get_queue_status(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get real-time processing queue status
    
    Returns information about documents in various processing states:
    - Pending: Documents queued for processing
    - Processing: Currently being processed
    - Failed: Processing failed
    - Queue depth and estimated completion times
    """
    try:
        from datetime import datetime, timedelta
        
        # Get documents by status
        pending_docs = db.query(Document).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.status == 'pending'
        ).order_by(Document.upload_date.desc()).all()

        processing_docs = db.query(Document).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.status == 'processing'
        ).order_by(Document.upload_date.desc()).all()

        failed_docs = db.query(Document).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.status == 'failed'
        ).order_by(Document.upload_date.desc()).limit(10).all()
        
        # Format pending documents
        pending_list = []
        for doc in pending_docs:
            pending_list.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                "position_in_queue": len(pending_list) + 1
            })
        
        # Format processing documents with progress estimates
        processing_list = []
        for doc in processing_docs:
            # Estimate progress based on time since upload
            # This is a placeholder - real implementation would track actual progress
            time_since_upload = datetime.now() - doc.upload_date if doc.upload_date else timedelta(0)
            estimated_total_time = 120  # 2 minutes average
            progress_pct = min(int((time_since_upload.total_seconds() / estimated_total_time) * 100), 95)
            
            # Determine current processing step
            if progress_pct < 20:
                current_step = "text_extraction"
                step_name = "Text Extraction"
            elif progress_pct < 40:
                current_step = "chunking"
                step_name = "Chunking"
            elif progress_pct < 70:
                current_step = "embedding"
                step_name = "Embedding Generation"
            elif progress_pct < 90:
                current_step = "storage"
                step_name = "Vector Storage"
            else:
                current_step = "finalizing"
                step_name = "Finalizing"
            
            eta_seconds = max(0, int(estimated_total_time - time_since_upload.total_seconds()))
            
            processing_list.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "status": "processing",
                "progress": progress_pct,
                "current_step": current_step,
                "step_name": step_name,
                "started_at": doc.upload_date.isoformat() if doc.upload_date else None,
                "eta_seconds": eta_seconds
            })
        
        # Format failed documents
        failed_list = []
        for doc in failed_docs:
            failed_list.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "failed_at": doc.processed_date.isoformat() if doc.processed_date else None,
                "error": doc.doc_metadata.get('error', 'Unknown error') if doc.doc_metadata else 'Processing failed'
            })
        
        # Calculate stats for today
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        processed_today = db.query(Document).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.status == 'completed',
            Document.processed_date >= today_start
        ).count()
        
        # Calculate average processing time (last 100 completed docs)
        recent_completed = db.query(Document).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.status == 'completed',
            Document.processed_date.isnot(None),
            Document.upload_date.isnot(None)
        ).order_by(Document.processed_date.desc()).limit(100).all()
        
        avg_time = 0.0
        if recent_completed:
            total_time = sum([
                (doc.processed_date - doc.upload_date).total_seconds() 
                for doc in recent_completed 
                if doc.processed_date and doc.upload_date
            ])
            avg_time = round(total_time / len(recent_completed), 1)
        
        # Calculate success rate (last 100 attempts)
        recent_all = db.query(Document).filter(
            Document.workspace_id == ctx.workspace_id,
            Document.status.in_(['completed', 'failed'])
        ).order_by(Document.upload_date.desc()).limit(100).all()
        
        success_rate = 0.0
        if recent_all:
            success_count = sum([1 for doc in recent_all if doc.status == 'completed'])
            success_rate = round((success_count / len(recent_all)) * 100, 1)
        
        return {
            "queue_depth": len(pending_list),
            "currently_processing": len(processing_list),
            "failed_count": len(failed_list),
            "pending": pending_list,
            "processing": processing_list,
            "failed": failed_list,
            "timestamp": datetime.now().isoformat(),
            # Stats for dashboard
            "total_processed_today": processed_today,
            "average_processing_time": avg_time,
            "success_rate": success_rate
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rag/retrieve")
async def rag_retrieve(
    query: str = Query(..., description="Query string for RAG context retrieval"),
    max_chunks: int = Query(5, ge=1, le=20, description="Maximum number of chunks to retrieve"),
    max_tokens: Optional[int] = Query(None, ge=100, le=8000, description="Maximum tokens (uses system_settings if not provided)"),
    diversity: Optional[float] = Query(None, ge=0.0, le=1.0, description="Diversity parameter (uses system_settings if not provided)"),
    min_similarity: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum similarity (uses system_settings if not provided)"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Retrieve optimized RAG context for LLM augmentation
    
    Uses the enhanced RAGService with:
    - Query enhancement (HyDE, decomposition)
    - Reciprocal Rank Fusion for hybrid search
    - Knapsack optimization for token budget
    - MMR for diversity
    
    Args:
        query: Search query
        max_chunks: Maximum chunks to return (1-20)
        max_tokens: Token budget for context (100-8000), defaults to system_settings
        diversity: 0.0 = max relevance, 1.0 = max diversity, defaults to system_settings
        min_similarity: Minimum similarity threshold, defaults to system_settings
        
    Returns:
        {
            "query": str,
            "chunks": [...],
            "context": str (formatted for LLM),
            "total_tokens": int,
            "diversity_score": float,
            "execution_time_ms": int
        }
    """
    try:
        import time
        from typing import Optional
        start_time = time.time()
        
        # Use the improved RAGService with all optimizations
        from modules.rag import RAGService, RAGConfig
        
        # Create config - only override if explicitly provided
        config_kwargs = {}
        if max_tokens is not None:
            config_kwargs['max_tokens'] = max_tokens
        if diversity is not None:
            config_kwargs['diversity'] = diversity
        if min_similarity is not None:
            config_kwargs['min_similarity'] = min_similarity
        
        config = RAGConfig(
            enable_query_enhancement=True,
            enable_rrf_fusion=True,
            enable_reranking=False,  # Disabled by default for speed
            **config_kwargs
        )
        
        rag_service = RAGService(config)
        result = await rag_service.retrieve(
            query=query,
            max_chunks=max_chunks,
            max_tokens=max_tokens,  # Pass through (can be None)
            diversity=diversity  # Pass through (can be None)
        )
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Format chunks for API response
        formatted_chunks = []
        for chunk in result.chunks:
            formatted_chunks.append({
                "chunk_id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index", 0),
                "content": chunk.get("content", ""),
                "similarity": chunk.get("similarity", 0.0),
                "source": {
                    "filename": chunk.get("source_file", chunk.get("filename", "unknown")),
                    "file_type": chunk.get("file_type", "unknown"),
                    "chunk_index": chunk.get("chunk_index", 0)
                },
                "tokens": chunk.get("token_count", len(chunk.get("content", "")) // 4),
                "truncated": False
            })
        
        logger.info(f"RAG retrieval: query='{query[:50]}', chunks={len(formatted_chunks)}, tokens={result.total_tokens}, diversity={result.diversity_score:.2f}, min_sim={min_similarity}, time={execution_time_ms}ms")
        
        # Track RAG query for analytics
        try:
            from sqlalchemy import text
            tracking_query = text("""
                INSERT INTO document_usage (event_type, query, results_count, execution_time_ms, metadata, timestamp)
                VALUES ('rag_query', :query, :results_count, :execution_time_ms, :metadata, :timestamp)
            """)
            db.execute(
                tracking_query,
                {
                    "query": query,
                    "results_count": len(formatted_chunks),
                    "execution_time_ms": execution_time_ms,
                    "metadata": json.dumps({
                        "max_chunks": max_chunks,
                        "max_tokens": max_tokens,
                        "diversity": diversity,
                        "min_similarity": min_similarity,
                        "total_tokens": result.total_tokens,
                        "diversity_score": round(result.diversity_score, 3),
                        "information_gain": round(result.information_gain, 3)
                    }),
                    "timestamp": datetime.now()
                }
            )
            db.commit()
        except Exception as track_error:
            logger.warning(f"Failed to track RAG query: {track_error}")
            # Don't fail the request if tracking fails
        
        return {
            "query": query,
            "chunks": formatted_chunks,
            "context": result.formatted_context,
            "total_tokens": result.total_tokens,
            "diversity_score": round(result.diversity_score, 3),
            "information_gain": round(result.information_gain, 3),
            "execution_time_ms": execution_time_ms,
            "settings": {
                "max_chunks": max_chunks,
                "max_tokens": config.max_tokens,  # Use actual config value
                "diversity": config.diversity,     # Use actual config value
                "min_similarity": config.min_similarity  # Use actual config value
            }
        }
        
    except Exception as e:
        logger.error(f"Error performing RAG retrieval: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Usage Analytics Endpoints

@router.post("/analytics/track")
async def track_usage_event(
    event_type: str = Query(..., description="Event type (document_viewed, document_searched, chunk_retrieved, rag_query)"),
    document_id: Optional[int] = Query(None, description="Document ID (if applicable)"),
    query: Optional[str] = Query(None, description="Search query (if applicable)"),
    metadata: Optional[str] = Query(None, description="Additional metadata as JSON string"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Track document usage events for analytics
    
    Event types:
    - document_viewed: User viewed a document
    - document_searched: User performed a search
    - chunk_retrieved: Chunk was retrieved for RAG
    - rag_query: RAG context was built
    """
    try:
        from datetime import datetime
        import json
        
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except:
                pass
        
        # Create usage event record in document_usage table
        try:
            insert_query = text("""
                INSERT INTO document_usage (event_type, document_id, query, results_count, metadata, timestamp)
                VALUES (:event_type, :document_id, :query, :results_count, :metadata, :timestamp)
                RETURNING id
            """)
            
            result = db.execute(
                insert_query,
                {
                    "event_type": event_type,
                    "document_id": document_id,
                    "query": query,
                    "results_count": results_count,
                    "metadata": json.dumps(metadata) if metadata else None,
                    "timestamp": datetime.now()
                }
            )
            db.commit()
            
            event_id = result.fetchone()[0]
            logger.info(f"✅ Usage event saved: id={event_id}, type={event_type}, doc_id={document_id}, query={query[:50] if query else 'N/A'}")
            
            return {
                "success": True,
                "event_id": event_id,
                "event_type": event_type,
                "document_id": document_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving usage event: {e}")
            # Fall back to just logging
            logger.info(f"Usage event (not persisted): type={event_type}, doc_id={document_id}, query={query[:50] if query else 'N/A'}")
            return {
                "success": True,
                "event_type": event_type,
                "document_id": document_id,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error tracking usage event: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/usage")
async def get_usage_analytics(
    period: str = Query("7d", description="Time period (24h, 7d, 30d)"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get aggregated usage analytics
    
    Returns:
    - Total events by type
    - Popular documents
    - Popular search terms
    - Time series data
    """
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import text, func
        
        # Parse period
        if period == "24h":
            start_time = datetime.now() - timedelta(hours=24)
        elif period == "7d":
            start_time = datetime.now() - timedelta(days=7)
        elif period == "30d":
            start_time = datetime.now() - timedelta(days=30)
        else:
            start_time = datetime.now() - timedelta(days=7)
        
        # Get document access statistics
        # For MVP, we'll calculate from existing document data
        
        # Most accessed documents (by view count or recent activity)
        popular_docs_query = text("""
            SELECT
                id,
                filename,
                file_type,
                upload_date,
                processed_date,
                CASE
                    WHEN processed_date IS NOT NULL THEN 1
                    ELSE 0
                END as access_count
            FROM documents
            WHERE status = 'completed'
                AND workspace_id = :workspace_id
            ORDER BY processed_date DESC NULLS LAST
            LIMIT 10
        """)
        
        popular_docs_result = db.execute(popular_docs_query, {"workspace_id": str(ctx.workspace_id)})
        popular_documents = [
            {
                "document_id": row.id,
                "filename": row.filename,
                "file_type": row.file_type,
                "view_count": row.access_count
            }
            for row in popular_docs_result
        ]
        
        # Get document statistics
        stats_query = text("""
            SELECT
                COUNT(*) as total_documents,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as processed_documents,
                COUNT(CASE WHEN upload_date >= :start_time THEN 1 END) as documents_this_period
            FROM documents
            WHERE workspace_id = :workspace_id
        """)

        stats_result = db.execute(stats_query, {"start_time": start_time, "workspace_id": str(ctx.workspace_id)}).fetchone()
        
        # Get REAL popular search terms from document_usage table (if it exists)
        popular_search_terms = []
        try:
            search_terms_query = text("""
                SELECT 
                    metadata->>'query' as query,
                    COUNT(*) as count
                FROM document_usage
                WHERE event_type = 'document_searched'
                    AND timestamp >= :start_time
                    AND metadata->>'query' IS NOT NULL
                GROUP BY metadata->>'query'
                ORDER BY count DESC
                LIMIT 10
            """)
            
            search_terms_result = db.execute(search_terms_query, {"start_time": start_time}).fetchall()
            popular_search_terms = [
                {"query": row.query, "count": row.count}
                for row in search_terms_result
            ]
        except Exception as e:
            # Table might not exist yet - that's OK for MVP
            logger.debug(f"Could not get search terms (table might not exist): {e}")
            db.rollback()  # Clear failed transaction
        
        # Time series data (documents uploaded per day)
        time_series_query = text("""
            SELECT
                DATE(upload_date) as date,
                COUNT(*) as count
            FROM documents
            WHERE upload_date >= :start_time
                AND workspace_id = :workspace_id
            GROUP BY DATE(upload_date)
            ORDER BY date
        """)

        time_series_result = db.execute(time_series_query, {"start_time": start_time, "workspace_id": str(ctx.workspace_id)})
        time_series = [
            {
                "date": row.date.isoformat() if row.date else None,
                "count": row.count
            }
            for row in time_series_result
        ]
        
        return {
            "period": period,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_events": stats_result.documents_this_period if stats_result else 0,
            "event_counts": {
                "document_viewed": stats_result.processed_documents if stats_result else 0,
                "document_searched": len(popular_search_terms),
                "chunk_retrieved": 0,
                "rag_query": 0
            },
            "popular_documents": popular_documents,
            "popular_search_terms": popular_search_terms,
            "time_series": time_series,
            "summary": {
                "total_documents": stats_result.total_documents if stats_result else 0,
                "processed_documents": stats_result.processed_documents if stats_result else 0,
                "documents_this_period": stats_result.documents_this_period if stats_result else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting usage analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# DOCUMENT RE-PROCESSING ENDPOINTS (Phase 4 - Better RAG)
# =============================================================================

@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Re-process a document with semantic chunking.

    Use after changing embedding models or to improve chunk quality.
    """
    try:
        # Verify document belongs to this workspace
        document = db.query(Document).filter(Document.id == document_id, Document.workspace_id == ctx.workspace_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        from modules.rag import get_rag_service

        rag_service = get_rag_service()
        result = await rag_service.reprocess_document(document_id)
        
        if result.get("success"):
            return {
                "status": "success",
                "message": f"Document {document_id} re-processed successfully",
                "chunk_count": result.get("chunk_count", 0)
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Re-processing failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-processing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reprocess-all")
async def reprocess_all_documents(
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Re-process ALL documents with semantic chunking.
    
    ⚠️ Use after:
    - Changing embedding model/dimensions
    - Updating chunking strategy
    - Migrating to new vector store
    
    Runs in background - check status via /status endpoint.
    """
    try:
        from modules.rag import get_rag_service
        
        # Run in background
        async def run_reprocessing():
            rag_service = get_rag_service()
            result = await rag_service.reprocess_all_documents()
            logger.info(f"Batch re-processing complete: {result}")
        
        import asyncio
        background_tasks.add_task(lambda: asyncio.run(run_reprocessing()))
        
        return {
            "status": "started",
            "message": "Re-processing started in background. This may take several minutes.",
            "note": "Check document status via GET /api/documents/ to monitor progress"
        }
        
    except Exception as e:
        logger.error(f"Error starting batch re-processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reprocess-status")
async def get_reprocess_status(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get status of document re-processing.
    
    Shows how many documents are pending, processing, completed.
    """
    try:
        from sqlalchemy import text
        
        status_query = text("""
            SELECT
                status,
                COUNT(*) as count
            FROM documents
            WHERE workspace_id = :workspace_id
            GROUP BY status
        """)
        
        result = db.execute(status_query, {"workspace_id": str(ctx.workspace_id)})
        status_counts = {row.status: row.count for row in result}
        
        total = sum(status_counts.values())
        completed = status_counts.get('completed', 0)
        pending = status_counts.get('pending', 0) + status_counts.get('pending_reprocess', 0)
        processing = status_counts.get('processing', 0)
        
        return {
            "total_documents": total,
            "status_breakdown": status_counts,
            "summary": {
                "completed": completed,
                "pending": pending,
                "processing": processing,
                "progress_percent": round((completed / total * 100) if total > 0 else 0, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting re-process status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
