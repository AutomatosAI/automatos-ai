
"""
Document Management API Routes
=============================

Enhanced REST API endpoints for document upload, processing, and management.
"""

import os
import hashlib
import tempfile
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from database.database import get_db
from models import Document, DocumentUploadResponse, DocumentResponse
from utils.document_manager import DocumentManager, DocumentStatus, DocumentType
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

# Initialize document manager
db_config = {
    "database": os.getenv("POSTGRES_DB", "orchestrator_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "secure_password_123"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}
openai_api_key = os.getenv("OPENAI_API_KEY", "demo_key")
doc_manager = DocumentManager(db_config, openai_api_key)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
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
        
        # Reset file pointer
        await file.seek(0)
        
        # Generate file hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Check for duplicate
        existing = db.query(Document).filter(Document.content_hash == content_hash).first()
        if existing:
            return DocumentUploadResponse(
                document_id=existing.id,
                filename=existing.filename,
                status="duplicate",
                message="Document already exists"
            )
        
        # Save file temporarily
        upload_dir = Path("/tmp/automotas_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{content_hash}_{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Determine file type
        file_extension = Path(file.filename).suffix.lower()
        file_type = "unknown"
        if file_extension in ['.pdf']:
            file_type = "pdf"
        elif file_extension in ['.txt', '.md']:
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
            from utils.document_manager import DocumentManager, DocumentType
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
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.get("/analytics")
async def get_document_analytics(db: Session = Depends(get_db)):
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
        total_docs = db.query(func.count(Document.id)).scalar() or 0
        
        # Documents by status
        status_counts = db.query(
            Document.status, 
            func.count(Document.id)
        ).group_by(Document.status).all()
        
        status_distribution = {status: count for status, count in status_counts}
        
        # Total storage used
        total_storage = db.query(func.sum(Document.file_size)).scalar() or 0
        
        # File type distribution
        file_type_counts = db.query(
            Document.file_type,
            func.count(Document.id)
        ).group_by(Document.file_type).all()
        
        file_types = {file_type: count for file_type, count in file_type_counts}
        
        # Total chunks processed
        total_chunks = db.query(func.sum(Document.chunk_count)).scalar() or 0
        
        # Recent uploads (last 24 hours)
        from datetime import datetime, timedelta
        recent_cutoff = datetime.utcnow() - timedelta(days=1)
        recent_uploads = db.query(func.count(Document.id)).filter(
            Document.upload_date >= recent_cutoff
        ).scalar() or 0
        
        # Average chunk count
        avg_chunks = db.query(func.avg(Document.chunk_count)).filter(
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
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List documents with filtering and pagination"""
    try:
        query = db.query(Document)
        
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
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get document by ID"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
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
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete document"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
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
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.post("/{document_id}/reprocess")
async def reprocess_document(document_id: int, db: Session = Depends(get_db)):
    """
    Reprocess a document - regenerate chunks and embeddings
    
    Note: This endpoint requires the original document file to still exist.
    For test documents with fake file paths, this will return an appropriate error.
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
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
            raise HTTPException(status_code=500, detail=f"Error reprocessing document: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reprocessing document: {str(e)}")

@router.get("/{document_id}/content")
async def get_document_content(document_id: int, db: Session = Depends(get_db)):
    """Get document content/chunks"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
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
        raise HTTPException(status_code=500, detail=f"Error getting document content: {str(e)}")
