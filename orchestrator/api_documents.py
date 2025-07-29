
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

from database import get_db
from models import Document, DocumentUploadResponse, DocumentResponse
from document_manager import DocumentManager, DocumentStatus, DocumentType
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

# Initialize document manager - temporarily disabled due to pgvector dependency
import os
# db_config = {
#     "host": "localhost",
#     "port": 5432,
#     "database": "orchestrator_db",
#     "user": "postgres",
#     "password": "secure_password_123"
# }
# openai_api_key = os.getenv("OPENAI_API_KEY", "***REMOVED***")
# doc_manager = DocumentManager(db_config, openai_api_key)
doc_manager = None  # Temporarily disabled

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    if doc_manager is None:
        raise HTTPException(status_code=503, detail="Document management service temporarily unavailable")
    
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
        document = Document(
            filename=file.filename,
            original_filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            file_path=str(file_path),
            content_hash=content_hash,
            status="uploaded",
            tags=tag_list,
            description=description,
            created_by="system"  # TODO: Get from auth context
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Process document asynchronously
        try:
            # Use existing document manager for processing
            result = await doc_manager.upload_document(
                file_path=str(file_path),
                filename=file.filename,
                file_type=file_type,
                description=description or "",
                tags=tag_list,
                created_by="system"
            )
            
            # Update document with processing results
            document.status = "processed"
            document.chunk_count = result.get("chunk_count", 0)
            db.commit()
            
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
    if doc_manager is None:
        return []  # Return empty list when service is unavailable
    
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
    """Reprocess a document"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.file_path or not os.path.exists(document.file_path):
            raise HTTPException(status_code=400, detail="Document file not found")
        
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
            
            return {"message": "Document reprocessed successfully"}
            
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
        
        # Get document chunks from the document manager
        chunks = await doc_manager.get_document_chunks(document_id)
        
        return {
            "document_id": document_id,
            "filename": document.filename,
            "chunk_count": len(chunks),
            "chunks": chunks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document content {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document content: {str(e)}")
