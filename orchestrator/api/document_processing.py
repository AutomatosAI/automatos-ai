"""
Real Document Processing Pipeline
=================================

Actual document processing with vector embeddings for RAG.
NO MOCK DATA - Real implementation or 501 errors.
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime

from database.database import get_db
from models import Document as DocumentModel
from utils.document_manager import DocumentManager
from context_engineering.manager import EnhancedContextManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["📄 Document Processing"])

# Initialize real services
import os
from config import Config

# Initialize configuration
config = Config()

# Database configuration
db_config = {
    "database": config.POSTGRES_DB,
    "user": config.POSTGRES_USER,
    "password": config.POSTGRES_PASSWORD,
    "host": config.POSTGRES_HOST,
    "port": config.POSTGRES_PORT
}
openai_api_key = config.OPENAI_API_KEY

# Initialize real services
document_manager = DocumentManager(db_config, openai_api_key)
context_manager = None  # Will initialize on first use

async def get_context_manager():
    """Get or initialize context manager"""
    global context_manager
    if not context_manager:
        context_manager = EnhancedContextManager()
        # Check if we have OpenAI key from credential resolver
        from services.credential_resolver import get_credential_resolver
        resolver = get_credential_resolver()
        openai_key = resolver.get_credential_field("development_openai", "api_key")
        
        if not openai_key:
            raise HTTPException(
                status_code=501,
                detail="OpenAI API key not configured. Cannot process documents without embeddings."
            )
        await context_manager.initialize()
    return context_manager

@router.post("/{document_id}/process")
async def process_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process document for RAG - REAL IMPLEMENTATION
    
    This will:
    1. Extract text from document
    2. Chunk the text
    3. Generate embeddings
    4. Store in vector database
    """
    try:
        # Get document from database
        doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if already processed
        if doc.status == "processed":
            return {
                "document_id": document_id,
                "status": "already_processed",
                "message": "Document has already been processed"
            }
        
        # Update status to processing
        doc.status = "processing"
        db.commit()
        
        # Check if we have the actual file content
        if not doc.file_path:
            # We don't have the file stored yet
            doc.status = "failed"
            db.commit()
            raise HTTPException(
                status_code=501,
                detail="File storage not implemented. Cannot process document without file content."
            )
        
        # Add to background processing
        background_tasks.add_task(
            process_document_async,
            document_id,
            doc.file_path
        )
        
        return {
            "document_id": document_id,
            "status": "processing_started",
            "message": "Document processing initiated in background"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_document_async(document_id: int, file_path: str):
    """
    Background task to process document
    REAL IMPLEMENTATION - No mock data
    """
    from database.database import SessionLocal
    
    db = SessionLocal()
    try:
        doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not doc:
            return
        
        # Get context manager
        ctx_manager = await get_context_manager()
        
        # Process with document manager
        chunks = await document_manager.process_document(file_path)
        
        if not chunks:
            doc.status = "failed"
            db.commit()
            return
        
        # Store chunks count
        doc.chunk_count = len(chunks)
        
        # Add to vector store
        await ctx_manager.add_documents(chunks)
        
        # Update status
        doc.status = "processed"
        doc.processed_date = datetime.utcnow()
        db.commit()
        
        logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        doc.status = "failed"
        db.commit()
    finally:
        db.close()

@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get processed chunks for a document
    Returns real data or appropriate error
    """
    try:
        doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if doc.status != "processed":
            return {
                "document_id": document_id,
                "status": doc.status,
                "message": "Document not yet processed",
                "chunks": []
            }
        
        # TODO: Retrieve actual chunks from vector store
        # For now, return honest response
        raise HTTPException(
            status_code=501,
            detail="Chunk retrieval not yet implemented. Document is processed but chunks not accessible via API."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/search")
async def search_document(
    document_id: int,
    query: str,
    db: Session = Depends(get_db)
):
    """
    Search within a specific document
    Real vector search or honest error
    """
    try:
        doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if doc.status != "processed":
            raise HTTPException(
                status_code=400,
                detail=f"Document not processed. Current status: {doc.status}"
            )
        
        # Get context manager
        ctx_manager = await get_context_manager()
        
        # Perform real search
        results = await ctx_manager.search(
            query,
            filter={"document_id": document_id}
        )
        
        if not results:
            return {
                "document_id": document_id,
                "query": query,
                "results": [],
                "message": "No relevant content found"
            }
        
        return {
            "document_id": document_id,
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
