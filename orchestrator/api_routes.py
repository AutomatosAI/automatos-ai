
"""
FastAPI Routes for Document Management
=====================================

This module provides REST API endpoints for the context engineering system:
- Document upload and management
- Context retrieval and search
- Admin operations and analytics
"""

import asyncio
import json
import logging
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from document_manager import DocumentManager, DocumentStatus, DocumentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    message: str

class DocumentInfo(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    processed_date: Optional[datetime]
    status: str
    chunk_count: int
    tags: List[str]
    description: str
    created_by: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int
    page: int
    per_page: int

class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)

class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    content: str
    similarity: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int

class DocumentStats(BaseModel):
    total_documents: int
    status_distribution: Dict[str, int]
    total_chunks: int
    file_type_distribution: Dict[str, int]

class ContextConfig(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=50)

# Global document manager instance
doc_manager: Optional[DocumentManager] = None

def get_document_manager() -> DocumentManager:
    """Dependency to get document manager instance"""
    if doc_manager is None:
        raise HTTPException(status_code=500, detail="Document manager not initialized")
    return doc_manager

# Initialize FastAPI app
app = FastAPI(
    title="Context Engineering API",
    description="Document management and context retrieval for AI orchestration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # DevOps UI origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize document manager on startup"""
    global doc_manager
    
    # Database configuration (should come from environment variables)
    db_config = {
        'host': 'localhost',
        'database': 'orchestrator_db',
        'user': 'postgres',
        'password': 'postgres'  # Change this to your actual password
    }
    
    # OpenAI API key (should come from environment variables)
    openai_api_key = "your_openai_api_key"  # Change this to your actual API key
    
    try:
        doc_manager = DocumentManager(db_config, openai_api_key)
        logger.info("Document manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document manager: {e}")
        raise

# Document Management Endpoints

@app.post("/api/admin/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    tags: str = Form(""),
    description: str = Form(""),
    created_by: str = Form("admin"),
    manager: DocumentManager = Depends(get_document_manager)
):
    """Upload a new document for processing"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.md', '.txt', '.py', '.json'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Upload and process document
            document_id = await manager.upload_document(
                file_path=temp_file_path,
                filename=file.filename,
                tags=tag_list,
                description=description,
                created_by=created_by
            )
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                status="processing",
                message="Document uploaded successfully and is being processed"
            )
            
        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/documents", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    manager: DocumentManager = Depends(get_document_manager)
):
    """List documents with optional filtering and pagination"""
    try:
        # Convert string parameters to enums if provided
        status_filter = DocumentStatus(status) if status else None
        file_type_filter = DocumentType(file_type) if file_type else None
        
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Get documents
        documents = manager.list_documents(
            status=status_filter,
            file_type=file_type_filter,
            limit=per_page,
            offset=offset
        )
        
        # Convert to response format
        document_infos = []
        for doc in documents:
            document_infos.append(DocumentInfo(
                id=doc['id'],
                filename=doc['filename'],
                file_type=doc['file_type'],
                file_size=doc['file_size'],
                upload_date=doc['upload_date'],
                processed_date=doc['processed_date'],
                status=doc['status'],
                chunk_count=doc['chunk_count'],
                tags=doc['tags'] or [],
                description=doc['description'] or "",
                created_by=doc['created_by']
            ))
        
        return DocumentListResponse(
            documents=document_infos,
            total=len(document_infos),  # This should be actual total count in production
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/documents/{document_id}", response_model=DocumentInfo)
async def get_document(
    document_id: int,
    manager: DocumentManager = Depends(get_document_manager)
):
    """Get document details by ID"""
    try:
        document = manager.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfo(
            id=document['id'],
            filename=document['filename'],
            file_type=document['file_type'],
            file_size=document['file_size'],
            upload_date=document['upload_date'],
            processed_date=document['processed_date'],
            status=document['status'],
            chunk_count=document['chunk_count'],
            tags=document['tags'] or [],
            description=document['description'] or "",
            created_by=document['created_by']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/admin/documents/{document_id}")
async def delete_document(
    document_id: int,
    manager: DocumentManager = Depends(get_document_manager)
):
    """Delete a document and all its chunks"""
    try:
        success = manager.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Context Retrieval Endpoints

@app.post("/api/context/search", response_model=SearchResponse)
async def search_context(
    request: SearchRequest,
    manager: DocumentManager = Depends(get_document_manager)
):
    """Search for relevant context based on query"""
    try:
        results = manager.search_documents(request.query, request.limit)
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result['chunk_id'],
                document_id=result['document_id'],
                filename=result['filename'],
                content=result['content'],
                similarity=result['similarity'],
                metadata=result['metadata']
            ))
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error searching context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/context/retrieve/{document_id}")
async def retrieve_document_context(
    document_id: int,
    query: Optional[str] = Query(None, description="Optional query for filtering chunks"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of chunks to return"),
    manager: DocumentManager = Depends(get_document_manager)
):
    """Retrieve context from a specific document"""
    try:
        # Verify document exists
        document = manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if query:
            # Search within the specific document
            all_results = manager.search_documents(query, limit * 2)  # Get more to filter
            results = [r for r in all_results if r['document_id'] == document_id][:limit]
        else:
            # Get all chunks from the document (simplified - would need separate method)
            results = []
        
        return {
            "document_id": document_id,
            "filename": document['filename'],
            "chunks": results,
            "total_chunks": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving context for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and Configuration Endpoints

@app.get("/api/admin/stats", response_model=DocumentStats)
async def get_document_stats(
    manager: DocumentManager = Depends(get_document_manager)
):
    """Get document and processing statistics"""
    try:
        stats = manager.get_document_stats()
        
        return DocumentStats(
            total_documents=stats['total_documents'],
            status_distribution=stats['status_distribution'],
            total_chunks=stats['total_chunks'],
            file_type_distribution=stats['file_type_distribution']
        )
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/config", response_model=ContextConfig)
async def get_context_config():
    """Get current context configuration"""
    # This would typically come from a database or config file
    return ContextConfig()

@app.post("/api/admin/config")
async def update_context_config(config: ContextConfig):
    """Update context configuration"""
    try:
        # In a real implementation, this would update the configuration
        # in the database and potentially restart processing components
        
        return {
            "message": "Configuration updated successfully",
            "config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "context-engineering-api"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "api_routes:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
