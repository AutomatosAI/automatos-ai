
"""
MCP Server Main Application
==========================

FastAPI-based MCP server that exposes all orchestration functionality
for headless operation with DeepAgent, Cursor, and other IDEs.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator import EnhancedOrchestrator
from src.utils.document_manager import DocumentManager
from enhanced_context_manager import EnhancedContextManager
from ssh_manager import EnhancedSSHManager, SecurityLevel
from security import get_audit_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
orchestrator = EnhancedOrchestrator()
doc_manager = DocumentManager()
context_manager = EnhancedContextManager()
ssh_manager = EnhancedSSHManager()
audit_logger = get_audit_logger()

# Security
security = HTTPBearer()

app = FastAPI(
    title="Multi-Agent Orchestration MCP Server",
    description="Headless MCP server for IDE integration with multi-agent orchestration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class WorkflowRequest(BaseModel):
    repository_url: str
    task_prompt: Optional[str] = None
    security_level: str = "medium"
    ssh_config: Optional[Dict[str, Any]] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    message: str
    created_at: str

class DocumentUploadRequest(BaseModel):
    filename: str
    content: str
    content_type: str = "text/plain"

class ContextSearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None

class SSHCommandRequest(BaseModel):
    command: str
    security_level: str = "medium"
    timeout: int = 30

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("MCP_API_TOKEN", "default-token")
    
    if token != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return token

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "components": {
            "orchestrator": "active",
            "document_manager": "active",
            "context_manager": "active",
            "ssh_manager": "active"
        }
    }

# Workflow management endpoints
@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Create and start a new workflow"""
    try:
        # Create workflow
        workflow_id = await orchestrator.create_workflow(
            repository_url=request.repository_url,
            task_prompt=request.task_prompt,
            security_level=SecurityLevel(request.security_level),
            ssh_config=request.ssh_config
        )
        
        # Start workflow in background
        background_tasks.add_task(
            orchestrator.execute_workflow,
            workflow_id
        )
        
        audit_logger.info(f"Workflow {workflow_id} created via MCP server")
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status="started",
            message="Workflow created and started successfully",
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows")
async def list_workflows(token: str = Depends(verify_token)):
    """List all workflows"""
    try:
        workflows = await orchestrator.list_workflows()
        return {"workflows": workflows}
    except Exception as e:
        logger.error(f"Failed to list workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{workflow_id}")
async def get_workflow_status(
    workflow_id: str,
    token: str = Depends(verify_token)
):
    """Get workflow status and details"""
    try:
        status = await orchestrator.get_workflow_status(workflow_id)
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/workflows/{workflow_id}")
async def stop_workflow(
    workflow_id: str,
    token: str = Depends(verify_token)
):
    """Stop a running workflow"""
    try:
        success = await orchestrator.stop_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        audit_logger.info(f"Workflow {workflow_id} stopped via MCP server")
        return {"message": "Workflow stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{workflow_id}/logs")
async def get_workflow_logs(
    workflow_id: str,
    token: str = Depends(verify_token)
):
    """Get workflow execution logs"""
    try:
        logs = await orchestrator.get_workflow_logs(workflow_id)
        if logs is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"logs": logs}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.post("/documents")
async def upload_document(
    request: DocumentUploadRequest,
    token: str = Depends(verify_token)
):
    """Upload a document"""
    try:
        doc_id = await doc_manager.upload_document(
            filename=request.filename,
            content=request.content.encode(),
            content_type=request.content_type
        )
        
        audit_logger.info(f"Document {request.filename} uploaded via MCP server")
        return {"document_id": doc_id, "message": "Document uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to upload document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(token: str = Depends(verify_token)):
    """List all documents"""
    try:
        documents = await doc_manager.list_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    token: str = Depends(verify_token)
):
    """Get document content"""
    try:
        document = await doc_manager.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    token: str = Depends(verify_token)
):
    """Delete a document"""
    try:
        success = await doc_manager.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        audit_logger.info(f"Document {doc_id} deleted via MCP server")
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Context engineering endpoints
@app.post("/context/search")
async def search_context(
    request: ContextSearchRequest,
    token: str = Depends(verify_token)
):
    """Search context database"""
    try:
        results = await context_manager.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to search context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/stats")
async def get_context_stats(token: str = Depends(verify_token)):
    """Get context database statistics"""
    try:
        stats = await context_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get context stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/configure")
async def configure_context(
    config: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """Configure context manager settings"""
    try:
        await context_manager.configure(config)
        audit_logger.info("Context manager configured via MCP server")
        return {"message": "Context manager configured successfully"}
    except Exception as e:
        logger.error(f"Failed to configure context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# SSH command execution endpoints
@app.post("/ssh/execute")
async def execute_ssh_command(
    request: SSHCommandRequest,
    token: str = Depends(verify_token)
):
    """Execute SSH command"""
    try:
        result = await ssh_manager.execute_command(
            command=request.command,
            security_level=SecurityLevel(request.security_level),
            timeout=request.timeout
        )
        
        audit_logger.info(f"SSH command executed via MCP server: {request.command}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to execute SSH command: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ssh/status")
async def get_ssh_status(token: str = Depends(verify_token)):
    """Get SSH connection status"""
    try:
        status = await ssh_manager.get_connection_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get SSH status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Progress tracking endpoints
@app.get("/progress/{workflow_id}")
async def get_workflow_progress(
    workflow_id: str,
    token: str = Depends(verify_token)
):
    """Get real-time workflow progress"""
    try:
        progress = await orchestrator.get_workflow_progress(workflow_id)
        if progress is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/{workflow_id}/stream")
async def stream_workflow_progress(
    workflow_id: str,
    token: str = Depends(verify_token)
):
    """Stream real-time workflow progress"""
    async def generate_progress():
        try:
            async for progress_update in orchestrator.stream_workflow_progress(workflow_id):
                yield f"data: {json.dumps(progress_update)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# System information endpoints
@app.get("/system/info")
async def get_system_info(token: str = Depends(verify_token)):
    """Get system information"""
    try:
        info = {
            "version": "2.0.0",
            "uptime": datetime.utcnow().isoformat(),
            "components": {
                "orchestrator": await orchestrator.get_status(),
                "document_manager": await doc_manager.get_status(),
                "context_manager": await context_manager.get_status(),
                "ssh_manager": await ssh_manager.get_status()
            }
        }
        return info
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/metrics")
async def get_system_metrics(token: str = Depends(verify_token)):
    """Get system performance metrics"""
    try:
        metrics = await orchestrator.get_system_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8002))
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting MCP Server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
