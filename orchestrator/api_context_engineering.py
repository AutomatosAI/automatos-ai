
"""
Context Engineering API Endpoints
=================================

Real RAG monitoring and configuration endpoints with David Kimaii's context engineering system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from database import get_db
from models import RAGConfiguration, Document
from rag_service import get_rag_service, RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/context", tags=["Context Engineering"])

@router.get("/stats")
async def get_context_stats(
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Get real-time context engineering statistics"""
    try:
        # Get retrieval stats from RAG service
        retrieval_stats = rag_service.get_retrieval_stats()
        
        # Get document count for vector embeddings estimate
        document_count = db.query(Document).count()
        total_chunks = db.query(Document).with_entities(
            db.func.sum(Document.chunk_count)
        ).scalar() or 0
        
        return {
            "contextQueries": retrieval_stats['total_queries'],
            "retrievalSuccess": retrieval_stats['success_rate'],
            "avgResponseTime": retrieval_stats['avg_response_time'],
            "vectorEmbeddings": int(total_chunks),
            "systemStatus": retrieval_stats['system_status'],
            "lastQueryTime": retrieval_stats['last_query_time']
        }
        
    except Exception as e:
        logger.error(f"Error getting context stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting context stats: {str(e)}")

@router.get("/performance")
async def get_rag_performance_data(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get RAG performance data for charts"""
    try:
        performance_data = rag_service.get_performance_data()
        
        # Ensure we have data for the last 24 hours
        if not performance_data:
            # Generate default data structure
            performance_data = []
            for hour in range(0, 24, 4):
                performance_data.append({
                    'time': f"{hour:02d}:00",
                    'queries': 0,
                    'success_rate': 0.0,
                    'avg_latency': 0.0
                })
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting performance data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance data: {str(e)}")

@router.get("/sources")
async def get_context_sources(
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Get context sources distribution"""
    try:
        return rag_service.get_context_sources(db)
        
    except Exception as e:
        logger.error(f"Error getting context sources: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting context sources: {str(e)}")

@router.get("/queries/recent")
async def get_recent_queries(
    limit: int = Query(default=10, ge=1, le=50),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get recent context queries"""
    try:
        return rag_service.get_recent_queries(limit)
        
    except Exception as e:
        logger.error(f"Error getting recent queries: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recent queries: {str(e)}")

@router.get("/patterns")
async def get_context_patterns(
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Get context patterns based on RAG configurations"""
    try:
        return await rag_service.get_context_patterns(db)
        
    except Exception as e:
        logger.error(f"Error getting context patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting context patterns: {str(e)}")

@router.post("/rag/{config_id}/test")
async def test_rag_configuration(
    config_id: int,
    query: str = Query(..., description="Test query for RAG system"),
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Test RAG configuration with real retrieval"""
    try:
        result = await rag_service.test_rag_config(config_id, query, db)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error testing RAG config {config_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing RAG config: {str(e)}")

@router.get("/system/health")
async def get_context_system_health(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get context engineering system health"""
    try:
        stats = rag_service.get_retrieval_stats()
        
        return {
            "status": "healthy" if stats['system_status'] == 'operational' else "degraded",
            "components": {
                "vector_store": "healthy" if rag_service.context_system else "not_initialized",
                "embedding_generator": "healthy" if rag_service.context_system else "not_initialized",
                "context_retriever": "healthy" if rag_service.context_system else "not_initialized",
                "learning_engine": "healthy" if rag_service.context_system else "not_initialized"
            },
            "metrics": {
                "total_queries": stats['total_queries'],
                "success_rate": stats['success_rate'],
                "avg_response_time": stats['avg_response_time'],
                "last_query": stats['last_query_time']
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")

@router.post("/initialize")
async def initialize_context_system(
    database_url: Optional[str] = None,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Initialize or reinitialize the context engineering system"""
    try:
        success = await rag_service.initialize(database_url)
        
        if success:
            return {
                "status": "success",
                "message": "Context engineering system initialized successfully",
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize context system")
            
    except Exception as e:
        logger.error(f"Error initializing context system: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing context system: {str(e)}")
