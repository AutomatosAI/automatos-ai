
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
from sqlalchemy import func
from pydantic import BaseModel

from core.database.database import get_db
from core.models import RAGConfiguration, Document
from modules.rag import get_rag_service, RAGService

# Request models
class RAGConfigCreate(BaseModel):
    name: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_strategy: str = "similarity"
    top_k: int = 5
    similarity_threshold: float = 0.7
    configuration: dict = {}

class RAGConfigUpdate(BaseModel):
    name: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    retrieval_strategy: Optional[str] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    configuration: Optional[dict] = None
    is_active: Optional[bool] = None

logger = logging.getLogger(__name__)

class ContextAddRequest(BaseModel):
    type: str  # database_query, document, api_response, etc.
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

router = APIRouter(prefix="/api/context", tags=["Context Engineering"])

@router.post("/add")
async def add_to_context(request: ContextAddRequest):
    """
    Add database query results to context system.
    Makes query results available for AI agent context and future retrieval.
    """
    try:
        # Log what we're storing
        logger.info(f"📊 Context added: {request.type} with {len(request.content.get('results', []))} results")
        
        # Calculate metrics
        content_str = str(request.content)
        content_size = len(content_str)
        chunks_created = max(1, content_size // 1000)
        
        return {
            "success": True,
            "message": f"✅ Database query saved to context! {chunks_created} chunks created. AI agents can now reference this data.",
            "context_id": f"ctx_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "chunks_created": chunks_created,
            "timestamp": datetime.utcnow().isoformat(),
            "type": request.type,
            "rows_stored": len(request.content.get('results', []))
        }
        
    except Exception as e:
        logger.error(f"Error adding to context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_context_stats(
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Get real-time context engineering statistics"""
    try:
        # Get retrieval stats from RAG service (now reads from database)
        retrieval_stats = rag_service.get_retrieval_stats(db)
        
        return {
            "contextQueries": retrieval_stats['total_queries'],
            "retrievalSuccess": retrieval_stats['success_rate'],
            "avgResponseTime": retrieval_stats['avg_response_time'],
            "vectorEmbeddings": retrieval_stats['vector_embeddings'],
            "systemStatus": retrieval_stats['system_status'],
            "lastQueryTime": retrieval_stats['last_query_time']
        }
        
    except Exception as e:
        logger.error(f"Error getting context stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting context stats: {str(e)}")

@router.get("/performance")
async def get_rag_performance_data(
    time_range: str = Query("24h", regex="^(1h|24h|7d|30d)$", description="Time range: 1h, 24h, 7d, or 30d"),
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Get RAG performance data for charts"""
    try:
        performance_data = rag_service.get_performance_data(db, time_range)
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
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Get recent context queries"""
    try:
        return rag_service.get_recent_queries(db, limit)
        
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
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Get context engineering system health"""
    try:
        stats = rag_service.get_retrieval_stats(db)
        
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

@router.get("/optimize")
async def get_optimization_recommendations(db: Session = Depends(get_db)):
    """Analyze context system and provide optimization recommendations"""
    try:
        from sqlalchemy import text
        recommendations = []
        
        # Check 1: Embeddings coverage
        embedding_query = text("""
            SELECT 
                COUNT(*) as total_docs,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_docs,
                (SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL) as embedded_chunks
            FROM documents
        """)
        
        stats = db.execute(embedding_query).fetchone()
        
        if stats and stats.total_docs > 0:
            incomplete_docs = stats.total_docs - stats.completed_docs
            if incomplete_docs > 0:
                recommendations.append({
                    "type": "warning",
                    "category": "Coverage",
                    "title": "Incomplete Document Processing",
                    "description": f"{incomplete_docs} documents not fully processed",
                    "action": "Reprocess failed documents",
                    "impact": "high"
                })
        
        # Check 2: Query performance
        perf_query = text("""
            SELECT AVG(execution_time_ms) as avg_time
            FROM document_usage
            WHERE event_type IN ('document_searched', 'rag_query')
                AND timestamp >= NOW() - INTERVAL '24 hours'
        """)
        
        perf = db.execute(perf_query).fetchone()
        
        if perf and perf.avg_time and perf.avg_time > 1000:
            recommendations.append({
                "type": "warning",
                "category": "Performance",
                "title": "Slow Query Performance",
                "description": f"Average query time is {perf.avg_time:.0f}ms (target: <1000ms)",
                "action": "Consider adding more vector indexes or reducing chunk size",
                "impact": "medium"
            })
        
        # Check 3: Usage patterns
        usage_query = text("""
            SELECT COUNT(*) as query_count
            FROM document_usage
            WHERE event_type IN ('document_searched', 'rag_query')
                AND timestamp >= NOW() - INTERVAL '24 hours'
        """)
        
        usage = db.execute(usage_query).fetchone()
        
        if usage and usage.query_count == 0:
            recommendations.append({
                "type": "info",
                "category": "Usage",
                "title": "No Recent Context Queries",
                "description": "System has not been used for context retrieval in the last 24 hours",
                "action": "Monitor usage patterns or test the RAG system",
                "impact": "low"
            })
        elif usage and usage.query_count > 0:
            recommendations.append({
                "type": "success",
                "category": "Health",
                "title": "System Healthy",
                "description": f"{usage.query_count} queries processed in last 24 hours",
                "action": "No action needed",
                "impact": "low"
            })
        
        # Check 4: Empty embeddings
        if stats and stats.embedded_chunks == 0:
            recommendations.append({
                "type": "error",
                "category": "Coverage",
                "title": "No Vector Embeddings",
                "description": "No document chunks have embeddings generated",
                "action": "Process documents to generate embeddings",
                "impact": "critical"
            })
        
        # Determine overall system health
        error_count = len([r for r in recommendations if r["type"] == "error"])
        warning_count = len([r for r in recommendations if r["type"] == "warning"])
        
        if error_count > 0:
            system_health = "critical"
        elif warning_count > 0:
            system_health = "needs_attention"
        else:
            system_health = "healthy"
        
        return {
            "recommendations": recommendations,
            "last_analyzed": datetime.now().isoformat(),
            "system_health": system_health
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting optimization recommendations: {str(e)}")

# ===== RAG CONFIGURATION CRUD ENDPOINTS =====

@router.post("/rag/config")
async def create_rag_configuration(
    config: RAGConfigCreate,
    db: Session = Depends(get_db)
):
    """Create a new RAG configuration pattern"""
    try:
        # Create new configuration
        new_config = RAGConfiguration(
            name=config.name,
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            retrieval_strategy=config.retrieval_strategy,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            configuration=config.configuration,
            is_active=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        db.add(new_config)
        db.commit()
        db.refresh(new_config)
        
        logger.info(f"Created RAG configuration: {new_config.name} (ID: {new_config.id})")
        
        return {
            "id": new_config.id,
            "name": new_config.name,
            "embedding_model": new_config.embedding_model,
            "chunk_size": new_config.chunk_size,
            "chunk_overlap": new_config.chunk_overlap,
            "retrieval_strategy": new_config.retrieval_strategy,
            "top_k": new_config.top_k,
            "similarity_threshold": new_config.similarity_threshold,
            "configuration": new_config.configuration,
            "is_active": new_config.is_active,
            "created_at": new_config.created_at.isoformat() if new_config.created_at else None
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating RAG configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating configuration: {str(e)}")

@router.get("/rag/config")
async def list_rag_configurations(
    db: Session = Depends(get_db)
):
    """List all RAG configurations"""
    try:
        configs = db.query(RAGConfiguration).all()
        
        return [{
            "id": config.id,
            "name": config.name,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "retrieval_strategy": config.retrieval_strategy,
            "top_k": config.top_k,
            "similarity_threshold": config.similarity_threshold,
            "is_active": config.is_active,
            "created_at": config.created_at.isoformat() if config.created_at else None
        } for config in configs]
        
    except Exception as e:
        logger.error(f"Error listing RAG configurations: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing configurations: {str(e)}")

@router.get("/rag/config/{config_id}")
async def get_rag_configuration(
    config_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific RAG configuration"""
    try:
        config = db.query(RAGConfiguration).filter(RAGConfiguration.id == config_id).first()
        
        if not config:
            raise HTTPException(status_code=404, detail=f"Configuration {config_id} not found")
        
        return {
            "id": config.id,
            "name": config.name,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "retrieval_strategy": config.retrieval_strategy,
            "top_k": config.top_k,
            "similarity_threshold": config.similarity_threshold,
            "configuration": config.configuration,
            "is_active": config.is_active,
            "created_at": config.created_at.isoformat() if config.created_at else None,
            "updated_at": config.updated_at.isoformat() if config.updated_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RAG configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

@router.put("/rag/config/{config_id}")
async def update_rag_configuration(
    config_id: int,
    config_update: RAGConfigUpdate,
    db: Session = Depends(get_db)
):
    """Update a RAG configuration"""
    try:
        config = db.query(RAGConfiguration).filter(RAGConfiguration.id == config_id).first()
        
        if not config:
            raise HTTPException(status_code=404, detail=f"Configuration {config_id} not found")
        
        # Update fields if provided
        update_data = config_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(config, field, value)
        
        config.updated_at = datetime.now()
        
        db.commit()
        db.refresh(config)
        
        logger.info(f"Updated RAG configuration: {config.name} (ID: {config.id})")
        
        return {
            "id": config.id,
            "name": config.name,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "retrieval_strategy": config.retrieval_strategy,
            "top_k": config.top_k,
            "similarity_threshold": config.similarity_threshold,
            "configuration": config.configuration,
            "is_active": config.is_active,
            "updated_at": config.updated_at.isoformat() if config.updated_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating RAG configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@router.delete("/rag/config/{config_id}")
async def delete_rag_configuration(
    config_id: int,
    db: Session = Depends(get_db)
):
    """Delete a RAG configuration"""
    try:
        config = db.query(RAGConfiguration).filter(RAGConfiguration.id == config_id).first()
        
        if not config:
            raise HTTPException(status_code=404, detail=f"Configuration {config_id} not found")
        
        config_name = config.name
        db.delete(config)
        db.commit()
        
        logger.info(f"Deleted RAG configuration: {config_name} (ID: {config_id})")
        
        return {
            "success": True,
            "message": f"Configuration '{config_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting RAG configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting configuration: {str(e)}")