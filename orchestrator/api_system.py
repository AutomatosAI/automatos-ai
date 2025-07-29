
"""
System Configuration and Health API Routes
==========================================

REST API endpoints for system configuration, health monitoring, and RAG management.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime
import psutil
import os

from database import get_db
from models import (
    SystemConfiguration, RAGConfiguration,
    SystemConfigCreate, SystemConfigResponse,
    RAGConfigCreate, RAGConfigResponse,
    SystemHealthResponse
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system"])

# System Configuration endpoints
@router.post("/config", response_model=SystemConfigResponse)
async def create_system_config(config_data: SystemConfigCreate, db: Session = Depends(get_db)):
    """Create or update system configuration"""
    try:
        # Check if config already exists
        existing = db.query(SystemConfiguration).filter(
            SystemConfiguration.config_key == config_data.config_key
        ).first()
        
        if existing:
            # Update existing
            existing.config_value = config_data.config_value
            existing.description = config_data.description
            existing.updated_by = "system"  # TODO: Get from auth context
            db.commit()
            db.refresh(existing)
            config = existing
        else:
            # Create new
            config = SystemConfiguration(
                config_key=config_data.config_key,
                config_value=config_data.config_value,
                description=config_data.description,
                updated_by="system"  # TODO: Get from auth context
            )
            db.add(config)
            db.commit()
            db.refresh(config)
        
        return SystemConfigResponse(
            id=config.id,
            config_key=config.config_key,
            config_value=config.config_value,
            description=config.description,
            is_active=config.is_active,
            created_at=config.created_at,
            updated_at=config.updated_at,
            updated_by=config.updated_by
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating system config: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating system config: {str(e)}")

@router.get("/config", response_model=List[SystemConfigResponse])
async def list_system_configs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """List system configurations"""
    try:
        query = db.query(SystemConfiguration)
        
        # Apply filters
        if active_only:
            query = query.filter(SystemConfiguration.is_active == True)
        if search:
            query = query.filter(
                or_(
                    SystemConfiguration.config_key.ilike(f"%{search}%"),
                    SystemConfiguration.description.ilike(f"%{search}%")
                )
            )
        
        configs = query.offset(skip).limit(limit).all()
        
        return [
            SystemConfigResponse(
                id=config.id,
                config_key=config.config_key,
                config_value=config.config_value,
                description=config.description,
                is_active=config.is_active,
                created_at=config.created_at,
                updated_at=config.updated_at,
                updated_by=config.updated_by
            ) for config in configs
        ]
        
    except Exception as e:
        logger.error(f"Error listing system configs: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing system configs: {str(e)}")

@router.get("/config/{config_key}", response_model=SystemConfigResponse)
async def get_system_config(config_key: str, db: Session = Depends(get_db)):
    """Get system configuration by key"""
    try:
        config = db.query(SystemConfiguration).filter(
            SystemConfiguration.config_key == config_key
        ).first()
        
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return SystemConfigResponse(
            id=config.id,
            config_key=config.config_key,
            config_value=config.config_value,
            description=config.description,
            is_active=config.is_active,
            created_at=config.created_at,
            updated_at=config.updated_at,
            updated_by=config.updated_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system config {config_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system config: {str(e)}")

@router.put("/config/{config_key}", response_model=SystemConfigResponse)
async def update_system_config(
    config_key: str, 
    config_data: SystemConfigCreate, 
    db: Session = Depends(get_db)
):
    """Update system configuration"""
    try:
        config = db.query(SystemConfiguration).filter(
            SystemConfiguration.config_key == config_key
        ).first()
        
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        config.config_value = config_data.config_value
        config.description = config_data.description
        config.updated_by = "system"  # TODO: Get from auth context
        
        db.commit()
        db.refresh(config)
        
        return SystemConfigResponse(
            id=config.id,
            config_key=config.config_key,
            config_value=config.config_value,
            description=config.description,
            is_active=config.is_active,
            created_at=config.created_at,
            updated_at=config.updated_at,
            updated_by=config.updated_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating system config {config_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating system config: {str(e)}")

# RAG Configuration endpoints
@router.post("/rag", response_model=RAGConfigResponse)
async def create_rag_config(rag_data: RAGConfigCreate, db: Session = Depends(get_db)):
    """Create RAG configuration"""
    try:
        rag_config = RAGConfiguration(
            name=rag_data.name,
            embedding_model=rag_data.embedding_model,
            chunk_size=rag_data.chunk_size,
            chunk_overlap=rag_data.chunk_overlap,
            retrieval_strategy=rag_data.retrieval_strategy,
            top_k=rag_data.top_k,
            similarity_threshold=rag_data.similarity_threshold,
            configuration=rag_data.configuration or {},
            created_by="system"  # TODO: Get from auth context
        )
        
        db.add(rag_config)
        db.commit()
        db.refresh(rag_config)
        
        return RAGConfigResponse(
            id=rag_config.id,
            name=rag_config.name,
            embedding_model=rag_config.embedding_model,
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
            retrieval_strategy=rag_config.retrieval_strategy,
            top_k=rag_config.top_k,
            similarity_threshold=rag_config.similarity_threshold,
            configuration=rag_config.configuration,
            is_active=rag_config.is_active,
            created_at=rag_config.created_at,
            updated_at=rag_config.updated_at,
            created_by=rag_config.created_by
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating RAG config: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating RAG config: {str(e)}")

@router.get("/rag", response_model=List[RAGConfigResponse])
async def list_rag_configs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """List RAG configurations"""
    try:
        query = db.query(RAGConfiguration)
        
        if active_only:
            query = query.filter(RAGConfiguration.is_active == True)
        
        configs = query.offset(skip).limit(limit).all()
        
        return [
            RAGConfigResponse(
                id=config.id,
                name=config.name,
                embedding_model=config.embedding_model,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                retrieval_strategy=config.retrieval_strategy,
                top_k=config.top_k,
                similarity_threshold=config.similarity_threshold,
                configuration=config.configuration,
                is_active=config.is_active,
                created_at=config.created_at,
                updated_at=config.updated_at,
                created_by=config.created_by
            ) for config in configs
        ]
        
    except Exception as e:
        logger.error(f"Error listing RAG configs: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing RAG configs: {str(e)}")

@router.get("/rag/{config_id}", response_model=RAGConfigResponse)
async def get_rag_config(config_id: int, db: Session = Depends(get_db)):
    """Get RAG configuration by ID"""
    try:
        config = db.query(RAGConfiguration).filter(RAGConfiguration.id == config_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="RAG configuration not found")
        
        return RAGConfigResponse(
            id=config.id,
            name=config.name,
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            retrieval_strategy=config.retrieval_strategy,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            configuration=config.configuration,
            is_active=config.is_active,
            created_at=config.created_at,
            updated_at=config.updated_at,
            created_by=config.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RAG config {config_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting RAG config: {str(e)}")

@router.post("/rag/{config_id}/test")
async def test_rag_config(
    config_id: int, 
    query: str = Query(..., description="Test query for RAG system"),
    db: Session = Depends(get_db)
):
    """Test RAG configuration with a query"""
    try:
        # Import and use real RAG service
        from rag_service import get_rag_service
        rag_service = await get_rag_service()
        
        # Use real RAG testing
        result = await rag_service.test_rag_config(config_id, query, db)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error testing RAG config {config_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing RAG config: {str(e)}")

# System Health endpoints
@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(db: Session = Depends(get_db)):
    """Get system health status"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check database connection
        db_status = "healthy"
        try:
            db.execute("SELECT 1")
        except Exception:
            db_status = "unhealthy"
        
        # Check services status
        services = {
            "database": db_status,
            "api": "healthy",
            "document_processor": "healthy",  # TODO: Check actual status
            "rag_system": "healthy"  # TODO: Check actual status
        }
        
        # Overall system status
        overall_status = "healthy" if all(status == "healthy" for status in services.values()) else "degraded"
        
        metrics = {
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "memory_available": f"{memory.available / (1024**3):.1f}GB",
            "disk_usage": f"{disk.percent}%",
            "disk_free": f"{disk.free / (1024**3):.1f}GB",
            "uptime": "N/A"  # TODO: Track actual uptime
        }
        
        return SystemHealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            services=services,
            metrics=metrics,
            version="1.0.0"  # TODO: Get from actual version
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")

@router.get("/metrics")
async def get_system_metrics():
    """Get detailed system metrics"""
    try:
        # CPU metrics
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network = psutil.net_io_counters()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "count": cpu_count,
                "usage_percent": cpu_percent,
                "average_usage": sum(cpu_percent) / len(cpu_percent)
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "percent": swap.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")
