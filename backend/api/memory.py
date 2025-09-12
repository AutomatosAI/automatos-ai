
"""
Advanced Memory System API Endpoints
====================================

REST API endpoints for the advanced memory management system including:
- Hierarchical memory operations
- Memory access optimization
- External knowledge management
- Memory consolidation
- System statistics and health monitoring
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
import asyncio

# Import memory system components
from memory.manager import AdvancedMemoryManager
from memory.memory_types import MemoryType, MemoryLevel
from memory.access_patterns import AccessPattern
from memory.augmentation import AugmentationStrategy
from memory.consolidation import ConsolidationStrategy

# Import models
from models import (
    MemoryItemCreate, MemoryItemResponse, 
    ExternalKnowledgeCreate, ExternalKnowledgeResponse
)

# Import database (if using persistence)
# from database import get_db

logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(prefix="/api/v1/memory", tags=["Advanced Memory"])

# Global memory manager instance (in production, this might be dependency injected)
memory_manager: Optional[AdvancedMemoryManager] = None

def get_memory_manager() -> AdvancedMemoryManager:
    """Get or create the memory manager instance"""
    global memory_manager
    if memory_manager is None:
        memory_manager = AdvancedMemoryManager(
            # Configure with reasonable defaults
            immediate_capacity=10,
            working_capacity=100,
            short_term_capacity=1000,
            long_term_capacity=50000,
            cache_size=1000,
            auto_consolidation=True,
            consolidation_interval_hours=6
        )
    return memory_manager

# Memory Storage Endpoints

@router.post("/store", response_model=Dict[str, Any])
async def store_memory(
    memory_data: MemoryItemCreate,
    auto_augment: bool = Query(True, description="Automatically augment with external knowledge"),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Store new memory item in the hierarchical memory system
    """
    try:
        manager = get_memory_manager()
        
        memory_id = await manager.store_memory(
            session_id=memory_data.session_id,
            content=memory_data.content,
            memory_type=MemoryType(memory_data.memory_type),
            importance=memory_data.importance,
            tags=memory_data.tags,
            auto_augment=auto_augment
        )
        
        return {
            "memory_id": memory_id,
            "session_id": memory_data.session_id,
            "status": "stored",
            "auto_augment": auto_augment,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")

@router.get("/retrieve/{session_id}", response_model=List[Dict[str, Any]])
async def retrieve_memories(
    session_id: str,
    query: Optional[str] = Query(None, description="Search query for relevant memories"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    max_items: int = Query(10, ge=1, le=100, description="Maximum number of items to retrieve"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance threshold"),
    access_pattern: str = Query("locality", description="Access optimization pattern"),
    include_augmented: bool = Query(True, description="Include augmented memories")
) -> List[Dict[str, Any]]:
    """
    Retrieve memories for a session with optimized access patterns
    """
    try:
        manager = get_memory_manager()
        
        # Convert string parameters to enums
        memory_type_enum = MemoryType(memory_type) if memory_type else None
        access_pattern_enum = AccessPattern(access_pattern) if access_pattern in [p.value for p in AccessPattern] else AccessPattern.LOCALITY_BASED
        
        memories = await manager.retrieve_memory(
            session_id=session_id,
            query=query,
            memory_type=memory_type_enum,
            max_items=max_items,
            min_importance=min_importance,
            access_pattern=access_pattern_enum,
            include_augmented=include_augmented
        )
        
        # Convert memories to response format
        response_memories = []
        for memory in memories:
            if hasattr(memory, 'original_memory'):  # AugmentedMemory
                memory_dict = {
                    "id": memory.original_memory.id,
                    "content": memory.original_memory.content,
                    "memory_type": memory.original_memory.memory_type.value,
                    "importance": memory.original_memory.importance,
                    "is_augmented": True,
                    "augmented_content": memory.augmented_content,
                    "similarity_score": memory.similarity_score,
                    "augmentation_source": memory.augmentation_source,
                    "timestamp": memory.timestamp.isoformat()
                }
            else:  # Regular MemoryItem
                memory_dict = {
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance,
                    "access_count": memory.access_count,
                    "tags": memory.tags,
                    "is_augmented": False,
                    "created_at": memory.creation_time.isoformat(),
                    "last_access": memory.last_access.isoformat()
                }
            
            response_memories.append(memory_dict)
        
        return response_memories
        
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")

# Memory Augmentation Endpoints

@router.post("/external-knowledge", response_model=Dict[str, Any])
async def add_external_knowledge(
    knowledge_data: ExternalKnowledgeCreate
) -> Dict[str, Any]:
    """
    Add external knowledge for memory augmentation
    """
    try:
        manager = get_memory_manager()
        
        knowledge_id = await manager.add_external_knowledge(
            content=knowledge_data.content,
            source=knowledge_data.source,
            metadata=knowledge_data.metadata
        )
        
        return {
            "knowledge_id": knowledge_id,
            "source": knowledge_data.source,
            "status": "added",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to add external knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add external knowledge: {str(e)}")

@router.post("/augment/{memory_id}", response_model=List[Dict[str, Any]])
async def augment_memory(
    memory_id: str,
    strategy: str = Query("hybrid", description="Augmentation strategy"),
    max_augmentations: int = Query(5, ge=1, le=20, description="Maximum augmentations to generate")
) -> List[Dict[str, Any]]:
    """
    Explicitly augment a specific memory item with external knowledge
    """
    try:
        manager = get_memory_manager()
        
        # Convert strategy to enum
        strategy_enum = AugmentationStrategy(strategy) if strategy in [s.value for s in AugmentationStrategy] else AugmentationStrategy.HYBRID
        
        augmented_memories = await manager.augment_memory_item(
            memory_id=memory_id,
            strategy=strategy_enum,
            max_augmentations=max_augmentations
        )
        
        # Convert to response format
        response_augmentations = []
        for aug_mem in augmented_memories:
            response_augmentations.append({
                "original_memory_id": aug_mem.original_memory.id,
                "augmented_content": aug_mem.augmented_content,
                "similarity_score": aug_mem.similarity_score,
                "augmentation_source": aug_mem.augmentation_source,
                "metadata": aug_mem.augmentation_metadata,
                "timestamp": aug_mem.timestamp.isoformat()
            })
        
        return response_augmentations
        
    except Exception as e:
        logger.error(f"Failed to augment memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to augment memory: {str(e)}")

# Memory Consolidation Endpoints

@router.post("/consolidate", response_model=Dict[str, Any])
async def consolidate_memories(
    session_id: Optional[str] = Query(None, description="Session ID to consolidate (optional)"),
    memory_level: Optional[str] = Query(None, description="Memory level to consolidate"),
    strategy: str = Query("hybrid", description="Consolidation strategy"),
    force: bool = Query(False, description="Force consolidation even if not needed"),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Consolidate memories to optimize storage and improve efficiency
    """
    try:
        manager = get_memory_manager()
        
        # Convert parameters to enums
        memory_level_enum = MemoryLevel(memory_level) if memory_level else None
        strategy_enum = ConsolidationStrategy(strategy) if strategy in [s.value for s in ConsolidationStrategy] else ConsolidationStrategy.HYBRID
        
        # Run consolidation (potentially in background for large operations)
        if background_tasks and not force:
            background_tasks.add_task(
                manager.consolidate_memories,
                session_id=session_id,
                memory_level=memory_level_enum,
                strategy=strategy_enum,
                force=force
            )
            
            return {
                "status": "consolidation_scheduled",
                "session_id": session_id,
                "memory_level": memory_level,
                "strategy": strategy,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            result = await manager.consolidate_memories(
                session_id=session_id,
                memory_level=memory_level_enum,
                strategy=strategy_enum,
                force=force
            )
            
            return {
                "status": "completed",
                "consolidation_result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Failed to consolidate memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to consolidate memories: {str(e)}")

# System Monitoring and Statistics

@router.get("/stats", response_model=Dict[str, Any])
async def get_memory_stats() -> Dict[str, Any]:
    """
    Get comprehensive memory system statistics
    """
    try:
        manager = get_memory_manager()
        stats = await manager.get_comprehensive_stats()
        
        return {
            "system_stats": {
                "total_memories": stats.total_memories,
                "memory_levels": stats.memory_levels,
                "system_performance": stats.system_performance
            },
            "access_metrics": {
                "hit_rate": stats.access_metrics.hit_rate if stats.access_metrics else 0,
                "miss_rate": stats.access_metrics.miss_rate if stats.access_metrics else 0,
                "avg_response_time": stats.access_metrics.avg_response_time if stats.access_metrics else 0,
                "cache_utilization": stats.access_metrics.cache_utilization if stats.access_metrics else 0,
                "locality_factor": stats.access_metrics.locality_factor if stats.access_metrics else 0
            },
            "consolidation_metrics": {
                "items_consolidated": stats.consolidation_metrics.items_consolidated if stats.consolidation_metrics else 0,
                "compression_ratio": stats.consolidation_metrics.compression_ratio if stats.consolidation_metrics else 1.0,
                "information_retention": stats.consolidation_metrics.information_retention if stats.consolidation_metrics else 1.0,
                "processing_time": stats.consolidation_metrics.processing_time if stats.consolidation_metrics else 0,
                "storage_saved": stats.consolidation_metrics.storage_saved if stats.consolidation_metrics else 0
            },
            "augmentation_stats": stats.augmentation_stats or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")

@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_system_performance(
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Optimize overall memory system performance
    """
    try:
        manager = get_memory_manager()
        
        if background_tasks:
            background_tasks.add_task(manager.optimize_system_performance)
            
            return {
                "status": "optimization_scheduled",
                "message": "System optimization running in background",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            optimization_result = await manager.optimize_system_performance()
            
            return {
                "status": "completed",
                "optimization_result": optimization_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Failed to optimize system performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize system: {str(e)}")

# System Management Endpoints

@router.post("/backup", response_model=Dict[str, Any])
async def backup_system_state(
    directory: str = Query("memory_backups", description="Backup directory"),
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Backup complete memory system state
    """
    try:
        manager = get_memory_manager()
        
        if background_tasks:
            background_tasks.add_task(manager.save_system_state, directory)
            
            return {
                "status": "backup_scheduled",
                "directory": directory,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            await manager.save_system_state(directory)
            
            return {
                "status": "backup_completed",
                "directory": directory,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Failed to backup system state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to backup system: {str(e)}")

@router.post("/restore", response_model=Dict[str, Any])
async def restore_system_state(
    directory: str = Query("memory_backups", description="Restore directory"),
    timestamp: Optional[str] = Query(None, description="Specific backup timestamp to restore")
) -> Dict[str, Any]:
    """
    Restore memory system state from backup
    """
    try:
        manager = get_memory_manager()
        
        await manager.load_system_state(directory, timestamp)
        
        return {
            "status": "restore_completed",
            "directory": directory,
            "timestamp": timestamp,
            "restored_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to restore system state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restore system: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def memory_system_health() -> Dict[str, Any]:
    """
    Get memory system health status
    """
    try:
        manager = get_memory_manager()
        stats = await manager.get_comprehensive_stats()
        
        # Determine health status based on metrics
        health_score = 0.0
        issues = []
        
        # Check memory utilization
        total_utilization = sum(stats.memory_levels.values())
        if total_utilization > 50000:  # High memory usage
            issues.append("High memory utilization")
        else:
            health_score += 0.3
        
        # Check access performance
        if stats.access_metrics and stats.access_metrics.hit_rate > 0.5:
            health_score += 0.3
        elif stats.access_metrics:
            issues.append("Low cache hit rate")
        
        # Check system performance
        if stats.system_performance:
            avg_response = stats.system_performance.get("avg_response_time", 0)
            if avg_response < 1.0:  # Under 1 second
                health_score += 0.4
            else:
                issues.append("Slow response times")
        
        status = "healthy" if health_score > 0.7 else "warning" if health_score > 0.4 else "critical"
        
        return {
            "status": status,
            "health_score": health_score,
            "issues": issues,
            "total_memories": stats.total_memories,
            "system_uptime_hours": stats.system_performance.get("uptime_hours", 0) if stats.system_performance else 0,
            "last_check": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

# Cleanup endpoint for development/testing
@router.post("/clear", response_model=Dict[str, Any])
async def clear_memory_system(
    confirm: bool = Query(False, description="Confirm clear operation")
) -> Dict[str, Any]:
    """
    Clear all memory system data (development/testing only)
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Must confirm clear operation with ?confirm=true"
        )
    
    try:
        global memory_manager
        if memory_manager:
            memory_manager.stop_background_services()
            memory_manager = None
        
        # Reinitialize clean manager
        memory_manager = get_memory_manager()
        
        return {
            "status": "cleared",
            "message": "Memory system cleared and reinitialized",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear memory system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear system: {str(e)}")
