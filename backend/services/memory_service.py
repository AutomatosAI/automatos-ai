
"""
Memory System Service
====================

Service layer for advanced memory system operations with database integration.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

# Import memory models and database
from models import MemoryItem as DBMemoryItem, ExternalKnowledge, MemoryItemCreate, MemoryItemResponse
from memory.manager import AdvancedMemoryManager
from memory.memory_types import MemoryType, MemoryLevel

logger = logging.getLogger(__name__)

class MemorySystemService:
    """
    Service for managing memory system with database persistence
    """
    
    def __init__(self):
        self.memory_manager = AdvancedMemoryManager()
        logger.info("Memory System Service initialized")
    
    async def store_memory_with_persistence(
        self,
        db: Session,
        memory_data: MemoryItemCreate,
        persist_to_db: bool = True
    ) -> str:
        """
        Store memory in both memory system and database
        """
        try:
            # Store in memory system
            memory_id = await self.memory_manager.store_memory(
                session_id=memory_data.session_id,
                content=memory_data.content,
                memory_type=MemoryType(memory_data.memory_type),
                importance=memory_data.importance,
                tags=memory_data.tags
            )
            
            # Optionally persist to database
            if persist_to_db:
                db_memory_item = DBMemoryItem(
                    id=memory_id,
                    session_id=memory_data.session_id,
                    content=memory_data.content,
                    memory_type=memory_data.memory_type,
                    memory_level=MemoryLevel.IMMEDIATE.value,  # Start at immediate level
                    importance=memory_data.importance,
                    tags=memory_data.tags
                )
                
                db.add(db_memory_item)
                db.commit()
                db.refresh(db_memory_item)
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory with persistence: {e}")
            if persist_to_db:
                db.rollback()
            raise
    
    async def retrieve_memories_with_persistence(
        self,
        db: Session,
        session_id: str,
        query: Optional[str] = None,
        include_db_memories: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories from both memory system and database
        """
        try:
            # Get from memory system
            memory_system_results = await self.memory_manager.retrieve_memory(
                session_id=session_id,
                query=query
            )
            
            results = []
            
            # Convert memory system results
            for memory in memory_system_results:
                if hasattr(memory, 'original_memory'):  # AugmentedMemory
                    results.append({
                        "id": memory.original_memory.id,
                        "content": memory.original_memory.content,
                        "memory_type": memory.original_memory.memory_type.value,
                        "importance": memory.original_memory.importance,
                        "is_augmented": True,
                        "augmented_content": memory.augmented_content,
                        "similarity_score": memory.similarity_score,
                        "source": "memory_system"
                    })
                else:  # Regular MemoryItem
                    results.append({
                        "id": memory.id,
                        "content": memory.content,
                        "memory_type": memory.memory_type.value,
                        "importance": memory.importance,
                        "access_count": memory.access_count,
                        "is_augmented": False,
                        "source": "memory_system"
                    })
            
            # Optionally include database memories
            if include_db_memories:
                db_query = db.query(DBMemoryItem).filter(
                    DBMemoryItem.session_id == session_id
                )
                
                if query:
                    # Simple text search in content (can be enhanced with full-text search)
                    db_query = db_query.filter(
                        DBMemoryItem.content.cast(str).contains(query)
                    )
                
                db_memories = db_query.all()
                
                for db_memory in db_memories:
                    results.append({
                        "id": db_memory.id,
                        "content": db_memory.content,
                        "memory_type": db_memory.memory_type,
                        "memory_level": db_memory.memory_level,
                        "importance": db_memory.importance,
                        "access_count": db_memory.access_count,
                        "tags": db_memory.tags,
                        "is_augmented": False,
                        "source": "database",
                        "created_at": db_memory.created_at.isoformat(),
                        "last_access": db_memory.last_access.isoformat()
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories with persistence: {e}")
            return []
    
    async def sync_memory_to_database(
        self,
        db: Session,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Sync memory system state to database for persistence
        """
        try:
            sync_stats = {
                "synced_count": 0,
                "updated_count": 0,
                "error_count": 0
            }
            
            # Get all memories from memory system for session
            memories = await self.memory_manager.retrieve_memory(
                session_id=session_id,
                max_items=1000  # Large number to get all
            )
            
            for memory in memories:
                try:
                    # Skip augmented memories for now
                    if hasattr(memory, 'original_memory'):
                        continue
                    
                    # Check if memory exists in database
                    existing = db.query(DBMemoryItem).filter(
                        DBMemoryItem.id == memory.id
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.content = memory.content
                        existing.importance = memory.importance
                        existing.access_count = memory.access_count
                        existing.tags = memory.tags
                        existing.last_access = memory.last_access
                        sync_stats["updated_count"] += 1
                    else:
                        # Create new record
                        # Determine memory level (simplified)
                        memory_level = MemoryLevel.WORKING.value  # Default
                        
                        db_memory = DBMemoryItem(
                            id=memory.id,
                            session_id=session_id,
                            content=memory.content,
                            memory_type=memory.memory_type.value,
                            memory_level=memory_level,
                            importance=memory.importance,
                            access_count=memory.access_count,
                            tags=memory.tags,
                            created_at=memory.creation_time,
                            last_access=memory.last_access
                        )
                        
                        db.add(db_memory)
                        sync_stats["synced_count"] += 1
                
                except Exception as item_error:
                    logger.error(f"Failed to sync memory item {memory.id}: {item_error}")
                    sync_stats["error_count"] += 1
            
            db.commit()
            
            return {
                "status": "completed",
                "stats": sync_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to sync memory to database: {e}")
            db.rollback()
            return {"status": "failed", "error": str(e)}
    
    async def add_external_knowledge_with_persistence(
        self,
        db: Session,
        content: Dict[str, Any],
        source: str = "external",
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Add external knowledge with database persistence
        """
        try:
            # Add to memory system
            knowledge_id = await self.memory_manager.add_external_knowledge(
                content=content,
                source=source,
                metadata=metadata
            )
            
            # Persist to database
            db_knowledge = ExternalKnowledge(
                content=content,
                source=source,
                knowledge_metadata=metadata or {}
            )
            
            db.add(db_knowledge)
            db.commit()
            db.refresh(db_knowledge)
            
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Failed to add external knowledge with persistence: {e}")
            db.rollback()
            raise
    
    async def get_memory_analytics(
        self,
        db: Session,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive memory analytics
        """
        try:
            # Get memory system stats
            system_stats = await self.memory_manager.get_comprehensive_stats()
            
            # Get database statistics
            db_stats = {}
            
            if session_id:
                db_count = db.query(DBMemoryItem).filter(
                    DBMemoryItem.session_id == session_id
                ).count()
                db_stats["session_memory_count"] = db_count
            
            total_db_count = db.query(DBMemoryItem).count()
            external_knowledge_count = db.query(ExternalKnowledge).count()
            
            db_stats.update({
                "total_database_memories": total_db_count,
                "external_knowledge_items": external_knowledge_count
            })
            
            return {
                "memory_system_stats": {
                    "total_memories": system_stats.total_memories,
                    "memory_levels": system_stats.memory_levels,
                    "system_performance": system_stats.system_performance
                },
                "database_stats": db_stats,
                "access_metrics": {
                    "hit_rate": system_stats.access_metrics.hit_rate if system_stats.access_metrics else 0,
                    "cache_utilization": system_stats.access_metrics.cache_utilization if system_stats.access_metrics else 0
                },
                "consolidation_metrics": {
                    "items_consolidated": system_stats.consolidation_metrics.items_consolidated if system_stats.consolidation_metrics else 0,
                    "compression_ratio": system_stats.consolidation_metrics.compression_ratio if system_stats.consolidation_metrics else 1.0
                },
                "augmentation_stats": system_stats.augmentation_stats or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory analytics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_memories(
        self,
        db: Session,
        days_threshold: int = 30,
        importance_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Clean up old, low-importance memories
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # Find old, low-importance memories in database
            old_memories = db.query(DBMemoryItem).filter(
                and_(
                    DBMemoryItem.created_at < cutoff_date,
                    DBMemoryItem.importance < importance_threshold,
                    DBMemoryItem.access_count < 5  # Rarely accessed
                )
            ).all()
            
            cleanup_stats = {
                "evaluated": len(old_memories),
                "deleted": 0,
                "errors": 0
            }
            
            for memory in old_memories:
                try:
                    db.delete(memory)
                    cleanup_stats["deleted"] += 1
                except Exception as delete_error:
                    logger.error(f"Failed to delete memory {memory.id}: {delete_error}")
                    cleanup_stats["errors"] += 1
            
            db.commit()
            
            return {
                "status": "completed",
                "stats": cleanup_stats,
                "cutoff_date": cutoff_date.isoformat(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}")
            db.rollback()
            return {"status": "failed", "error": str(e)}
