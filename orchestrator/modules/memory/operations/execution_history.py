"""
Execution History Search
========================

Search for similar past task executions to learn from patterns.
Migrated from context_engineering/manager.py

This enables:
- Finding similar past tasks
- Learning from success/failure patterns
- Applying past solutions to new problems
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from execution history search"""
    execution_id: str
    task_description: str
    success: bool
    similarity_score: float
    timestamp: datetime
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None


class ExecutionHistorySearch:
    """
    Search for similar past executions using vector similarity.
    
    Uses workflow_executions table with embeddings for semantic search.
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self._embedding_manager = None
        
    def _ensure_embedding_manager(self):
        """Lazy load embedding manager"""
        if self._embedding_manager is None:
            from core.llm import create_embedding_manager
            self._embedding_manager = create_embedding_manager()
        return self._embedding_manager
    
    async def search_similar_executions(
        self,
        query: str,
        k: int = 5,
        success_only: bool = True,
        workflow_id: Optional[int] = None
    ) -> List[ExecutionResult]:
        """
        Search for similar past executions.
        
        Args:
            query: Search query (task description or similar)
            k: Number of results to return
            success_only: Only return successful executions
            workflow_id: Filter by specific workflow
            
        Returns:
            List of similar execution results
        """
        try:
            if not self.db:
                logger.warning("No database session provided for execution history search")
                return []
            
            embedding_manager = self._ensure_embedding_manager()
            
            # Generate query embedding
            query_embedding = await embedding_manager.generate_embedding(query)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build query with filters
            conditions = ["we.embedding IS NOT NULL"]
            params = {"limit": k}
            
            if success_only:
                conditions.append("we.status = 'completed'")
            
            if workflow_id:
                conditions.append("we.workflow_id = :workflow_id")
                params["workflow_id"] = workflow_id
            
            where_clause = " AND ".join(conditions)
            
            from sqlalchemy import text
            
            result = self.db.execute(text(f"""
                SELECT 
                    we.id,
                    we.workflow_id,
                    we.status,
                    we.input_data,
                    we.output_data,
                    we.error_message,
                    we.started_at,
                    we.completed_at,
                    w.name as workflow_name,
                    w.description as task_description,
                    1 - (we.embedding <=> '{embedding_str}'::vector) as similarity
                FROM workflow_executions we
                LEFT JOIN workflows w ON w.id = we.workflow_id
                WHERE {where_clause}
                ORDER BY we.embedding <=> '{embedding_str}'::vector
                LIMIT :limit
            """), params)
            
            executions = []
            for row in result.fetchall():
                duration_ms = None
                if row.started_at and row.completed_at:
                    duration_ms = int((row.completed_at - row.started_at).total_seconds() * 1000)
                
                executions.append(ExecutionResult(
                    execution_id=str(row.id),
                    task_description=row.task_description or f"Workflow {row.workflow_id}",
                    success=row.status == 'completed',
                    similarity_score=float(row.similarity) if row.similarity else 0.0,
                    timestamp=row.started_at,
                    input_data=row.input_data,
                    output_data=row.output_data,
                    error_message=row.error_message,
                    duration_ms=duration_ms
                ))
            
            logger.info(f"Found {len(executions)} similar executions for query: {query[:50]}...")
            return executions
            
        except Exception as e:
            logger.error(f"Execution history search failed: {e}")
            return []
    
    async def get_successful_patterns(
        self,
        task_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get successful execution patterns for a task type.
        
        Useful for learning what worked in the past.
        """
        try:
            if not self.db:
                return []
            
            from sqlalchemy import text
            
            result = self.db.execute(text("""
                SELECT 
                    we.input_data,
                    we.output_data,
                    w.name,
                    w.description,
                    COUNT(*) as success_count
                FROM workflow_executions we
                JOIN workflows w ON w.id = we.workflow_id
                WHERE we.status = 'completed'
                    AND w.description ILIKE :task_pattern
                GROUP BY we.input_data, we.output_data, w.name, w.description
                ORDER BY success_count DESC
                LIMIT :limit
            """), {"task_pattern": f"%{task_type}%", "limit": limit})
            
            patterns = []
            for row in result.fetchall():
                patterns.append({
                    "workflow_name": row.name,
                    "description": row.description,
                    "input_pattern": row.input_data,
                    "output_pattern": row.output_data,
                    "success_count": row.success_count
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get successful patterns: {e}")
            return []


# Singleton accessor
_execution_history: Optional[ExecutionHistorySearch] = None


def get_execution_history_search(db_session=None) -> ExecutionHistorySearch:
    """Get execution history search instance"""
    global _execution_history
    if _execution_history is None or db_session is not None:
        _execution_history = ExecutionHistorySearch(db_session)
    return _execution_history

