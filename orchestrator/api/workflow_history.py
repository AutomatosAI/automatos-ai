"""
Workflow History and Execution Data API
========================================
Provides endpoints for retrieving historical workflow execution data,
including memory operations, communications, and learning insights.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from database.database import get_db
from database.models import WorkflowExecution, Workflow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/workflows", tags=["workflow-history"])

@router.get("/{workflow_id}/executions")
async def get_workflow_executions(
    workflow_id: int,
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get execution history for a workflow.
    Returns list of executions with their metadata.
    """
    try:
        executions = db.query(WorkflowExecution)\
            .filter(WorkflowExecution.workflow_id == workflow_id)\
            .order_by(desc(WorkflowExecution.created_at))\
            .limit(limit)\
            .offset(offset)\
            .all()
        
        result = []
        for exec in executions:
            exec_data = {
                "id": exec.id,
                "workflow_id": exec.workflow_id,
                "status": exec.status,
                "created_at": exec.created_at.isoformat() if exec.created_at else None,
                "updated_at": exec.updated_at.isoformat() if exec.updated_at else None,
                "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                "execution_time": (exec.completed_at - exec.created_at).total_seconds() if exec.completed_at and exec.created_at else None,
                "summary": {
                    "total_subtasks": exec.input_data.get("total_subtasks", 0) if exec.input_data else 0,
                    "completed_subtasks": exec.input_data.get("completed_subtasks", 0) if exec.input_data else 0,
                    "success_rate": exec.input_data.get("analytics", {}).get("overall_success_rate", 0) if exec.input_data else 0,
                    "total_cost": exec.input_data.get("analytics", {}).get("total_cost", 0) if exec.input_data else 0
                }
            }
            result.append(exec_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching workflow executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions/{execution_id}")
async def get_execution_details(
    execution_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed execution data including memory, communication, and learning data.
    """
    try:
        execution = db.query(WorkflowExecution)\
            .filter(WorkflowExecution.id == execution_id)\
            .first()
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Extract orchestrator logs
        orchestrator_logs = []
        if execution.input_data and "orchestrator_logs" in execution.input_data:
            orchestrator_logs = execution.input_data["orchestrator_logs"]
        
        # Extract memory data
        memory_data = {}
        if execution.input_data and "memory_retrieval" in execution.input_data:
            memory_data = execution.input_data["memory_retrieval"]
        
        # Extract learning data
        learning_data = {}
        if execution.input_data and "learning_system" in execution.input_data:
            learning_data = execution.input_data["learning_system"]
        
        # Extract communication data
        communication_data = []
        if execution.input_data and "communications" in execution.input_data:
            communication_data = execution.input_data["communications"]
        
        # Extract analytics
        analytics = {}
        if execution.input_data and "analytics" in execution.input_data:
            analytics = execution.input_data["analytics"]
        
        return {
            "id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status,
            "created_at": execution.created_at.isoformat() if execution.created_at else None,
            "updated_at": execution.updated_at.isoformat() if execution.updated_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "orchestrator_logs": orchestrator_logs,
            "input_data": {
                "memory_retrieval": memory_data,
                "learning_system": learning_data,
                "communications": communication_data,
                "analytics": analytics,
                "decomposition": execution.input_data.get("decomposition", {}) if execution.input_data else {},
                "context_engineering": execution.input_data.get("context_engineering", {}) if execution.input_data else {},
                "agent_execution": execution.input_data.get("agent_execution", {}) if execution.input_data else {}
            },
            "output_data": execution.output_data,
            "result": execution.result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching execution details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workflow_id}/live-progress")
async def get_live_progress(
    workflow_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get live progress data for an executing workflow.
    This endpoint is polled during execution to get real-time updates.
    """
    try:
        # Get the most recent execution for this workflow
        execution = db.query(WorkflowExecution)\
            .filter(WorkflowExecution.workflow_id == workflow_id)\
            .filter(WorkflowExecution.status.in_(["started", "running", "in_progress"]))\
            .order_by(desc(WorkflowExecution.created_at))\
            .first()
        
        if not execution:
            # No active execution, return empty progress
            return {
                "status": "idle",
                "progress": 0,
                "metrics": {},
                "memory_activities": [],
                "communications": []
            }
        
        # Extract live data from the execution
        input_data = execution.input_data or {}
        
        # Calculate progress
        total_subtasks = input_data.get("total_subtasks", 0)
        completed_subtasks = input_data.get("completed_subtasks", 0)
        progress = (completed_subtasks / total_subtasks * 100) if total_subtasks > 0 else 0
        
        # Extract recent memory activities
        memory_activities = []
        if "recent_memory_operations" in input_data:
            memory_activities = input_data["recent_memory_operations"][-10:]  # Last 10 operations
        
        # Extract recent communications
        communications = []
        if "recent_communications" in input_data:
            communications = input_data["recent_communications"][-10:]  # Last 10 messages
        
        # Get active agents
        active_agents = []
        if "active_agents" in input_data:
            active_agents = input_data["active_agents"]
        
        # Build metrics
        metrics = {
            "total_tasks": total_subtasks,
            "completed_tasks": completed_subtasks,
            "active_agents": len(active_agents),
            "memory_operations": input_data.get("memory_operation_count", 0),
            "messages_exchanged": input_data.get("message_count", 0),
            "patterns_learned": input_data.get("patterns_learned", 0),
            "avg_response_time": input_data.get("avg_response_time", 0),
            "success_rate": input_data.get("current_success_rate", 0),
            "token_usage": input_data.get("total_tokens", 0),
            "estimated_cost": input_data.get("estimated_cost", 0)
        }
        
        return {
            "status": execution.status,
            "progress": progress,
            "current_step": input_data.get("current_step", ""),
            "completed_steps": completed_subtasks,
            "total_steps": total_subtasks,
            "active_agents": active_agents,
            "metrics": metrics,
            "memory_activities": memory_activities,
            "communications": communications
        }
        
    except Exception as e:
        logger.error(f"Error fetching live progress: {e}")
        return {
            "status": "error",
            "progress": 0,
            "metrics": {},
            "memory_activities": [],
            "communications": [],
            "error": str(e)
        }

@router.post("/{workflow_id}/store-orchestration-event")
async def store_orchestration_event(
    workflow_id: int,
    event_type: str,
    event_data: Dict[str, Any],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Store orchestration events (memory, communication, learning) in real-time.
    Called by the orchestrator during execution.
    """
    try:
        # Get the active execution
        execution = db.query(WorkflowExecution)\
            .filter(WorkflowExecution.workflow_id == workflow_id)\
            .filter(WorkflowExecution.status.in_(["started", "running", "in_progress"]))\
            .order_by(desc(WorkflowExecution.created_at))\
            .first()
        
        if not execution:
            raise HTTPException(status_code=404, detail="No active execution found")
        
        # Initialize input_data if needed
        if not execution.input_data:
            execution.input_data = {}
        
        # Store event based on type
        if event_type == "memory_operation":
            if "recent_memory_operations" not in execution.input_data:
                execution.input_data["recent_memory_operations"] = []
            execution.input_data["recent_memory_operations"].append({
                **event_data,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 100 operations
            execution.input_data["recent_memory_operations"] = execution.input_data["recent_memory_operations"][-100:]
            
            # Update counter
            execution.input_data["memory_operation_count"] = execution.input_data.get("memory_operation_count", 0) + 1
        
        elif event_type == "communication":
            if "recent_communications" not in execution.input_data:
                execution.input_data["recent_communications"] = []
            execution.input_data["recent_communications"].append({
                **event_data,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 100 messages
            execution.input_data["recent_communications"] = execution.input_data["recent_communications"][-100:]
            
            # Update counter
            execution.input_data["message_count"] = execution.input_data.get("message_count", 0) + 1
        
        elif event_type == "learning_insight":
            if "learning_insights" not in execution.input_data:
                execution.input_data["learning_insights"] = []
            execution.input_data["learning_insights"].append({
                **event_data,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update counter
            execution.input_data["patterns_learned"] = execution.input_data.get("patterns_learned", 0) + 1
        
        elif event_type == "metric_update":
            # Update metrics directly
            for key, value in event_data.items():
                execution.input_data[key] = value
        
        # Mark the object as modified so SQLAlchemy knows to update it
        from sqlalchemy.orm import attributes
        attributes.flag_modified(execution, "input_data")
        
        # Commit the changes
        db.commit()
        
        return {"status": "success", "message": f"Event {event_type} stored successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing orchestration event: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
