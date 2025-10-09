"""
Execution History API Endpoints
================================

Provides detailed execution history and analytics for the UI components.
Fixes the issue of execution data not reaching frontend properly.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from database.database import get_db
from database.models import WorkflowExecution, Workflow, Agent
from services.workflow_analytics_service import WorkflowAnalyticsService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/execution-history", tags=["execution-history"])


@router.get("/workflow/{workflow_id}/latest")
async def get_latest_execution(workflow_id: int, db: Session = Depends(get_db)):
    """Get the most recent execution for a workflow with full details"""
    try:
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.workflow_id == workflow_id
        ).order_by(desc(WorkflowExecution.started_at)).first()
        
        if not execution:
            return {"message": "No executions found for this workflow"}
        
        return format_execution_details(execution, db)
        
    except Exception as e:
        logger.error(f"Error getting latest execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{workflow_id}/all")
async def get_all_executions(
    workflow_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get all executions for a workflow with pagination"""
    try:
        executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.workflow_id == workflow_id
        ).order_by(desc(WorkflowExecution.started_at)).offset(skip).limit(limit).all()
        
        total = db.query(WorkflowExecution).filter(
            WorkflowExecution.workflow_id == workflow_id
        ).count()
        
        return {
            "executions": [format_execution_summary(e, db) for e in executions],
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/{execution_id}/details")
async def get_execution_details(execution_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific execution"""
    try:
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.id == execution_id
        ).first()
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return format_execution_details(execution, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/execution/{execution_id}/stages")
async def get_execution_stages(execution_id: int, db: Session = Depends(get_db)):
    """Get the 9-stage pipeline details for an execution"""
    try:
        execution = db.query(WorkflowExecution).filter(
            WorkflowExecution.id == execution_id
        ).first()
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        input_data = execution.input_data or {}
        output_data = execution.output_data or {}
        
        stages = [
            {
                "stage": 1,
                "name": "Task Decomposition",
                "status": "completed" if input_data.get("decomposition", {}).get("is_real") else "skipped",
                "data": input_data.get("decomposition", {}),
                "metrics": {
                    "task_count": input_data.get("decomposition", {}).get("task_count", 0),
                    "execution_strategy": input_data.get("decomposition", {}).get("execution_strategy", "parallel")
                }
            },
            {
                "stage": 2,
                "name": "Agent Selection",
                "status": "completed" if input_data.get("agent_selection", {}).get("is_real") else "skipped",
                "data": input_data.get("agent_selection", {}),
                "metrics": {
                    "unique_agents": input_data.get("agent_selection", {}).get("summary", {}).get("unique_agents", 0),
                    "avg_match_score": input_data.get("agent_selection", {}).get("summary", {}).get("avg_match_score", 0)
                }
            },
            {
                "stage": 3,
                "name": "Context Engineering",
                "status": "completed" if input_data.get("context_engineering", {}).get("is_real") else "skipped",
                "data": input_data.get("context_engineering", {}),
                "metrics": {
                    "total_sources": input_data.get("context_engineering", {}).get("summary", {}).get("total_sources_retrieved", 0),
                    "avg_quality": input_data.get("context_engineering", {}).get("summary", {}).get("avg_context_quality", 0)
                }
            },
            {
                "stage": 4,
                "name": "Agent Execution",
                "status": "completed" if input_data.get("agent_execution", {}).get("is_real") else "skipped",
                "data": input_data.get("agent_execution", {}),
                "metrics": {
                    "success_rate": input_data.get("agent_execution", {}).get("summary", {}).get("success_rate", 0),
                    "total_tokens": input_data.get("agent_execution", {}).get("summary", {}).get("total_tokens_used", 0),
                    "execution_time_ms": input_data.get("agent_execution", {}).get("summary", {}).get("total_execution_time_ms", 0)
                }
            },
            {
                "stage": 5,
                "name": "Result Aggregation",
                "status": "completed" if input_data.get("result_aggregation", {}).get("is_real") else "skipped",
                "data": input_data.get("result_aggregation", {}),
                "quality_scores": input_data.get("result_aggregation", {}).get("quality_scores", {})
            },
            {
                "stage": 6,
                "name": "Learning Update",
                "status": "completed" if input_data.get("learning_system", {}).get("is_real") else "skipped",
                "data": input_data.get("learning_system", {}),
                "metrics": {
                    "total_updates": input_data.get("learning_system", {}).get("updates", {}).get("total_updates", 0),
                    "agents_improved": len(input_data.get("learning_system", {}).get("updates", {}).get("agent_performance_updates", []))
                }
            },
            {
                "stage": 7,
                "name": "Quality Assessment",
                "status": "completed" if output_data.get("quality_scores") else "skipped",
                "quality_scores": output_data.get("quality_scores", {})
            },
            {
                "stage": 8,
                "name": "Memory Storage",
                "status": "completed" if input_data.get("memory_storage", {}).get("is_real") else "skipped",
                "data": input_data.get("memory_storage", {}),
                "metrics": {
                    "experiences_stored": input_data.get("memory_storage", {}).get("results", {}).get("total_experiences", 0),
                    "agents_with_memory": len(input_data.get("memory_storage", {}).get("results", {}).get("per_agent", {}))
                }
            },
            {
                "stage": 9,
                "name": "Memory Consolidation",
                "status": "completed" if input_data.get("memory_consolidation", {}).get("is_real") else "skipped",
                "data": input_data.get("memory_consolidation", {}),
                "metrics": {
                    "patterns_extracted": input_data.get("memory_consolidation", {}).get("results", {}).get("patterns_extracted", 0),
                    "agents_consolidated": len(input_data.get("memory_consolidation", {}).get("results", {}).get("agents_consolidated", []))
                }
            }
        ]
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "stages": stages,
            "overall_status": execution.status,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution stages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def format_execution_summary(execution: WorkflowExecution, db: Session) -> Dict[str, Any]:
    """Format execution data for summary view"""
    workflow = db.query(Workflow).filter(Workflow.id == execution.workflow_id).first()
    
    input_data = execution.input_data or {}
    output_data = execution.output_data or {}
    
    return {
        "id": execution.id,
        "workflow_name": workflow.name if workflow else "Unknown",
        "status": execution.status,
        "started_at": execution.started_at.isoformat() if execution.started_at else None,
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "duration_seconds": (
            (execution.completed_at - execution.started_at).total_seconds()
            if execution.completed_at and execution.started_at else None
        ),
        "quality_score": output_data.get("quality_scores", {}).get("overall", 0),
        "subtasks_completed": output_data.get("steps_completed", 0),
        "tokens_used": input_data.get("agent_execution", {}).get("summary", {}).get("total_tokens_used", 0),
        "agents_used": input_data.get("agent_selection", {}).get("summary", {}).get("unique_agents", 0)
    }


def format_execution_details(execution: WorkflowExecution, db: Session) -> Dict[str, Any]:
    """Format execution data for detailed view"""
    workflow = db.query(Workflow).filter(Workflow.id == execution.workflow_id).first()
    
    input_data = execution.input_data or {}
    output_data = execution.output_data or {}
    
    # Get agent details
    agent_assignments = input_data.get("agent_selection", {}).get("assignments", {})
    agents_used = set()
    for subtask_id, matches in agent_assignments.items():
        for match in matches:
            agents_used.add(match.get("agent_name"))
    
    # Extract subtask results
    subtasks = output_data.get("subtasks", [])
    
    # Memory integration summary
    memory_summary = input_data.get("memory_integration_summary", {})
    
    return {
        "id": execution.id,
        "workflow": {
            "id": workflow.id if workflow else None,
            "name": workflow.name if workflow else "Unknown",
            "description": workflow.description if workflow else None
        },
        "status": execution.status,
        "started_at": execution.started_at.isoformat() if execution.started_at else None,
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "duration_seconds": (
            (execution.completed_at - execution.started_at).total_seconds()
            if execution.completed_at and execution.started_at else None
        ),
        "execution_log": execution.execution_log,
        "error_message": execution.error_message,
        "quality_scores": output_data.get("quality_scores", {}),
        "metrics": {
            "subtasks_total": len(subtasks),
            "subtasks_completed": sum(1 for s in subtasks if s.get("completed")),
            "agents_used": len(agents_used),
            "agents_list": list(agents_used),
            "tokens_used": input_data.get("agent_execution", {}).get("summary", {}).get("total_tokens_used", 0),
            "execution_time_ms": input_data.get("agent_execution", {}).get("summary", {}).get("total_execution_time_ms", 0),
            "cost": execution.metadata.get("analytics", {}).get("total_cost", 0) if execution.metadata else 0
        },
        "memory_integration": {
            "memories_retrieved": memory_summary.get("memories_retrieved", 0),
            "experiences_stored": memory_summary.get("experiences_stored", 0),
            "patterns_extracted": memory_summary.get("patterns_extracted", 0),
            "knowledge_nodes_created": memory_summary.get("knowledge_nodes_created", 0)
        },
        "stages_completed": {
            "task_decomposition": input_data.get("decomposition", {}).get("is_real", False),
            "agent_selection": input_data.get("agent_selection", {}).get("is_real", False),
            "context_engineering": input_data.get("context_engineering", {}).get("is_real", False),
            "agent_execution": input_data.get("agent_execution", {}).get("is_real", False),
            "result_aggregation": input_data.get("result_aggregation", {}).get("is_real", False),
            "learning_update": input_data.get("learning_system", {}).get("is_real", False),
            "memory_storage": input_data.get("memory_storage", {}).get("is_real", False),
            "memory_consolidation": input_data.get("memory_consolidation", {}).get("is_real", False)
        },
        "subtasks": subtasks,
        "input_data": input_data,
        "output_data": output_data
    }
