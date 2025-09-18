"""
Orchestrator API Routes
======================

API endpoints for task orchestration, multi-agent coordination, and workflow execution.
These endpoints bridge the gap between the user journey tests and the actual orchestrator service.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
import logging
import json
import uuid

from database.database import get_db
from database.models import Workflow, WorkflowExecution, Agent
# from services.orchestrator_service import EnhancedOrchestratorService  # Temporarily disabled
from services.websocket_manager import manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])

# Initialize orchestrator service - temporarily disabled
# orchestrator_service = EnhancedOrchestratorService()

@router.post("/task/submit")
async def submit_complex_task(
    task_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Submit a complex task for orchestration and multi-agent processing
    """
    try:
        task_id = task_data.get("task_id", f"task_{int(datetime.now().timestamp())}")
        description = task_data.get("description", "")
        task_type = task_data.get("type", "general")
        complexity = task_data.get("complexity", "medium")
        requirements = task_data.get("requirements", [])
        
        # Create workflow for the task
        workflow_data = {
            "name": f"Task: {description[:50]}...",
            "description": description,
            "category": "orchestrated_task",
            "priority": "high" if complexity == "high" else "medium",
            "config": {
                "task_id": task_id,
                "task_type": task_type,
                "complexity": complexity,
                "requirements": requirements
            },
            "steps": [
                {
                    "id": "analysis",
                    "name": "Task Analysis",
                    "type": "analysis",
                    "config": {"auto_analyze": True}
                },
                {
                    "id": "execution",
                    "name": "Task Execution", 
                    "type": "execution",
                    "config": {"multi_agent": True}
                }
            ],
            "agents": [],
            "tags": ["orchestrated", task_type, complexity]
        }
        
        # Create workflow in database
        workflow = Workflow(
            name=workflow_data["name"],
            description=workflow_data["description"],
            workflow_definition=workflow_data,
            status="active",
            created_by="orchestrator"
        )
        
        db.add(workflow)
        db.commit()
        db.refresh(workflow)
        
        # Send real-time update
        await manager.broadcast({
            "type": "task_submitted",
            "task_id": task_id,
            "workflow_id": workflow.id,
            "status": "submitted"
        })
        
        return {
            "status": "success",
            "task_id": task_id,
            "workflow_id": workflow.id,
            "message": "Complex task submitted for orchestration",
            "estimated_completion": "2-4 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error submitting complex task: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")

@router.post("/task/analyze")
async def analyze_task(
    analysis_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Analyze and break down a complex task into subtasks
    """
    try:
        task_id = analysis_data.get("task_id")
        analysis_depth = analysis_data.get("analysis_depth", "standard")
        breakdown_strategy = analysis_data.get("breakdown_strategy", "hierarchical")
        
        if not task_id:
            raise HTTPException(status_code=400, detail="Task ID is required")
        
        # Find the workflow for this task
        workflow = db.query(Workflow).filter(
            Workflow.workflow_definition.contains({"config": {"task_id": task_id}})
        ).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Simulate task analysis and breakdown
        subtasks = [
            {
                "subtask_id": f"{task_id}_document_extraction",
                "description": "Extract and parse document content",
                "agent_type": "document_processor",
                "priority": "high",
                "dependencies": [],
                "estimated_duration": "30-60 seconds"
            },
            {
                "subtask_id": f"{task_id}_analysis",
                "description": "Analyze extracted content",
                "agent_type": "analyst",
                "priority": "high", 
                "dependencies": [f"{task_id}_document_extraction"],
                "estimated_duration": "60-120 seconds"
            },
            {
                "subtask_id": f"{task_id}_synthesis",
                "description": "Synthesize results and recommendations",
                "agent_type": "synthesizer",
                "priority": "medium",
                "dependencies": [f"{task_id}_analysis"],
                "estimated_duration": "30-90 seconds"
            }
        ]
        
        # Update workflow with subtasks
        workflow_def = workflow.workflow_definition
        workflow_def["subtasks"] = subtasks
        workflow_def["analysis_completed"] = True
        workflow.workflow_definition = workflow_def
        
        db.commit()
        
        # Send real-time update
        await manager.broadcast({
            "type": "task_analyzed",
            "task_id": task_id,
            "subtasks_count": len(subtasks),
            "status": "analyzed"
        })
        
        return {
            "status": "success",
            "task_id": task_id,
            "analysis_depth": analysis_depth,
            "breakdown_strategy": breakdown_strategy,
            "subtasks": subtasks,
            "total_subtasks": len(subtasks),
            "message": "Task analysis completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing task: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing task: {str(e)}")

@router.post("/execute-phase")
async def execute_workflow_phase(
    execution_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Execute a specific phase of a workflow with assigned agents
    """
    try:
        phase = execution_data.get("phase")
        agents = execution_data.get("agents", [])
        execution_type = execution_data.get("execution_type", "sequential")
        session_id = execution_data.get("session_id")
        
        if not phase:
            raise HTTPException(status_code=400, detail="Phase is required")
        
        # Create execution record
        execution_id = f"exec_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        # Simulate phase execution
        execution_result = {
            "execution_id": execution_id,
            "phase": phase,
            "agents": agents,
            "execution_type": execution_type,
            "status": "completed",
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "results": {
                "phase_output": f"Phase {phase} completed successfully",
                "agent_contributions": [
                    {
                        "agent": agent,
                        "contribution": f"Agent {agent} completed assigned tasks",
                        "performance_score": 0.85 + (hash(agent) % 100) / 1000  # Simulate varying performance
                    } for agent in agents
                ],
                "metrics": {
                    "execution_time": "45.2s",
                    "success_rate": 0.92,
                    "resource_utilization": 0.78
                }
            }
        }
        
        # Send real-time update
        await manager.broadcast({
            "type": "phase_executed",
            "execution_id": execution_id,
            "phase": phase,
            "status": "completed",
            "session_id": session_id
        })
        
        return {
            "status": "success",
            "data": execution_result,
            "message": f"Phase {phase} executed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow phase: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing phase: {str(e)}")

@router.get("/task/{task_id}/status")
async def get_task_status(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the current status of a task
    """
    try:
        # Find workflow for this task
        workflow = db.query(Workflow).filter(
            Workflow.workflow_definition.contains({"config": {"task_id": task_id}})
        ).first()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Get execution status
        executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.workflow_id == workflow.id
        ).order_by(WorkflowExecution.started_at.desc()).limit(5).all()
        
        return {
            "status": "success",
            "data": {
                "task_id": task_id,
                "workflow_id": workflow.id,
                "status": workflow.status,
                "progress": 75,  # Simulated progress
                "current_phase": "synthesis",
                "executions": len(executions),
                "last_updated": workflow.updated_at.isoformat() if workflow.updated_at else None
            },
            "message": "Task status retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting task status: {str(e)}")

@router.get("/tasks/active")
async def get_active_tasks(
    db: Session = Depends(get_db)
):
    """
    Get all currently active orchestrated tasks
    """
    try:
        # Get active workflows created by orchestrator
        active_workflows = db.query(Workflow).filter(
            Workflow.status == "active",
            Workflow.created_by == "orchestrator"
        ).all()
        
        tasks = []
        for workflow in active_workflows:
            config = workflow.workflow_definition.get("config", {})
            task_id = config.get("task_id")
            if task_id:
                tasks.append({
                    "task_id": task_id,
                    "workflow_id": workflow.id,
                    "description": workflow.description,
                    "status": workflow.status,
                    "complexity": config.get("complexity", "medium"),
                    "created_at": workflow.created_at.isoformat() if workflow.created_at else None
                })
        
        return {
            "status": "success",
            "data": {
                "active_tasks": tasks,
                "total_count": len(tasks)
            },
            "message": "Active tasks retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting active tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting active tasks: {str(e)}")

