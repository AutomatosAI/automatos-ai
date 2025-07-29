
"""
Workflow Management API Routes
=============================

REST API endpoints for managing workflows and workflow executions.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_

from database import get_db
from models import (
    Workflow, WorkflowExecution, Agent, workflow_agents,
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    WorkflowExecutionCreate, WorkflowExecutionResponse,
    WorkflowStatus, ExecutionStatus
)
from websocket_manager import manager
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["workflows"])

# Workflow endpoints
@router.post("/", response_model=WorkflowResponse)
async def create_workflow(workflow_data: WorkflowCreate, db: Session = Depends(get_db)):
    """Create a new workflow"""
    try:
        # Create workflow
        workflow = Workflow(
            name=workflow_data.name,
            description=workflow_data.description,
            workflow_definition=workflow_data.workflow_definition,
            created_by="system"  # TODO: Get from auth context
        )
        
        db.add(workflow)
        db.flush()  # Get the ID
        
        # Add agents if provided
        if workflow_data.agent_ids:
            agents = db.query(Agent).filter(Agent.id.in_(workflow_data.agent_ids)).all()
            workflow.agents.extend(agents)
        
        db.commit()
        db.refresh(workflow)
        
        # Load agents for response
        workflow_with_agents = db.query(Workflow).options(joinedload(Workflow.agents)).filter(Workflow.id == workflow.id).first()
        
        return WorkflowResponse(
            id=workflow_with_agents.id,
            name=workflow_with_agents.name,
            description=workflow_with_agents.description,
            workflow_definition=workflow_with_agents.workflow_definition,
            status=workflow_with_agents.status,
            created_at=workflow_with_agents.created_at,
            updated_at=workflow_with_agents.updated_at,
            created_by=workflow_with_agents.created_by,
            agents=[{
                "id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "status": agent.status
            } for agent in workflow_with_agents.agents]
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating workflow: {str(e)}")

@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[WorkflowStatus] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List workflows with filtering and pagination"""
    try:
        query = db.query(Workflow).options(joinedload(Workflow.agents))
        
        # Apply filters
        if status:
            query = query.filter(Workflow.status == status.value)
        if search:
            query = query.filter(
                or_(
                    Workflow.name.ilike(f"%{search}%"),
                    Workflow.description.ilike(f"%{search}%")
                )
            )
        
        workflows = query.offset(skip).limit(limit).all()
        
        return [
            WorkflowResponse(
                id=workflow.id,
                name=workflow.name,
                description=workflow.description,
                workflow_definition=workflow.workflow_definition,
                status=workflow.status,
                created_at=workflow.created_at,
                updated_at=workflow.updated_at,
                created_by=workflow.created_by,
                agents=[{
                    "id": agent.id,
                    "name": agent.name,
                    "agent_type": agent.agent_type,
                    "status": agent.status
                } for agent in workflow.agents]
            ) for workflow in workflows
        ]
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing workflows: {str(e)}")

@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Get workflow by ID"""
    try:
        workflow = db.query(Workflow).options(joinedload(Workflow.agents)).filter(Workflow.id == workflow_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            workflow_definition=workflow.workflow_definition,
            status=workflow.status,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            created_by=workflow.created_by,
            agents=[{
                "id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "status": agent.status
            } for agent in workflow.agents]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting workflow: {str(e)}")

@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(workflow_id: int, workflow_data: WorkflowUpdate, db: Session = Depends(get_db)):
    """Update workflow"""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Update fields
        if workflow_data.name is not None:
            workflow.name = workflow_data.name
        if workflow_data.description is not None:
            workflow.description = workflow_data.description
        if workflow_data.workflow_definition is not None:
            workflow.workflow_definition = workflow_data.workflow_definition
        if workflow_data.status is not None:
            workflow.status = workflow_data.status.value
        
        # Update agents
        if workflow_data.agent_ids is not None:
            workflow.agents.clear()
            if workflow_data.agent_ids:
                agents = db.query(Agent).filter(Agent.id.in_(workflow_data.agent_ids)).all()
                workflow.agents.extend(agents)
        
        db.commit()
        db.refresh(workflow)
        
        # Load agents for response
        workflow_with_agents = db.query(Workflow).options(joinedload(Workflow.agents)).filter(Workflow.id == workflow.id).first()
        
        return WorkflowResponse(
            id=workflow_with_agents.id,
            name=workflow_with_agents.name,
            description=workflow_with_agents.description,
            workflow_definition=workflow_with_agents.workflow_definition,
            status=workflow_with_agents.status,
            created_at=workflow_with_agents.created_at,
            updated_at=workflow_with_agents.updated_at,
            created_by=workflow_with_agents.created_by,
            agents=[{
                "id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "status": agent.status
            } for agent in workflow_with_agents.agents]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating workflow: {str(e)}")

@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Delete workflow"""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        db.delete(workflow)
        db.commit()
        
        return {"message": "Workflow deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting workflow: {str(e)}")

# Workflow execution endpoints
@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: int, 
    execution_data: WorkflowExecutionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Execute a workflow"""
    try:
        # Validate workflow exists
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Validate agent exists
        agent = db.query(Agent).filter(Agent.id == execution_data.agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            agent_id=execution_data.agent_id,
            input_data=execution_data.input_data or {},
            status=ExecutionStatus.PENDING.value
        )
        
        db.add(execution)
        db.commit()
        db.refresh(execution)
        
        # Start execution in background
        background_tasks.add_task(run_workflow_execution, execution.id)
        
        return WorkflowExecutionResponse(
            id=execution.id,
            workflow_id=execution.workflow_id,
            agent_id=execution.agent_id,
            status=execution.status,
            input_data=execution.input_data,
            output_data=execution.output_data,
            execution_log=execution.execution_log,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            error_message=execution.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing workflow: {str(e)}")

@router.get("/{workflow_id}/executions", response_model=List[WorkflowExecutionResponse])
async def list_workflow_executions(
    workflow_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[ExecutionStatus] = None,
    db: Session = Depends(get_db)
):
    """List workflow executions"""
    try:
        query = db.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id)
        
        if status:
            query = query.filter(WorkflowExecution.status == status.value)
        
        executions = query.order_by(WorkflowExecution.started_at.desc()).offset(skip).limit(limit).all()
        
        return [
            WorkflowExecutionResponse(
                id=execution.id,
                workflow_id=execution.workflow_id,
                agent_id=execution.agent_id,
                status=execution.status,
                input_data=execution.input_data,
                output_data=execution.output_data,
                execution_log=execution.execution_log,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                error_message=execution.error_message
            ) for execution in executions
        ]
        
    except Exception as e:
        logger.error(f"Error listing workflow executions for {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing executions: {str(e)}")

@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(execution_id: int, db: Session = Depends(get_db)):
    """Get workflow execution by ID"""
    try:
        execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return WorkflowExecutionResponse(
            id=execution.id,
            workflow_id=execution.workflow_id,
            agent_id=execution.agent_id,
            status=execution.status,
            input_data=execution.input_data,
            output_data=execution.output_data,
            execution_log=execution.execution_log,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            error_message=execution.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution {execution_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting execution: {str(e)}")

async def run_workflow_execution(execution_id: int):
    """Background task to run workflow execution"""
    from database import get_db_session
    
    try:
        with get_db_session() as db:
            execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
            if not execution:
                return
            
            # Update status to running
            execution.status = ExecutionStatus.RUNNING.value
            db.commit()
            
            # Send WebSocket update
            await manager.broadcast({
                "type": "execution_status",
                "data": {
                    "execution_id": execution_id,
                    "status": "running",
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Simulate workflow execution (replace with actual logic)
            await asyncio.sleep(2)  # Simulate processing time
            
            # Update execution with results
            execution.status = ExecutionStatus.COMPLETED.value
            execution.completed_at = datetime.now()
            execution.output_data = {
                "result": "Workflow completed successfully",
                "processed_steps": len(execution.workflow.workflow_definition.get("steps", [])),
                "execution_time": "2.1s"
            }
            execution.execution_log = "Workflow executed successfully with all steps completed."
            
            db.commit()
            
            # Send completion WebSocket update
            await manager.broadcast({
                "type": "execution_completed",
                "data": {
                    "execution_id": execution_id,
                    "status": "completed",
                    "output": execution.output_data,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
    except Exception as e:
        logger.error(f"Error in workflow execution {execution_id}: {e}")
        
        try:
            with get_db_session() as db:
                execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
                if execution:
                    execution.status = ExecutionStatus.FAILED.value
                    execution.completed_at = datetime.now()
                    execution.error_message = str(e)
                    db.commit()
                    
                    # Send error WebSocket update
                    await manager.broadcast({
                        "type": "execution_failed",
                        "data": {
                            "execution_id": execution_id,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                    })
        except Exception as inner_e:
            logger.error(f"Error updating failed execution {execution_id}: {inner_e}")
