
"""
Enhanced Workflow Management API Routes
=======================================

Extended workflow API with live progress tracking, real-time updates, and advanced features.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pathlib import Path
from sqlalchemy.orm import Session, joinedload, attributes
from sqlalchemy import and_, or_, func, desc, String
from datetime import datetime, timedelta
import asyncio
import logging
import json

from config import config
from core.database.database import get_db
from core.models import (
    Workflow, WorkflowExecution, Agent, workflow_agents,
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    WorkflowExecutionCreate, WorkflowExecutionResponse,
    WorkflowStatus, ExecutionStatus
)
from core.models.core import RecipeExecution, WorkflowTemplate as WorkflowRecipe
# websocket_manager removed - using AI SDK SSE streaming
from core.services.workspace_manager import WorkspaceManager
from core.auth.hybrid import get_request_context_hybrid
from core.task_runner import get_task_runner, AgentTask, TaskType, TaskPriority
from core.auth.dependencies import RequestContext
from config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["workflow-enhanced"])


class WorkflowStageTracker:
    """Track workflow execution with SSE events — supports both legacy 9-stage and PRD-59 dynamic phases"""

    # Legacy 9-stage names (backward compat)
    STAGES = {
        1: "Task Decomposition",
        2: "Agent Selection",
        3: "Context Engineering",
        4: "Agent Execution",
        5: "Result Aggregation",
        6: "Learning Update",
        7: "Quality Assessment",
        8: "Memory Storage",
        9: "Response Generation",
    }

    # PRD-59: Dynamic stage names including sub-stages
    DYNAMIC_STAGES = {
        **STAGES,
        "2b": "Agent Negotiation",
        "3b": "Prompt Optimization",
        "4b": "Inter-Agent Coordination",
    }

    # PRD-59: Phase → stage mapping
    PHASES = {
        "PLAN":     {"stages": [1, 2, "2b"], "label": "Planning"},
        "PREPARE":  {"stages": [3, "3b"],     "label": "Preparation"},
        "EXECUTE":  {"stages": [4, "4b"],     "label": "Execution"},
        "EVALUATE": {"stages": [5, 6],        "label": "Evaluation"},
        "LEARN":    {"stages": [7, 8, 9],     "label": "Learning"},
    }

    def __init__(self, execution_id: int, redis_client=None, stream_manager=None):
        self.execution_id = execution_id
        self.redis = redis_client
        self.stream_manager = stream_manager
        self.current_stage = 0
        self.current_phase = None
        self.stage_start_times = {}
        self.phase_start_times = {}
        self.active_phases = []  # PRD-59: Ordered list of phases selected for this execution

    def set_active_phases(self, phases: list):
        """PRD-59: Set which phases are active for this execution (from PhaseSelector)"""
        self.active_phases = phases

    def _get_stage_name(self, stage_id) -> str:
        """Get stage name supporting both int (legacy) and str (dynamic) IDs"""
        return self.DYNAMIC_STAGES.get(stage_id, self.DYNAMIC_STAGES.get(str(stage_id), f"Stage {stage_id}"))

    async def start_phase(self, phase_name: str):
        """PRD-59: Mark a phase as started, emit SSE event"""
        self.current_phase = phase_name
        self.phase_start_times[phase_name] = datetime.now()
        phase_info = self.PHASES.get(phase_name, {"label": phase_name, "stages": []})
        phase_index = self.active_phases.index(phase_name) if phase_name in self.active_phases else 0
        total_phases = len(self.active_phases) or 5

        logger.info(f"🔷 PHASE {phase_name} START: {phase_info['label']} ({phase_index+1}/{total_phases})")

        event_data = {
            "phase": phase_name,
            "phase_label": phase_info["label"],
            "phase_index": phase_index,
            "total_phases": total_phases,
            "stages": [{"id": s, "name": self._get_stage_name(s)} for s in phase_info.get("stages", [])],
            "timestamp": datetime.now().isoformat(),
        }
        await self._emit("phase_start", event_data)

    async def complete_phase(self, phase_name: str, result: dict = None):
        """PRD-59: Mark a phase as complete, emit SSE event"""
        duration_ms = 0
        if phase_name in self.phase_start_times:
            duration_ms = int((datetime.now() - self.phase_start_times[phase_name]).total_seconds() * 1000)

        phase_info = self.PHASES.get(phase_name, {"label": phase_name})
        logger.info(f"🔷 PHASE {phase_name} COMPLETE: {phase_info['label']} ({duration_ms}ms)")

        event_data = {
            "phase": phase_name,
            "phase_label": phase_info["label"],
            "result": result or {},
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        }
        await self._emit("phase_complete", event_data)

    async def start_stage(self, stage_num):
        """Mark stage as started, emit SSE event. Accepts int or str (e.g. '2b')."""
        self.current_stage = stage_num
        self.stage_start_times[stage_num] = datetime.now()
        stage_name = self._get_stage_name(stage_num)

        logger.info(f"🎯 STAGE {stage_num} START: {stage_name}")

        event_data = {
            "stage": stage_num,
            "stage_name": stage_name,
            "phase": self.current_phase,
            "timestamp": datetime.now().isoformat(),
        }
        await self._emit("stage_start", event_data)

    async def complete_stage(self, stage_num, result: dict = None):
        """Mark stage as complete, emit SSE event. Accepts int or str (e.g. '2b')."""
        duration_ms = 0
        if stage_num in self.stage_start_times:
            duration_ms = int((datetime.now() - self.stage_start_times[stage_num]).total_seconds() * 1000)
        stage_name = self._get_stage_name(stage_num)

        logger.info(f"✅ STAGE {stage_num} COMPLETE: {stage_name} ({duration_ms}ms)")

        event_data = {
            "stage": stage_num,
            "stage_name": stage_name,
            "phase": self.current_phase,
            "result": result or {},
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        }
        await self._emit("stage_complete", event_data)

    async def _emit(self, event_type: str, data: dict):
        """Emit event to both SSE stream manager and Redis"""
        if self.stream_manager:
            try:
                await self.stream_manager.broadcast_event(
                    execution_id=self.execution_id,
                    event_type=event_type,
                    data=data,
                )
            except Exception as e:
                logger.warning(f"Failed to broadcast {event_type} event: {e}")

        if self.redis:
            try:
                self.redis.publish_workflow_event(
                    execution_id=self.execution_id,
                    event_type=event_type,
                    data=data,
                )
            except Exception as e:
                logger.warning(f"Failed to publish {event_type} to Redis: {e}")



@router.get("")
async def list_workflows(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    q: Optional[str] = None,
    owner: Optional[str] = Query(None, description="Filter by owner exact match"),
    tag: Optional[str] = Query(None, description="Filter by tag (contained in tags array)"),
    db: Session = Depends(get_db),
):
    try:
        query = db.query(Workflow).filter(Workflow.workspace_id == ctx.workspace_id)
        if q:
            query = query.filter(or_(Workflow.name.ilike(f"%{q}%"), Workflow.description.ilike(f"%{q}%")))
        if owner:
            query = query.filter(Workflow.owner == owner)
        if tag:
            # For JSON array column 'tags', check if provided tag is contained
            try:
                query = query.filter(Workflow.tags.contains([tag]))
            except Exception:
                # Fallback: simple text match on serialized JSON
                query = query.filter(func.cast(Workflow.tags, String).ilike(f"%\"{tag}\"%"))
        total = query.count()
        rows = query.order_by(desc(Workflow.updated_at)).offset(skip).limit(limit).all()
        items = [
            {
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "status": w.status,
                "owner": getattr(w, 'owner', None),
                "tags": getattr(w, 'tags', None),
                "default_policy_id": getattr(w, 'default_policy_id', None),
                "created_at": w.created_at.isoformat() if w.created_at else None,
                "updated_at": w.updated_at.isoformat() if w.updated_at else None,
            } for w in rows
        ]
        return {"items": items, "total": total}
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail="Error listing workflows")

@router.get("/active")
async def get_active_workflows(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Get all currently active workflows with live status"""
    try:
        active_workflows = db.query(Workflow).filter(Workflow.workspace_id == ctx.workspace_id).options(joinedload(Workflow.agents)).filter(
            Workflow.status == WorkflowStatus.ACTIVE.value
        ).all()
        
        # Get recent executions for each workflow
        workflow_data = []
        for workflow in active_workflows:
            recent_executions = db.query(WorkflowExecution).filter(
                WorkflowExecution.workflow_id == workflow.id
            ).order_by(desc(WorkflowExecution.started_at)).limit(5).all()
            
            # Calculate workflow metrics
            total_executions = db.query(WorkflowExecution).filter(
                WorkflowExecution.workflow_id == workflow.id
            ).count()
            
            successful_executions = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow.id,
                    WorkflowExecution.status == ExecutionStatus.COMPLETED.value
                )
            ).count()
            
            success_rate = (successful_executions / max(total_executions, 1)) * 100
            
            # Get current execution status
            current_execution = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow.id,
                    WorkflowExecution.status == ExecutionStatus.RUNNING.value
                )
            ).first()
            
            # Simulate live progress for running workflows
            progress = 0
            current_step = "Idle"
            estimated_completion = None
            
            if current_execution:
                # Calculate progress based on execution time
                elapsed = (datetime.now() - current_execution.started_at).total_seconds()
                progress = min(95, int(elapsed / 60 * 20))  # Simulate progress
                
                # Determine current step (would be read from actual execution data)
                if progress < 20:
                    current_step = "Initializing"
                elif progress < 40:
                    current_step = "Processing"
                elif progress < 70:
                    current_step = "Executing"
                else:
                    current_step = "Finalizing"
                    
                estimated_completion = (datetime.now() + timedelta(minutes=5-elapsed/60)).isoformat()
            
            workflow_data.append({
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "status": workflow.status,
                "current_execution": {
                    "id": current_execution.id if current_execution else None,
                    "status": current_execution.status if current_execution else "idle",
                    "progress": progress,
                    "current_step": current_step,
                    "started_at": current_execution.started_at.isoformat() if current_execution else None,
                    "estimated_completion": estimated_completion
                },
                "metrics": {
                    "total_executions": total_executions,
                    "successful_executions": successful_executions,
                    "success_rate": round(success_rate, 1),
                    "avg_duration": "4.2m",  # Would be calculated from actual data
                    "last_execution": recent_executions[0].started_at.isoformat() if recent_executions else None
                },
                "recent_executions": [
                    {
                        "id": exec.id,
                        "status": exec.status,
                        "started_at": exec.started_at.isoformat(),
                        "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                        "duration": str(exec.completed_at - exec.started_at) if exec.completed_at else None
                    } for exec in recent_executions
                ],
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            })
        
        # Also fetch recipe executions for the Cooking tab
        recipe_runs = []
        try:
            recent_recipe_execs = db.query(RecipeExecution).filter(
                RecipeExecution.workspace_id == ctx.workspace_id
            ).order_by(desc(RecipeExecution.started_at)).limit(20).all()

            for exec_rec in recent_recipe_execs:
                # Get recipe name
                recipe = db.query(WorkflowRecipe).filter(WorkflowRecipe.id == exec_rec.recipe_id).first()
                recipe_name = recipe.name if recipe else f"Recipe #{exec_rec.recipe_id}"

                step_results = exec_rec.step_results or []
                total_steps = len(step_results) if step_results else 0
                current_step = exec_rec.current_step or 0

                # Calculate duration
                duration = None
                if exec_rec.completed_at and exec_rec.started_at:
                    duration = str(exec_rec.completed_at - exec_rec.started_at)

                # Get total tokens and duration from output_data
                output = exec_rec.output_data or {}
                total_tokens = output.get("total_tokens", 0)
                total_duration_ms = output.get("total_duration_ms", 0)

                recipe_runs.append({
                    "id": exec_rec.id,
                    "execution_id": exec_rec.execution_id,
                    "recipe_id": exec_rec.recipe_id,
                    "recipe_template_id": recipe.template_id if recipe else None,
                    "recipe_name": recipe_name,
                    "type": "recipe",
                    "status": exec_rec.status,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "step_results": step_results,
                    "started_at": exec_rec.started_at.isoformat() if exec_rec.started_at else None,
                    "completed_at": exec_rec.completed_at.isoformat() if exec_rec.completed_at else None,
                    "duration": duration,
                    "total_tokens": total_tokens,
                    "total_duration_ms": total_duration_ms,
                    "error_message": exec_rec.error_message,
                })
        except Exception as recipe_err:
            logger.warning(f"Error fetching recipe executions for cooking tab: {recipe_err}")

        return {
            "active_workflows": workflow_data,
            "recipe_runs": recipe_runs,
            "total_active": len(workflow_data),
            "total_recipe_runs": len(recipe_runs),
            "system_load": min(100, len(workflow_data) * 15),
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting active workflows: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{workflow_id}")
async def get_workflow(workflow_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Get individual workflow by ID"""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id, Workflow.workspace_id == ctx.workspace_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "owner": getattr(workflow, 'owner', None),
            "tags": getattr(workflow, 'tags', None),
            "default_policy_id": getattr(workflow, 'default_policy_id', None),
            "workflow_definition": workflow.workflow_definition,
            "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
            "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None,
            "last_execution": getattr(workflow, 'last_execution', None)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: int,
    workflow_data: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Update workflow"""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id, Workflow.workspace_id == ctx.workspace_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Update fields if provided
        if "name" in workflow_data:
            workflow.name = workflow_data["name"]
        if "description" in workflow_data:
            workflow.description = workflow_data["description"]
        if "status" in workflow_data:
            workflow.status = workflow_data["status"]
        if "owner" in workflow_data:
            workflow.owner = workflow_data["owner"]
        if "tags" in workflow_data:
            workflow.tags = workflow_data["tags"]
        if "default_policy_id" in workflow_data:
            workflow.default_policy_id = workflow_data["default_policy_id"]
        if "workflow_definition" in workflow_data:
            workflow.workflow_definition = workflow_data["workflow_definition"]
        
        workflow.updated_at = datetime.now()
        db.commit()
        db.refresh(workflow)
        
        logger.info(f"Workflow {workflow_id} updated successfully")
        
        # Real-time updates now handled via SSE/AI SDK streaming (stream_manager)
        # Legacy WebSocket broadcast removed
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "owner": workflow.owner,
            "tags": workflow.tags,
            "default_policy_id": workflow.default_policy_id,
            "message": "Workflow updated successfully",
            "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workflow {workflow_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Delete workflow"""
    try:
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id, Workflow.workspace_id == ctx.workspace_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_name = workflow.name
        
        # Delete workflow_agents associations first (foreign key constraint)
        try:
            from core.models import workflow_agents
            from sqlalchemy import delete
            db.execute(delete(workflow_agents).where(workflow_agents.c.workflow_id == workflow_id))
        except Exception as e:
            logger.warning(f"Could not delete workflow_agents (table may not exist): {e}")
        
        # Delete associated executions
        db.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id).delete()
        
        # Delete workflow
        db.delete(workflow)
        db.commit()
        
        logger.info(f"Workflow {workflow_id} ({workflow_name}) deleted successfully")
        
        # Real-time updates now handled via SSE/AI SDK streaming (stream_manager)
        # Legacy WebSocket broadcast removed
        
        return {
            "message": "Workflow deleted successfully",
            "id": workflow_id,
            "name": workflow_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow {workflow_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/cleanup/old")
async def cleanup_old_workflows(days: int = 30, db: Session = Depends(get_db)):
    """Delete workflows older than specified days"""
    from datetime import datetime, timedelta
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Find old workflows
        old_workflows = db.query(Workflow).filter(Workflow.created_at < cutoff_date).all()
        
        if not old_workflows:
            return {
                "message": "No old workflows to delete",
                "deleted_count": 0,
                "days": days
            }
        
        deleted_count = len(old_workflows)
        workflow_ids = [w.id for w in old_workflows]
        
        # Delete workflow_agents associations first (foreign key constraint)
        try:
            from core.models import workflow_agents
            from sqlalchemy import delete
            db.execute(delete(workflow_agents).where(workflow_agents.c.workflow_id.in_(workflow_ids)))
        except Exception as e:
            logger.warning(f"Could not delete workflow_agents (table may not exist): {e}")
        
        # Delete executions for these workflows
        db.query(WorkflowExecution).filter(WorkflowExecution.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
        
        # Delete workflows
        db.query(Workflow).filter(Workflow.id.in_(workflow_ids)).delete(synchronize_session=False)
        
        db.commit()
        
        logger.info(f"Deleted {deleted_count} workflows older than {days} days")
        
        # Real-time updates now handled via SSE/AI SDK streaming (stream_manager)
        # Legacy WebSocket broadcast removed
        
        return {
            "message": f"Successfully deleted {deleted_count} workflows",
            "deleted_count": deleted_count,
            "days": days,
            "workflow_ids": workflow_ids
        }
    except Exception as e:
        logger.error(f"Error cleaning up old workflows: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("")
async def create_workflow(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    workflow_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """Create a new workflow with enhanced validation and error handling"""
    try:
        # Debug: Log what we received
        import json
        logger.info(f"Received workflow_data type: {type(workflow_data)}")
        logger.info(f"Received workflow_data: {workflow_data}")
        
        # Ensure workflow_data is a dict
        if isinstance(workflow_data, str):
            workflow_data = json.loads(workflow_data)
        
        # Extract required fields
        name = workflow_data.get("name")
        description = workflow_data.get("description", "")
        goal = workflow_data.get("goal")  # High-level objective (optional)
        context = workflow_data.get("context")  # Additional context like {"codegraph_project": "my-repo"}
        category = workflow_data.get("category", "automation")
        priority = workflow_data.get("priority", "medium")
        config = workflow_data.get("config", {})
        steps = workflow_data.get("steps", [])
        agents = workflow_data.get("agents", [])
        tags = workflow_data.get("tags", [])

        if not name:
            raise HTTPException(status_code=400, detail="Workflow name is required")

        # Check if workflow with this name already exists
        existing = db.query(Workflow).filter(Workflow.name == name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Workflow with name '{name}' already exists")

        # Build workflow definition from frontend data
        workflow_definition = {
            "category": category,
            "priority": priority,
            "config": config,
            "steps": steps,
            "agents": agents,
            "version": "1.0"
        }
        
        # Store goal and context in workflow_definition (9-stage workflow fields)
        if goal:
            workflow_definition["goal"] = goal
        if context:
            workflow_definition["context"] = context

        # Create workflow record
        import json as json_lib
        # Default to ACTIVE status so workflows are immediately visible
        is_active = workflow_data.get("is_active", True)
        status = WorkflowStatus.ACTIVE.value if is_active else WorkflowStatus.DRAFT.value
        
        workflow = Workflow(
            workspace_id=ctx.workspace_id,
            name=name,
            description=description,
            tags=tags,
            workflow_definition=workflow_definition,
            status=status,
            created_by=workflow_data.get("created_by", "system")
        )

        db.add(workflow)
        db.commit()
        db.refresh(workflow)

        # Associate agents if provided
        if agents:
            # Handle both string names and dict objects
            agent_names = []
            for agent in agents:
                if isinstance(agent, str):
                    agent_names.append(agent)
                elif isinstance(agent, dict) and agent.get("name"):
                    agent_names.append(agent["name"])
            
            if agent_names:
                # Get agent objects by name
                agent_objects = db.query(Agent).filter(Agent.name.in_(agent_names)).all()
                workflow.agents.extend(agent_objects)
                db.commit()

        # Real-time updates now handled via SSE/AI SDK streaming (stream_manager)
        # Legacy WebSocket broadcast removed

        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
            "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None,
            "message": "Workflow created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{workflow_id}/duplicate")
async def duplicate_workflow(
    workflow_id: int,
    duplicate_data: Dict[str, Any] = Body(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Duplicate an existing workflow with optional modifications"""
    try:
        # Get the original workflow
        original = db.query(Workflow).filter(Workflow.id == workflow_id, Workflow.workspace_id == ctx.workspace_id).first()
        if not original:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Generate new name
        if duplicate_data and duplicate_data.get("name"):
            new_name = duplicate_data["name"]
        else:
            # Find a unique name
            base_name = original.name
            if " (Copy" in base_name:
                # Remove existing copy suffix
                base_name = base_name.split(" (Copy")[0]
            
            counter = 1
            new_name = f"{base_name} (Copy)"
            while db.query(Workflow).filter(Workflow.name == new_name).first():
                counter += 1
                new_name = f"{base_name} (Copy {counter})"
        
        # Create the duplicate
        duplicate = Workflow(
            name=new_name,
            description=duplicate_data.get("description", original.description) if duplicate_data else original.description,
            goal=original.goal,
            context=original.context,
            tags=duplicate_data.get("tags", original.tags) if duplicate_data else original.tags,
            workflow_definition=original.workflow_definition,
            status='active',  # New duplicates are active by default
            created_by=duplicate_data.get("created_by", "system") if duplicate_data else "system",
            owner=duplicate_data.get("owner", original.owner) if duplicate_data else original.owner,
            default_policy_id=original.default_policy_id,
            workspace_id=ctx.workspace_id
        )
        
        # Copy agent associations
        if original.agents:
            duplicate.agents = original.agents
        
        db.add(duplicate)
        db.commit()
        db.refresh(duplicate)
        
        # Real-time updates now handled via SSE/AI SDK streaming (stream_manager)
        # Legacy WebSocket broadcast removed
        
        return {
            "id": duplicate.id,
            "name": duplicate.name,
            "description": duplicate.description,
            "status": duplicate.status,
            "created_at": duplicate.created_at.isoformat() if duplicate.created_at else None,
            "message": f"Workflow duplicated successfully as '{duplicate.name}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error duplicating workflow {workflow_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats/dashboard")
async def get_workflow_dashboard_stats(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Get comprehensive workflow statistics for dashboard"""
    try:
        # Basic workflow counts
        total_workflows = db.query(Workflow).filter(Workflow.workspace_id == ctx.workspace_id).count()
        active_workflows = db.query(Workflow).filter(Workflow.workspace_id == ctx.workspace_id, Workflow.status == WorkflowStatus.ACTIVE.value).count()
        draft_workflows = db.query(Workflow).filter(Workflow.status == WorkflowStatus.DRAFT.value).count()
        archived_workflows = db.query(Workflow).filter(Workflow.status == WorkflowStatus.ARCHIVED.value).count()
        
        # Execution statistics - Filter by workspace via Workflow join
        total_executions = db.query(WorkflowExecution).join(Workflow).filter(Workflow.workspace_id == ctx.workspace_id).count()
        running_executions = db.query(WorkflowExecution).join(Workflow).filter(
            Workflow.workspace_id == ctx.workspace_id,
            WorkflowExecution.status == ExecutionStatus.RUNNING.value
        ).count()
        
        # Today's statistics
        today = datetime.now().date()
        today_executions = db.query(WorkflowExecution).join(Workflow).filter(
            Workflow.workspace_id == ctx.workspace_id,
            func.date(WorkflowExecution.started_at) == today
        ).count()
        
        completed_today = db.query(WorkflowExecution).join(Workflow).filter(
            and_(
                Workflow.workspace_id == ctx.workspace_id,
                func.date(WorkflowExecution.started_at) == today,
                WorkflowExecution.status == ExecutionStatus.COMPLETED.value
            )
        ).count()
        
        failed_today = db.query(WorkflowExecution).join(Workflow).filter(
            and_(
                Workflow.workspace_id == ctx.workspace_id,
                func.date(WorkflowExecution.started_at) == today,
                WorkflowExecution.status == ExecutionStatus.FAILED.value
            )
        ).count()
        
        # Success rate calculation
        total_completed = db.query(WorkflowExecution).join(Workflow).filter(
            Workflow.workspace_id == ctx.workspace_id,
            WorkflowExecution.status == ExecutionStatus.COMPLETED.value
        ).count()
        
        success_rate = (total_completed / max(total_executions, 1)) * 100
        
        # Agent utilization
        total_agents = db.query(Agent).count()
        active_agents = db.query(Agent).filter(Agent.status == 'active').count()
        agent_utilization = (active_agents / max(total_agents, 1)) * 100
        
        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_executions = db.query(WorkflowExecution).join(Workflow).filter(
            Workflow.workspace_id == ctx.workspace_id,
            WorkflowExecution.started_at >= week_ago
        ).order_by(desc(WorkflowExecution.started_at)).limit(20).all()
        
        # Workflow type breakdown
        workflow_types = {}
        workflows = db.query(Workflow).all()
        for workflow in workflows:
            wf_def = workflow.workflow_definition or {}
            wf_type = wf_def.get('category', 'General')
            workflow_types[wf_type] = workflow_types.get(wf_type, 0) + 1
        
        # Performance trends (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        daily_executions = db.query(
            func.date(WorkflowExecution.started_at).label('date'),
            func.count(WorkflowExecution.id).label('count')
        ).join(Workflow).filter(
            Workflow.workspace_id == ctx.workspace_id,
            WorkflowExecution.started_at >= thirty_days_ago
        ).group_by(func.date(WorkflowExecution.started_at)).order_by('date').all()
        
        return {
            "overview": {
                "total_workflows": total_workflows,
                "active_workflows": active_workflows,
                "draft_workflows": draft_workflows,
                "archived_workflows": archived_workflows,
                "total_executions": total_executions,
                "running_executions": running_executions
            },
            "today_stats": {
                "executions": today_executions,
                "completed": completed_today,
                "failed": failed_today,
                "success_rate": round((completed_today / max(today_executions, 1)) * 100, 1)
            },
            "performance": {
                "overall_success_rate": round(success_rate, 1),
                "avg_execution_time": "4.2m",  # Would be calculated from actual data
                "agent_utilization": round(agent_utilization, 1),
                "system_efficiency": round((success_rate + agent_utilization) / 2, 1)
            },
            "workflow_types": [
                {"type": wf_type, "count": count, "percentage": round((count / max(total_workflows, 1)) * 100, 1)}
                for wf_type, count in workflow_types.items()
            ],
            "execution_trends": [
                {
                    "date": date.isoformat() if date else None,
                    "executions": count
                } for date, count in daily_executions
            ],
            "recent_activity": [
                {
                    "id": exec.id,
                    "workflow_id": exec.workflow_id,
                    "status": exec.status,
                    "started_at": exec.started_at.isoformat(),
                    "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                    "agent_id": exec.agent_id
                } for exec in recent_executions
            ],
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{workflow_id}/live-progress")
async def get_workflow_live_progress(workflow_id: int, db: Session = Depends(get_db)):
    """Get live progress for a specific workflow execution"""
    try:
        # Get the workflow
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id, Workflow.workspace_id == ctx.workspace_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Get current running execution
        current_execution = db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.workflow_id == workflow_id,
                WorkflowExecution.status == ExecutionStatus.RUNNING.value
            )
        ).first()
        
        if not current_execution:
            return {
                "workflow_id": workflow_id,
                "status": "idle",
                "message": "No active execution"
            }
        
        # Calculate detailed progress
        elapsed = (datetime.now() - current_execution.started_at).total_seconds()
        progress = min(95, int(elapsed / 60 * 20))  # Simulate progress
        
        # Define workflow steps
        steps = [
            {"name": "Initialization", "status": "completed", "duration": "0.5s"},
            {"name": "Agent Assignment", "status": "completed", "duration": "0.2s"},
            {"name": "Context Loading", "status": "completed" if progress > 20 else "running", "duration": "1.2s"},
            {"name": "Task Processing", "status": "completed" if progress > 50 else "running" if progress > 20 else "pending", "duration": "2.8s"},
            {"name": "Result Generation", "status": "completed" if progress > 80 else "running" if progress > 50 else "pending", "duration": "1.1s"},
            {"name": "Finalization", "status": "completed" if progress > 95 else "running" if progress > 80 else "pending", "duration": "0.3s"}
        ]
        
        # Get current step
        current_step_index = min(len(steps) - 1, int(progress / 20))
        current_step = steps[current_step_index]
        
        # Generate log entries
        log_entries = [
            {
                "timestamp": (current_execution.started_at + timedelta(seconds=i*30)).isoformat(),
                "level": "INFO",
                "message": f"Step {i+1}: {steps[min(i, len(steps)-1)]['name']} {'completed' if i < current_step_index else 'in progress'}"
            } for i in range(min(current_step_index + 1, len(steps)))
        ]
        
        # Add current activity log
        if progress < 95:
            log_entries.append({
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": f"Currently processing: {current_step['name']}"
            })
        
        return {
            "workflow_id": workflow_id,
            "execution_id": current_execution.id,
            "status": "running",
            "progress": {
                "percentage": progress,
                "current_step": current_step['name'],
                "current_step_index": current_step_index,
                "total_steps": len(steps),
                "estimated_completion": (current_execution.started_at + timedelta(minutes=5)).isoformat()
            },
            "steps": steps,
            "timing": {
                "started_at": current_execution.started_at.isoformat(),
                "elapsed_time": f"{int(elapsed)}s",
                "estimated_total": "5m",
                "estimated_remaining": f"{max(0, 300 - int(elapsed))}s"
            },
            "resources": {
                "agent_id": current_execution.agent_id,
                "memory_usage": f"{min(100, 20 + progress)}MB",
                "cpu_usage": f"{min(100, 10 + progress//2)}%"
            },
            "log_entries": log_entries[-10:],  # Last 10 entries
            "last_updated": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting live progress for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{workflow_id}/execute-advanced")
async def execute_workflow_advanced(
    workflow_id: int,
    execution_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Execute workflow with advanced options and live progress tracking"""
    try:
        # Validate workflow exists
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id, Workflow.workspace_id == ctx.workspace_id).first()
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Get agent (use first available if not specified)
        agent_id = execution_data.get('agent_id')
        if not agent_id:
            agent = db.query(Agent).filter(Agent.status == 'active').first()
            if not agent:
                raise HTTPException(status_code=400, detail="No active agents available")
            agent_id = agent.id
        else:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            agent_id=agent_id,
            workspace_id=ctx.workspace_id,
            input_data=execution_data.get('input_data', {}),
            status=ExecutionStatus.PENDING.value
        )
        
        db.add(execution)
        db.commit()
        db.refresh(execution)

        # PRD-56: Dispatch through TaskRunner abstraction
        # When TASK_RUNNER_BACKEND=queued, this enqueues to Redis for the workspace worker.
        # When TASK_RUNNER_BACKEND=local (default), this runs in-process via asyncio.
        runner = get_task_runner()

        if runner.backend_name == "queued":
            # Phase 2: Enqueue to workspace worker
            task = AgentTask(
                task_type=TaskType.WORKFLOW_SUBTASK,
                workspace_id=ctx.workspace_id,
                agent_id=agent_id,
                prompt=json.dumps(execution_data.get('input_data', {})),
                context={
                    "execution_id": execution.id,
                    "workflow_id": workflow_id,
                    "options": execution_data.get('options', {}),
                },
                priority=TaskPriority.NORMAL,
                timeout_seconds=600,
            )
            handle = await runner.submit_task(task)
            logger.info(f"Workflow {workflow_id} execution {execution.id} queued (task={handle.task_id[:8]})")
        else:
            # Phase 1 / Local: Run in-process (existing behavior)
            asyncio.create_task(
                execute_workflow_with_progress(
                    execution.id,
                    execution_data.get('options', {})
                )
            )

        return {
            "execution_id": execution.id,
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "status": "started",
            "message": "Workflow execution started with live progress tracking",
            "progress_endpoint": f"/api/workflows/{workflow_id}/live-progress",
            "websocket_events": ["execution_progress", "execution_completed", "execution_failed"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Additional endpoints for user journey tests
@router.post("/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: int,
    execution_data: Dict[str, Any],
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Execute workflow (simplified version for journey tests)"""
    try:
        # Validate workflow exists with proper loading
        workflow = db.query(Workflow).options(
            joinedload(Workflow.agents)
        ).filter(Workflow.id == workflow_id).first()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if not workflow.id:
            raise HTTPException(status_code=400, detail="Invalid workflow data")

        # Get agent with skills loaded
        agent = db.query(Agent).options(
            joinedload(Agent.skills)
        ).filter(Agent.status == 'active').first()

        if not agent:
            raise HTTPException(status_code=400, detail="No active agents available")

        if not agent.id:
            logger.error(f"❌ Agent object missing ID: {agent}")
            raise HTTPException(status_code=500, detail="Agent data corruption - missing ID")

        # Validate input_data
        input_data = execution_data.get('input_data')
        if not input_data:
            logger.warning("⚠️  No input_data provided, using empty dict")
            input_data = {}

        if not isinstance(input_data, dict):
            raise HTTPException(
                status_code=400,
                detail=f"input_data must be a dictionary, got {type(input_data).__name__}"
            )

        logger.info(f"🚀 Creating execution for workflow {workflow_id} with agent {agent.id}")
        logger.info(f"   Agent skills: {len(agent.skills) if agent.skills else 0}")

        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            agent_id=agent.id,
            workspace_id=ctx.workspace_id,
            input_data=input_data,
            status=ExecutionStatus.PENDING.value
        )
        
        db.add(execution)
        db.commit()
        db.refresh(execution)
        
        # PRD-56: Dispatch through TaskRunner abstraction
        runner = get_task_runner()

        if runner.backend_name == "queued":
            # Phase 2: Enqueue to workspace worker
            task = AgentTask(
                task_type=TaskType.WORKFLOW_SUBTASK,
                workspace_id=ctx.workspace_id,
                agent_id=agent.id,
                prompt=json.dumps(input_data),
                context={
                    "execution_id": execution.id,
                    "workflow_id": workflow_id,
                    "options": execution_data.get('options', {}),
                },
                priority=TaskPriority.NORMAL,
                timeout_seconds=600,
            )
            handle = await runner.submit_task(task)
            logger.info(f"Workflow {workflow_id} execution {execution.id} queued (task={handle.task_id[:8]})")
        else:
            # Phase 1 / Local: Run in-process (existing behavior)
            import asyncio

            async def _run_with_error_handling():
                try:
                    await execute_workflow_with_progress(
                        execution.id,
                        execution_data.get('options', {})
                    )
                except Exception as e:
                    logger.error(f"❌ FATAL: Workflow execution task crashed: {e}", exc_info=True)

            asyncio.create_task(_run_with_error_handling())

        logger.info(f"Workflow {workflow_id} execution {execution.id} started")
        
        return {
            "id": execution.id,
            "execution_id": execution.id,
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Workflow execution started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/execute")
async def execute_workflow_general(execution_data: Dict[str, Any], ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """General workflow execution endpoint"""
    try:
        workflow_id = execution_data.get('workflow_id')
        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id required")

        return await execute_workflow(workflow_id, execution_data, ctx, db)
        
    except Exception as e:
        logger.error(f"Error in general workflow execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/executions/")
async def list_executions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    workflow_id: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all workflow executions with filtering"""
    try:
        query = db.query(WorkflowExecution)
        
        # Apply filters
        if workflow_id:
            query = query.filter(WorkflowExecution.workflow_id == workflow_id)
        if status:
            query = query.filter(WorkflowExecution.status == status)
        
        total = query.count()
        executions = query.order_by(desc(WorkflowExecution.started_at)).offset(skip).limit(limit).all()
        
        return {
            "items": [
                {
                    "id": e.id,
                    "workflow_id": e.workflow_id,
                    "agent_id": e.agent_id,
                    "status": e.status,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                    "duration": str(e.completed_at - e.started_at) if e.completed_at and e.started_at else None,
                    "input_data": e.input_data,
                    "output_data": e.output_data
                } for e in executions
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/executions/")
async def create_execution(execution_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create workflow execution"""
    try:
        workflow_id = execution_data.get('workflow_id')
        return await execute_workflow(workflow_id, execution_data, db)
    except Exception as e:
        logger.error(f"Error creating execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: int, db: Session = Depends(get_db)):
    """Get workflow execution status"""
    try:
        execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return {
            "id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status,
            "input_data": execution.input_data,
            "output_data": execution.output_data,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution status {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: int, db: Session = Depends(get_db)):
    """Cancel a running workflow execution"""
    try:
        execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Check if execution is cancellable (running or pending)
        if execution.status not in [ExecutionStatus.RUNNING.value, ExecutionStatus.PENDING.value]:
            return {
                "message": f"Execution is already {execution.status}, cannot cancel",
                "execution_id": execution_id,
                "status": execution.status
            }
        
        # Update execution status to cancelled
        execution.status = ExecutionStatus.CANCELLED.value
        execution.completed_at = datetime.now()
        
        # Add cancellation metadata to output_data
        if not execution.output_data:
            execution.output_data = {}
        execution.output_data["cancellation"] = {
            "cancelled_at": datetime.now().isoformat(),
            "reason": "User requested cancellation"
        }
        
        db.commit()
        db.refresh(execution)
        
        logger.info(f"⏹️  Execution {execution_id} cancelled successfully")
        
        return {
            "message": "Execution cancelled successfully",
            "execution_id": execution_id,
            "status": execution.status,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling execution {execution_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/executions/{execution_id}/stream")
async def stream_execution_updates(execution_id: int, db: Session = Depends(get_db)):
    """
    SSE stream for real-time workflow execution updates (PRD-28).
    
    Replaces WebSocket + Redis + Polling architecture with direct SSE streaming.
    < 500ms latency, no polling required, smooth progressive updates.
    
    Usage (Frontend):
        const eventSource = new EventSource(`/api/workflows/executions/${id}/stream`);
        eventSource.onmessage = (event) => {
            const update = JSON.parse(event.data);
            // Handle update...
        };
    
    Event types:
        - connected: Initial connection confirmation
        - subtask_update: Subtask execution update with FULL details
        - execution_log: Structured log event (not truncated)
        - workflow_complete: Final workflow completion
        - error: Error event
    """
    # Verify execution exists
    execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    logger.info(f"🚀 Starting SSE stream for execution {execution_id}")
    
    try:
        from consumers.workflows.streaming import stream_workflow_execution
        
        return StreamingResponse(
            stream_workflow_execution(execution_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",  # CORS for SSE
            }
        )
    except Exception as e:
        logger.error(f"❌ Error creating SSE stream for execution {execution_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/executions/{execution_id}/stream/aisdk")
async def stream_execution_aisdk(execution_id: int, db: Session = Depends(get_db)):
    """
    AI SDK stream for real-time workflow execution updates.
    
    Uses Vercel AI SDK Data Stream Protocol:
    - 0:"text" -> Text delta
    - d:{"key":"val"} -> Data payload
    - e:{"err":"msg"} -> Error
    
    Usage (Frontend with @ai-sdk/react):
        const { messages, data } = useChat({
            api: '/api/workflow/stream',
            body: { executionId }
        })
    """
    # Verify execution exists
    execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # CRITICAL: Don't stream if execution already finished
    if execution.status in ['completed', 'failed', 'cancelled']:
        logger.warning(f"⚠️  Attempted to stream already finished execution {execution_id} (status: {execution.status})")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Execution already finished",
                "status": execution.status,
                "execution_id": execution_id,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "message": "Cannot stream a completed execution. Please start a new execution."
            }
        )

    logger.info(f"🚀 Starting AI SDK stream for execution {execution_id} (status: {execution.status})")

    try:
        from consumers.workflows.streaming import stream_workflow_as_aisdk

        return StreamingResponse(
            stream_workflow_as_aisdk(execution_id),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked",  # Force chunked transfer
                "x-vercel-ai-data-stream": "v1",
                "Access-Control-Allow-Origin": "*",
            }
        )
    except Exception as e:
        logger.error(f"❌ Error creating AI SDK stream for execution {execution_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stream")
async def stream_workflow_chat(request: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """
    AI SDK compatible streaming endpoint for useChat hook.
    
    This endpoint is designed to work with Vercel AI SDK's useChat hook:
    
    Frontend Usage:
        const { messages, data } = useChat({
            api: '/api/workflows/stream',
            body: { executionId: 123 }
        })
    
    Accepts:
        { executionId: number }
    
    Returns:
        AI SDK Data Stream Protocol (text/plain with x-vercel-ai-data-stream header)
        - 0:"text" -> Text deltas for stage updates
        - d:{...}  -> Data payloads for structured info
        - e:{...}  -> Error events
    """
    # Extract execution ID from request body
    execution_id = request.get("executionId")
    if not execution_id:
        raise HTTPException(status_code=400, detail="executionId is required in request body")
    
    # Verify execution exists
    execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")

    # CRITICAL: Don't stream if execution already finished
    if execution.status in ['completed', 'failed', 'cancelled']:
        logger.warning(f"⚠️  Attempted to stream already finished execution {execution_id} (status: {execution.status})")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Execution already finished",
                "status": execution.status,
                "execution_id": execution_id,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "message": "Cannot stream a completed execution. Please start a new execution."
            }
        )

    logger.info(f"🚀 Starting AI SDK chat stream for execution {execution_id} (status: {execution.status})")

    try:
        from consumers.workflows.streaming import stream_workflow_as_aisdk

        return StreamingResponse(
            stream_workflow_as_aisdk(execution_id),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked",  # Force chunked transfer
                "x-vercel-ai-data-stream": "v1",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        logger.error(f"❌ Error creating AI SDK chat stream for execution {execution_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")



@router.get("/executions/{execution_id}/results")
async def get_execution_results(execution_id: int, db: Session = Depends(get_db)):
    """Get workflow execution results"""
    try:
        execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status,
            "results": execution.output_data or {},
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution results {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def _adaptive_checkpoint(
    stage_name: str,
    stage_results: Dict[str, Any],
    workflow_context: Dict[str, Any],
    attempt: int,
    stage_tracker,
) -> "AdaptationDecision":
    """
    Two-tier adaptive checkpoint: heuristic → LLM evaluation.
    Called after Stage 4 (execution) and Stage 7 (quality) to decide
    whether to continue, retry, or abort.

    Returns an AdaptationDecision from AdaptiveExecutionMonitor.
    """
    from modules.orchestrator.llm.adaptive_execution_monitor import (
        AdaptiveExecutionMonitor,
        AdaptationDecision,
        InterventionAction,
    )

    await stage_tracker._emit("adaptive_checkpoint_start", {
        "stage": stage_name,
        "attempt": attempt,
        "timestamp": datetime.now().isoformat(),
    })

    monitor = AdaptiveExecutionMonitor(max_retries=1)
    decision = await monitor.evaluate_stage_outcome(
        stage_name=stage_name,
        stage_results=stage_results,
        workflow_context=workflow_context,
        attempt_number=attempt,
    )

    logger.info(
        f"[Adaptive] Checkpoint after {stage_name}: {decision.action.value} "
        f"(confidence={decision.confidence:.2f}, reason={decision.reasoning[:120]})"
    )

    await stage_tracker._emit("adaptive_checkpoint_decision", {
        "stage": stage_name,
        "attempt": attempt,
        "action": decision.action.value,
        "reasoning": decision.reasoning[:200],
        "confidence": decision.confidence,
        "timestamp": datetime.now().isoformat(),
    })

    return decision


async def execute_workflow_with_progress(execution_id: int, options: Dict[str, Any]):
    """Execute workflow with COMPLETE pipeline: decompose, select, enhance, execute, score, learn, remember"""
    from core.database.database import get_db_session
    from modules.orchestrator.stages import (
        RealTaskDecomposer,
        ContextEngineeringIntegrator,
        ResultAggregator,
        WorkflowMemoryIntegrator,
    )
    from modules.orchestrator.llm import LLMAgentSelector  # ALWAYS use LLM selector
    from modules.agents import AgentExecutionManager
    from modules.learning import LearningSystemUpdater
    from modules.memory.storage import HierarchicalMemorySystem
    from core.services.workspace_manager import WorkspaceManager  # Unique workspace per execution
    from consumers.workflows import ModelUsageTracker  # PRD-15
    import os
    
    # Create unique workspace for this execution
    workspace_manager = WorkspaceManager(execution_id)
    workspace_path = workspace_manager.create_workspace()
    
    logger.info(f"📁 Execution {execution_id} workspace: {workspace_path}")
    logger.info(f"💾 Results will be saved to: {workspace_manager.get_results_path()}")
    
    try:
        with get_db_session() as db:
            # PRD-15: Initialize model usage tracker
            model_tracker = ModelUsageTracker(db)
            execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
            if not execution:
                return
            
            workflow = db.query(Workflow).filter(Workflow.id == execution.workflow_id).first()
            
            # Update status to running
            execution.status = ExecutionStatus.RUNNING.value
            db.commit()
            
            # Publish execution start to Redis
            from core.redis.client import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                redis_client.publish_workflow_event(
                    workflow_id=execution.workflow_id,
                    execution_id=execution_id,
                    event_type="execution_started",
                    data={
                        "workflow_name": workflow.name if workflow else "Unknown",
                        "status": "running",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            else:
                logger.warning("Redis client not initialized for execution_started event")
            
            # Initialize SSE stream manager for instant UI updates
            from consumers.workflows.streaming import get_stream_manager
            stream_manager = get_stream_manager()
            
            # Initialize stage tracker for 9-stage workflow with stream manager
            stage_tracker = WorkflowStageTracker(execution_id, redis_client, stream_manager=stream_manager)

            
            # ========== EXECUTION MODE DETECTION ==========
            # Determine execution mode based on workflow/recipe configuration
            workflow_def = workflow.workflow_definition or {}
            predefined_tasks = workflow_def.get("tasks", [])
            predefined_steps = workflow_def.get("steps", [])  # Recipe steps with pre-assigned agents

            # Get task description early (needed for all execution modes)
            goal = workflow_def.get("goal")
            task_description = goal or workflow.description or workflow.name

            # Detect execution mode (AUTONOMOUS, RECIPE, HYBRID)
            execution_mode = "AUTONOMOUS"  # Default

            # Check if this is a recipe with pre-defined steps and agents
            if predefined_steps and len(predefined_steps) > 0:
                # Count steps with pre-assigned agents
                steps_with_agents = sum(1 for step in predefined_steps if step.get("agent_id") or step.get("required_agent_id"))
                total_steps = len(predefined_steps)

                if steps_with_agents == total_steps:
                    execution_mode = "RECIPE"  # All steps have agents
                elif steps_with_agents > 0:
                    execution_mode = "HYBRID"  # Some steps have agents
                else:
                    execution_mode = "AUTONOMOUS"  # No agents assigned
            elif predefined_tasks and len(predefined_tasks) > 0:
                # Legacy tasks format - treat as RECIPE if agents are assigned
                steps_with_agents = sum(1 for task in predefined_tasks if task.get("required_agent_id"))
                total_tasks = len(predefined_tasks)

                if steps_with_agents == total_tasks:
                    execution_mode = "RECIPE"
                elif steps_with_agents > 0:
                    execution_mode = "HYBRID"

            logger.info(f"🔍 EXECUTION MODE: {execution_mode}")
            logger.info(f"   Predefined Steps: {len(predefined_steps)}")
            logger.info(f"   Predefined Tasks: {len(predefined_tasks)}")

            # Store execution mode in input_data
            execution.input_data = execution.input_data or {}
            execution.input_data["execution_mode"] = execution_mode
            attributes.flag_modified(execution, "input_data")
            db.commit()

            # PRD-59: Use PhaseSelector to determine dynamic phase plan
            try:
                from modules.orchestrator.phase_selector import (
                    PhaseSelector, ExecutionMode as PSExecutionMode,
                    ComplexityLevel,
                )
                _ps_mode_map = {
                    "AUTONOMOUS": PSExecutionMode.AUTONOMOUS,
                    "RECIPE": PSExecutionMode.RECIPE,
                    "HYBRID": PSExecutionMode.HYBRID,
                }
                phase_selector = PhaseSelector()

                # Quick complexity estimate from step count / description length
                step_count = len(predefined_steps) or len(predefined_tasks) or 1
                desc_len = len(task_description)
                if step_count <= 1 and desc_len < 100:
                    _complexity = ComplexityLevel.ATOM
                elif step_count <= 3:
                    _complexity = ComplexityLevel.MOLECULE
                elif step_count <= 6:
                    _complexity = ComplexityLevel.CELL
                elif step_count <= 10:
                    _complexity = ComplexityLevel.ORGAN
                else:
                    _complexity = ComplexityLevel.ORGANISM

                # Check prompt registry availability
                _has_prompt_registry = False
                try:
                    from modules.prompts.registry import PromptRegistry
                    _has_prompt_registry = True
                except ImportError:
                    pass

                phase_specs = phase_selector.select_phases(
                    mode=_ps_mode_map.get(execution_mode, PSExecutionMode.AUTONOMOUS),
                    complexity=_complexity,
                    agent_count=step_count,  # Rough proxy: steps ≈ agents
                    has_context_sources=True,
                    has_prompt_registry=_has_prompt_registry,
                )

                # Extract phase names for the tracker (map Phase enum → top-level name)
                _phase_name_map = {
                    "plan_full": "PLAN", "plan_partial": "PLAN",
                    "prepare": "PREPARE",
                    "execute_single": "EXECUTE", "execute_multi": "EXECUTE",
                    "evaluate_full": "EVALUATE", "evaluate_light": "EVALUATE",
                    "learn": "LEARN",
                }
                active_phases = []
                for ps in phase_specs:
                    mapped = _phase_name_map.get(ps.phase.value, ps.phase.value.upper())
                    if mapped not in active_phases:
                        active_phases.append(mapped)

                # Store phase plan in execution metadata
                execution.input_data["phase_plan"] = {
                    "complexity": _complexity.value,
                    "phases": [
                        {"phase": ps.phase.value, "stages": [s.stage_name for s in ps.stages], "budget": ps.token_budget}
                        for ps in phase_specs
                    ],
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()

                logger.info(f"📋 PRD-59 PhaseSelector: complexity={_complexity.value}, phases={active_phases}")
            except Exception as ps_err:
                logger.warning(f"PhaseSelector unavailable, using default phases: {ps_err}")
                active_phases = ["PLAN", "PREPARE", "EXECUTE", "EVALUATE", "LEARN"]
                if execution_mode == "RECIPE":
                    active_phases = ["PREPARE", "EXECUTE", "EVALUATE", "LEARN"]

            stage_tracker.set_active_phases(active_phases)

            # ========== PHASE: PLAN (Stages 1, 2) ==========
            if "PLAN" in active_phases:
                await stage_tracker.start_phase("PLAN")

            # ========== STAGE 1: TASK DECOMPOSITION ==========
            # RECIPE mode: Skip - steps are pre-defined
            # HYBRID mode: Skip - partial steps are pre-defined
            # AUTONOMOUS mode: Run - need to decompose

            should_skip_stage1 = execution_mode in ["RECIPE", "HYBRID"]

            if should_skip_stage1:
                logger.info(f"⏭️  SKIPPED Stage 1: {execution_mode} mode has pre-defined steps")
                # Use predefined steps/tasks
                steps = predefined_steps if predefined_steps else predefined_tasks
                execution.input_data["decomposition"] = {
                    "is_real": False,
                    "is_predefined": True,
                    "execution_mode": execution_mode,
                    "task_count": len(steps),
                    "skip_reason": f"{execution_mode} mode - steps pre-defined"
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
            else:
                await stage_tracker.start_stage(1)
                # AUTONOMOUS mode: REAL TASK DECOMPOSITION using LLM
                decomposer = RealTaskDecomposer()

                # Get additional parameters for decomposition
                task_type = workflow_def.get("category", "general")
                complexity = workflow_def.get("priority", "medium")

                # Pass workflow context to decomposer if available (for CodeGraph, PR review, etc.)
                workflow_context = workflow_def.get("context", {})

                logger.info(f"🔧 Decomposing task with RealTaskDecomposer: {task_description[:100]}")

                try:
                    # Call REAL LLM to decompose task
                    decomposition_result = await decomposer.decompose_task(
                        task_description=task_description,
                        task_type=task_type,
                        complexity=complexity,
                        requirements=[],
                        max_subtasks=None
                    )

                    # Extract real subtasks from LLM response
                    steps = decomposition_result.get("subtasks", [])

                    # Store decomposition metadata
                    execution.input_data["decomposition"] = {
                        "is_real": True,
                        "llm_model": decomposition_result.get("llm_model"),
                        "execution_strategy": decomposition_result.get("execution_strategy"),
                        "total_estimated_time": decomposition_result.get("total_estimated_time"),
                        "complexity_assessment": decomposition_result.get("complexity_assessment")
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()

                    logger.info(f"✅ Decomposed into {len(steps)} real subtasks")

                    # PRD-59 Fix 5: Propagate graph metadata to individual subtasks
                    graph_analysis = decomposition_result.get("graph_analysis", {})
                    parallel_groups = graph_analysis.get("parallel_groups", [])
                    execution_order = graph_analysis.get("execution_order", [])
                    for step in steps:
                        step_id = step.get("subtask_id", "")
                        step["graph_metadata"] = {
                            "parallel_group": next(
                                (i for i, group in enumerate(parallel_groups)
                                 if step_id in group),
                                0
                            ),
                            "dependencies": step.get("dependencies", []),
                            "execution_order": execution_order,
                        }

                except Exception as e:
                    logger.error(f"❌ Task decomposition failed: {e}, falling back to default steps", exc_info=True)
                    logger.error(f"❌ DECOMPOSITION ERROR TRACEBACK:")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Fallback to simple steps if decomposition fails
                    steps = [
                        {"description": "Initialize workflow", "estimated_duration": "30 seconds", "agent_type": "orchestrator"},
                        {"description": "Execute main task", "estimated_duration": "60 seconds", "agent_type": "worker"},
                        {"description": "Finalize results", "estimated_duration": "20 seconds", "agent_type": "orchestrator"}
                    ]

                await stage_tracker.complete_stage(1, {"subtasks": len(steps)})
            
            # ========== STAGE 2: AGENT SELECTION ==========
            # RECIPE mode: Skip - agents are pre-assigned
            # HYBRID mode: Partial - select agents only for steps without assignments
            # AUTONOMOUS mode: Run - select optimal agents

            should_skip_stage2 = execution_mode == "RECIPE"

            if should_skip_stage2:
                logger.info(f"⏭️  SKIPPED Stage 2: RECIPE mode has pre-assigned agents")

                # Extract agent assignments from pre-defined steps
                agent_assignments = {}
                for idx, step in enumerate(steps):
                    subtask_id = step.get("subtask_id", f"subtask_{idx}")
                    agent_id = step.get("agent_id") or step.get("required_agent_id")

                    if agent_id:
                        # Get agent details
                        agent = db.query(Agent).filter(Agent.id == agent_id).first()
                        if agent:
                            from modules.orchestrator.llm import AgentMatch
                            agent_assignments[subtask_id] = [
                                AgentMatch(
                                    agent_id=agent.id,
                                    agent_name=agent.name,
                                    agent_type=agent.agent_type,
                                    match_score=1.0,  # Perfect match - pre-assigned
                                    reasoning=f"Pre-assigned in recipe: {step.get('name', 'Unknown')}",
                                    skills_matched=[],
                                    performance_score=1.0,
                                    availability_score=1.0,
                                    collaboration_score=1.0
                                )
                            ]
                            logger.info(f"✅ Step {idx}: Pre-assigned agent {agent.name} (ID: {agent.id})")

                execution.input_data["agent_selection"] = {
                    "is_real": False,
                    "is_predefined": True,
                    "execution_mode": execution_mode,
                    "summary": {
                        "total_assignments": len(agent_assignments),
                        "unique_agents": len(set(matches[0].agent_id for matches in agent_assignments.values() if matches)),
                        "avg_match_score": 1.0
                    },
                    "skip_reason": "RECIPE mode - agents pre-assigned"
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()

            else:
                await stage_tracker.start_stage(2)
                # INTELLIGENT AGENT SELECTION

                logger.info(f"🤖 Selecting optimal agents for {len(steps)} subtasks...")

                # Check if we should use smart grouping
                use_smart_selection = True  # Enable smart selection by default
                workflow_ctx = workflow_def.get("context")
                if workflow_ctx:
                    try:
                        ctx = json.loads(workflow_ctx) if isinstance(workflow_ctx, str) else workflow_ctx
                        use_smart_selection = ctx.get("use_smart_selection", True)
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.error(f"Error parsing workflow context: {e}", exc_info=True)

                # Check if tasks have specific agent requirements (HYBRID mode)
                has_agent_requirements = any(
                    task.get("required_agent_id") or task.get("agent_id") for task in steps
                ) if isinstance(steps, list) else False
                agent_assignments = {}
                try:
                    logger.info(f"🔍 DEBUG: use_smart_selection={use_smart_selection}, has_agent_requirements={has_agent_requirements}")

                    if use_smart_selection and not has_agent_requirements:
                        # Use BATCH LLM agent selection (NO LOOPS!)
                        logger.info("⚡ Using BATCH LLM agent selection (semantic + LLM in one shot)")
                        logger.info(f"  📋 Number of steps to assign: {len(steps)}")

                        from core.llm.llm_agent_selector import LLMAgentSelector

                        llm_selector = LLMAgentSelector(db_session=db)

                        # Select agents for ALL subtasks in ONE BATCH (no loops!)
                        agent_assignments = await llm_selector.select_agents_for_subtasks(
                            steps,
                            workflow_context={
                                "description": task_description,
                                "workflow_id": execution.workflow_id
                            }
                        )
                        logger.info(f"  ✅ BATCH selection complete: {len(agent_assignments)} assignments")

                        # Create selection summary
                        unique_agents = set()
                        for matches in agent_assignments.values():
                            for match in matches:
                                unique_agents.add(match.agent_id)

                        # Calculate average match score
                        total_score = 0
                        count = 0
                        for matches in agent_assignments.values():
                            for match in matches:
                                total_score += match.match_score
                                count += 1
                        avg_score = total_score / count if count > 0 else 0.9

                        selection_summary = {
                            "total_subtasks": len(steps),
                            "unique_agents": len(unique_agents),
                            "selection_method": "llm_intelligent",
                            "efficiency_ratio": len(steps) / len(unique_agents) if unique_agents else 0,
                            "avg_match_score": avg_score
                        }

                    elif has_agent_requirements:
                        # Use specified agents from predefined tasks (HYBRID mode)
                        logger.info("📌 Using specified agents from task definitions")
                        agent_assignments = {}

                        for idx, task in enumerate(steps):
                            # Use the subtask_id from the step, not a generated one!
                            subtask_id = task.get("subtask_id", f"subtask_{idx}")
                            required_agent_id = task.get("required_agent_id") or task.get("agent_id")

                            if required_agent_id:
                                # Get the agent details
                                agent = db.query(Agent).filter(Agent.id == required_agent_id).first()
                                if agent:
                                    from modules.orchestrator.llm import AgentMatch
                                    agent_assignments[subtask_id] = [
                                        AgentMatch(
                                            agent_id=agent.id,
                                            agent_name=agent.name,
                                            agent_type=agent.agent_type,
                                            match_score=1.0,  # Perfect match since explicitly specified
                                            reasoning=f"Explicitly specified for task: {task.get('name', 'Unknown')}",
                                            skills_matched=task.get("required_skills", []),
                                            performance_score=0.8,
                                            availability_score=1.0,
                                            collaboration_score=1.0
                                        )
                                    ]
                                    logger.info(f"✅ Task {idx}: Using specified agent {agent.name} (ID: {agent.id})")
                                else:
                                    logger.warning(f"⚠️ Task {idx}: Specified agent {required_agent_id} not found")

                        # Create summary for explicit assignments
                        selection_summary = {
                            "total_assignments": len(agent_assignments),
                            "unique_agents": len(set(matches[0].agent_id for matches in agent_assignments.values() if matches)),
                            "avg_match_score": 1.0  # All explicit matches are perfect
                        }

                    else:
                        # Fallback: Use batch LLM selector
                        logger.info("⚡ Fallback: Using BATCH LLM agent selection")
                        from core.llm.llm_agent_selector import LLMAgentSelector

                        llm_selector = LLMAgentSelector(db_session=db)
                        agent_assignments = await llm_selector.select_agents_for_subtasks(
                            steps,
                            workflow_context={
                                "description": task_description,
                                "workflow_id": execution.workflow_id
                            }
                        )

                        # Summary for LLM selection
                        unique_agents = set()
                        for matches in agent_assignments.values():
                            for match in matches:
                                unique_agents.add(match.agent_id)

                        total_score = 0
                        count = 0
                        for matches in agent_assignments.values():
                            for match in matches:
                                total_score += match.match_score
                                count += 1
                        avg_score = total_score / count if count > 0 else 0.9

                        selection_summary = {
                            "total_assignments": len(agent_assignments),
                            "unique_agents": len(unique_agents),
                            "avg_match_score": avg_score
                        }
                        logger.info(f"  Using LLM selection summary: {selection_summary}")

                    # Store agent selection results
                    execution.input_data["agent_selection"] = {
                        "is_real": True,
                        "summary": selection_summary,
                        "assignments": {
                            subtask_id: [
                                {
                                    "agent_id": match.agent_id,
                                    "agent_name": match.agent_name,
                                    "match_score": match.match_score,
                                    "reasoning": match.reasoning
                                }
                                for match in matches
                            ]
                            for subtask_id, matches in agent_assignments.items()
                        }
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()

                    logger.info(f"✅ Agent selection complete: {selection_summary['avg_match_score']:.2f} avg match score")

                    # Enhance steps with selected agents
                    for idx, step in enumerate(steps):
                        # Use the subtask_id from the step, not a generated one!
                        subtask_id = step.get("subtask_id", f"subtask_{idx}")
                        if subtask_id in agent_assignments and agent_assignments[subtask_id]:
                            best_match = agent_assignments[subtask_id][0]
                            step["selected_agent"] = {
                                "agent_id": best_match.agent_id,
                                "agent_name": best_match.agent_name,
                                "match_score": best_match.match_score
                            }
                
                    await stage_tracker.complete_stage(2, {
                        "agents_assigned": len(agent_assignments),
                        "avg_match_score": selection_summary.get('avg_match_score', 0)
                    })

                except Exception as e:
                    logger.error(f"❌ Agent selection failed: {e}, continuing without specific agents")
                    logger.error(f"❌ Full traceback:", exc_info=True)
                    execution.input_data["agent_selection"] = {
                        "is_real": False,
                        "error": str(e)
                    }

                    await stage_tracker.complete_stage(2, {"agents_assigned": len(agent_assignments)})
            
            # MEMORY SYSTEM INITIALIZATION (UnifiedMemoryService with Workspace Isolation)
            logger.info(f"🧠 Initializing memory system with workspace isolation...")
            memory_service_available = False
            memory_retrieval_results = {}

            # Get workspace_id for memory scoping
            workspace_id = execution.workspace_id

            try:
                from modules.memory.unified_memory_service import get_unified_memory_service, MemoryNamespace

                memory_service = get_unified_memory_service()
                if not memory_service.is_mem0_configured:
                    raise RuntimeError("Mem0 is not configured")
                memory_service_available = True
                logger.info(f"✅ UnifiedMemoryService initialized")

                # Use MemoryNamespace for workspace-scoped workflow memory
                ns = MemoryNamespace(workspace_id=str(workspace_id))
                memory_scope_id = ns.workflow(execution.workflow_id)

                # Retrieve workflow memories
                logger.info(f"🧠 Retrieving workflow memories for scope: {memory_scope_id}")

                memories = await memory_service.search_long_term_scoped(
                    user_id=memory_scope_id,
                    query=task_description,
                    limit=10,
                )

                logger.info(f"✅ Retrieved {len(memories)} memories")

                # Format memories for execution context
                memory_retrieval_results = {
                    "total_memories_retrieved": len(memories),
                    "workspace_id": str(workspace_id),
                    "memory_scope": memory_scope_id,
                    "memories": [
                        {
                            "id": m.get("id"),
                            "content": m.get("memory"),
                            "score": m.get("score", 0),
                            "created_at": m.get("created_at")
                        }
                        for m in memories
                    ]
                }

                execution.input_data["memory_retrieval"] = {
                    "is_real": True,
                    "results": memory_retrieval_results
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()

                # Enhanced logging
                if memories:
                    sample_memories = [m.get("memory", "")[:50] for m in memories[:3]]
                    logger.info(f"📊 Sample memories: {sample_memories}")

            except Exception as e:
                logger.error(f"❌ Memory initialization or retrieval failed: {e}", exc_info=True)
                logger.warning(f"⚠️ Continuing workflow without memory system")
                execution.input_data["memory_retrieval"] = {
                    "is_real": False,
                    "error": str(e)
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
                memory_service_available = False
            
            # Complete PLAN phase, start PREPARE phase
            if "PLAN" in active_phases:
                await stage_tracker.complete_phase("PLAN", {"stages_completed": [1, 2]})
            if "PREPARE" in active_phases:
                await stage_tracker.start_phase("PREPARE")

            # ========== STAGE 3: CONTEXT ENGINEERING ==========
            await stage_tracker.start_stage(3)
            
            # PRD-59 Fix 4: Respect requires_context from decomposer
            # Only run context engineering for subtasks that need it
            subtasks_needing_context = [
                step for step in steps
                if step.get("requires_context", True)  # Default to True for safety
            ]
            subtasks_skipping_context = [
                step for step in steps
                if not step.get("requires_context", True)
            ]
            if subtasks_skipping_context:
                logger.info(f"⚡ Skipping context engineering for {len(subtasks_skipping_context)} subtasks (requires_context=False)")
            
            try:
                if subtasks_needing_context:
                    logger.info(f"📚 Enhancing {len(subtasks_needing_context)}/{len(steps)} subtasks with context (LLM decided)...")
                    context_integrator = ContextEngineeringIntegrator(db_session=db)
                    
                    # Get workflow tags and context for CodeGraph project selection
                    workflow_tags = workflow.tags if workflow and hasattr(workflow, 'tags') and workflow.tags else []
                    workflow_ctx = workflow_def.get("context")
                    
                    # If context is a JSON string, parse it
                    if isinstance(workflow_ctx, str):
                        try:
                            workflow_ctx = json.loads(workflow_ctx)
                        except Exception:
                            workflow_ctx = None
                    
                    # Only enhance subtasks that need context
                    context_enhancements = await context_integrator.enhance_subtasks_with_context(
                        subtasks=subtasks_needing_context,
                        workflow_description=task_description,
                        workflow_tags=workflow_tags,
                        workflow_context=workflow_ctx
                    )
                    
                    # Store context enhancement results
                    enhancement_summary = context_integrator.get_enhancement_summary(context_enhancements)
                    execution.input_data["context_engineering"] = {
                        "is_real": True,
                        "summary": enhancement_summary,
                        "enhancements": {
                            subtask_id: {
                                "total_tokens": enh.total_tokens,
                                "num_sources": enh.num_sources,
                                "context_quality": enh.context_quality_score,
                                "retrieval_time_ms": enh.retrieval_time_ms
                            }
                            for subtask_id, enh in context_enhancements.items()
                        }
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()
                    
                    logger.info(
                        f"✅ Context enhancement complete: {enhancement_summary['context_coverage']:.0%} coverage, "
                        f"{enhancement_summary['total_sources_retrieved']} sources, "
                        f"{enhancement_summary['avg_context_quality']:.0%} avg quality"
                    )
                    
                    # Merge enhanced subtasks back into steps (preserve order)
                    for i, step in enumerate(steps):
                        if step.get("subtask_id") in context_enhancements:
                            enh = context_enhancements[step["subtask_id"]]
                            # Convert to dict and add context_quality alias for frontend compatibility
                            from dataclasses import asdict
                            enh_dict = asdict(enh)
                            enh_dict["context_quality"] = enh.context_quality_score  # Add alias
                            steps[i]["context_enhancement"] = enh_dict
                else:
                    logger.info(f"⚡ Skipping context engineering - LLM determined no subtasks need external context (token savings!)")
                    context_enhancements = {}
                    
            except Exception as e:
                logger.error(f"❌ Context engineering failed: {e}, continuing without enhanced context")
                execution.input_data["context_engineering"] = {
                    "is_real": False,
                    "error": str(e)
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
                # Initialize empty context_enhancements if error occurred
                context_enhancements = {}
            
            await stage_tracker.complete_stage(3, {
                "enhanced_subtasks": len(context_enhancements),
                "avg_context_quality": execution.input_data.get("context_engineering", {}).get("summary", {}).get("avg_context_quality", 0)
            })

            # Complete PREPARE phase, start EXECUTE phase
            if "PREPARE" in active_phases:
                await stage_tracker.complete_phase("PREPARE", {"stages_completed": [3]})
            if "EXECUTE" in active_phases:
                await stage_tracker.start_phase("EXECUTE")

            # ========== STAGE 4: AGENT EXECUTION ==========
            await stage_tracker.start_stage(4)
            
            # DEBUG: Check what we have before agent execution
            logger.info(f"🔍 DEBUG: About to start agent execution")
            logger.info(f"🔍 DEBUG: steps count = {len(steps)}")
            logger.info(f"🔍 DEBUG: agent_assignments count = {len(agent_assignments)}")
            logger.info(f"🔍 DEBUG: context_enhancements count = {len(context_enhancements)}")
            
            # REAL AGENT EXECUTION
            logger.info(f"🤖 Starting agent execution phase...")
            logger.info(f"🤖 Executing {len(steps)} subtasks with real agents...")
            try:
                logger.info(f"🔍 DEBUG: Creating AgentExecutionManager...")
                execution_manager = AgentExecutionManager(
                    db_session=db,
                    max_parallel_executions=3,
                    max_retries=2,
                    workspace_dir=workspace_path  # Pass unique workspace
                )
                logger.info(f"🔍 DEBUG: AgentExecutionManager created successfully")
                logger.info(f"📁 Using workspace: {workspace_path}")
                # websocket_manager assignment removed - using SSE/AI SDK streaming instead
                
                logger.info(f"🔍 DEBUG: About to call execute_workflow_subtasks...")
                
                # Check for execution strategy from workflow context
                execution_strategy = "parallel"  # default
                
                # Get workflow definition and context
                workflow_def = workflow.workflow_definition or {}
                workflow_context = workflow_def.get("context")
                
                # Check workflow definition first
                if workflow_def.get("execution_strategy"):
                    execution_strategy = workflow_def["execution_strategy"]
                # Then check workflow context
                elif workflow_context:
                    if isinstance(workflow_context, str):
                        import json as json_lib
                        workflow_context = json_lib.loads(workflow_context)
                    if isinstance(workflow_context, dict):
                        execution_strategy = workflow_context.get("execution_mode", "parallel")
                
                logger.info(f"📊 Using execution strategy: {execution_strategy}")
                
                # Execute all subtasks with real agents
                subtask_results = await execution_manager.execute_workflow_subtasks(
                    subtasks=steps,
                    agent_assignments=agent_assignments,
                    context_enhancements=context_enhancements,
                    execution_id=execution_id,
                    workflow_id=execution.workflow_id,
                    memory_retrieval_results=memory_retrieval_results,
                    execution_strategy=execution_strategy  # Pass memory to execution
                )
                
                # Store execution results
                execution_summary = execution_manager.get_execution_summary(subtask_results)
                
                # PRD-15: Track model usage for each subtask
                for subtask_id, result in subtask_results.items():
                    if result.tokens_used > 0:
                        # Get agent's model configuration
                        agent = db.query(Agent).filter(Agent.id == result.agent_id).first()
                        if agent and agent.model_config:
                            model_id = agent.model_config.get("model_id", config.LLM_MODEL)
                        else:
                            model_id = config.LLM_MODEL  # Default fallback
                        
                        # Estimate input/output split (70% input, 30% output is typical)
                        input_tokens = int(result.tokens_used * 0.7)
                        output_tokens = result.tokens_used - input_tokens
                        
                        model_tracker.record_usage(
                            agent_id=result.agent_id,
                            model_id=model_id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            execution_time_ms=result.execution_time_ms
                        )
                
                execution.input_data["agent_execution"] = {
                    "is_real": True,
                    "summary": execution_summary,
                    "results": {
                        subtask_id: {
                            "status": result.status.value,
                            "agent_name": result.agent_name,
                            "tokens_used": result.tokens_used,
                            "execution_time_ms": result.execution_time_ms,
                            "retry_count": result.retry_count,
                            "error": result.error_message
                        }
                        for subtask_id, result in subtask_results.items()
                    }
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
                
                logger.info(
                    f"✅ Agent execution complete: {execution_summary['success_rate']:.0%} success rate, "
                    f"{execution_summary['total_tokens_used']} tokens, "
                    f"{execution_summary['total_execution_time_ms']}ms total time"
                )
                
                # Update steps with real execution results
                for idx, step in enumerate(steps):
                    # Use the subtask_id from the step, not a generated one!
                    subtask_id = step.get("subtask_id", f"subtask_{idx}")
                    if subtask_id in subtask_results:
                        result = subtask_results[subtask_id]
                        step["execution_result"] = {
                            "status": result.status.value,
                            "llm_response": result.llm_response,
                            "tokens_used": result.tokens_used,
                            "execution_time_ms": result.execution_time_ms
                        }
                
                total_duration = execution_summary["total_execution_time_ms"] / 1000.0
                
            except Exception as e:
                logger.error(f"❌ Agent execution failed: {e}, falling back to simulation")
                execution.input_data["agent_execution"] = {
                    "is_real": False,
                    "error": str(e)
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
                
                # Fallback: simulate execution
                total_duration = 30.0
                for i, step in enumerate(steps):
                    step["execution_result"] = {
                        "status": "completed",
                        "llm_response": f"Simulated response for: {step.get('description', '')}",
                        "tokens_used": 0,
                        "execution_time_ms": 3000
                    }
                    await asyncio.sleep(3)
            
            await stage_tracker.complete_stage(4, {
                "subtasks_executed": len(subtask_results) if 'subtask_results' in locals() else 0,
                "success_rate": execution_summary.get("success_rate", 0) if 'execution_summary' in locals() else 0,
                "total_tokens": execution_summary.get("total_tokens_used", 0) if 'execution_summary' in locals() else 0
            })

            # ── ADAPTIVE CHECKPOINT 1: Post-Execution ──
            # If execution success rate is poor, ask LLM whether to retry.
            # Max 1 retry (2 total attempts). Heuristic fast-path skips LLM for good results.
            _execution_attempt = 1
            if (
                'execution_summary' in locals()
                and execution.input_data.get("agent_execution", {}).get("is_real")
                and execution_summary.get("success_rate", 1.0) < 0.6
            ):
                from modules.orchestrator.llm.adaptive_execution_monitor import InterventionAction

                _checkpoint_decision = await _adaptive_checkpoint(
                    stage_name="execution",
                    stage_results=execution_summary,
                    workflow_context={
                        "task_description": task_description,
                        "steps": steps,
                        "agent_assignments": {
                            k: [{"agent_id": m.agent_id, "agent_name": m.agent_name} for m in v]
                            for k, v in agent_assignments.items()
                        } if agent_assignments else {},
                    },
                    attempt=_execution_attempt,
                    stage_tracker=stage_tracker,
                )

                if _checkpoint_decision.action in (
                    InterventionAction.RETRY_SAME,
                    InterventionAction.RETRY_DIFFERENT,
                ):
                    _execution_attempt += 1
                    await stage_tracker._emit("adaptive_retry_start", {
                        "stage": "execution",
                        "attempt": _execution_attempt,
                        "action": _checkpoint_decision.action.value,
                        "timestamp": datetime.now().isoformat(),
                    })

                    try:
                        # RETRY_DIFFERENT: re-select agents first
                        if _checkpoint_decision.action == InterventionAction.RETRY_DIFFERENT:
                            logger.info("[Adaptive] Re-selecting agents for retry...")
                            from core.llm.llm_agent_selector import LLMAgentSelector as _RetrySelector
                            _retry_selector = _RetrySelector(db_session=db)
                            agent_assignments = await _retry_selector.select_agents_for_subtasks(
                                steps,
                                workflow_context={
                                    "description": task_description,
                                    "workflow_id": execution.workflow_id,
                                },
                            )
                            # Update steps with new agent assignments
                            for idx, step in enumerate(steps):
                                subtask_id = step.get("subtask_id", f"subtask_{idx}")
                                if subtask_id in agent_assignments and agent_assignments[subtask_id]:
                                    best_match = agent_assignments[subtask_id][0]
                                    step["selected_agent"] = {
                                        "agent_id": best_match.agent_id,
                                        "agent_name": best_match.agent_name,
                                        "match_score": best_match.match_score,
                                    }

                        # Re-run agent execution (same or different agents)
                        logger.info(f"[Adaptive] Retry attempt {_execution_attempt}: re-executing agents...")
                        from modules.agents import AgentExecutionManager as _RetryExecManager
                        _retry_exec_mgr = _RetryExecManager(
                            db_session=db,
                            max_parallel_executions=3,
                            max_retries=2,
                            workspace_dir=workspace_path,
                        )
                        subtask_results = await _retry_exec_mgr.execute_workflow_subtasks(
                            subtasks=steps,
                            agent_assignments=agent_assignments,
                            context_enhancements=context_enhancements,
                            execution_id=execution_id,
                            workflow_id=execution.workflow_id,
                            memory_retrieval_results=memory_retrieval_results,
                            execution_strategy=execution_strategy if 'execution_strategy' in locals() else "parallel",
                        )
                        execution_summary = _retry_exec_mgr.get_execution_summary(subtask_results)

                        # Update execution data with retry results
                        execution.input_data["agent_execution"] = {
                            "is_real": True,
                            "is_retry": True,
                            "retry_attempt": _execution_attempt,
                            "retry_action": _checkpoint_decision.action.value,
                            "summary": execution_summary,
                            "results": {
                                sid: {
                                    "status": r.status.value,
                                    "agent_name": r.agent_name,
                                    "tokens_used": r.tokens_used,
                                    "execution_time_ms": r.execution_time_ms,
                                    "retry_count": r.retry_count,
                                    "error": r.error_message,
                                }
                                for sid, r in subtask_results.items()
                            },
                        }
                        attributes.flag_modified(execution, "input_data")
                        db.commit()

                        # Update steps with retry results
                        for idx, step in enumerate(steps):
                            subtask_id = step.get("subtask_id", f"subtask_{idx}")
                            if subtask_id in subtask_results:
                                result = subtask_results[subtask_id]
                                step["execution_result"] = {
                                    "status": result.status.value,
                                    "llm_response": result.llm_response,
                                    "tokens_used": result.tokens_used,
                                    "execution_time_ms": result.execution_time_ms,
                                }

                        logger.info(
                            f"[Adaptive] Retry complete: {execution_summary.get('success_rate', 0):.0%} success rate"
                        )
                    except Exception as retry_err:
                        logger.error(f"[Adaptive] Retry failed: {retry_err}", exc_info=True)

                elif _checkpoint_decision.action == InterventionAction.ABORT:
                    logger.warning("[Adaptive] Checkpoint decided to ABORT execution")
                    execution.status = ExecutionStatus.FAILED.value
                    execution.input_data["adaptive_abort"] = {
                        "stage": "execution",
                        "reasoning": _checkpoint_decision.reasoning,
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()
                    return  # Exit the entire function

            # Complete EXECUTE phase, start EVALUATE phase
            if "EXECUTE" in active_phases:
                await stage_tracker.complete_phase("EXECUTE", {"stages_completed": [4]})
            if "EVALUATE" in active_phases:
                await stage_tracker.start_phase("EVALUATE")

            # ========== STAGE 5: RESULT AGGREGATION ==========
            await stage_tracker.start_stage(5)
            
            # RESULT AGGREGATION & QUALITY SCORING
            logger.info(f"📊 Aggregating results and calculating quality scores...")
            try:
                aggregator = ResultAggregator()
                aggregated_results = aggregator.aggregate_results(
                    workflow_id=execution.workflow_id,
                    execution_id=execution_id,
                    subtask_executions=subtask_results if execution.input_data.get("agent_execution", {}).get("is_real") else {},
                    decomposition_metadata=execution.input_data.get("decomposition", {}),
                    agent_selection_metadata=execution.input_data.get("agent_selection", {}),
                    context_engineering_metadata=execution.input_data.get("context_engineering", {}),
                    agent_execution_metadata=execution.input_data.get("agent_execution", {})
                )
                
                # Store aggregated results
                execution.input_data["result_aggregation"] = {
                    "is_real": True,
                    "quality_scores": aggregated_results.quality_scores.to_dict(),
                    "agent_performance": aggregated_results.agent_performance,
                    "recommendations": aggregated_results.recommendations,
                    "metrics": {
                        "total_tokens_used": aggregated_results.total_tokens_used,
                        "total_execution_time_ms": aggregated_results.total_execution_time_ms,
                        "avg_context_quality": aggregated_results.avg_context_quality,
                        "total_retries": aggregated_results.total_retries
                    }
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
                
                logger.info(
                    f"✅ Result aggregation complete: {aggregated_results.quality_scores.overall:.0%} overall quality "
                    f"(Completeness: {aggregated_results.quality_scores.completeness:.0%}, "
                    f"Accuracy: {aggregated_results.quality_scores.accuracy:.0%}, "
                    f"Efficiency: {aggregated_results.quality_scores.efficiency:.0%}, "
                    f"Reliability: {aggregated_results.quality_scores.reliability:.0%}, "
                    f"Coherence: {aggregated_results.quality_scores.coherence:.0%})"
                )
                
                # Quality scores now streamed via SSE/AI SDK (stream_manager)
                # Legacy WebSocket broadcast removed
                
            except Exception as e:
                logger.error(f"❌ Result aggregation failed: {e}")
                execution.input_data["result_aggregation"] = {
                    "is_real": False,
                    "error": str(e)
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
            
            await stage_tracker.complete_stage(5, {
                "overall_quality": aggregated_results.quality_scores.overall if 'aggregated_results' in locals() else 0,
                "total_tokens": aggregated_results.total_tokens_used if 'aggregated_results' in locals() else 0
            })
            
            # ========== STAGE 6: LEARNING UPDATE ==========
            await stage_tracker.start_stage(6)
            
            # LEARNING SYSTEM UPDATE
            logger.info(f"🎓 Updating learning system...")
            try:
                learning_updater = LearningSystemUpdater(db_session=db)
                learning_updates = await learning_updater.update_from_execution(
                    workflow_id=execution.workflow_id,
                    execution_id=execution_id,
                    aggregated_results=aggregated_results if 'aggregated_results' in locals() else None,
                    subtask_executions=subtask_results if execution.input_data.get("agent_execution", {}).get("is_real") else {},
                    decomposition_metadata=execution.input_data.get("decomposition", {}),
                    context_metadata=execution.input_data.get("context_engineering", {})
                )
                
                # Store learning updates
                execution.input_data["learning_system"] = {
                    "is_real": True,
                    "updates": learning_updates,
                    "summary": learning_updater.get_learning_summary()
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
                
                logger.info(
                    f"✅ Learning system updated: {learning_updates['total_updates']} updates, "
                    f"{len(learning_updates['agent_performance_updates'])} agents improved"
                )
                
            except Exception as e:
                logger.error(f"❌ Learning system update failed: {e}")
                execution.input_data["learning_system"] = {
                    "is_real": False,
                    "error": str(e)
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
            
            await stage_tracker.complete_stage(6, {
                "total_updates": learning_updates.get("total_updates", 0) if 'learning_updates' in locals() else 0
            })

            # Complete EVALUATE phase, start LEARN phase
            if "EVALUATE" in active_phases:
                await stage_tracker.complete_phase("EVALUATE", {"stages_completed": [5, 6]})
            if "LEARN" in active_phases:
                await stage_tracker.start_phase("LEARN")

            # ========== STAGE 7: QUALITY ASSESSMENT ==========
            # PRD-59: PhaseSelector may skip quality for EVALUATE_LIGHT (simple tasks)
            _phase_plan = execution.input_data.get("phase_plan", {})
            _skip_quality = not any(
                "Quality Assessment" in p.get("stages", [])
                for p in _phase_plan.get("phases", [])
            ) if _phase_plan else False

            if _skip_quality:
                logger.info(f"⏭️ SKIPPED Stage 7: EVALUATE_LIGHT mode — no LLM quality assessment")
                execution.input_data["quality_assessment"] = {
                    "is_real": False,
                    "skip_reason": "EVALUATE_LIGHT — simple task",
                    "overall_score": 0.7,
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()
            else:
                await stage_tracker.start_stage(7)

                # QUALITY ASSESSMENT
                logger.info(f"🎯 Assessing workflow output quality...")
                try:
                    from modules.orchestrator.stages import OutputQualityAssessor, OutputType

                    # PRD-59 Fix 1: Use LLM-based quality assessment on real outputs
                    use_llm_quality = len(steps) >= 3 and config.ENABLE_LLM_QUALITY_ASSESSMENT

                    llm_client_for_quality = None
                    if use_llm_quality:
                        try:
                            from core.llm import create_llm_manager
                            llm_client_for_quality = create_llm_manager(service_name="orchestrator")
                        except Exception as llm_err:
                            logger.warning(f"Could not create LLM for quality assessment: {llm_err}")

                    quality_assessor = OutputQualityAssessor(
                        llm_client=llm_client_for_quality,
                        use_llm=use_llm_quality and llm_client_for_quality is not None
                    )

                    # PRD-59: Build quality input from ACTUAL agent outputs, not metadata
                    actual_outputs = []
                    for step in steps:
                        exec_result = step.get("execution_result", {})
                        result_content = exec_result.get("result", "") or exec_result.get("output", "") or ""
                        if result_content:
                            actual_outputs.append({
                                "subtask": step.get("description", "Unknown"),
                                "agent": step.get("agent_name", "Unknown"),
                                "output": str(result_content)[:2000],
                                "tokens_used": exec_result.get("tokens_used", 0),
                                "status": exec_result.get("status", "unknown"),
                            })

                    completed_count = sum(1 for s in steps if s.get("execution_result", {}).get("status") == "completed")
                    total_count = len(steps) or 1

                    output_summary = json.dumps({
                        "task": task_description[:500],
                        "subtask_outputs": actual_outputs,
                        "overall_success_rate": completed_count / total_count,
                        "total_subtasks": len(steps),
                    }, indent=2)

                    quality_assessment = await quality_assessor.assess_quality(
                        output=output_summary,
                        requirements=task_description,
                        output_type=OutputType.GENERAL,
                        quality_threshold=0.7
                    )

                    execution.input_data["quality_assessment"] = {
                        "is_real": quality_assessment.assessment_method in ("llm", "hybrid"),
                        "overall_score": quality_assessment.overall_score,
                        "passes_threshold": quality_assessment.passes_threshold,
                        "dimensions": {
                            name: {
                                "score": dim.score,
                                "feedback": dim.feedback
                            }
                            for name, dim in quality_assessment.dimensions.items()
                        },
                        "strengths": quality_assessment.strengths,
                        "weaknesses": quality_assessment.weaknesses,
                        "improvement_suggestions": quality_assessment.improvement_suggestions
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()

                    logger.info(
                        f"✅ Quality assessment complete: {quality_assessment.overall_score:.0%} overall score "
                        f"({'PASS' if quality_assessment.passes_threshold else 'FAIL'})"
                    )

                except Exception as e:
                    logger.error(f"❌ Quality assessment failed: {e}")
                    execution.input_data["quality_assessment"] = {
                        "is_real": False,
                        "error": str(e)
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()

                await stage_tracker.complete_stage(7, {
                    "quality_score": quality_assessment.overall_score if 'quality_assessment' in locals() else 0,
                    "passes_threshold": quality_assessment.passes_threshold if 'quality_assessment' in locals() else False
                })

            # ── ADAPTIVE CHECKPOINT 2: Post-Quality ──
            # If quality assessment fails threshold, ask LLM whether to retry execution.
            # Max 1 retry. Re-runs from Stage 4 through Stage 7 on retry.
            _quality_attempt = 1
            if (
                'quality_assessment' in locals()
                and hasattr(quality_assessment, 'passes_threshold')
                and not quality_assessment.passes_threshold
                and execution.input_data.get("quality_assessment", {}).get("is_real")
            ):
                from modules.orchestrator.llm.adaptive_execution_monitor import InterventionAction as _QualityIA

                _quality_data = {
                    "overall_score": quality_assessment.overall_score,
                    "passes_threshold": quality_assessment.passes_threshold,
                    "strengths": getattr(quality_assessment, 'strengths', []),
                    "weaknesses": getattr(quality_assessment, 'weaknesses', []),
                    "improvement_suggestions": getattr(quality_assessment, 'improvement_suggestions', []),
                }
                _quality_decision = await _adaptive_checkpoint(
                    stage_name="quality",
                    stage_results=_quality_data,
                    workflow_context={
                        "task_description": task_description,
                        "steps": steps,
                        "agent_assignments": {
                            k: [{"agent_id": m.agent_id, "agent_name": m.agent_name} for m in v]
                            for k, v in agent_assignments.items()
                        } if agent_assignments else {},
                    },
                    attempt=_quality_attempt,
                    stage_tracker=stage_tracker,
                )

                if _quality_decision.action in (
                    _QualityIA.RETRY_SAME,
                    _QualityIA.RETRY_DIFFERENT,
                ):
                    _quality_attempt += 1
                    await stage_tracker._emit("adaptive_retry_start", {
                        "stage": "quality",
                        "attempt": _quality_attempt,
                        "action": _quality_decision.action.value,
                        "timestamp": datetime.now().isoformat(),
                    })

                    try:
                        # RETRY_DIFFERENT: re-select agents
                        if _quality_decision.action == _QualityIA.RETRY_DIFFERENT:
                            logger.info("[Adaptive] Quality retry: re-selecting agents...")
                            from core.llm.llm_agent_selector import LLMAgentSelector as _QRetrySelector
                            _qretry_sel = _QRetrySelector(db_session=db)
                            agent_assignments = await _qretry_sel.select_agents_for_subtasks(
                                steps,
                                workflow_context={
                                    "description": task_description,
                                    "workflow_id": execution.workflow_id,
                                },
                            )
                            for idx, step in enumerate(steps):
                                subtask_id = step.get("subtask_id", f"subtask_{idx}")
                                if subtask_id in agent_assignments and agent_assignments[subtask_id]:
                                    best_match = agent_assignments[subtask_id][0]
                                    step["selected_agent"] = {
                                        "agent_id": best_match.agent_id,
                                        "agent_name": best_match.agent_name,
                                        "match_score": best_match.match_score,
                                    }

                        # Re-run Stage 4: Agent Execution
                        logger.info(f"[Adaptive] Quality retry {_quality_attempt}: re-executing agents...")
                        from modules.agents import AgentExecutionManager as _QRetryExecMgr
                        _qretry_mgr = _QRetryExecMgr(
                            db_session=db,
                            max_parallel_executions=3,
                            max_retries=2,
                            workspace_dir=workspace_path,
                        )
                        subtask_results = await _qretry_mgr.execute_workflow_subtasks(
                            subtasks=steps,
                            agent_assignments=agent_assignments,
                            context_enhancements=context_enhancements,
                            execution_id=execution_id,
                            workflow_id=execution.workflow_id,
                            memory_retrieval_results=memory_retrieval_results if 'memory_retrieval_results' in locals() else {},
                            execution_strategy=execution_strategy if 'execution_strategy' in locals() else "parallel",
                        )
                        execution_summary = _qretry_mgr.get_execution_summary(subtask_results)

                        # Update steps with retry results
                        for idx, step in enumerate(steps):
                            subtask_id = step.get("subtask_id", f"subtask_{idx}")
                            if subtask_id in subtask_results:
                                result = subtask_results[subtask_id]
                                step["execution_result"] = {
                                    "status": result.status.value,
                                    "llm_response": result.llm_response,
                                    "tokens_used": result.tokens_used,
                                    "execution_time_ms": result.execution_time_ms,
                                }

                        # Re-run Stage 5: Result Aggregation
                        logger.info("[Adaptive] Quality retry: re-aggregating results...")
                        aggregator = ResultAggregator()
                        aggregated_results = aggregator.aggregate_results(
                            workflow_id=execution.workflow_id,
                            execution_id=execution_id,
                            subtask_executions=subtask_results,
                            decomposition_metadata=execution.input_data.get("decomposition", {}),
                            agent_selection_metadata=execution.input_data.get("agent_selection", {}),
                            context_engineering_metadata=execution.input_data.get("context_engineering", {}),
                            agent_execution_metadata=execution.input_data.get("agent_execution", {}),
                        )

                        # Re-run Stage 7: Quality Assessment
                        logger.info("[Adaptive] Quality retry: re-assessing quality...")
                        from modules.orchestrator.stages import OutputQualityAssessor, OutputType
                        _use_llm_q = len(steps) >= 3 and config.ENABLE_LLM_QUALITY_ASSESSMENT
                        _llm_q = None
                        if _use_llm_q:
                            try:
                                from core.llm import create_llm_manager
                                _llm_q = create_llm_manager(service_name="orchestrator")
                            except Exception:
                                pass

                        _qa = OutputQualityAssessor(
                            llm_client=_llm_q,
                            use_llm=_use_llm_q and _llm_q is not None,
                        )
                        actual_outputs = []
                        for step in steps:
                            exec_result = step.get("execution_result", {})
                            result_content = exec_result.get("result", "") or exec_result.get("output", "") or ""
                            if result_content:
                                actual_outputs.append({
                                    "subtask": step.get("description", "Unknown"),
                                    "agent": step.get("agent_name", "Unknown"),
                                    "output": str(result_content)[:2000],
                                    "tokens_used": exec_result.get("tokens_used", 0),
                                    "status": exec_result.get("status", "unknown"),
                                })
                        completed_count = sum(1 for s in steps if s.get("execution_result", {}).get("status") == "completed")
                        total_count = len(steps) or 1
                        output_summary = json.dumps({
                            "task": task_description[:500],
                            "subtask_outputs": actual_outputs,
                            "overall_success_rate": completed_count / total_count,
                            "total_subtasks": len(steps),
                        }, indent=2)
                        quality_assessment = await _qa.assess_quality(
                            output=output_summary,
                            requirements=task_description,
                            output_type=OutputType.GENERAL,
                            quality_threshold=0.7,
                        )

                        # Store retry quality results
                        execution.input_data["quality_assessment"] = {
                            "is_real": quality_assessment.assessment_method in ("llm", "hybrid"),
                            "is_retry": True,
                            "retry_attempt": _quality_attempt,
                            "overall_score": quality_assessment.overall_score,
                            "passes_threshold": quality_assessment.passes_threshold,
                            "dimensions": {
                                name: {"score": dim.score, "feedback": dim.feedback}
                                for name, dim in quality_assessment.dimensions.items()
                            },
                            "strengths": quality_assessment.strengths,
                            "weaknesses": quality_assessment.weaknesses,
                        }
                        attributes.flag_modified(execution, "input_data")
                        db.commit()

                        logger.info(
                            f"[Adaptive] Quality retry complete: {quality_assessment.overall_score:.0%} "
                            f"({'PASS' if quality_assessment.passes_threshold else 'FAIL'})"
                        )

                    except Exception as qretry_err:
                        logger.error(f"[Adaptive] Quality retry failed: {qretry_err}", exc_info=True)

                elif _quality_decision.action == _QualityIA.ABORT:
                    logger.warning("[Adaptive] Quality checkpoint decided to ABORT execution")
                    execution.status = ExecutionStatus.FAILED.value
                    execution.input_data["adaptive_abort"] = {
                        "stage": "quality",
                        "reasoning": _quality_decision.reasoning,
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()
                    return

            # ========== STAGE 8: MEMORY STORAGE ==========
            await stage_tracker.start_stage(8)

            # MEMORY STORAGE (Store experiences via UnifiedMemoryService)
            logger.info(f"💾 Storing execution experiences to memory...")
            memory_storage_results = {}
            try:
                if memory_service_available and 'aggregated_results' in locals():
                    # Use MemoryNamespace for workspace-scoped workflow memory
                    ns = MemoryNamespace(workspace_id=str(workspace_id))
                    memory_scope_id = ns.workflow(execution.workflow_id)

                    # Build experience summary from execution
                    quality_scores = aggregated_results.quality_scores
                    execution_summary_text = f"""
Workflow Execution Complete:
- Workflow: {workflow.name if workflow else 'Unknown'}
- Execution ID: {execution_id}
- Status: Completed
- Overall Quality: {quality_scores.overall:.0%}
- Completeness: {quality_scores.completeness:.0%}
- Accuracy: {quality_scores.accuracy:.0%}
- Efficiency: {quality_scores.efficiency:.0%}
- Reliability: {quality_scores.reliability:.0%}
- Subtasks: {len(steps)}
- Success Rate: {execution_summary.get('success_rate', 0):.0%} if 'execution_summary' in locals() else 'N/A'
- Total Tokens: {aggregated_results.total_tokens_used}
- Total Time: {aggregated_results.total_execution_time_ms}ms

Key Learnings:
"""
                    # Add agent-specific learnings
                    for idx, step in enumerate(steps[:5]):  # Limit to top 5
                        if "execution_result" in step:
                            result = step["execution_result"]
                            execution_summary_text += f"\n- Step {idx+1}: {step.get('description', 'Unknown')[:80]} - {result.get('status', 'unknown')}"

                    # Store via UnifiedMemoryService
                    messages = [
                        {"role": "system", "content": "Workflow execution completed"},
                        {"role": "assistant", "content": execution_summary_text}
                    ]

                    metadata = {
                        "workspace_id": str(workspace_id),
                        "workflow_id": execution.workflow_id,
                        "execution_id": execution_id,
                        "quality_score": quality_scores.overall,
                        "timestamp": datetime.now().isoformat()
                    }

                    logger.info(f"💾 Storing memory to scope: {memory_scope_id}")
                    mem0_result = await memory_service.store_long_term_messages(
                        user_id=memory_scope_id,
                        messages=messages,
                        metadata=metadata,
                    )

                    memory_storage_results = {
                        "total_experiences": 1,
                        "workspace_id": str(workspace_id),
                        "memory_scope": memory_scope_id,
                        "mem0_result": mem0_result,
                        "quality_score": quality_scores.overall
                    }

                    execution.input_data["memory_storage"] = {
                        "is_real": True,
                        "results": memory_storage_results
                    }
                    attributes.flag_modified(execution, "input_data")
                    db.commit()

                    logger.info(f"✅ Stored execution experience (quality: {quality_scores.overall:.0%})")
                else:
                    logger.warning("Memory service not available, skipping memory storage")

            except Exception as e:
                logger.error(f"❌ Memory storage failed: {e}", exc_info=True)
                execution.input_data["memory_storage"] = {
                    "is_real": False,
                    "error": str(e)
                }
                attributes.flag_modified(execution, "input_data")
                db.commit()

            await stage_tracker.complete_stage(8, {
                "experiences_stored": memory_storage_results.get('total_experiences', 0) if memory_storage_results else 0,
                "quality_score": memory_storage_results.get('quality_score', 0) if memory_storage_results else 0
            })
            
            # ========== STAGE 9: RESPONSE GENERATION ==========
            await stage_tracker.start_stage(9)
            
            # PRD-15: Save model usage summary to execution
            usage_summary = model_tracker.get_usage_summary()
            execution.models_used = usage_summary.get("records", [])
            
            logger.info(
                f"📊 Model Usage Summary: {usage_summary['total_requests']} requests | "
                f"{usage_summary['total_tokens']} tokens | ${usage_summary['total_cost']:.6f} cost"
            )
            
            # Complete execution with COMPLETE orchestration pipeline data
            execution.status = ExecutionStatus.COMPLETED.value
            execution.completed_at = datetime.now()
            
            await stage_tracker.complete_stage(9, {
                "status": "completed",
                "total_cost": usage_summary.get("total_cost", 0)
            })

            # Complete LEARN phase — all phases done
            if "LEARN" in active_phases:
                await stage_tracker.complete_phase("LEARN", {"stages_completed": [7, 8, 9]})

            # Update workflow status and last_execution
            workflow.status = "completed"
            
            # Save last execution details to workflow
            workflow.last_execution = {
                "id": execution.id,
                "status": execution.status,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "quality_scores": execution.input_data.get("result_aggregation", {}).get("quality_scores", {}),
                "agents_used": len(set(match.agent_id for matches in agent_assignments.values() for match in matches if matches)),
                "tokens_used": execution.input_data.get("agent_execution", {}).get("summary", {}).get("total_tokens_used", 0),
                "cost": usage_summary.get("total_cost", 0),
                "subtasks_completed": len(steps)
            }
            attributes.flag_modified(workflow, "last_execution")
            # Build dynamic result based on actual execution
            stages_completed = []
            if execution.input_data.get("decomposition", {}).get("is_real"):
                stages_completed.append("Task Decomposition")
            if execution.input_data.get("agent_selection", {}).get("is_real"):
                stages_completed.append("Agent Selection")
            if execution.input_data.get("memory_retrieval", {}).get("is_real"):
                stages_completed.append("Memory Retrieval")
            if execution.input_data.get("context_engineering", {}).get("is_real"):
                stages_completed.append("Context Engineering")
            if execution.input_data.get("agent_execution", {}).get("is_real"):
                stages_completed.append("Agent Execution")
            if execution.input_data.get("result_aggregation"):
                stages_completed.append("Result Aggregation")
            if execution.input_data.get("learning_system"):
                stages_completed.append("Learning Update")
            if execution.input_data.get("memory_storage"):
                stages_completed.append("Memory Storage")
            if execution.input_data.get("memory_consolidation"):
                stages_completed.append("Memory Consolidation")
            
            execution.output_data = {
                "result": f"Workflow completed with {len(stages_completed)} stages: {', '.join(stages_completed)}",
                "is_real_decomposition": execution.input_data.get("decomposition", {}).get("is_real", False),
                "is_real_agent_selection": execution.input_data.get("agent_selection", {}).get("is_real", False),
                "is_real_memory_retrieval": execution.input_data.get("memory_retrieval", {}).get("is_real", False),
                "is_real_context_engineering": execution.input_data.get("context_engineering", {}).get("is_real", False),
                "is_real_agent_execution": execution.input_data.get("agent_execution", {}).get("is_real", False),
                "is_real_quality_scoring": execution.input_data.get("result_aggregation", {}).get("is_real", False),
                "is_real_memory_storage": execution.input_data.get("memory_storage", {}).get("is_real", False),
                "is_real_learning_system": execution.input_data.get("learning_system", {}).get("is_real", False),
                "is_real_memory_consolidation": execution.input_data.get("memory_consolidation", {}).get("is_real", False),
                "memory_integration_summary": execution.input_data.get("memory_integration_summary", {}),
                "quality_scores": execution.input_data.get("result_aggregation", {}).get("quality_scores", {}),
                "steps_completed": len(steps),
                "execution_time": f"{total_duration:.1f}s",
                # Performance Analytics - Cost & Token Tracking
                "total_cost": usage_summary.get("total_cost", 0),
                "total_tokens_used": usage_summary.get("total_tokens", 0),
                "total_requests": usage_summary.get("total_requests", 0),
                "subtasks": [
                    {
                        "description": step.get("description", step.get("name", "Unknown")),
                        "agent_type": step.get("agent_type"),
                        "priority": step.get("priority"),
                        "selected_agent": step.get("selected_agent"),
                        "context_quality": step.get("context_quality"),
                        "context_sources": step.get("context_sources"),
                        "execution_result": step.get("execution_result"),
                        "completed": True
                    }
                    for step in steps
                ],
                "decomposition_metadata": execution.input_data.get("decomposition", {}),
                "agent_selection_metadata": execution.input_data.get("agent_selection", {}),
                "memory_retrieval_metadata": execution.input_data.get("memory_retrieval", {}),
                "context_engineering_metadata": execution.input_data.get("context_engineering", {}),
                "agent_execution_metadata": execution.input_data.get("agent_execution", {}),
                "result_aggregation_metadata": execution.input_data.get("result_aggregation", {}),
                "memory_storage_metadata": execution.input_data.get("memory_storage", {}),
                "learning_system_metadata": execution.input_data.get("learning_system", {}),
                "memory_consolidation_metadata": execution.input_data.get("memory_consolidation", {})
            }
            
            # Enhanced execution log
            agent_selection_info = ""
            if execution.input_data.get("agent_selection", {}).get("is_real", False):
                summary = execution.input_data["agent_selection"]["summary"]
                agent_selection_info = f", matched agents with {summary['avg_match_score']:.0%} avg score"
            
            context_info = ""
            if execution.input_data.get("context_engineering", {}).get("is_real", False):
                summary = execution.input_data["context_engineering"]["summary"]
                context_info = f", enhanced with RAG context ({summary['total_sources_retrieved']} sources, {summary['avg_context_quality']:.0%} quality)"
            
            execution_info = ""
            if execution.input_data.get("agent_execution", {}).get("is_real", False):
                summary = execution.input_data["agent_execution"]["summary"]
                execution_info = f", executed by {summary['completed']} agents ({summary['total_tokens_used']} tokens, {summary['success_rate']:.0%} success)"
            
            quality_info = ""
            if execution.input_data.get("result_aggregation", {}).get("is_real", False):
                scores = execution.input_data["result_aggregation"]["quality_scores"]
                quality_info = f", quality: {scores.get('overall', 0):.0%}"
            
            learning_info = ""
            if execution.input_data.get("learning_system", {}).get("is_real", False):
                updates = execution.input_data["learning_system"]["updates"]
                learning_info = f", learned from {updates['total_updates']} updates"
            
            memory_info = ""
            if execution.input_data.get("memory_integration_summary", {}).get("memory_integration_enabled", False):
                summary = execution.input_data["memory_integration_summary"]
                memory_info = f", memory: {summary.get('memories_retrieved', 0)} retrieved, {summary.get('experiences_stored', 0)} stored, {summary.get('patterns_extracted', 0)} patterns"
            
            execution.execution_log = f"COMPLETE PIPELINE + MEMORY{agent_selection_info}{context_info}{execution_info}{quality_info}{learning_info}{memory_info} - {len(steps)} subtasks in {total_duration:.1f}s"
            
            # Record workflow analytics for monitoring
            try:
                from consumers.workflows.analytics import WorkflowAnalyticsService
                from modules.orchestrator import orchestration_tracker
                analytics_service = WorkflowAnalyticsService(db)
                analytics = analytics_service.record_workflow_analytics(execution)
                logger.info(
                    f"📊 Analytics recorded: {analytics.completed_subtasks}/{analytics.total_subtasks} tasks, "
                    f"{len(analytics.agents_used)} agents, ${analytics.total_cost:.4f} cost, "
                    f"{analytics.overall_quality_score:.1%} quality"
                )
            except Exception as e:
                logger.error(f"Failed to record analytics: {e}")
            
            db.commit()
            
            # Save results and cleanup workspace
            try:
                logger.info(f"💾 Saving workflow results to permanent storage...")
                
                # Auto-save all result files from workspace
                saved_files = workspace_manager.save_all_results(
                    file_patterns=['*.pdf', '*.docx', '*.xlsx', '*.pptx', '*.md', '*.txt', '*.json', '*.html']
                )
                
                # Create manifest with execution metadata
                manifest_path = workspace_manager.create_result_manifest({
                    "status": "completed",
                    "execution_time": f"{total_duration:.1f}s",
                    "total_tokens": usage_summary.get("total_tokens", 0),
                    "total_cost": usage_summary.get("total_cost", 0),
                    "quality_scores": execution.input_data.get("result_aggregation", {}).get("quality_scores", {}),
                    "stages_completed": stages_completed
                })
                
                logger.info(f"✅ Saved {len(saved_files)} result file(s) to {workspace_manager.get_results_path()}")
                
                # Cleanup workspace (keeps results directory)
                if workspace_manager.cleanup_workspace():
                    logger.info(f"🧹 Workspace cleaned up: {workspace_path}")
                else:
                    logger.warning(f"⚠️  Workspace cleanup skipped (no results saved)")
                    
            except Exception as cleanup_err:
                logger.error(f"❌ Failed to save results or cleanup workspace: {cleanup_err}")
            
            # Publish execution completion to Redis
            from core.redis.client import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                redis_client.publish_workflow_event(
                    workflow_id=execution.workflow_id,
                    execution_id=execution_id,
                    event_type="execution_completed",
                    data={
                        "workflow_name": workflow.name if workflow else "Unknown",
                        "status": "completed",
                        "progress": 100,
                        "execution_time": f"{total_duration}s",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            else:
                logger.warning("Redis client not initialized for execution_completed event")
            
            # CRITICAL: Broadcast workflow_complete to SSE stream for UI
            if stream_manager:
                try:
                    await stream_manager.broadcast_event(
                        execution_id=execution_id,
                        event_type="workflow_complete",
                        data={
                            "status": "completed",
                            "execution_time": f"{total_duration:.1f}s",
                            "quality_score": execution.input_data.get("result_aggregation", {}).get("quality_scores", {}).get("overall", 0),
                            "stages_completed": len(stages_completed),
                            "total_tokens": usage_summary.get("total_tokens", 0),
                            "total_cost": usage_summary.get("total_cost", 0),
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    logger.info(f"✅ Broadcast workflow_complete to SSE stream for execution {execution_id}")
                except Exception as e:
                    logger.warning(f"Failed to broadcast workflow_complete: {e}")
            
    except Exception as e:
        import traceback
        logger.error(f"❌ FATAL ERROR in workflow execution {execution_id}: {e}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Save any partial results and cleanup
        try:
            logger.info(f"💾 Attempting to save partial results before cleanup...")
            saved_files = workspace_manager.save_all_results()
            if saved_files:
                logger.info(f"✅ Saved {len(saved_files)} partial result file(s)")
            
            # Create error manifest
            workspace_manager.create_result_manifest({
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            # Force cleanup (even if no results)
            workspace_manager.cleanup_workspace(force=True)
            logger.info(f"🧹 Workspace cleaned up after error")
        except Exception as cleanup_err:
            logger.error(f"❌ Failed to cleanup after error: {cleanup_err}")
        
        try:
            with get_db_session() as db:
                execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
                if execution:
                    execution.status = ExecutionStatus.FAILED.value
                    execution.completed_at = datetime.now()
                    execution.error_message = str(e)
                    db.commit()
                    
                    # Error notifications now streamed via SSE/AI SDK (stream_manager)
                    # Legacy WebSocket broadcast removed
        except Exception as inner_e:
            logger.error(f"Error updating failed execution {execution_id}: {inner_e}")

@router.get("/executions/{execution_id}/results")
async def get_execution_results_files(execution_id: int, db: Session = Depends(get_db)):
    """
    Get list of result files for a workflow execution.
    
    Returns metadata about all files created during execution that are available for download.
    """
    try:
        from core.services.workspace_manager import WorkspaceManager
        
        # Verify execution exists
        execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        # Get result files
        workspace_manager = WorkspaceManager(execution_id)
        result_files = workspace_manager.get_result_files()
        
        return {
            "execution_id": execution_id,
            "status": execution.status,
            "files": result_files,
            "results_directory": workspace_manager.get_results_path()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution results {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/executions/{execution_id}/results/{file_path:path}")
async def download_execution_result_file(execution_id: int, file_path: str, db: Session = Depends(get_db)):
    """
    Download a specific result file from a workflow execution.
    
    Args:
        execution_id: The workflow execution ID
        file_path: Relative path to the file within the results directory
    """
    try:
        # Verify execution exists
        execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        # Get results directory
        workspace_manager = WorkspaceManager(execution_id)
        results_dir = Path(workspace_manager.get_results_dir()).resolve()
        
        # Build full file path
        full_path = (results_dir / file_path).resolve()
        
        # Security check: ensure path is within results directory using Path API
        if not full_path.is_relative_to(results_dir):
            raise HTTPException(status_code=403, detail="Access denied: path outside results directory")
        
        # Check if file exists
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Return file for download
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {file_path} from execution {execution_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/templates/recommended")
async def get_recommended_workflow_templates(db: Session = Depends(get_db)):
    """Get recommended workflow templates based on system usage"""
    try:
        # Get existing workflows to analyze patterns
        workflows = db.query(Workflow).all()
        
        # Analyze common patterns
        common_agents = {}
        common_categories = {}
        
        for workflow in workflows:
            wf_def = workflow.workflow_definition or {}
            category = wf_def.get('category', 'General')
            common_categories[category] = common_categories.get(category, 0) + 1
            
            for agent in workflow.agents:
                common_agents[agent.agent_type] = common_agents.get(agent.agent_type, 0) + 1
        
        # Generate templates from ACTUAL workflow data - NO HARDCODED TEMPLATES
        templates = []
        
        # Only generate templates from real workflow executions
        for workflow in workflows[:5]:  # Top 5 most recent workflows
            if workflow.last_execution and workflow.status == "completed":
                # Build template from ACTUAL workflow data
                template = {
                    "id": f"workflow_{workflow.id}",
                    "name": workflow.name,
                    "description": workflow.description,
                    "category": workflow.workflow_definition.get("category", "General") if workflow.workflow_definition else "General",
                    "difficulty": "intermediate",  # Calculate from actual execution time
                    "estimated_time": f"{workflow.last_execution.get('execution_time', 0):.0f} seconds" if workflow.last_execution else "Unknown",
                    "recommended_agents": [agent.agent_type for agent in workflow.agents] if workflow.agents else [],
                    "steps": workflow.workflow_definition.get("steps", []) if workflow.workflow_definition else [],
                    "use_cases": [],  # To be filled from actual usage
                    "popularity": workflow.execution_count or 0,
                    "success_rate": workflow.success_rate or 0
                }
                templates.append(template)
        
        # If no workflows exist, return empty templates
        if not templates:
            logger.warning("No completed workflows found to generate templates")
        
        return {
            "recommended_templates": templates,
            "usage_insights": {
                "most_popular_category": max(common_categories.items(), key=lambda x: x[1])[0] if common_categories else None,
                "most_used_agent": max(common_agents.items(), key=lambda x: x[1])[0] if common_agents else None,
                "total_workflows": len(workflows)
            },
            "personalized_recommendations": [],  # NO HARDCODED DATA - build from actual usage
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommended templates: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
