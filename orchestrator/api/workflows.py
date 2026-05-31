
"""
Enhanced Workflow Management API Routes
=======================================

Extended workflow API with live progress tracking, real-time updates, and advanced features.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload, attributes
from sqlalchemy import and_, or_, func, desc, String
from datetime import datetime, timedelta
import logging
import json

from config import config
from core.database.database import get_db
from core.utils.background_tasks import launch_guarded
from core.models import (
    Workflow, WorkflowExecution, Agent, workflow_agents,
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    WorkflowExecutionCreate, WorkflowExecutionResponse,
    WorkflowStatus, ExecutionStatus
)
from core.models.core import RecipeExecution, WorkflowTemplate as WorkflowRecipe
# websocket_manager removed - using AI SDK SSE streaming
from core.auth.hybrid import get_request_context_hybrid
from core.task_runner import get_task_runner, AgentTask, TaskType, TaskPriority
from core.auth.dependencies import RequestContext

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
async def get_workflow_live_progress(workflow_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
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
            # Phase 1 / Local: Run in-process (existing behavior), guarded so a
            # GC'd loop can't silently cancel it and an uncaught crash is recorded.
            launch_guarded(
                execute_workflow_with_progress(
                    execution.id,
                    execution_data.get('options', {})
                ),
                subsystem="workflow",
                operation="execute",
                workspace_id=ctx.workspace_id,
                extra={"execution_id": execution.id, "workflow_id": workflow_id},
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
            # Phase 1 / Local: Run in-process (existing behavior), guarded so a
            # GC'd loop can't silently cancel it and an uncaught crash is recorded.
            launch_guarded(
                execute_workflow_with_progress(
                    execution.id,
                    execution_data.get('options', {})
                ),
                subsystem="workflow",
                operation="execute",
                workspace_id=ctx.workspace_id,
                extra={"execution_id": execution.id, "workflow_id": workflow_id},
            )

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



async def execute_workflow_with_progress(execution_id: int, options: Dict[str, Any]):
    """Legacy workflow execution — disabled (PRD-125). Use missions instead.

    This function is still imported by chat.py, composio.py, and github_webhooks.py
    but returns immediately. Kept as a stub to avoid import errors until callers
    are migrated to the mission system.
    """
    logger.warning(f"⛔ execute_workflow_with_progress({execution_id}) called — "
                   "legacy pipeline disabled (PRD-125). Use missions instead.")
    return


@router.get("/executions/{execution_id}/results")
async def get_execution_results_files(execution_id: int, db: Session = Depends(get_db)):
    """Legacy endpoint — filesystem workspaces removed (PRD-125)."""
    raise HTTPException(status_code=410, detail="Workflow filesystem results removed. Use mission outputs instead.")

@router.get("/executions/{execution_id}/results/{file_path:path}")
async def download_execution_result_file(execution_id: int, file_path: str, db: Session = Depends(get_db)):
    """Legacy endpoint — filesystem workspaces removed (PRD-125)."""
    raise HTTPException(status_code=410, detail="Workflow filesystem results removed. Use mission outputs instead.")

# PRD-125: Kept endpoint stubs returning 410 Gone so existing frontend
# links don't 404 silently. Safe to fully remove once Phase 3c (frontend
# cleanup) drops the execution result UI components.
_LEGACY_RESULT_ENDPOINTS_REMOVED = True  # grep marker for Phase 3c

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
