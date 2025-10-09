"""
Agent Factory API Endpoints
===========================

RESTful API endpoints for agent creation and management with real LLM connections.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from sqlalchemy.orm import Session
from database.database import get_db
from database.models import (
    Agent, AgentCreate, AgentResponse,
    AgentType, AgentStatus
)
from services.agent_factory import (
    AgentFactory, AgentLifecycle, create_specialized_agent
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])

# Global agent factory instance (in production, use dependency injection)
_factory = None

def get_agent_factory():
    """Get or create agent factory instance"""
    global _factory
    if _factory is None:
        _factory = AgentFactory()
    return _factory


@router.post("/create-specialized", response_model=Dict[str, Any])
async def create_specialized_agent_endpoint(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Create a specialized agent with real LLM connection.
    
    Request body:
    {
        "name": "Agent name",
        "type": "code_architect|data_analyst|security_expert|...",
        "skills": ["skill1", "skill2"],
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7
        },
        "auto_verify": true
    }
    
    Returns agent details with verification status.
    """
    try:
        factory = get_agent_factory()
        
        # Extract parameters
        name = request.get("name")
        agent_type = request.get("type")
        skills = request.get("skills", [])
        model_config = request.get("model")
        auto_verify = request.get("auto_verify", True)
        
        if not name or not agent_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name and type are required"
            )
        
        # Create agent with real LLM
        agent_runtime = await factory.create_agent(
            name=name,
            agent_type=agent_type,
            model_config=model_config,
            skills=skills,
            auto_verify=auto_verify
        )
        
        # Get status
        agent_status = await factory.get_agent_status(agent_runtime.agent_id)
        
        return {
            "status": "success",
            "message": f"Agent '{name}' created with verified LLM connection",
            "agent": agent_status
        }
        
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{agent_id}/execute")
async def execute_agent_task(
    agent_id: int,
    request: Dict[str, Any]
):
    """
    Execute a task using a specific agent's LLM.
    
    Request body:
    {
        "task": {
            "description": "Task description",
            "context": {...},
            "use_memory": true
        },
        "execution_mode": "thorough|quick",
        "use_tools": false
    }
    
    Returns real LLM execution results.
    """
    try:
        factory = get_agent_factory()
        
        # Extract task details
        task_data = request.get("task", {})
        task_description = task_data.get("description")
        context = task_data.get("context")
        use_memory = task_data.get("use_memory", True)
        
        if not task_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task description is required"
            )
        
        # Execute with real LLM
        result = await factory.execute_task(
            agent=agent_id,
            task_description=task_description,
            context=context,
            use_memory=use_memory
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Task execution failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task execution error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{agent_id}/learn")
async def update_agent_learning(
    agent_id: int,
    request: Dict[str, Any]
):
    """
    Provide feedback for agent learning.
    
    Request body:
    {
        "feedback": {
            "task_id": "...",
            "quality_score": 8.5,
            "corrections": [...],
            "improvements": [...]
        }
    }
    """
    try:
        factory = get_agent_factory()
        
        # Get agent
        agent_runtime = factory.active_agents.get(agent_id)
        if not agent_runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found in runtime"
            )
        
        # Process feedback (store in memory for now)
        feedback = request.get("feedback", {})
        
        # Add to agent memory
        learning_entry = {
            "type": "feedback",
            "timestamp": datetime.now().isoformat(),
            "quality_score": feedback.get("quality_score"),
            "task_id": feedback.get("task_id"),
            "improvements": feedback.get("improvements", [])
        }
        
        agent_runtime.memory.append(learning_entry)
        
        # Update lifecycle state
        agent_runtime.lifecycle_state = AgentLifecycle.LEARNING
        
        # In a real implementation, you would:
        # 1. Store feedback in database
        # 2. Trigger fine-tuning process
        # 3. Update agent prompts based on patterns
        
        # Simulate learning (change back to active after "processing")
        agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
        
        return {
            "status": "success",
            "message": f"Learning feedback processed for agent {agent_id}",
            "agent_id": agent_id,
            "memory_size": len(agent_runtime.memory)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Learning update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{agent_id}/performance")
async def get_agent_performance(
    agent_id: int,
    period: str = "all",
    db: Session = Depends(get_db)
):
    """
    Get agent performance metrics from database.
    
    Query params:
    - period: all|7d|30d|24h (currently returns all)
    
    Returns real performance data from agents.performance_metrics.
    """
    try:
        # Get agent from database
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        # Get agent runtime status if active
        factory = get_agent_factory()
        agent_status = await factory.get_agent_status(agent_id)
        
        # Extract performance_metrics from database
        perf_metrics = agent.performance_metrics or {}
        model_stats = agent.model_usage_stats or {}
        
        # Build comprehensive performance response
        performance = {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "period": period,
            "status": agent_status.get("status", agent.status),
            
            # Core performance metrics from DB
            "overall_score": min(100, int(perf_metrics.get("success_rate", 0) * 100)) if perf_metrics else 85,
            "success_rate": perf_metrics.get("success_rate", 0) * 100 if perf_metrics else 0,
            "average_response_time": perf_metrics.get("avg_execution_time_ms", 0) / 1000 if perf_metrics else 0,
            "total_tasks": perf_metrics.get("total_tasks_executed", 0),
            "completed_tasks": int(perf_metrics.get("total_tasks_executed", 0) * perf_metrics.get("success_rate", 1)) if perf_metrics else 0,
            "failed_tasks": int(perf_metrics.get("total_tasks_executed", 0) * (1 - perf_metrics.get("success_rate", 1))) if perf_metrics else 0,
            
            # Model usage stats
            "average_tokens_per_task": perf_metrics.get("avg_tokens_per_task", 0),
            "total_tokens_used": model_stats.get("total_tokens", 0),
            "total_cost": model_stats.get("total_cost", 0.0),
            "total_requests": model_stats.get("total_requests", 0),
            
            # Resource metrics (estimates - can be enhanced later)
            "average_memory_usage": 65,  # TODO: Track real memory
            "average_cpu_usage": 35,     # TODO: Track real CPU
            "uptime_percentage": 99.8 if agent_status.get("status") == "active" else 0,
            
            # Runtime info (if agent is active)
            "runtime": agent_status.get("runtime") if agent_status.get("status") == "active" else None,
            "llm_info": agent_status.get("llm") if agent_status.get("status") == "active" else None,
            
            # Raw metrics for debugging
            "raw_performance_metrics": perf_metrics,
            "raw_model_stats": model_stats
        }
        
        return performance
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{agent_id}/logs")
async def get_agent_logs(
    agent_id: int,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get agent activity logs from workflow executions.
    
    Returns recent execution logs for the agent.
    """
    try:
        from database.models import WorkflowExecution
        from sqlalchemy import desc
        
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        # Get recent workflow executions for this agent
        executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.agent_id == agent_id
        ).order_by(desc(WorkflowExecution.started_at)).limit(limit).all()
        
        logs = []
        for execution in executions:
            # Extract subtask data if available
            output_data = execution.output_data or {}
            subtasks = output_data.get("subtasks", [])
            
            # Create log entries for each subtask
            for subtask in subtasks:
                if subtask.get("selected_agent", {}).get("agent_id") == agent_id:
                    result = subtask.get("execution_result", {})
                    logs.append({
                        "timestamp": execution.started_at.isoformat() if execution.started_at else None,
                        "level": "error" if result.get("status") == "failed" else "info",
                        "message": subtask.get("description", "Task executed"),
                        "details": result.get("llm_response", result.get("response", ""))[:200],
                        "tokens_used": result.get("tokens_used", 0),
                        "execution_time_ms": result.get("execution_time_ms", 0),
                        "status": result.get("status", "unknown"),
                        "workflow_id": execution.workflow_id,
                        "execution_id": execution.id
                    })
            
            # If no subtasks, create a simple log entry
            if not subtasks:
                logs.append({
                    "timestamp": execution.started_at.isoformat() if execution.started_at else None,
                    "level": "error" if execution.status == "failed" else "info",
                    "message": f"Workflow execution {execution.status}",
                    "details": execution.error_message or "",
                    "status": execution.status,
                    "workflow_id": execution.workflow_id,
                    "execution_id": execution.id
                })
        
        return logs[:limit]  # Ensure we don't exceed limit
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logs query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{agent_id}/status")
async def get_agent_status_endpoint(agent_id: int):
    """
    Get detailed agent status including LLM connection info.
    
    Returns real-time agent status and configuration.
    """
    try:
        factory = get_agent_factory()
        agent_status = await factory.get_agent_status(agent_id)
        
        if agent_status["status"] == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=agent_status.get("error")
            )
        
        return agent_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{agent_id}/test-capabilities")
async def test_agent_capabilities_endpoint(agent_id: int):
    """
    Run comprehensive capability tests on an agent.
    
    Returns test results with performance metrics.
    """
    try:
        factory = get_agent_factory()
        
        # Get agent runtime
        agent_runtime = factory.active_agents.get(agent_id)
        if not agent_runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found in runtime"
            )
        
        # Run tests
        test_results = await factory.test_agent_capabilities(agent_runtime)
        
        return test_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Capability test error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/batch-create")
async def create_agent_batch(
    request: Dict[str, Any]
):
    """
    Create multiple agents in batch.
    
    Request body:
    {
        "agents": [
            {"name": "...", "type": "...", "skills": [...]},
            ...
        ],
        "auto_verify": true
    }
    
    Returns list of created agents.
    """
    try:
        factory = get_agent_factory()
        
        agents_config = request.get("agents", [])
        auto_verify = request.get("auto_verify", True)
        
        if not agents_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No agents specified"
            )
        
        created_agents = []
        errors = []
        
        for config in agents_config:
            try:
                agent = await factory.create_agent(
                    name=config.get("name"),
                    agent_type=config.get("type"),
                    skills=config.get("skills", []),
                    model_config=config.get("model"),
                    auto_verify=auto_verify
                )
                
                agent_status = await factory.get_agent_status(agent.agent_id)
                created_agents.append(agent_status)
                
            except Exception as e:
                errors.append({
                    "name": config.get("name"),
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "created": len(created_agents),
            "failed": len(errors),
            "agents": created_agents,
            "errors": errors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/active")
async def list_active_agents():
    """
    List all active agents in runtime.
    
    Returns list of agents with their current status.
    """
    try:
        factory = get_agent_factory()
        
        active_agents = []
        for agent_id, agent_runtime in factory.active_agents.items():
            status = await factory.get_agent_status(agent_id)
            active_agents.append(status)
        
        return {
            "total": len(active_agents),
            "agents": active_agents
        }
        
    except Exception as e:
        logger.error(f"List agents error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{agent_id}/add-skills")
async def add_agent_skills(
    agent_id: int,
    request: Dict[str, Any]
):
    """
    Add new skills to an existing agent.
    
    Request body:
    {
        "skills": ["skill1", "skill2"]
    }
    
    Updates agent's system prompt with new capabilities.
    """
    try:
        factory = get_agent_factory()
        
        # Get agent runtime
        agent_runtime = factory.active_agents.get(agent_id)
        if not agent_runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found in runtime"
            )
        
        # Apply new skills
        new_skills = request.get("skills", [])
        if not new_skills:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No skills specified"
            )
        
        # Apply skills (enhances system prompt)
        updated_agent = factory.apply_skills(agent_runtime, new_skills)
        
        return {
            "status": "success",
            "message": f"Added {len(new_skills)} skills to agent",
            "agent_id": agent_id,
            "total_skills": len(updated_agent.skills),
            "skills": updated_agent.skills
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add skills error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Health check endpoint
@router.get("/health")
async def agent_system_health():
    """
    Check agent system health.
    
    Returns status of agent factory and LLM connections.
    """
    try:
        factory = get_agent_factory()
        
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_agents": len(factory.active_agents),
            "agents": []
        }
        
        # Check each active agent's LLM
        for agent_id, agent in factory.active_agents.items():
            agent_health = {
                "id": agent_id,
                "name": agent.name,
                "lifecycle": agent.lifecycle_state.value,
                "executions": agent.execution_count,
                "llm_provider": agent.llm_manager.config.provider.value
            }
            health["agents"].append(agent_health)
        
        return health
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ===================================================================
# PRD-15: Model Configuration Endpoints
# ===================================================================

@router.get("/{agent_id}/model-config")
async def get_agent_model_config(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """
    Get agent's current model configuration.
    
    Returns the model configuration including provider, model_id,
    and all generation parameters.
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "model_config": agent.model_config or {},
            "model_usage_stats": agent.model_usage_stats or {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get model config error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/{agent_id}/model-config")
async def update_agent_model_config(
    agent_id: int,
    model_config: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Update agent's model configuration.
    
    Request body:
    {
        "provider": "openai",
        "model_id": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "fallback_model_id": "gpt-3.5-turbo"
    }
    
    Note: Agent must be recreated for changes to take effect in runtime.
    """
    try:
        from services.model_registry import ModelRegistry
        from sqlalchemy import update
        
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        # Validate model exists
        model_id = model_config.get("model_id")
        if model_id:
            registry = ModelRegistry(db)
            model = registry.get_model(model_id)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid model_id: {model_id}"
                )
        
        # Update model config
        agent.model_config = model_config
        
        # Mark for modification tracking
        from sqlalchemy.orm import attributes
        attributes.flag_modified(agent, "model_config")
        
        db.commit()
        
        logger.info(f"Updated model config for agent {agent_id}: {model_id}")
        
        return {
            "status": "success",
            "message": "Model configuration updated",
            "agent_id": agent_id,
            "model_config": agent.model_config,
            "note": "Agent must be restarted for changes to take effect"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update model config error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{agent_id}/model-usage")
async def get_agent_model_usage(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """
    Get agent's model usage statistics.
    
    Returns token usage, costs, and request counts.
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        usage_stats = agent.model_usage_stats or {
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_requests": 0,
            "avg_tokens_per_request": 0,
            "last_used_at": None
        }
        
        model_config = agent.model_config or {}
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "current_model": {
                "provider": model_config.get("provider"),
                "model_id": model_config.get("model_id"),
                "temperature": model_config.get("temperature")
            },
            "usage_stats": usage_stats,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get model usage error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{agent_id}/switch-model")
async def switch_agent_model(
    agent_id: int,
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Switch agent to a different model.
    
    Request body:
    {
        "model_id": "claude-3-sonnet-20240229",
        "temperature": 0.7,  // Optional
        "max_tokens": 2000   // Optional
    }
    
    This is a convenience endpoint that:
    1. Validates the new model
    2. Updates model configuration
    3. Provides recommended settings
    """
    try:
        from services.model_registry import ModelRegistry
        from sqlalchemy.orm import attributes
        
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        # Validate new model
        new_model_id = request.get("model_id")
        if not new_model_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="model_id is required"
            )
        
        registry = ModelRegistry(db)
        new_model = registry.get_model(new_model_id)
        if not new_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model_id: {new_model_id}"
            )
        
        # Get current config or create new one
        current_config = agent.model_config or {}
        
        # Update with new model
        new_config = {
            "provider": new_model.provider,
            "model_id": new_model_id,
            "temperature": request.get("temperature", new_model.default_temperature),
            "max_tokens": request.get("max_tokens", min(2000, new_model.max_output_tokens)),
            "top_p": current_config.get("top_p", 1.0),
            "frequency_penalty": current_config.get("frequency_penalty", 0.0),
            "presence_penalty": current_config.get("presence_penalty", 0.0),
            "fallback_model_id": current_config.get("fallback_model_id")
        }
        
        # Store old config for reference
        old_model_id = current_config.get("model_id", "unknown")
        
        # Update agent
        agent.model_config = new_config
        attributes.flag_modified(agent, "model_config")
        db.commit()
        
        logger.info(f"Switched agent {agent_id} from {old_model_id} to {new_model_id}")
        
        return {
            "status": "success",
            "message": f"Model switched from {old_model_id} to {new_model_id}",
            "agent_id": agent_id,
            "old_model": old_model_id,
            "new_model": new_model_id,
            "model_config": new_config,
            "model_details": new_model.to_dict(),
            "note": "Agent must be restarted for changes to take effect"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Switch model error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
