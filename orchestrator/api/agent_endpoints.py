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
    period: str = "all"
):
    """
    Get agent performance metrics.
    
    Query params:
    - period: all|7d|30d|24h
    
    Returns performance data and statistics.
    """
    try:
        factory = get_agent_factory()
        
        # Get agent status
        agent_status = await factory.get_agent_status(agent_id)
        
        if agent_status["status"] == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=agent_status.get("error")
            )
        
        # Add performance data
        performance = {
            "agent_id": agent_id,
            "period": period,
            "status": status["status"],
            "runtime": status.get("runtime"),
            "metrics": status.get("metrics", {}),
            "llm_info": status.get("llm")
        }
        
        # In production, you would query performance data from database
        # based on the period parameter
        
        return performance
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance query error: {str(e)}")
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
