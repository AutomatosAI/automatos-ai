
"""
Agent Management API Routes
===========================

REST API endpoints for managing agents with enhanced functionality.
Supports new agent fields, skills management, and templates.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_

from database import get_db
from models import (
    Agent, Skill, Pattern, agent_skills,
    AgentCreate, AgentUpdate, AgentResponse,
    SkillCreate, SkillUpdate, SkillResponse,
    PatternCreate, PatternResponse,
    AgentStatus, AgentType, SkillType, PriorityLevel
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])

def _build_agent_response(agent: Agent) -> AgentResponse:
    """Helper function to build AgentResponse with all fields"""
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        agent_type=agent.agent_type,
        status=agent.status,
        configuration=agent.configuration,
        performance_metrics=agent.performance_metrics,
        priority_level=agent.priority_level or "medium",
        max_concurrent_tasks=agent.max_concurrent_tasks or 5,
        auto_start=agent.auto_start or False,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        created_by=agent.created_by,
        skills=[{
            "id": skill.id,
            "name": skill.name,
            "skill_type": skill.skill_type,
            "category": skill.category or "development"
        } for skill in agent.skills]
    )

@router.post("/", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate, db: Session = Depends(get_db)):
    """Create a new agent with enhanced fields"""
    try:
        # Check if agent name already exists
        existing = db.query(Agent).filter(Agent.name == agent_data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Agent with this name already exists")
        
        # Create agent with new fields
        agent = Agent(
            name=agent_data.name,
            description=agent_data.description,
            agent_type=agent_data.agent_type.value,
            configuration=agent_data.configuration or {},
            priority_level=agent_data.priority_level.value if agent_data.priority_level else "medium",
            max_concurrent_tasks=agent_data.max_concurrent_tasks or 5,
            auto_start=agent_data.auto_start or False,
            created_by="api"
        )
        
        db.add(agent)
        db.flush()  # Get the ID
        
        # Add skills if provided
        if agent_data.skill_ids:
            skills = db.query(Skill).filter(
                Skill.id.in_(agent_data.skill_ids),
                Skill.is_active == True
            ).all()
            if len(skills) != len(agent_data.skill_ids):
                found_ids = [skill.id for skill in skills]
                missing_ids = [sid for sid in agent_data.skill_ids if sid not in found_ids]
                raise HTTPException(status_code=404, detail=f"Skills not found: {missing_ids}")
            agent.skills.extend(skills)
        
        db.commit()
        db.refresh(agent)
        
        # Load skills for response
        agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
        
        return _build_agent_response(agent_with_skills)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[AgentStatus] = None,
    agent_type: Optional[AgentType] = None,
    priority_level: Optional[PriorityLevel] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List agents with enhanced filtering and pagination"""
    try:
        query = db.query(Agent).options(joinedload(Agent.skills))
        
        # Apply filters
        if status:
            query = query.filter(Agent.status == status.value)
        
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type.value)
        
        if priority_level:
            query = query.filter(Agent.priority_level == priority_level.value)
        
        if search:
            search_filter = or_(
                Agent.name.ilike(f"%{search}%"),
                Agent.description.ilike(f"%{search}%")
            )
            query = query.filter(search_filter)
        
        agents = query.offset(skip).limit(limit).all()
        
        return [_build_agent_response(agent) for agent in agents]
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, db: Session = Depends(get_db)):
    """Get a specific agent by ID"""
    try:
        agent = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return _build_agent_response(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int, 
    agent_update: AgentUpdate, 
    db: Session = Depends(get_db)
):
    """Update an existing agent with enhanced fields"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update basic fields
        update_data = agent_update.dict(exclude_unset=True, exclude={"skill_ids"})
        for field, value in update_data.items():
            if field in ["status", "priority_level"] and value:
                setattr(agent, field, value.value)
            else:
                setattr(agent, field, value)
        
        # Update skills if provided
        if agent_update.skill_ids is not None:
            # Clear existing skills
            agent.skills.clear()
            
            # Add new skills
            if agent_update.skill_ids:
                skills = db.query(Skill).filter(
                    Skill.id.in_(agent_update.skill_ids),
                    Skill.is_active == True
                ).all()
                if len(skills) != len(agent_update.skill_ids):
                    found_ids = [skill.id for skill in skills]
                    missing_ids = [sid for sid in agent_update.skill_ids if sid not in found_ids]
                    raise HTTPException(status_code=404, detail=f"Skills not found: {missing_ids}")
                agent.skills.extend(skills)
        
        db.commit()
        db.refresh(agent)
        
        # Load skills for response
        agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
        
        return _build_agent_response(agent_with_skills)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete an agent (soft delete by setting status to inactive)"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Soft delete
        agent.status = "inactive"
        db.commit()
        
        return {"message": "Agent deleted successfully", "agent_id": agent_id}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/start")
async def start_agent(agent_id: int, db: Session = Depends(get_db)):
    """Start an agent (set status to active)"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent.status = "active"
        db.commit()
        
        return {"message": "Agent started successfully", "agent_id": agent_id, "status": "active"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error starting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: int, db: Session = Depends(get_db)):
    """Stop an agent (set status to inactive)"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent.status = "inactive"
        db.commit()
        
        return {"message": "Agent stopped successfully", "agent_id": agent_id, "status": "inactive"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error stopping agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}/performance")
async def get_agent_performance(agent_id: int, db: Session = Depends(get_db)):
    """Get agent performance metrics"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get execution statistics
        from models import WorkflowExecution
        total_executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.agent_id == agent_id
        ).count()
        
        successful_executions = db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.agent_id == agent_id,
                WorkflowExecution.status == 'completed'
            )
        ).count()
        
        failed_executions = db.query(WorkflowExecution).filter(
            and_(
                WorkflowExecution.agent_id == agent_id,
                WorkflowExecution.status == 'failed'
            )
        ).count()
        
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
        
        performance_data = {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": round(success_rate, 2),
            "current_status": agent.status,
            "priority_level": agent.priority_level,
            "max_concurrent_tasks": agent.max_concurrent_tasks,
            "stored_metrics": agent.performance_metrics or {}
        }
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/clone")
async def clone_agent(
    agent_id: int, 
    new_name: str,
    db: Session = Depends(get_db)
):
    """Clone an existing agent with a new name"""
    try:
        # Get original agent
        original_agent = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent_id).first()
        if not original_agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if new name already exists
        existing = db.query(Agent).filter(Agent.name == new_name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Agent with this name already exists")
        
        # Create cloned agent
        cloned_agent = Agent(
            name=new_name,
            description=f"Clone of {original_agent.name}",
            agent_type=original_agent.agent_type,
            configuration=original_agent.configuration,
            priority_level=original_agent.priority_level,
            max_concurrent_tasks=original_agent.max_concurrent_tasks,
            auto_start=False,  # Don't auto-start clones
            created_by="api"
        )
        
        db.add(cloned_agent)
        db.flush()
        
        # Copy skills
        cloned_agent.skills.extend(original_agent.skills)
        
        db.commit()
        db.refresh(cloned_agent)
        
        # Load skills for response
        cloned_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == cloned_agent.id).first()
        
        return _build_agent_response(cloned_with_skills)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error cloning agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
@router.get("/test123")
async def test_simple_route():
    return {"message": "Simple test route works"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agents_api"}
