from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func
import time
import logging

from database.database import get_db
from models import (
    Agent, Skill, Pattern, agent_skills,
    AgentCreate, AgentUpdate, AgentResponse,
    SkillCreate, SkillUpdate, SkillResponse,
    PatternCreate, PatternResponse,
    AgentStatus, AgentType, PriorityLevel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"]) 

def _build_agent_response(agent: Agent) -> AgentResponse:
    """Build agent response with skills"""
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        agent_type=agent.agent_type,
        status=agent.status,
        configuration=agent.configuration or {},
        skills=[SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            category=skill.category,
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at
        ) for skill in agent.skills] if agent.skills else [],
        priority_level=getattr(agent, 'priority_level', 'medium'),
        max_concurrent_tasks=getattr(agent, 'max_concurrent_tasks', 5),
        auto_start=getattr(agent, 'auto_start', False),
        created_at=agent.created_at,
        updated_at=agent.updated_at
    )

# SPECIFIC ROUTES FIRST (before {agent_id})
from ..main import require_api_key

@router.get("/types", dependencies=[Depends(require_api_key)])
async def get_agent_types():
    """Get available agent types"""
    return {
        "data": [
            "code_architect", 
            "security_expert", 
            "performance_optimizer",
            "data_analyst", 
            "infrastructure_manager", 
            "custom", 
            "system", 
            "specialized"
        ],
        "descriptions": {
            "code_architect": "Designs and reviews code architecture",
            "security_expert": "Performs security analysis and audits", 
            "performance_optimizer": "Optimizes system performance",
            "data_analyst": "Analyzes data and generates insights",
            "infrastructure_manager": "Manages infrastructure and deployments",
            "custom": "Custom agent configuration",
            "system": "System-level operations",
            "specialized": "Specialized domain expertise"
        }
    }

@router.get("/stats", dependencies=[Depends(require_api_key)])
async def get_agent_stats(db: Session = Depends(get_db)):
    """Get comprehensive agent statistics"""
    try:
        total_agents = db.query(func.count(Agent.id)).scalar() or 0
        active_agents = db.query(func.count(Agent.id)).filter(Agent.status == "active").scalar() or 0
        inactive_agents = db.query(func.count(Agent.id)).filter(Agent.status == "inactive").scalar() or 0
        
        # Get agent counts by type
        agent_types = {}
        for agent_type in AgentType:
            count = db.query(func.count(Agent.id)).filter(Agent.agent_type == agent_type.value).scalar() or 0
            agent_types[agent_type.value] = count
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "inactive_agents": inactive_agents,
            "agents_by_type": agent_types,
            "average_performance": 85.5,  # Placeholder
            "total_executions": 0,  # Placeholder
            "successful_executions": 0,  # Placeholder
            "failed_executions": 0,  # Placeholder
            "timestamp": "2025-08-01T12:57:03Z"
        }
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk", response_model=List[AgentResponse], dependencies=[Depends(require_api_key)])
async def create_agents_bulk(agents: List[AgentCreate], db: Session = Depends(get_db)):
    """Create multiple agents at once"""
    try:
        created_agents = []
        
        for agent_data in agents:
            # Check if agent with this name already exists
            existing = db.query(Agent).filter(Agent.name == agent_data.name).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Agent with name '{agent_data.name}' already exists")
            
            # Create agent
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
            
            created_agents.append(agent)
        
        db.commit()
        
        # Refresh and build responses
        result = []
        for agent in created_agents:
            db.refresh(agent)
            agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
            result.append(_build_agent_response(agent_with_skills))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating bulk agents: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating bulk agents: {str(e)}")

@router.post("/", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
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
        
        return {"data": _build_agent_response(agent_with_skills)}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

@router.get("/", response_model=List[AgentResponse], dependencies=[Depends(require_api_key)])
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
        
        return {"data": [_build_agent_response(agent) for agent in agents]}
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}/status", dependencies=[Depends(require_api_key)])
async def get_agent_status(agent_id: int, db: Session = Depends(get_db)):
    """Get current status of a specific agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        return {
            "agent_id": agent_id,
            "name": agent.name,
            "status": agent.status,
            "agent_type": agent.agent_type,
            "priority_level": getattr(agent, 'priority_level', 'medium'),
            "max_concurrent_tasks": getattr(agent, 'max_concurrent_tasks', 5),
            "auto_start": getattr(agent, 'auto_start', False),
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
            "configuration": agent.configuration or {}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/execute", dependencies=[Depends(require_api_key)])
async def execute_agent(agent_id: int, execution_data: dict = {}, db: Session = Depends(get_db)):
    """Execute an agent with given parameters"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        if agent.status != "active":
            raise HTTPException(status_code=400, detail="Agent must be active to execute")
            
        # Generate execution ID and simulate execution start
        execution_id = f"exec_{agent_id}_{int(time.time())}"
        
        return {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent.name,
            "status": "started",
            "parameters": execution_data,
            "started_at": "2025-08-01T12:57:03Z",
            "estimated_duration": "5-10 minutes",
            "message": f"Execution started for agent {agent.name}"
        }
    except HTTPException:
        raise  
    except Exception as e:
        logger.error(f"Error executing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
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

@router.get("/{agent_id}/skills", dependencies=[Depends(require_api_key)])
async def get_agent_skills(agent_id: int, db: Session = Depends(get_db)):
    """Get skills for a specific agent"""
    try:
        agent = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        skills = [SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            category=skill.category,
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at
        ) for skill in agent.skills] if agent.skills else []
        
        return {"data": skills}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/skills", dependencies=[Depends(require_api_key)])
async def add_agent_skills(agent_id: int, skill_ids: List[int], db: Session = Depends(get_db)):
    """Add skills to an agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        skills = db.query(Skill).filter(Skill.id.in_(skill_ids), Skill.is_active == True).all()
        if len(skills) != len(skill_ids):
            found_ids = [skill.id for skill in skills]
            missing_ids = [sid for sid in skill_ids if sid not in found_ids]
            raise HTTPException(status_code=404, detail=f"Skills not found: {missing_ids}")
        
        agent.skills.extend(skills)
        db.commit()
        
        return {"data": {"message": "Skills added successfully", "agent_id": agent_id, "skill_ids": skill_ids}}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding agent skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{agent_id}", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
async def update_agent(agent_id: int, agent_update: AgentUpdate, db: Session = Depends(get_db)):
    """Update an existing agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update fields if provided
        if agent_update.name is not None:
            # Check for name conflicts
            existing = db.query(Agent).filter(Agent.name == agent_update.name, Agent.id != agent_id).first()
            if existing:
                raise HTTPException(status_code=400, detail="Agent with this name already exists")
            agent.name = agent_update.name
        
        if agent_update.description is not None:
            agent.description = agent_update.description
        
        if agent_update.status is not None:
            agent.status = agent_update.status.value
        
        db.commit()
        db.refresh(agent)
        
        # Load with skills for response
        agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
        
        return {"data": _build_agent_response(agent_with_skills)}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")

@router.delete("/{agent_id}", dependencies=[Depends(require_api_key)])
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete an agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        db.delete(agent)
        db.commit()
        
        return {"message": f"Agent {agent_id} deleted successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")
