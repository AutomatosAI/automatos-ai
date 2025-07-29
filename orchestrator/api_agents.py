
"""
Agent Management API Routes
===========================

REST API endpoints for managing agents, skills, and patterns.
Enhanced with professional agent type support.
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
    AgentStatus, AgentType, SkillType
)
from agents import create_agent as create_agent_instance, AGENT_REGISTRY
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])

# Agent endpoints
@router.post("/", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate, db: Session = Depends(get_db)):
    """Create a new agent"""
    try:
        # Create agent
        agent = Agent(
            name=agent_data.name,
            description=agent_data.description,
            agent_type=agent_data.agent_type.value,
            configuration=agent_data.configuration or {},
            created_by="system"  # TODO: Get from auth context
        )
        
        db.add(agent)
        db.flush()  # Get the ID
        
        # Add skills if provided
        if agent_data.skill_ids:
            skills = db.query(Skill).filter(Skill.id.in_(agent_data.skill_ids)).all()
            agent.skills.extend(skills)
        
        db.commit()
        db.refresh(agent)
        
        # Load skills for response
        agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
        
        return AgentResponse(
            id=agent_with_skills.id,
            name=agent_with_skills.name,
            description=agent_with_skills.description,
            agent_type=agent_with_skills.agent_type,
            status=agent_with_skills.status,
            configuration=agent_with_skills.configuration,
            performance_metrics=agent_with_skills.performance_metrics,
            created_at=agent_with_skills.created_at,
            updated_at=agent_with_skills.updated_at,
            created_by=agent_with_skills.created_by,
            skills=[{
                "id": skill.id,
                "name": skill.name,
                "skill_type": skill.skill_type
            } for skill in agent_with_skills.skills]
        )
        
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
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List agents with filtering and pagination"""
    try:
        query = db.query(Agent).options(joinedload(Agent.skills))
        
        # Apply filters
        if status:
            query = query.filter(Agent.status == status.value)
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type.value)
        if search:
            query = query.filter(
                or_(
                    Agent.name.ilike(f"%{search}%"),
                    Agent.description.ilike(f"%{search}%")
                )
            )
        
        agents = query.offset(skip).limit(limit).all()
        
        return [
            AgentResponse(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                agent_type=agent.agent_type,
                status=agent.status,
                configuration=agent.configuration,
                performance_metrics=agent.performance_metrics,
                created_at=agent.created_at,
                updated_at=agent.updated_at,
                created_by=agent.created_by,
                skills=[{
                    "id": skill.id,
                    "name": skill.name,
                    "skill_type": skill.skill_type
                } for skill in agent.skills]
            ) for agent in agents
        ]
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, db: Session = Depends(get_db)):
    """Get agent by ID"""
    try:
        agent = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            agent_type=agent.agent_type,
            status=agent.status,
            configuration=agent.configuration,
            performance_metrics=agent.performance_metrics,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            created_by=agent.created_by,
            skills=[{
                "id": skill.id,
                "name": skill.name,
                "skill_type": skill.skill_type
            } for skill in agent.skills]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting agent: {str(e)}")

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: int, agent_data: AgentUpdate, db: Session = Depends(get_db)):
    """Update agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update fields
        if agent_data.name is not None:
            agent.name = agent_data.name
        if agent_data.description is not None:
            agent.description = agent_data.description
        if agent_data.status is not None:
            agent.status = agent_data.status.value
        if agent_data.configuration is not None:
            agent.configuration = agent_data.configuration
        
        # Update skills
        if agent_data.skill_ids is not None:
            agent.skills.clear()
            if agent_data.skill_ids:
                skills = db.query(Skill).filter(Skill.id.in_(agent_data.skill_ids)).all()
                agent.skills.extend(skills)
        
        db.commit()
        db.refresh(agent)
        
        # Load skills for response
        agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
        
        return AgentResponse(
            id=agent_with_skills.id,
            name=agent_with_skills.name,
            description=agent_with_skills.description,
            agent_type=agent_with_skills.agent_type,
            status=agent_with_skills.status,
            configuration=agent_with_skills.configuration,
            performance_metrics=agent_with_skills.performance_metrics,
            created_at=agent_with_skills.created_at,
            updated_at=agent_with_skills.updated_at,
            created_by=agent_with_skills.created_by,
            skills=[{
                "id": skill.id,
                "name": skill.name,
                "skill_type": skill.skill_type
            } for skill in agent_with_skills.skills]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")

@router.delete("/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        db.delete(agent)
        db.commit()
        
        return {"message": "Agent deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")

# Professional Agent Management Endpoints
@router.get("/types", response_model=List[dict])
async def get_agent_types():
    """Get available professional agent types with their capabilities"""
    try:
        agent_types = []
        
        for agent_type, agent_class in AGENT_REGISTRY.items():
            # Create a temporary instance to get metadata
            temp_agent = agent_class(
                agent_id=0,
                name="temp",
                description="temp"
            )
            
            agent_types.append({
                "type": agent_type,
                "name": agent_type.replace('_', ' ').title(),
                "description": temp_agent.description,
                "default_skills": temp_agent.default_skills,
                "specializations": temp_agent.specializations,
                "capabilities": list(temp_agent.capabilities.keys())
            })
        
        return agent_types
        
    except Exception as e:
        logger.error(f"Error getting agent types: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting agent types: {str(e)}")

@router.post("/{agent_id}/execute", response_model=dict)
async def execute_agent_task(
    agent_id: int, 
    task: dict,
    db: Session = Depends(get_db)
):
    """Execute a task using a professional agent"""
    try:
        # Get agent from database
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create agent instance
        agent_instance = create_agent_instance(
            agent_type=agent.agent_type,
            agent_id=agent.id,
            name=agent.name,
            description=agent.description,
            configuration=agent.configuration or {}
        )
        
        # Execute task
        result = await agent_instance.execute_task(task)
        
        # Update agent performance metrics in database
        agent.performance_metrics = agent_instance.performance_metrics
        db.commit()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing task for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing task: {str(e)}")

@router.get("/{agent_id}/status", response_model=dict)
async def get_agent_status(agent_id: int, db: Session = Depends(get_db)):
    """Get detailed status information for a professional agent"""
    try:
        # Get agent from database
        agent = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create agent instance to get current status
        agent_instance = create_agent_instance(
            agent_type=agent.agent_type,
            agent_id=agent.id,
            name=agent.name,
            description=agent.description,
            configuration=agent.configuration or {}
        )
        
        # Get comprehensive status
        status_info = agent_instance.get_status_info()
        
        # Add database information
        status_info.update({
            "database_info": {
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat(),
                "created_by": agent.created_by,
                "skills_count": len(agent.skills)
            }
        })
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting agent status: {str(e)}")

@router.post("/{agent_id}/validate-task", response_model=dict)
async def validate_agent_task(
    agent_id: int,
    task: dict,
    db: Session = Depends(get_db)
):
    """Validate if an agent can handle a specific task"""
    try:
        # Get agent from database
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Create agent instance
        agent_instance = create_agent_instance(
            agent_type=agent.agent_type,
            agent_id=agent.id,
            name=agent.name,
            description=agent.description,
            configuration=agent.configuration or {}
        )
        
        # Validate task
        can_handle = agent_instance.validate_task(task)
        estimated_time = agent_instance.estimate_execution_time(task)
        
        return {
            "can_handle": can_handle,
            "estimated_execution_time": estimated_time,
            "required_capabilities": task.get("required_capability"),
            "required_skills": task.get("required_skills", []),
            "agent_capabilities": agent_instance.get_available_capabilities(),
            "agent_skills": list(agent_instance.skills.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating task for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating task: {str(e)}")

@router.get("/professional-skills", response_model=List[dict])
async def get_professional_skills():
    """Get all available professional skills across agent types"""
    try:
        all_skills = {}
        
        for agent_type, agent_class in AGENT_REGISTRY.items():
            # Create temporary instance to get skills
            temp_agent = agent_class(
                agent_id=0,
                name="temp",
                description="temp"
            )
            
            for skill_name, skill in temp_agent.skills.items():
                if skill_name not in all_skills:
                    all_skills[skill_name] = {
                        "name": skill.name,
                        "skill_type": skill.skill_type.value,
                        "description": skill.description,
                        "parameters": skill.parameters,
                        "agent_types": [agent_type]
                    }
                else:
                    all_skills[skill_name]["agent_types"].append(agent_type)
        
        return list(all_skills.values())
        
    except Exception as e:
        logger.error(f"Error getting professional skills: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting professional skills: {str(e)}")

# Skill endpoints
@router.post("/skills", response_model=SkillResponse)
async def create_skill(skill_data: SkillCreate, db: Session = Depends(get_db)):
    """Create a new skill"""
    try:
        skill = Skill(
            name=skill_data.name,
            description=skill_data.description,
            skill_type=skill_data.skill_type.value,
            implementation=skill_data.implementation,
            parameters=skill_data.parameters or {},
            created_by="system"  # TODO: Get from auth context
        )
        
        db.add(skill)
        db.commit()
        db.refresh(skill)
        
        return SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            implementation=skill.implementation,
            parameters=skill.parameters,
            performance_data=skill.performance_data,
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
            created_by=skill.created_by
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating skill: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating skill: {str(e)}")

@router.get("/skills", response_model=List[SkillResponse])
async def list_skills(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    skill_type: Optional[SkillType] = None,
    search: Optional[str] = None,
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """List skills with filtering and pagination"""
    try:
        query = db.query(Skill)
        
        # Apply filters
        if active_only:
            query = query.filter(Skill.is_active == True)
        if skill_type:
            query = query.filter(Skill.skill_type == skill_type.value)
        if search:
            query = query.filter(
                or_(
                    Skill.name.ilike(f"%{search}%"),
                    Skill.description.ilike(f"%{search}%")
                )
            )
        
        skills = query.offset(skip).limit(limit).all()
        
        return [
            SkillResponse(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                skill_type=skill.skill_type,
                implementation=skill.implementation,
                parameters=skill.parameters,
                performance_data=skill.performance_data,
                is_active=skill.is_active,
                created_at=skill.created_at,
                updated_at=skill.updated_at,
                created_by=skill.created_by
            ) for skill in skills
        ]
        
    except Exception as e:
        logger.error(f"Error listing skills: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing skills: {str(e)}")

@router.get("/skills/{skill_id}", response_model=SkillResponse)
async def get_skill(skill_id: int, db: Session = Depends(get_db)):
    """Get skill by ID"""
    try:
        skill = db.query(Skill).filter(Skill.id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        return SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            implementation=skill.implementation,
            parameters=skill.parameters,
            performance_data=skill.performance_data,
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
            created_by=skill.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting skill {skill_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting skill: {str(e)}")

@router.put("/skills/{skill_id}", response_model=SkillResponse)
async def update_skill(skill_id: int, skill_data: SkillUpdate, db: Session = Depends(get_db)):
    """Update skill"""
    try:
        skill = db.query(Skill).filter(Skill.id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        # Update fields
        if skill_data.name is not None:
            skill.name = skill_data.name
        if skill_data.description is not None:
            skill.description = skill_data.description
        if skill_data.implementation is not None:
            skill.implementation = skill_data.implementation
        if skill_data.parameters is not None:
            skill.parameters = skill_data.parameters
        if skill_data.is_active is not None:
            skill.is_active = skill_data.is_active
        
        db.commit()
        db.refresh(skill)
        
        return SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            implementation=skill.implementation,
            parameters=skill.parameters,
            performance_data=skill.performance_data,
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
            created_by=skill.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating skill {skill_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating skill: {str(e)}")

# Pattern endpoints
@router.post("/patterns", response_model=PatternResponse)
async def create_pattern(pattern_data: PatternCreate, db: Session = Depends(get_db)):
    """Create a new coordination pattern"""
    try:
        pattern = Pattern(
            name=pattern_data.name,
            description=pattern_data.description,
            pattern_type=pattern_data.pattern_type,
            pattern_data=pattern_data.pattern_data,
            created_by="system"  # TODO: Get from auth context
        )
        
        db.add(pattern)
        db.commit()
        db.refresh(pattern)
        
        return PatternResponse(
            id=pattern.id,
            name=pattern.name,
            description=pattern.description,
            pattern_type=pattern.pattern_type,
            pattern_data=pattern.pattern_data,
            usage_count=pattern.usage_count,
            effectiveness_score=pattern.effectiveness_score,
            is_active=pattern.is_active,
            created_at=pattern.created_at,
            updated_at=pattern.updated_at,
            created_by=pattern.created_by
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating pattern: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating pattern: {str(e)}")

@router.get("/patterns", response_model=List[PatternResponse])
async def list_patterns(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    pattern_type: Optional[str] = None,
    search: Optional[str] = None,
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """List coordination patterns with filtering and pagination"""
    try:
        query = db.query(Pattern)
        
        # Apply filters
        if active_only:
            query = query.filter(Pattern.is_active == True)
        if pattern_type:
            query = query.filter(Pattern.pattern_type == pattern_type)
        if search:
            query = query.filter(
                or_(
                    Pattern.name.ilike(f"%{search}%"),
                    Pattern.description.ilike(f"%{search}%")
                )
            )
        
        patterns = query.offset(skip).limit(limit).all()
        
        return [
            PatternResponse(
                id=pattern.id,
                name=pattern.name,
                description=pattern.description,
                pattern_type=pattern.pattern_type,
                pattern_data=pattern.pattern_data,
                usage_count=pattern.usage_count,
                effectiveness_score=pattern.effectiveness_score,
                is_active=pattern.is_active,
                created_at=pattern.created_at,
                updated_at=pattern.updated_at,
                created_by=pattern.created_by
            ) for pattern in patterns
        ]
        
    except Exception as e:
        logger.error(f"Error listing patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing patterns: {str(e)}")
