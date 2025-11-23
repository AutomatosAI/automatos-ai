from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, text
import time
import logging

from database.database import get_db
from models import PriorityLevel
from models import Agent, Skill, Pattern, agent_skills
# Import MCP tool models from database.models (SQLAlchemy models)
from models import AgentToolAssignment, MCPTool
# Import Pydantic models from database.models (not models.py)
from models import (
    AgentCreate, AgentUpdate, AgentResponse,
    SkillCreate, SkillUpdate, SkillResponse,
    PatternCreate, PatternResponse,
    AgentStatus, AgentType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"]) 

def _normalize_tags(raw_tags) -> List[str]:
    """Normalize incoming tags into a list of unique, lower-trimmed strings."""
    if raw_tags is None:
        return []
    items: List[str] = []
    if isinstance(raw_tags, str):
        items = [segment.strip() for segment in raw_tags.split(',')]
    elif isinstance(raw_tags, (list, tuple, set)):
        for value in raw_tags:
            if isinstance(value, str):
                items.extend([segment.strip() for segment in value.split(',')])
    else:
        return []

    # Deduplicate while preserving order
    seen = set()
    normalized = []
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _build_agent_response(agent: Agent) -> AgentResponse:
    """Build agent response with skills and tools"""
    # PRD-15: Debug logging for model_config
    model_cfg = getattr(agent, 'model_config', None)
    logger.info(f"Agent {agent.id} model_config: {model_cfg}")
    
    # Build tools list from tool_assignments relationship
    tools = []
    if hasattr(agent, 'tool_assignments') and agent.tool_assignments:
        for assignment in agent.tool_assignments:
            if assignment.enabled and hasattr(assignment, 'tool'):
                tools.append({
                    "id": assignment.tool.id,
                    "name": assignment.tool.name,
                    "description": assignment.tool.description,
                    "provider": assignment.tool.provider,
                    "category": assignment.tool.category,
                    "icon": assignment.tool.icon,
                    "permissions": assignment.permissions or {},
                    "configuration": assignment.configuration or {},
                    "assigned_at": assignment.assigned_at
                })
    
    # Read-time adapter: Remove tags from configuration if present (legacy data cleanup)
    # agent.tags is the single source of truth, configuration should not contain tags
    configuration = agent.configuration.copy() if agent.configuration else {}
    if "tags" in configuration:
        configuration.pop("tags", None)
        logger.debug(f"Removed legacy tags from configuration for agent {agent.id}")
    
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        agent_type=agent.agent_type,
        status=agent.status,
        configuration=configuration,
        skills=[SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            category=skill.category,
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at
        ).model_dump() for skill in agent.skills] if agent.skills else [],
        tools=tools,  # Add tools to response
        priority_level=getattr(agent, 'priority_level', 'medium') or 'medium',
        max_concurrent_tasks=getattr(agent, 'max_concurrent_tasks', 5) or 5,
        auto_start=getattr(agent, 'auto_start', False) or False,
        tags=_normalize_tags(agent.tags) if getattr(agent, 'tags', None) else [],
        created_at=agent.created_at,
        updated_at=agent.updated_at or agent.created_at,
        performance_metrics=agent.performance_metrics or {},
        created_by=agent.created_by,
        agent_model_config=getattr(agent, 'model_config', None),  # PRD-15: Include model config (field renamed to agent_model_config)
)

# SPECIFIC ROUTES FIRST (before {agent_id})
# from main import require_api_key

@router.get("/types", )
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

@router.get("/stats", )
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

@router.post("/bulk", response_model=List[AgentResponse], )
async def create_agents_bulk(agents: List[AgentCreate], db: Session = Depends(get_db)):
    """Create multiple agents at once"""
    try:
        created_agents = []
        
        for agent_data in agents:
            tags = _normalize_tags(getattr(agent_data, 'tags', None))
            # Check if agent with this name already exists
            existing = db.query(Agent).filter(Agent.name == agent_data.name).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Agent with name '{agent_data.name}' already exists")
            
            # Create agent
            agent = Agent(
                name=agent_data.name,
                description=agent_data.description,
                agent_type=agent_data.agent_type,  # Already a string, no .value needed
                configuration=agent_data.configuration or {},
                priority_level=agent_data.priority_level if agent_data.priority_level else "medium",  # Already a string
                max_concurrent_tasks=agent_data.max_concurrent_tasks or 5,
                auto_start=agent_data.auto_start or False,
                tags=tags,
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
            
            # Note: agent.tags is the single source of truth for tags.
            # Tags are NOT stored in agent.configuration to avoid duplicate state.
            # Legacy clients reading tags from configuration should migrate to use agent.tags.
            
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

@router.post("/", response_model=AgentResponse, )
async def create_agent(agent_data: AgentCreate, db: Session = Depends(get_db)):
    """Create a new agent with enhanced fields"""
    print("🚀 API CALL: create_agent function called!")
    try:
        logger.info(f"🔧 Creating agent: {agent_data.name}, tool_ids: {agent_data.tool_ids}")
        
        # Check if agent name already exists
        existing = db.query(Agent).filter(Agent.name == agent_data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Agent with this name already exists")
        
        tags = _normalize_tags(agent_data.tags if hasattr(agent_data, 'tags') else None)
        
        # Create agent with new fields
        agent = Agent(
            name=agent_data.name,
            description=agent_data.description,
            agent_type=agent_data.agent_type,  # Now accepts any string
            configuration=agent_data.configuration or {},
            priority_level=agent_data.priority_level if agent_data.priority_level else "medium",  # Already a string
            max_concurrent_tasks=agent_data.max_concurrent_tasks or 5,
            auto_start=agent_data.auto_start or False,
            tags=tags,
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

        # Note: agent.tags is the single source of truth for tags.
        # Tags are NOT stored in agent.configuration to avoid duplicate state.
        # Legacy clients reading tags from configuration should migrate to use agent.tags.
        
        # Add tools if provided (NEW FEATURE)
        if agent_data.tool_ids:
            logger.info(f"🛠️ Processing tool_ids: {agent_data.tool_ids}")
            # Enhanced validation: Check that all tool IDs exist and are active
            tools = db.query(MCPTool).filter(
                MCPTool.id.in_(agent_data.tool_ids),
                MCPTool.status == "active"
            ).all()
            logger.info(f"🔍 Found {len(tools)} active tools out of {len(agent_data.tool_ids)} requested")
            if len(tools) != len(agent_data.tool_ids):
                found_ids = [tool.id for tool in tools]
                missing_ids = [tid for tid in agent_data.tool_ids if tid not in found_ids]
                # Check if missing tools exist but are inactive
                inactive_tools = db.query(MCPTool).filter(
                    MCPTool.id.in_(missing_ids),
                    MCPTool.status != "active"
                ).all()
                if inactive_tools:
                    inactive_names = [tool.name for tool in inactive_tools]
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Tools are inactive and cannot be assigned: {inactive_names}"
                    )
                else:
                    raise HTTPException(status_code=404, detail=f"Tools not found: {missing_ids}")
            
            # Additional validation: Check for duplicate tool IDs
            if len(set(agent_data.tool_ids)) != len(agent_data.tool_ids):
                raise HTTPException(status_code=400, detail="Duplicate tool IDs are not allowed")
            
            # Create tool assignments
            for tool in tools:
                assignment = AgentToolAssignment(
                    agent_id=agent.id,
                    tool_id=tool.id,
                    enabled=True,
                    permissions={"read": True, "write": True, "execute": True},
                    configuration={}
                )
                db.add(assignment)
        
        db.commit()
        db.refresh(agent)
        
        # Load skills and tools for response
        agent_with_skills_and_tools = db.query(Agent).options(
            joinedload(Agent.skills),
            joinedload(Agent.tool_assignments).joinedload(AgentToolAssignment.tool)
        ).filter(Agent.id == agent.id).first()
        
        return _build_agent_response(agent_with_skills_and_tools)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

@router.get("/", response_model=List[AgentResponse], )
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
        query = db.query(Agent).options(
            joinedload(Agent.skills),
            joinedload(Agent.tool_assignments).joinedload(AgentToolAssignment.tool)
        )
        
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

@router.get("/{agent_id}/status", )
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

@router.post("/{agent_id}/execute", )
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

@router.get("/{agent_id}", response_model=AgentResponse, )
async def get_agent(agent_id: int, db: Session = Depends(get_db)):
    """Get a specific agent by ID with skills and tools"""
    try:
        agent = db.query(Agent).options(
            joinedload(Agent.skills),
            joinedload(Agent.tool_assignments).joinedload(AgentToolAssignment.tool)
        ).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return _build_agent_response(agent)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}/skills", )
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
        ).model_dump() for skill in agent.skills] if agent.skills else []
        
        return {"data": skills}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/skills", )
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

@router.put("/{agent_id}", response_model=AgentResponse, )
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

        if agent_update.tags is not None:
            tags = _normalize_tags(agent_update.tags)
            agent.tags = tags
            # Remove tags from configuration if present (cleanup legacy data)
            if agent.configuration and "tags" in agent.configuration:
                config = agent.configuration.copy()
                config.pop("tags", None)
                agent.configuration = config
        
        db.commit()
        db.refresh(agent)
        
        # Load with skills for response
        agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
        
        return _build_agent_response(agent_with_skills)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")

@router.delete("/{agent_id}", )
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete an agent and all related records"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Delete related records first (tables without CASCADE) - use savepoints to handle errors
        # Order matters: delete in correct order to avoid FK violations
        
        deletions = [
            ("agent_skills", "DELETE FROM agent_skills WHERE agent_id = :agent_id", True),
            ("workflow_agents", "DELETE FROM workflow_agents WHERE agent_id = :agent_id", True),
            ("memory_items", "DELETE FROM memory_items WHERE agent_id = :agent_id", True),
            ("tasks", "DELETE FROM tasks WHERE agent_id = :agent_id", False),
            ("workflow_executions", "DELETE FROM workflow_executions WHERE agent_id = :agent_id", False),
        ]
        
        for table_name, sql_stmt, required in deletions:
            savepoint = db.begin_nested()  # Create savepoint for this deletion
            try:
                db.execute(text(sql_stmt), {"agent_id": agent_id})
                savepoint.commit()  # Commit savepoint
            except Exception as e:
                savepoint.rollback()  # Rollback savepoint, but keep main transaction
                if required:
                    logger.error(f"Error deleting {table_name} for agent {agent_id}: {e}")
                    raise HTTPException(status_code=500, detail=f"Error deleting {table_name}: {str(e)}")
                else:
                    logger.warning(f"Error deleting {table_name} for agent {agent_id}: {e}")
                    # Continue for optional tables
        
        # Now delete the agent (other relationships have CASCADE)
        db.delete(agent)
        db.commit()
        
        return {"message": f"Agent {agent_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")
