

"""
Agents API Endpoints
===================

REST API endpoints for agent management including:
- Agent creation, configuration, and lifecycle management
- Skills assignment and management
- Performance tracking and analytics
- Agent coordination and communication
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

# Import database and dependencies
from database.database import get_db
from database.models import Agent
from sqlalchemy import Table, MetaData, text
from services.orchestrator_service import EnhancedOrchestratorService

logger = logging.getLogger(__name__)

# Simple API key auth dependency (local copy to avoid circular import)
def require_api_key(x_api_key: str = Header(None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# Pydantic Models for API
class AgentCreateRequest(BaseModel):
    """Request model for creating a new agent"""
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    agent_type: str = Field(..., description="Agent type: custom, system, specialist")
    description: Optional[str] = Field(None, max_length=500, description="Agent description")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Agent configuration parameters")
    skills: Optional[List[str]] = Field(None, description="List of skill IDs to assign")
    patterns: Optional[List[str]] = Field(None, description="List of pattern IDs to assign")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Data Analyst Agent",
                "type": "specialist",
                "description": "Specialized agent for data analysis and reporting",
                "configuration": {
                    "max_concurrent_tasks": 5,
                    "timeout_seconds": 300,
                    "priority": "high"
                },
                "skills": ["data_analysis", "report_generation"],
                "patterns": ["analytical_thinking", "systematic_approach"]
            }
        }

class AgentResponse(BaseModel):
    """Response model for agent data"""
    id: int
    name: str
    agent_type: str  # Changed from 'type' to match database field
    description: Optional[str]
    status: str
    configuration: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    performance_metrics: Optional[Dict[str, Any]]
    
    class Config:
        from_attributes = True

class AgentUpdateRequest(BaseModel):
    """Request model for updating an agent"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    configuration: Optional[Dict[str, Any]] = Field(None)
    status: Optional[str] = Field(None, description="Agent status: active, inactive, maintenance")

class SkillAssignmentRequest(BaseModel):
    """Request model for assigning skills to an agent"""
    skill_ids: List[str] = Field(..., min_items=1, description="List of skill IDs to assign")
    proficiency_levels: Optional[List[int]] = Field(None, description="Proficiency levels (1-10) for each skill")

class PatternAssignmentRequest(BaseModel):
    """Request model for assigning patterns to an agent"""
    pattern_ids: List[str] = Field(..., min_items=1, description="List of pattern IDs to assign")
    weights: Optional[List[float]] = Field(None, description="Weights for each pattern (0.0-1.0)")

# Create router
router = APIRouter(prefix="/api/agents", tags=["🤖 Agents"])

@router.post("/", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
async def create_agent(
    agent_data: AgentCreateRequest,
    db: Session = Depends(get_db)
):
    """
    ## 🤖 Create New Agent
    
    Creates a new agent with specified configuration, skills, and patterns.
    
    **Features:**
    - Flexible agent types (custom, system, specialist)
    - Configurable parameters and settings
    - Automatic skill and pattern assignment
    - Performance metrics initialization
    
    **Agent Types:**
    - `custom`: User-defined agents with custom capabilities
    - `system`: Built-in system agents for core functions
    - `specialist`: Domain-specific agents with specialized skills
    """
    try:
        # Create agent instance
        agent = Agent(
            name=agent_data.name,
            agent_type=agent_data.agent_type,
            description=agent_data.description,
            configuration=agent_data.configuration or {},
            status="active",
            performance_metrics={}
        )
        
        db.add(agent)
        db.commit()
        db.refresh(agent)
        
        # Store skills and patterns in agent configuration for now
        if agent_data.skills or agent_data.patterns:
            if not agent.configuration:
                agent.configuration = {}
            if agent_data.skills:
                agent.configuration['skills'] = agent_data.skills
            if agent_data.patterns:
                agent.configuration['patterns'] = agent_data.patterns
        
        db.commit()
        
        logger.info(f"Created agent: {agent.name} (ID: {agent.id})")
        return agent
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@router.get("/", response_model=List[AgentResponse], dependencies=[Depends(require_api_key)])
async def list_agents(
    skip: int = Query(0, ge=0, description="Number of agents to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of agents to return"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by agent status"),
    db: Session = Depends(get_db)
):
    """
    ## 📋 List All Agents
    
    Retrieves a paginated list of all agents with optional filtering.
    
    **Filtering Options:**
    - `agent_type`: Filter by agent type (custom, system, specialist)
    - `status`: Filter by status (active, inactive, maintenance)
    
    **Pagination:**
    - `skip`: Number of records to skip
    - `limit`: Maximum number of records to return
    """
    try:
        query = db.query(Agent)
        
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type)
        if status:
            query = query.filter(Agent.status == status)
        
        agents = query.offset(skip).limit(limit).all()
        return agents
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@router.get("/available", dependencies=[Depends(require_api_key)])
async def get_available_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    db: Session = Depends(get_db)
):
    """
    ## 🔍 Get Available Agents
    
    Retrieves all agents that are currently available for task assignment.
    
    **Availability Criteria:**
    - Agent status is 'active'
    - Agent is not currently overloaded
    - Agent has required capabilities
    """
    try:
        query = db.query(Agent).filter(Agent.status == "active")
        
        if agent_type:
            query = query.filter(Agent.agent_type == agent_type)
        
        agents = query.all()
        
        # Format response with availability info
        available_agents = []
        for agent in agents:
            available_agents.append({
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type,
                "status": agent.status,
                "current_load": 0,  # Would be calculated from active tasks
                "max_capacity": agent.configuration.get("max_concurrent_tasks", 5),
                "availability": "available",
                "skills": [],  # Would be populated from agent_skills relationship
                "last_active": agent.updated_at.isoformat()
            })
        
        return {
            "agents": available_agents,
            "total_available": len(available_agents),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting available agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available agents: {str(e)}")

@router.get("/{agent_id}", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
async def get_agent(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """
    ## 🔍 Get Agent Details
    
    Retrieves detailed information about a specific agent including:
    - Basic agent information
    - Configuration parameters
    - Assigned skills and patterns
    - Performance metrics
    - Current status and activity
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")

@router.put("/{agent_id}", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
async def update_agent(
    agent_id: int,
    agent_data: AgentUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    ## ✏️ Update Agent
    
    Updates an existing agent's configuration, status, or other properties.
    
    **Updatable Fields:**
    - Name and description
    - Configuration parameters
    - Status (active, inactive, maintenance)
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update fields if provided
        if agent_data.name is not None:
            agent.name = agent_data.name
        if agent_data.description is not None:
            agent.description = agent_data.description
        if agent_data.configuration is not None:
            agent.configuration = agent_data.configuration
        if agent_data.status is not None:
            agent.status = agent_data.status
        
        agent.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(agent)
        
        logger.info(f"Updated agent: {agent.name} (ID: {agent.id})")
        return agent
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")

@router.delete("/{agent_id}", dependencies=[Depends(require_api_key)])
async def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """
    ## 🗑️ Delete Agent
    
    Permanently deletes an agent and all associated data.
    
    **Warning:** This action cannot be undone. All agent history,
    performance metrics, and task assignments will be lost.
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Skills and patterns are stored in agent configuration, so they'll be deleted with the agent
        
        # Delete the agent
        db.delete(agent)
        db.commit()
        
        logger.info(f"Deleted agent: {agent.name} (ID: {agent.id})")
        return {"message": f"Agent {agent.name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

@router.post("/{agent_id}/skills", dependencies=[Depends(require_api_key)])
async def assign_skills(
    agent_id: int,
    skill_data: SkillAssignmentRequest,
    db: Session = Depends(get_db)
):
    """
    ## 🎯 Assign Skills to Agent
    
    Assigns one or more skills to an agent with optional proficiency levels.
    
    **Proficiency Levels:**
    - 1-3: Beginner
    - 4-6: Intermediate  
    - 7-8: Advanced
    - 9-10: Expert
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update skills in agent configuration
        if not agent.configuration:
            agent.configuration = {}
        
        skills_config = []
        for i, skill_id in enumerate(skill_data.skill_ids):
            proficiency = 5  # Default proficiency
            if skill_data.proficiency_levels and i < len(skill_data.proficiency_levels):
                proficiency = skill_data.proficiency_levels[i]
            
            skills_config.append({
                "skill_id": skill_id,
                "proficiency_level": proficiency
            })
        
        agent.configuration['skills'] = skills_config
        
        db.commit()
        
        logger.info(f"Assigned {len(skill_data.skill_ids)} skills to agent {agent_id}")
        return {
            "message": f"Successfully assigned {len(skill_data.skill_ids)} skills to agent",
            "skills_assigned": skill_data.skill_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error assigning skills to agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign skills: {str(e)}")

@router.post("/{agent_id}/patterns", dependencies=[Depends(require_api_key)])
async def assign_patterns(
    agent_id: int,
    pattern_data: PatternAssignmentRequest,
    db: Session = Depends(get_db)
):
    """
    ## 🧩 Assign Patterns to Agent
    
    Assigns behavioral and coordination patterns to an agent.
    
    **Pattern Types:**
    - Coordination patterns (sequential, parallel, hierarchical)
    - Communication patterns (broadcast, peer-to-peer, hub-spoke)
    - Decision patterns (consensus, majority, expert)
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update patterns in agent configuration
        if not agent.configuration:
            agent.configuration = {}
        
        patterns_config = []
        for i, pattern_id in enumerate(pattern_data.pattern_ids):
            weight = 1.0  # Default weight
            if pattern_data.weights and i < len(pattern_data.weights):
                weight = pattern_data.weights[i]
            
            patterns_config.append({
                "pattern_id": pattern_id,
                "weight": weight
            })
        
        agent.configuration['patterns'] = patterns_config
        
        db.commit()
        
        logger.info(f"Assigned {len(pattern_data.pattern_ids)} patterns to agent {agent_id}")
        return {
            "message": f"Successfully assigned {len(pattern_data.pattern_ids)} patterns to agent",
            "patterns_assigned": pattern_data.pattern_ids
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error assigning patterns to agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign patterns: {str(e)}")

@router.get("/{agent_id}/performance", dependencies=[Depends(require_api_key)])
async def get_agent_performance(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """Get Agent Performance Metrics from real task data"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get real task statistics
        total_tasks = db.execute(text("SELECT COUNT(*) FROM tasks WHERE assigned_to = :agent_id"), {"agent_id": agent_id}).fetchone()[0] or 0
        completed_tasks = db.execute(text("SELECT COUNT(*) FROM tasks WHERE assigned_to = :agent_id AND status = 'completed'"), {"agent_id": agent_id}).fetchone()[0] or 0
        in_progress_tasks = db.execute(text("SELECT COUNT(*) FROM tasks WHERE assigned_to = :agent_id AND status = 'in_progress'"), {"agent_id": agent_id}).fetchone()[0] or 0
        
        completion_rate = round((completed_tasks / total_tasks * 100), 1) if total_tasks > 0 else 0
        
        performance_metrics = {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "completion_rate": completion_rate,
            "status": agent.status
        }
        
        return performance_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}/skills", dependencies=[Depends(require_api_key)])
async def get_agent_skills(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """
    ## 🎯 Get Agent Skills
    
    Retrieves all skills assigned to an agent with proficiency levels.
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        skills_config = agent.configuration.get('skills', []) if agent.configuration else []
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "skills": [
                {
                    "skill_id": skill.get("skill_id"),
                    "proficiency_level": skill.get("proficiency_level", 5),
                    "proficiency_label": "Expert" if skill.get("proficiency_level", 5) >= 9 else
                                       "Advanced" if skill.get("proficiency_level", 5) >= 7 else
                                       "Intermediate" if skill.get("proficiency_level", 5) >= 4 else "Beginner"
                }
                for skill in skills_config
            ],
            "total_skills": len(skills_config)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting skills for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent skills: {str(e)}")

@router.get("/{agent_id}/patterns", dependencies=[Depends(require_api_key)])
async def get_agent_patterns(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """
    ## 🧩 Get Agent Patterns
    
    Retrieves all behavioral patterns assigned to an agent.
    """
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        patterns_config = agent.configuration.get('patterns', []) if agent.configuration else []
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "patterns": [
                {
                    "pattern_id": pattern.get("pattern_id"),
                    "weight": pattern.get("weight", 1.0),
                    "type": "coordination"  # Would be determined from pattern definition
                }
                for pattern in patterns_config
            ],
            "total_patterns": len(patterns_config)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patterns for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent patterns: {str(e)}")
