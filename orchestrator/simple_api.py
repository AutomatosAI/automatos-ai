#!/usr/bin/env python3
"""
Simple API Server for Automotas AI Testing
==========================================

A minimal FastAPI server for testing connectivity between frontend and backend.
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    agent_type: str
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = None

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = None

class SkillCreate(BaseModel):
    name: str
    description: Optional[str] = None
    skill_type: str
    implementation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class SkillUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    implementation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

# In-memory storage for agents and skills
agents_storage = {}
skills_storage = {}
next_agent_id = 6  # Start after existing mock data
next_skill_id = 12  # Start after existing mock skills

# Create FastAPI app
app = FastAPI(
    title="Automotas AI Simple API",
    description="Simple API for testing connectivity",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "automotas-ai-simple-api"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Automotas AI Simple API", "version": "1.0.0", "docs": "/docs"}

# Initialize default agents and skills data
def initialize_default_data():
    """Initialize default agents and skills data"""
    from datetime import datetime
    
    # Default skills
    default_skills = [
        {"id": 1, "name": "code_analysis", "description": "Analyze code quality and structure", "skill_type": "technical", "implementation": "static_analysis", "parameters": {"languages": ["python", "javascript", "java"]}, "performance_data": {"success_rate": 0.98}, "is_active": True, "created_at": "2024-01-15T10:30:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 2, "name": "architecture_design", "description": "Design system architecture and patterns", "skill_type": "cognitive", "implementation": "pattern_matching", "parameters": {"patterns": ["mvc", "microservices", "event_driven"]}, "performance_data": {"success_rate": 0.94}, "is_active": True, "created_at": "2024-01-15T10:30:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 3, "name": "best_practices", "description": "Apply coding best practices and standards", "skill_type": "analytical", "implementation": "rule_engine", "parameters": {"standards": ["pep8", "eslint", "sonar"]}, "performance_data": {"success_rate": 0.97}, "is_active": True, "created_at": "2024-01-15T10:30:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 4, "name": "data_processing", "description": "Process and clean data sets", "skill_type": "technical", "implementation": "pandas_processing", "parameters": {"formats": ["csv", "json", "parquet"]}, "performance_data": {"success_rate": 0.92}, "is_active": True, "created_at": "2024-02-01T09:15:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 5, "name": "pattern_recognition", "description": "Identify patterns in data", "skill_type": "analytical", "implementation": "ml_algorithms", "parameters": {"algorithms": ["clustering", "regression", "classification"]}, "performance_data": {"success_rate": 0.85}, "is_active": True, "created_at": "2024-02-01T09:15:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 6, "name": "vulnerability_scanning", "description": "Scan for security vulnerabilities", "skill_type": "technical", "implementation": "security_scanner", "parameters": {"scan_types": ["static", "dynamic", "dependency"]}, "performance_data": {"success_rate": 0.99}, "is_active": True, "created_at": "2024-01-20T14:45:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 7, "name": "threat_modeling", "description": "Model potential security threats", "skill_type": "cognitive", "implementation": "threat_analysis", "parameters": {"frameworks": ["stride", "pasta", "octave"]}, "performance_data": {"success_rate": 0.96}, "is_active": True, "created_at": "2024-01-20T14:45:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 8, "name": "performance_analysis", "description": "Analyze system performance metrics", "skill_type": "analytical", "implementation": "metrics_analysis", "parameters": {"metrics": ["cpu", "memory", "io", "network"]}, "performance_data": {"success_rate": 0.94}, "is_active": True, "created_at": "2024-03-10T11:20:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 9, "name": "bottleneck_detection", "description": "Identify performance bottlenecks", "skill_type": "technical", "implementation": "profiling_tools", "parameters": {"tools": ["profiler", "tracer", "monitor"]}, "performance_data": {"success_rate": 0.91}, "is_active": True, "created_at": "2024-03-10T11:20:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 10, "name": "deployment", "description": "Deploy applications and services", "skill_type": "operational", "implementation": "deployment_automation", "parameters": {"platforms": ["kubernetes", "docker", "aws"]}, "performance_data": {"success_rate": 0.97}, "is_active": True, "created_at": "2024-02-15T16:30:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"},
        {"id": 11, "name": "monitoring", "description": "Monitor infrastructure health", "skill_type": "operational", "implementation": "monitoring_tools", "parameters": {"tools": ["prometheus", "grafana", "elk"]}, "performance_data": {"success_rate": 0.95}, "is_active": True, "created_at": "2024-02-15T16:30:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system"}
    ]
    
    # Store skills
    for skill in default_skills:
        skills_storage[skill["id"]] = skill
    
    # Default agents
    default_agents = [
        {"id": 1, "name": "Code Architect", "description": "Expert in software architecture and code design patterns", "agent_type": "code_architect", "status": "active", "configuration": {"max_concurrent_tasks": 5, "specialization": "system_design"}, "performance_metrics": {"success_rate": 0.965, "tasks_completed": 1247, "average_execution_time": 45.2, "error_rate": 0.035}, "created_at": "2024-01-15T10:30:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system", "skills": [1, 2, 3]},
        {"id": 2, "name": "Data Analyst", "description": "Specialized in data processing and analysis", "agent_type": "data_analyst", "status": "active", "configuration": {"max_concurrent_tasks": 3, "specialization": "business_intelligence"}, "performance_metrics": {"success_rate": 0.879, "tasks_completed": 445, "average_execution_time": 67.8, "error_rate": 0.121}, "created_at": "2024-02-01T09:15:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system", "skills": [4, 5]},
        {"id": 3, "name": "Security Expert", "description": "Cybersecurity specialist for threat detection and prevention", "agent_type": "security_expert", "status": "active", "configuration": {"max_concurrent_tasks": 4, "specialization": "web_security"}, "performance_metrics": {"success_rate": 0.981, "tasks_completed": 567, "average_execution_time": 32.1, "error_rate": 0.019}, "created_at": "2024-01-20T14:45:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system", "skills": [6, 7]},
        {"id": 4, "name": "Performance Optimizer", "description": "Optimize system and application performance", "agent_type": "performance_optimizer", "status": "active", "configuration": {"max_concurrent_tasks": 2, "specialization": "database_optimization"}, "performance_metrics": {"success_rate": 0.927, "tasks_completed": 234, "average_execution_time": 89.5, "error_rate": 0.073}, "created_at": "2024-03-10T11:20:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system", "skills": [8, 9]},
        {"id": 5, "name": "Infrastructure Manager", "description": "Manage cloud infrastructure and deployments", "agent_type": "infrastructure_manager", "status": "active", "configuration": {"max_concurrent_tasks": 6, "specialization": "kubernetes"}, "performance_metrics": {"success_rate": 0.954, "tasks_completed": 789, "average_execution_time": 52.3, "error_rate": 0.046}, "created_at": "2024-02-15T16:30:00Z", "updated_at": "2024-07-28T14:00:00Z", "created_by": "system", "skills": [10, 11]}
    ]
    
    # Store agents
    for agent in default_agents:
        agents_storage[agent["id"]] = agent

# Initialize data on startup
initialize_default_data()

# Helper function to get agent with skills populated
def get_agent_with_skills(agent_id: int):
    """Get agent with skills populated from skills_storage"""
    if agent_id not in agents_storage:
        return None
    
    agent = agents_storage[agent_id].copy()
    agent_skills = []
    
    for skill_id in agent.get("skills", []):
        if skill_id in skills_storage:
            agent_skills.append(skills_storage[skill_id])
    
    agent["skills"] = agent_skills
    return agent

# Agents endpoints
@app.get("/api/agents")
async def get_agents(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    agent_type: Optional[str] = None,
    search: Optional[str] = None
):
    """Get all agents with filtering and pagination"""
    try:
        agents = []
        
        for agent_id, agent_data in agents_storage.items():
            # Apply filters
            if status and agent_data.get("status") != status:
                continue
            if agent_type and agent_data.get("agent_type") != agent_type:
                continue
            if search:
                search_lower = search.lower()
                if not (search_lower in agent_data.get("name", "").lower() or 
                       search_lower in agent_data.get("description", "").lower()):
                    continue
            
            # Get agent with skills populated
            agent_with_skills = get_agent_with_skills(agent_id)
            if agent_with_skills:
                agents.append(agent_with_skills)
        
        # Apply pagination
        paginated_agents = agents[skip:skip + limit]
        
        logger.info(f"Retrieved {len(paginated_agents)} agents")
        return paginated_agents
        
    except Exception as e:
        logger.error(f"Error retrieving agents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving agents: {str(e)}")

# Agent types endpoint
@app.get("/api/agents/types")
async def get_agent_types():
    """Get available agent types"""
    return [
        "code_architect",
        "security_expert", 
        "performance_optimizer",
        "data_analyst",
        "infrastructure_manager",
        "custom"
    ]

# Skills endpoints (must come before agent detail endpoint)
@app.get("/api/agents/skills")
async def get_skills(
    skip: int = 0,
    limit: int = 100,
    skill_type: Optional[str] = None,
    search: Optional[str] = None,
    active_only: bool = True
):
    """Get all skills with filtering and pagination"""
    try:
        skills = []
        
        for skill_id, skill_data in skills_storage.items():
            # Apply filters
            if active_only and not skill_data.get("is_active", True):
                continue
            if skill_type and skill_data.get("skill_type") != skill_type:
                continue
            if search:
                search_lower = search.lower()
                if not (search_lower in skill_data.get("name", "").lower() or 
                       search_lower in skill_data.get("description", "").lower()):
                    continue
            
            skills.append(skill_data)
        
        # Apply pagination
        paginated_skills = skills[skip:skip + limit]
        
        logger.info(f"Retrieved {len(paginated_skills)} skills")
        return paginated_skills
        
    except Exception as e:
        logger.error(f"Error retrieving skills: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving skills: {str(e)}")

@app.get("/api/agents/skills/{skill_id}")
async def get_skill(skill_id: int):
    """Get skill by ID"""
    try:
        if skill_id not in skills_storage:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        skill = skills_storage[skill_id]
        logger.info(f"Retrieved skill {skill_id}")
        return skill
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving skill {skill_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving skill: {str(e)}")

@app.post("/api/agents/skills")
async def create_skill(skill_data: SkillCreate):
    """Create a new skill"""
    try:
        global next_skill_id
        from datetime import datetime
        
        # Create new skill
        new_skill = {
            "id": next_skill_id,
            "name": skill_data.name,
            "description": skill_data.description,
            "skill_type": skill_data.skill_type,
            "implementation": skill_data.implementation,
            "parameters": skill_data.parameters or {},
            "performance_data": {},
            "is_active": True,
            "created_at": datetime.now().isoformat() + "Z",
            "updated_at": datetime.now().isoformat() + "Z",
            "created_by": "system"
        }
        
        # Store skill
        skills_storage[next_skill_id] = new_skill
        next_skill_id += 1
        
        logger.info(f"Created skill {new_skill['id']}: {new_skill['name']}")
        return new_skill
        
    except Exception as e:
        logger.error(f"Error creating skill: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating skill: {str(e)}")

@app.put("/api/agents/skills/{skill_id}")
async def update_skill(skill_id: int, skill_data: SkillUpdate):
    """Update skill"""
    try:
        if skill_id not in skills_storage:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        from datetime import datetime
        
        # Update skill fields
        skill = skills_storage[skill_id]
        if skill_data.name is not None:
            skill["name"] = skill_data.name
        if skill_data.description is not None:
            skill["description"] = skill_data.description
        if skill_data.implementation is not None:
            skill["implementation"] = skill_data.implementation
        if skill_data.parameters is not None:
            skill["parameters"] = skill_data.parameters
        if skill_data.is_active is not None:
            skill["is_active"] = skill_data.is_active
        
        skill["updated_at"] = datetime.now().isoformat() + "Z"
        
        logger.info(f"Updated skill {skill_id}")
        return skill
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating skill {skill_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating skill: {str(e)}")

@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: int):
    """Get agent by ID"""
    try:
        agent = get_agent_with_skills(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        logger.info(f"Retrieved agent {agent_id}")
        return agent
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent: {str(e)}")

@app.post("/api/agents")
async def create_agent(agent_data: AgentCreate):
    """Create a new agent"""
    try:
        global next_agent_id
        from datetime import datetime
        
        # Create new agent
        new_agent = {
            "id": next_agent_id,
            "name": agent_data.name,
            "description": agent_data.description,
            "agent_type": agent_data.agent_type,
            "status": "active",
            "configuration": agent_data.configuration or {},
            "performance_metrics": {
                "success_rate": 0.0,
                "tasks_completed": 0,
                "average_execution_time": 0.0,
                "error_rate": 0.0
            },
            "created_at": datetime.now().isoformat() + "Z",
            "updated_at": datetime.now().isoformat() + "Z",
            "created_by": "system",
            "skills": agent_data.skill_ids or []
        }
        
        # Store agent
        agents_storage[next_agent_id] = new_agent
        next_agent_id += 1
        
        # Return agent with skills populated
        agent_with_skills = get_agent_with_skills(new_agent["id"])
        
        logger.info(f"Created agent {new_agent['id']}: {new_agent['name']}")
        return agent_with_skills
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: int, agent_data: AgentUpdate):
    """Update agent"""
    try:
        if agent_id not in agents_storage:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        from datetime import datetime
        
        # Update agent fields
        agent = agents_storage[agent_id]
        if agent_data.name is not None:
            agent["name"] = agent_data.name
        if agent_data.description is not None:
            agent["description"] = agent_data.description
        if agent_data.status is not None:
            agent["status"] = agent_data.status
        if agent_data.configuration is not None:
            agent["configuration"] = agent_data.configuration
        if agent_data.skill_ids is not None:
            agent["skills"] = agent_data.skill_ids
        
        agent["updated_at"] = datetime.now().isoformat() + "Z"
        
        # Return agent with skills populated
        agent_with_skills = get_agent_with_skills(agent_id)
        
        logger.info(f"Updated agent {agent_id}")
        return agent_with_skills
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: int):
    """Delete agent"""
    try:
        if agent_id not in agents_storage:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        del agents_storage[agent_id]
        
        logger.info(f"Deleted agent {agent_id}")
        return {"message": "Agent deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")

# Skills endpoints (must come before agent detail endpoint)
@app.get("/api/agents/skills")
async def get_skills(
    skip: int = 0,
    limit: int = 100,
    skill_type: Optional[str] = None,
    search: Optional[str] = None,
    active_only: bool = True
):
    """Get all skills with filtering and pagination"""
    try:
        skills = []
        
        for skill_id, skill_data in skills_storage.items():
            # Apply filters
            if active_only and not skill_data.get("is_active", True):
                continue
            if skill_type and skill_data.get("skill_type") != skill_type:
                continue
            if search:
                search_lower = search.lower()
                if not (search_lower in skill_data.get("name", "").lower() or 
                       search_lower in skill_data.get("description", "").lower()):
                    continue
            
            skills.append(skill_data)
        
        # Apply pagination
        paginated_skills = skills[skip:skip + limit]
        
        logger.info(f"Retrieved {len(paginated_skills)} skills")
        return paginated_skills
        
    except Exception as e:
        logger.error(f"Error retrieving skills: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving skills: {str(e)}")

@app.get("/api/agents/skills/{skill_id}")
async def get_skill(skill_id: int):
    """Get skill by ID"""
    try:
        if skill_id not in skills_storage:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        skill = skills_storage[skill_id]
        logger.info(f"Retrieved skill {skill_id}")
        return skill
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving skill {skill_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving skill: {str(e)}")

@app.post("/api/agents/skills")
async def create_skill(skill_data: SkillCreate):
    """Create a new skill"""
    try:
        global next_skill_id
        from datetime import datetime
        
        # Create new skill
        new_skill = {
            "id": next_skill_id,
            "name": skill_data.name,
            "description": skill_data.description,
            "skill_type": skill_data.skill_type,
            "implementation": skill_data.implementation,
            "parameters": skill_data.parameters or {},
            "performance_data": {},
            "is_active": True,
            "created_at": datetime.now().isoformat() + "Z",
            "updated_at": datetime.now().isoformat() + "Z",
            "created_by": "system"
        }
        
        # Store skill
        skills_storage[next_skill_id] = new_skill
        next_skill_id += 1
        
        logger.info(f"Created skill {new_skill['id']}: {new_skill['name']}")
        return new_skill
        
    except Exception as e:
        logger.error(f"Error creating skill: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating skill: {str(e)}")

@app.put("/api/agents/skills/{skill_id}")
async def update_skill(skill_id: int, skill_data: SkillUpdate):
    """Update skill"""
    try:
        if skill_id not in skills_storage:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        from datetime import datetime
        
        # Update skill fields
        skill = skills_storage[skill_id]
        if skill_data.name is not None:
            skill["name"] = skill_data.name
        if skill_data.description is not None:
            skill["description"] = skill_data.description
        if skill_data.implementation is not None:
            skill["implementation"] = skill_data.implementation
        if skill_data.parameters is not None:
            skill["parameters"] = skill_data.parameters
        if skill_data.is_active is not None:
            skill["is_active"] = skill_data.is_active
        
        skill["updated_at"] = datetime.now().isoformat() + "Z"
        
        logger.info(f"Updated skill {skill_id}")
        return skill
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating skill {skill_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating skill: {str(e)}")

# Workflows endpoints
@app.get("/api/workflows")
async def get_workflows():
    """Get all workflows"""
    return [
        {"id": 1, "name": "Development Pipeline", "status": "active", "steps": 5},
        {"id": 2, "name": "Data Processing", "status": "completed", "steps": 3},
        {"id": 3, "name": "Security Audit", "status": "pending", "steps": 4}
    ]

@app.post("/api/workflows")
async def create_workflow():
    """Create a new workflow"""
    return {"message": "Workflow created successfully", "id": 4}

# Documents endpoints
@app.get("/api/documents")
async def get_documents():
    """Get all documents"""
    return [
        {"id": 1, "name": "API Documentation", "type": "markdown", "size": "2.5MB"},
        {"id": 2, "name": "System Architecture", "type": "pdf", "size": "1.8MB"},
        {"id": 3, "name": "User Guide", "type": "docx", "size": "3.2MB"}
    ]

@app.post("/api/documents")
async def upload_document():
    """Upload a new document"""
    return {"message": "Document uploaded successfully", "id": 4}

# System endpoints
@app.get("/api/system/health")
async def system_health():
    """Get system health status"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "database": "connected",
            "redis": "connected"
        },
        "metrics": {
            "cpu_usage": "15%",
            "memory_usage": "45%",
            "disk_usage": "60%"
        }
    }

@app.get("/api/system/metrics")
async def system_metrics():
    """Get system metrics with detailed resource information"""
    import psutil
    import random
    
    # Get real system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()
    
    return {
        "cpu": {
            "average_usage": cpu_percent,
            "cores": psutil.cpu_count(),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 2400
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100
        },
        "network": {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        },
        "metrics": {
            "requests_per_minute": 120,
            "active_agents": 4,
            "running_workflows": 2,
            "total_documents": 156
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
