"""
Agent Registry - Central registry for agent capabilities
========================================================

Provides fast, cached access to agent skills, tools, and performance data.
This is the SINGLE SOURCE OF TRUTH for agent capabilities in the system.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func

from models import Agent, Skill, AgentToolAssignment, WorkflowExecution

logger = logging.getLogger(__name__)


@dataclass
class AgentCapabilities:
    """Complete agent capability profile"""
    agent_id: int
    name: str
    agent_type: str
    status: str
    
    # Skills and tools
    skills: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)  # All tool names (from skills + assignments)
    
    # Performance metrics
    performance: Dict[str, float] = field(default_factory=dict)
    
    # Availability
    availability: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = None
    last_updated: datetime = None


class AgentRegistry:
    """
    CENTRAL REGISTRY for all agent capabilities.
    
    Provides:
    - Fast agent lookup by ID, skill, tool, type
    - Cached capability profiles (5 min TTL)
    - Performance metrics aggregation
    - Availability checking
    
    Usage:
        registry = AgentRegistry(db_session)
        capabilities = registry.get_agent_capabilities(agent_id)
        agents = registry.find_agents_with_tool("create_pdf")
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.cache: Dict[int, AgentCapabilities] = {}
        self.cache_timestamps: Dict[int, datetime] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        logger.info("🏛️ AgentRegistry initialized")
    
    def get_agent_capabilities(self, agent_id: int, force_refresh: bool = False) -> Optional[AgentCapabilities]:
        """
        Get complete agent capability profile (cached).
        
        Returns: AgentCapabilities with skills, tools, performance, availability
        """
        # Check cache
        if not force_refresh and agent_id in self.cache:
            cache_age = datetime.now() - self.cache_timestamps.get(agent_id, datetime.min)
            if cache_age < self.cache_ttl:
                logger.debug(f"📦 Cache hit for agent {agent_id}")
                return self.cache[agent_id]
        
        # Load from database
        agent = self.db.query(Agent).options(
            joinedload(Agent.skills),
            joinedload(Agent.tool_assignments)
        ).filter(Agent.id == agent_id).first()
        
        if not agent:
            logger.warning(f"⚠️ Agent {agent_id} not found")
            return None
        
        # Extract skills with tool information
        skills = []
        all_tool_names = set()
        
        for skill in agent.skills:
            skill_info = {
                "name": skill.name,
                "description": skill.description,
                "tools": []
            }
            
            # Extract tool names from tools_schema
            if skill.tools_schema:
                try:
                    import json
                    tools_schema = skill.tools_schema
                    if isinstance(tools_schema, str):
                        tools_schema = json.loads(tools_schema)
                    
                    if isinstance(tools_schema, list):
                        for tool_spec in tools_schema:
                            if isinstance(tool_spec, dict) and 'function' in tool_spec:
                                tool_name = tool_spec['function'].get('name')
                                if tool_name:
                                    skill_info["tools"].append(tool_name)
                                    all_tool_names.add(tool_name)
                except Exception as e:
                    logger.error(f"Error extracting tools from skill {skill.name}: {e}")
            
            skills.append(skill_info)
        
        # Get performance metrics
        performance = self._get_agent_performance(agent_id)
        
        # Get availability
        availability = self._get_agent_availability(agent_id)
        
        # Create capabilities object
        capabilities = AgentCapabilities(
            agent_id=agent.id,
            name=agent.name,
            agent_type=agent.agent_type,
            status=agent.status,
            skills=skills,
            tools=list(all_tool_names),
            performance=performance,
            availability=availability,
            created_at=agent.created_at if hasattr(agent, 'created_at') else None,
            last_updated=datetime.now()
        )
        
        # Update cache
        self.cache[agent_id] = capabilities
        self.cache_timestamps[agent_id] = datetime.now()
        
        logger.debug(f"✅ Loaded capabilities for agent {agent_id}: {len(skills)} skills, {len(all_tool_names)} tools")
        
        return capabilities
    
    def _get_agent_performance(self, agent_id: int) -> Dict[str, float]:
        """Calculate agent performance metrics from execution history"""
        try:
            # Get recent executions (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            executions = self.db.query(WorkflowExecution).filter(
                WorkflowExecution.agent_id == agent_id,
                WorkflowExecution.created_at >= thirty_days_ago
            ).all()
            
            if not executions:
                return {
                    "success_rate": 0.0,
                    "total_tasks": 0,
                    "average_quality": 0.0
                }
            
            total = len(executions)
            successful = sum(1 for e in executions if e.status == 'completed')
            
            # Try to get quality scores from results
            quality_scores = []
            for execution in executions:
                if execution.results and isinstance(execution.results, dict):
                    quality = execution.results.get('quality_score')
                    if quality is not None:
                        quality_scores.append(float(quality))
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            return {
                "success_rate": successful / total if total > 0 else 0.0,
                "total_tasks": total,
                "successful_tasks": successful,
                "average_quality": avg_quality
            }
        except Exception as e:
            logger.error(f"Error calculating performance for agent {agent_id}: {e}")
            return {
                "success_rate": 0.0,
                "total_tasks": 0,
                "average_quality": 0.0
            }
    
    def _get_agent_availability(self, agent_id: int) -> Dict[str, Any]:
        """Check agent availability and current workload"""
        try:
            # Count currently running tasks
            running_tasks = self.db.query(WorkflowExecution).filter(
                WorkflowExecution.agent_id == agent_id,
                WorkflowExecution.status.in_(['pending', 'running'])
            ).count()
            
            # Get agent max concurrent tasks (default 5)
            agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
            max_concurrent = agent.max_concurrent_tasks if hasattr(agent, 'max_concurrent_tasks') else 5
            
            workload = running_tasks / max_concurrent if max_concurrent > 0 else 0.0
            
            return {
                "status": agent.status if agent else "unknown",
                "current_tasks": running_tasks,
                "max_concurrent": max_concurrent,
                "workload": min(workload, 1.0),  # 0.0-1.0
                "is_available": workload < 0.8  # Available if < 80% capacity
            }
        except Exception as e:
            logger.error(f"Error checking availability for agent {agent_id}: {e}")
            return {
                "status": "unknown",
                "current_tasks": 0,
                "workload": 0.0,
                "is_available": True
            }
    
    def find_agents_with_tool(self, tool_name: str) -> List[Agent]:
        """Find all agents that have access to specific tool"""
        try:
            # Get all active agents with skills
            agents = self.db.query(Agent).options(
                joinedload(Agent.skills)
            ).filter(Agent.status == 'active').all()
            
            matching_agents = []
            for agent in agents:
                capabilities = self.get_agent_capabilities(agent.id)
                if capabilities and tool_name in capabilities.tools:
                    matching_agents.append(agent)
            
            logger.info(f"🔍 Found {len(matching_agents)} agents with tool '{tool_name}'")
            return matching_agents
        except Exception as e:
            logger.error(f"Error finding agents with tool {tool_name}: {e}")
            return []
    
    def find_agents_with_skill(self, skill_name: str) -> List[Agent]:
        """Find all agents with specific skill"""
        try:
            agents = self.db.query(Agent).join(Agent.skills).filter(
                Skill.name.ilike(f"%{skill_name}%"),
                Agent.status == 'active'
            ).all()
            
            logger.info(f"🔍 Found {len(agents)} agents with skill '{skill_name}'")
            return agents
        except Exception as e:
            logger.error(f"Error finding agents with skill {skill_name}: {e}")
            return []
    
    def get_agents_by_type(self, agent_type: str) -> List[Agent]:
        """Get all agents of specific type (researcher, writer, etc.)"""
        try:
            agents = self.db.query(Agent).filter(
                Agent.agent_type == agent_type,
                Agent.status == 'active'
            ).all()
            
            logger.info(f"🔍 Found {len(agents)} agents of type '{agent_type}'")
            return agents
        except Exception as e:
            logger.error(f"Error finding agents by type {agent_type}: {e}")
            return []
    
    def get_all_agents(self, include_inactive: bool = False) -> List[Agent]:
        """Get all agents"""
        try:
            query = self.db.query(Agent)
            if not include_inactive:
                query = query.filter(Agent.status == 'active')
            
            agents = query.all()
            logger.info(f"🔍 Found {len(agents)} agents (include_inactive={include_inactive})")
            return agents
        except Exception as e:
            logger.error(f"Error getting all agents: {e}")
            return []
    
    def invalidate_cache(self, agent_id: Optional[int] = None):
        """Invalidate cache for specific agent or all agents"""
        if agent_id:
            self.cache.pop(agent_id, None)
            self.cache_timestamps.pop(agent_id, None)
            logger.debug(f"🗑️ Cache invalidated for agent {agent_id}")
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.debug(f"🗑️ All agent cache cleared")


# Singleton instance (optional, can be instantiated per-request)
_registry_instance: Optional[AgentRegistry] = None


def get_agent_registry(db_session: Session) -> AgentRegistry:
    """Get or create AgentRegistry instance"""
    # Note: We don't use singleton here because db_session changes per request
    # Each request should create its own registry instance
    return AgentRegistry(db_session)

