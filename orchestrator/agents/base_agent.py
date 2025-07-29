
"""
Base Agent Implementation
========================

Abstract base class for all professional agents with common functionality,
skill management, and execution patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    TRAINING = "training"

class SkillType(Enum):
    COGNITIVE = "cognitive"
    TECHNICAL = "technical"
    COMMUNICATION = "communication"
    ANALYTICAL = "analytical"
    OPERATIONAL = "operational"

@dataclass
class AgentSkill:
    """Represents a specific skill that an agent can perform"""
    name: str
    skill_type: SkillType
    description: str
    implementation: Optional[str] = None
    parameters: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class AgentCapability:
    """Represents a high-level capability composed of multiple skills"""
    name: str
    description: str
    required_skills: List[str]
    optional_skills: List[str] = None
    complexity_level: int = 1  # 1-5 scale
    
    def __post_init__(self):
        if self.optional_skills is None:
            self.optional_skills = []

class BaseAgent(ABC):
    """Abstract base class for all professional agents"""
    
    def __init__(self, 
                 agent_id: int,
                 name: str,
                 description: str,
                 configuration: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.configuration = configuration or {}
        self.status = AgentStatus.INACTIVE
        self.skills: Dict[str, AgentSkill] = {}
        self.capabilities: Dict[str, AgentCapability] = {}
        self.performance_metrics = {
            'tasks_completed': 0,
            'success_rate': 0.0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0,
            'error_count': 0
        }
        self.created_at = datetime.now()
        self.last_activity = None
        
        # Initialize agent-specific skills and capabilities
        self._initialize_skills()
        self._initialize_capabilities()
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier"""
        pass
    
    @property
    @abstractmethod
    def default_skills(self) -> List[str]:
        """Return list of default skills for this agent type"""
        pass
    
    @property
    @abstractmethod
    def specializations(self) -> List[str]:
        """Return list of specializations for this agent type"""
        pass
    
    @abstractmethod
    def _initialize_skills(self):
        """Initialize agent-specific skills"""
        pass
    
    @abstractmethod
    def _initialize_capabilities(self):
        """Initialize agent-specific capabilities"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using agent's capabilities"""
        pass
    
    def add_skill(self, skill: AgentSkill):
        """Add a skill to the agent"""
        self.skills[skill.name] = skill
        logger.info(f"Added skill '{skill.name}' to agent '{self.name}'")
    
    def remove_skill(self, skill_name: str):
        """Remove a skill from the agent"""
        if skill_name in self.skills:
            del self.skills[skill_name]
            logger.info(f"Removed skill '{skill_name}' from agent '{self.name}'")
    
    def has_skill(self, skill_name: str) -> bool:
        """Check if agent has a specific skill"""
        return skill_name in self.skills
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        if capability_name not in self.capabilities:
            return False
        
        capability = self.capabilities[capability_name]
        return all(self.has_skill(skill) for skill in capability.required_skills)
    
    def get_available_capabilities(self) -> List[str]:
        """Get list of capabilities the agent can currently perform"""
        return [name for name, cap in self.capabilities.items() 
                if self.has_capability(name)]
    
    def update_performance_metrics(self, execution_time: float, success: bool):
        """Update agent performance metrics after task execution"""
        self.performance_metrics['tasks_completed'] += 1
        self.performance_metrics['total_execution_time'] += execution_time
        self.performance_metrics['average_execution_time'] = (
            self.performance_metrics['total_execution_time'] / 
            self.performance_metrics['tasks_completed']
        )
        
        if success:
            # Update success rate
            total_tasks = self.performance_metrics['tasks_completed']
            current_successes = (self.performance_metrics['success_rate'] * (total_tasks - 1))
            self.performance_metrics['success_rate'] = (current_successes + 1) / total_tasks
        else:
            self.performance_metrics['error_count'] += 1
            # Update success rate
            total_tasks = self.performance_metrics['tasks_completed']
            current_successes = (self.performance_metrics['success_rate'] * (total_tasks - 1))
            self.performance_metrics['success_rate'] = current_successes / total_tasks
        
        self.last_activity = datetime.now()
    
    def set_status(self, status: AgentStatus):
        """Update agent status"""
        old_status = self.status
        self.status = status
        logger.info(f"Agent '{self.name}' status changed from {old_status.value} to {status.value}")
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive status information"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'type': self.agent_type,
            'status': self.status.value,
            'skills': list(self.skills.keys()),
            'capabilities': self.get_available_capabilities(),
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }
    
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate if the agent can handle the given task"""
        required_capability = task.get('required_capability')
        if required_capability and not self.has_capability(required_capability):
            return False
        
        required_skills = task.get('required_skills', [])
        return all(self.has_skill(skill) for skill in required_skills)
    
    def estimate_execution_time(self, task: Dict[str, Any]) -> float:
        """Estimate execution time for a task based on complexity and agent performance"""
        base_time = task.get('estimated_duration', 60.0)  # Default 1 minute
        complexity_multiplier = task.get('complexity', 1.0)
        
        # Adjust based on agent's average execution time
        if self.performance_metrics['average_execution_time'] > 0:
            performance_factor = self.performance_metrics['average_execution_time'] / 60.0
        else:
            performance_factor = 1.0
        
        return base_time * complexity_multiplier * performance_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'description': self.description,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'configuration': self.configuration,
            'skills': {name: {
                'name': skill.name,
                'type': skill.skill_type.value,
                'description': skill.description,
                'parameters': skill.parameters,
                'performance_metrics': skill.performance_metrics
            } for name, skill in self.skills.items()},
            'capabilities': {name: {
                'name': cap.name,
                'description': cap.description,
                'required_skills': cap.required_skills,
                'optional_skills': cap.optional_skills,
                'complexity_level': cap.complexity_level
            } for name, cap in self.capabilities.items()},
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAgent':
        """Create agent from dictionary representation"""
        # This is a factory method that should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement from_dict method")
