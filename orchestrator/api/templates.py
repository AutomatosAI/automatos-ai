
"""
Agent Templates API
==================

API endpoints for agent templates and creation wizards.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from database.database import get_db
from models import AgentTemplate, AgentType, PriorityLevel, SkillCategory
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/templates", tags=["templates"])

# Predefined agent templates
AGENT_TEMPLATES = [
    {
        "name": "Code Architect",
        "description": "Specialized in software architecture, design patterns, and code quality",
        "agent_type": AgentType.CODE_ARCHITECT,
        "default_skills": ["Code Review", "Design Patterns", "API Development", "Best Practices"],
        "default_configuration": {
            "programming_languages": ["python", "javascript", "java"],
            "frameworks": ["fastapi", "react", "spring"],
            "focus_areas": ["architecture", "patterns", "quality"]
        },
        "priority_level": PriorityLevel.HIGH,
        "max_concurrent_tasks": 3
    },
    {
        "name": "Security Expert",
        "description": "Focused on security analysis, vulnerability assessment, and threat modeling",
        "agent_type": AgentType.SECURITY_EXPERT,
        "default_skills": ["Vulnerability Scanning", "Threat Modeling", "Penetration Testing", "Security Monitoring"],
        "default_configuration": {
            "security_frameworks": ["OWASP", "NIST", "ISO27001"],
            "scan_types": ["static", "dynamic", "interactive"],
            "focus_areas": ["vulnerabilities", "compliance", "monitoring"]
        },
        "priority_level": PriorityLevel.CRITICAL,
        "max_concurrent_tasks": 2
    },
    {
        "name": "Performance Optimizer",
        "description": "Specializes in system performance analysis and optimization",
        "agent_type": AgentType.PERFORMANCE_OPTIMIZER,
        "default_skills": ["Performance Optimization", "Monitoring Setup", "Scaling Management", "Real-time Analytics"],
        "default_configuration": {
            "optimization_areas": ["cpu", "memory", "io", "network"],
            "monitoring_tools": ["prometheus", "grafana", "datadog"],
            "focus_areas": ["performance", "scalability", "efficiency"]
        },
        "priority_level": PriorityLevel.HIGH,
        "max_concurrent_tasks": 4
    },
    {
        "name": "Data Analyst",
        "description": "Expert in data processing, analysis, and business intelligence",
        "agent_type": AgentType.DATA_ANALYST,
        "default_skills": ["Data Processing", "Statistical Analysis", "Data Visualization", "Business Intelligence"],
        "default_configuration": {
            "data_formats": ["csv", "json", "parquet", "sql"],
            "analysis_tools": ["pandas", "numpy", "plotly", "tableau"],
            "focus_areas": ["analysis", "visualization", "insights"]
        },
        "priority_level": PriorityLevel.MEDIUM,
        "max_concurrent_tasks": 5
    },
    {
        "name": "Infrastructure Manager",
        "description": "Manages deployment, scaling, and infrastructure automation",
        "agent_type": AgentType.INFRASTRUCTURE_MANAGER,
        "default_skills": ["Deployment Automation", "Container Orchestration", "Infrastructure as Code", "Backup Management"],
        "default_configuration": {
            "platforms": ["kubernetes", "docker", "aws", "terraform"],
            "deployment_strategies": ["blue-green", "canary", "rolling"],
            "focus_areas": ["deployment", "scaling", "automation"]
        },
        "priority_level": PriorityLevel.HIGH,
        "max_concurrent_tasks": 3
    },
    {
        "name": "Full-Stack Developer",
        "description": "Versatile agent capable of both frontend and backend development",
        "agent_type": AgentType.CUSTOM,
        "default_skills": ["Code Review", "API Development", "Testing Automation", "Documentation"],
        "default_configuration": {
            "frontend_technologies": ["react", "vue", "angular"],
            "backend_technologies": ["nodejs", "python", "java"],
            "focus_areas": ["development", "testing", "documentation"]
        },
        "priority_level": PriorityLevel.MEDIUM,
        "max_concurrent_tasks": 4
    },
    {
        "name": "DevOps Engineer",
        "description": "Combines development and operations for continuous integration and deployment",
        "agent_type": AgentType.CUSTOM,
        "default_skills": ["Deployment Automation", "Infrastructure as Code", "Monitoring Setup", "Version Control"],
        "default_configuration": {
            "ci_cd_tools": ["jenkins", "gitlab-ci", "github-actions"],
            "infrastructure_tools": ["terraform", "ansible", "kubernetes"],
            "focus_areas": ["automation", "deployment", "monitoring"]
        },
        "priority_level": PriorityLevel.HIGH,
        "max_concurrent_tasks": 3
    },
    {
        "name": "Quality Assurance",
        "description": "Focused on testing, quality assurance, and compliance",
        "agent_type": AgentType.CUSTOM,
        "default_skills": ["Testing Automation", "Code Review", "Compliance Auditing", "Documentation"],
        "default_configuration": {
            "testing_frameworks": ["pytest", "jest", "selenium"],
            "quality_metrics": ["coverage", "complexity", "maintainability"],
            "focus_areas": ["testing", "quality", "compliance"]
        },
        "priority_level": PriorityLevel.MEDIUM,
        "max_concurrent_tasks": 5
    }
]

@router.get("/", response_model=List[AgentTemplate])
async def get_agent_templates():
    """Get all available agent templates for the creation wizard"""
    try:
        templates = []
        for template_data in AGENT_TEMPLATES:
            template = AgentTemplate(
                name=template_data["name"],
                description=template_data["description"],
                agent_type=template_data["agent_type"],
                default_skills=template_data["default_skills"],
                default_configuration=template_data["default_configuration"],
                priority_level=template_data["priority_level"],
                max_concurrent_tasks=template_data["max_concurrent_tasks"]
            )
            templates.append(template)
        
        return templates
        
    except Exception as e:
        logger.error(f"Error getting agent templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{agent_type}")
async def get_template_by_type(agent_type: AgentType):
    """Get a specific template by agent type"""
    try:
        for template_data in AGENT_TEMPLATES:
            if template_data["agent_type"] == agent_type:
                return AgentTemplate(
                    name=template_data["name"],
                    description=template_data["description"],
                    agent_type=template_data["agent_type"],
                    default_skills=template_data["default_skills"],
                    default_configuration=template_data["default_configuration"],
                    priority_level=template_data["priority_level"],
                    max_concurrent_tasks=template_data["max_concurrent_tasks"]
                )
        
        raise HTTPException(status_code=404, detail="Template not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/skills/suggestions")
async def get_skill_suggestions(
    agent_type: AgentType,
    db: Session = Depends(get_db)
):
    """Get skill suggestions based on agent type"""
    try:
        from models import Skill
        
        # Map agent types to skill categories
        type_to_categories = {
            AgentType.CODE_ARCHITECT: [SkillCategory.DEVELOPMENT],
            AgentType.SECURITY_EXPERT: [SkillCategory.SECURITY],
            AgentType.PERFORMANCE_OPTIMIZER: [SkillCategory.INFRASTRUCTURE, SkillCategory.ANALYTICS],
            AgentType.DATA_ANALYST: [SkillCategory.ANALYTICS],
            AgentType.INFRASTRUCTURE_MANAGER: [SkillCategory.INFRASTRUCTURE],
            AgentType.CUSTOM: [SkillCategory.DEVELOPMENT, SkillCategory.INFRASTRUCTURE],
            AgentType.SYSTEM: [SkillCategory.DEVELOPMENT, SkillCategory.SECURITY, SkillCategory.INFRASTRUCTURE],
            AgentType.SPECIALIZED: [SkillCategory.ANALYTICS]
        }
        
        suggested_categories = type_to_categories.get(agent_type, [SkillCategory.DEVELOPMENT])
        
        # Get skills from suggested categories
        category_values = [cat.value for cat in suggested_categories]
        suggested_skills = db.query(Skill).filter(
            Skill.category.in_(category_values),
            Skill.is_active == True
        ).all()
        
        # Group by category
        skills_by_category = {}
        for skill in suggested_skills:
            if skill.category not in skills_by_category:
                skills_by_category[skill.category] = []
            
            skills_by_category[skill.category].append({
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "skill_type": skill.skill_type
            })
        
        return {
            "agent_type": agent_type.value,
            "suggested_categories": [cat.value for cat in suggested_categories],
            "skills_by_category": skills_by_category
        }
        
    except Exception as e:
        logger.error(f"Error getting skill suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/creation-wizard/config")
async def get_creation_wizard_config():
    """Get configuration options for the agent creation wizard"""
    try:
        return {
            "agent_types": [
                {
                    "value": agent_type.value,
                    "label": agent_type.value.replace("_", " ").title(),
                    "description": f"Agent specialized in {agent_type.value.replace('_', ' ')}"
                } for agent_type in AgentType
            ],
            "priority_levels": [
                {
                    "value": priority.value,
                    "label": priority.value.title(),
                    "description": f"{priority.value.title()} priority level"
                } for priority in PriorityLevel
            ],
            "skill_categories": [
                {
                    "value": category.value,
                    "label": category.value.title(),
                    "description": f"Skills related to {category.value}"
                } for category in SkillCategory
            ],
            "default_settings": {
                "max_concurrent_tasks": {
                    "min": 1,
                    "max": 100,
                    "default": 5,
                    "step": 1
                },
                "auto_start": {
                    "default": False,
                    "description": "Automatically start agent when system boots"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting creation wizard config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
