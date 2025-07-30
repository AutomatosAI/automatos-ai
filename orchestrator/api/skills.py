
"""
Skills Management API
====================

API endpoints for managing skills, skill categories, and agent-skill associations.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from database import get_db
from models import (
    Skill, Agent, SkillCreate, SkillUpdate, SkillResponse, 
    SkillsByCategory, SkillCategory, agent_skills
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["skills"])

@router.get("/skills/categories", response_model=List[SkillsByCategory])
async def get_skills_by_categories(
    category: Optional[SkillCategory] = None,
    db: Session = Depends(get_db)
):
    """Get skills organized by categories"""
    try:
        if category:
            # Get skills for specific category
            skills = db.query(Skill).filter(
                Skill.category == category.value,
                Skill.is_active == True
            ).all()
            
            skills_data = [
                SkillResponse(
                    id=skill.id,
                    name=skill.name,
                    description=skill.description,
                    skill_type=skill.skill_type,
                    category=skill.category,
                    implementation=skill.implementation,
                    parameters=skill.parameters,
                    performance_data=skill.performance_data,
                    is_active=skill.is_active,
                    created_at=skill.created_at,
                    updated_at=skill.updated_at,
                    created_by=skill.created_by
                ) for skill in skills
            ]
            
            return [SkillsByCategory(category=category.value, skills=skills_data)]
        
        else:
            # Get all skills organized by categories
            categories_data = []
            for cat in SkillCategory:
                skills = db.query(Skill).filter(
                    Skill.category == cat.value,
                    Skill.is_active == True
                ).all()
                
                skills_data = [
                    SkillResponse(
                        id=skill.id,
                        name=skill.name,
                        description=skill.description,
                        skill_type=skill.skill_type,
                        category=skill.category,
                        implementation=skill.implementation,
                        parameters=skill.parameters,
                        performance_data=skill.performance_data,
                        is_active=skill.is_active,
                        created_at=skill.created_at,
                        updated_at=skill.updated_at,
                        created_by=skill.created_by
                    ) for skill in skills
                ]
                
                if skills_data:  # Only include categories that have skills
                    categories_data.append(SkillsByCategory(category=cat.value, skills=skills_data))
            
            return categories_data
            
    except Exception as e:
        logger.error(f"Error getting skills by categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}/skills", response_model=List[SkillResponse])
async def get_agent_skills(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """Get skills associated with a specific agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get agent's skills
        skills = db.query(Skill).join(agent_skills).filter(
            agent_skills.c.agent_id == agent_id,
            Skill.is_active == True
        ).all()
        
        return [
            SkillResponse(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                skill_type=skill.skill_type,
                category=skill.category,
                implementation=skill.implementation,
                parameters=skill.parameters,
                performance_data=skill.performance_data,
                is_active=skill.is_active,
                created_at=skill.created_at,
                updated_at=skill.updated_at,
                created_by=skill.created_by
            ) for skill in skills
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/skills")
async def add_skills_to_agent(
    agent_id: int,
    skill_ids: List[int],
    db: Session = Depends(get_db)
):
    """Add skills to an agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Verify all skills exist
        skills = db.query(Skill).filter(
            Skill.id.in_(skill_ids),
            Skill.is_active == True
        ).all()
        
        if len(skills) != len(skill_ids):
            found_ids = [skill.id for skill in skills]
            missing_ids = [sid for sid in skill_ids if sid not in found_ids]
            raise HTTPException(
                status_code=404, 
                detail=f"Skills not found: {missing_ids}"
            )
        
        # Add skills to agent (avoid duplicates)
        current_skill_ids = [skill.id for skill in agent.skills]
        new_skills = [skill for skill in skills if skill.id not in current_skill_ids]
        
        agent.skills.extend(new_skills)
        db.commit()
        
        return {
            "message": f"Added {len(new_skills)} skills to agent",
            "agent_id": agent_id,
            "added_skill_ids": [skill.id for skill in new_skills],
            "total_skills": len(agent.skills)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding skills to agent: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{agent_id}/skills/{skill_id}")
async def remove_skill_from_agent(
    agent_id: int,
    skill_id: int,
    db: Session = Depends(get_db)
):
    """Remove a skill from an agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if skill exists and is associated with agent
        skill = db.query(Skill).filter(Skill.id == skill_id).first()
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        if skill not in agent.skills:
            raise HTTPException(
                status_code=404, 
                detail="Skill not associated with this agent"
            )
        
        # Remove skill from agent
        agent.skills.remove(skill)
        db.commit()
        
        return {
            "message": "Skill removed from agent",
            "agent_id": agent_id,
            "removed_skill_id": skill_id,
            "remaining_skills": len(agent.skills)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing skill from agent: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/skills", response_model=List[SkillResponse])
async def get_all_skills(
    category: Optional[SkillCategory] = None,
    skill_type: Optional[str] = None,
    is_active: Optional[bool] = True,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all skills with optional filtering"""
    try:
        query = db.query(Skill)
        
        if category:
            query = query.filter(Skill.category == category.value)
        
        if skill_type:
            query = query.filter(Skill.skill_type == skill_type)
        
        if is_active is not None:
            query = query.filter(Skill.is_active == is_active)
        
        skills = query.offset(skip).limit(limit).all()
        
        return [
            SkillResponse(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                skill_type=skill.skill_type,
                category=skill.category,
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
        logger.error(f"Error getting all skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/skills", response_model=SkillResponse)
async def create_skill(
    skill: SkillCreate,
    db: Session = Depends(get_db)
):
    """Create a new skill"""
    try:
        # Check if skill name already exists
        existing = db.query(Skill).filter(Skill.name == skill.name).first()
        if existing:
            raise HTTPException(
                status_code=400, 
                detail="Skill with this name already exists"
            )
        
        db_skill = Skill(
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type.value,
            category=skill.category.value,
            implementation=skill.implementation,
            parameters=skill.parameters,
            created_by="api"
        )
        
        db.add(db_skill)
        db.commit()
        db.refresh(db_skill)
        
        return SkillResponse(
            id=db_skill.id,
            name=db_skill.name,
            description=db_skill.description,
            skill_type=db_skill.skill_type,
            category=db_skill.category,
            implementation=db_skill.implementation,
            parameters=db_skill.parameters,
            performance_data=db_skill.performance_data,
            is_active=db_skill.is_active,
            created_at=db_skill.created_at,
            updated_at=db_skill.updated_at,
            created_by=db_skill.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating skill: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/skills/{skill_id}", response_model=SkillResponse)
async def update_skill(
    skill_id: int,
    skill_update: SkillUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing skill"""
    try:
        db_skill = db.query(Skill).filter(Skill.id == skill_id).first()
        if not db_skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        # Update fields if provided
        update_data = skill_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == "category" and value:
                setattr(db_skill, field, value.value)
            else:
                setattr(db_skill, field, value)
        
        db.commit()
        db.refresh(db_skill)
        
        return SkillResponse(
            id=db_skill.id,
            name=db_skill.name,
            description=db_skill.description,
            skill_type=db_skill.skill_type,
            category=db_skill.category,
            implementation=db_skill.implementation,
            parameters=db_skill.parameters,
            performance_data=db_skill.performance_data,
            is_active=db_skill.is_active,
            created_at=db_skill.created_at,
            updated_at=db_skill.updated_at,
            created_by=db_skill.created_by
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating skill: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/skills/{skill_id}")
async def delete_skill(
    skill_id: int,
    db: Session = Depends(get_db)
):
    """Delete a skill (soft delete by setting is_active=False)"""
    try:
        db_skill = db.query(Skill).filter(Skill.id == skill_id).first()
        if not db_skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        
        # Soft delete
        db_skill.is_active = False
        db.commit()
        
        return {"message": "Skill deleted successfully", "skill_id": skill_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting skill: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
