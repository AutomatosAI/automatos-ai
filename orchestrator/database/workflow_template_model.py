"""
Workflow Template Database Model
=================================

Model for storing and managing workflow templates that users can browse,
customize, and use to create new workflows.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float
from sqlalchemy.sql import func
from database.database import Base


class WorkflowTemplate(Base):
    """
    Workflow templates that users can use to quickly create workflows.
    Templates include pre-configured steps, agents, and settings.
    """
    __tablename__ = 'workflow_templates'
    
    # Primary identification
    id = Column(Integer, primary_key=True)
    template_id = Column(String(100), unique=True, nullable=False, index=True)  # e.g., "ai-code-review"
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    
    # Categorization
    category = Column(String(100), nullable=False, index=True)  # Development, Data Processing, etc.
    tags = Column(JSON, default=list)  # ["code-review", "security", "automation"]
    difficulty = Column(String(50), default='intermediate')  # beginner, intermediate, advanced
    
    # Template definition
    template_definition = Column(JSON, nullable=False)  # Full workflow structure
    # Contains: { steps: [], agents: [], config: {}, variables: [] }
    
    # Recommended configuration
    recommended_agents = Column(JSON, default=list)  # Agent types that work well with this template
    estimated_time = Column(String(50))  # "5-10 minutes"
    required_tools = Column(JSON, default=list)  # Tools that must be installed
    
    # Usage and popularity
    use_count = Column(Integer, default=0)  # How many workflows created from this template
    success_rate = Column(Float, default=0.0)  # Average success rate of workflows using this template
    popularity = Column(Integer, default=0)  # Popularity score (0-100)
    average_rating = Column(Float, default=0.0)  # User ratings (0-5)
    
    # Visibility and access
    is_public = Column(Boolean, default=True)  # Public templates visible to all users
    is_featured = Column(Boolean, default=False)  # Featured on templates page
    is_system = Column(Boolean, default=False)  # System templates (can't be deleted)
    
    # Metadata
    icon = Column(String(50))  # Emoji or icon identifier
    preview_image = Column(String(500))  # URL to preview image
    documentation_url = Column(String(500))  # Link to detailed documentation
    
    # Versioning
    version = Column(String(50), default='1.0')
    changelog = Column(JSON, default=list)  # Version history
    
    # Timestamps and ownership
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(255), nullable=False)
    last_used_at = Column(DateTime)
    
    def to_dict(self):
        """Convert template to dictionary for API responses"""
        return {
            'id': self.id,
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'tags': self.tags or [],
            'difficulty': self.difficulty,
            'template_definition': self.template_definition or {},
            'recommended_agents': self.recommended_agents or [],
            'estimated_time': self.estimated_time,
            'required_tools': self.required_tools or [],
            'use_count': self.use_count,
            'success_rate': self.success_rate,
            'popularity': self.popularity,
            'average_rating': self.average_rating,
            'is_public': self.is_public,
            'is_featured': self.is_featured,
            'is_system': self.is_system,
            'icon': self.icon,
            'preview_image': self.preview_image,
            'documentation_url': self.documentation_url,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
        }

