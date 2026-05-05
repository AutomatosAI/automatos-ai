"""
System Settings Database Models
==============================

SQLAlchemy models for system-wide configuration settings.
Replaces .env file with database-backed settings.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, Float
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from .core import Base


class SettingCategory(str, Enum):
    """Categories for system settings.

    PRD-136: LLM configuration is collapsed to three tiers — Auto, System,
    Embeddings. The legacy per-service categories (codegraph, chatbot,
    complexity_assessor, coordination, knowledge_graph, memory_management)
    have been merged into `system_llm`. Embeddings is lifted out of `general`
    into its own tier.
    """
    GENERAL = "general"

    # LLM tiers (PRD-136)
    ORCHESTRATOR_LLM = "orchestrator_llm"  # Auto — premium, user-facing
    SYSTEM_LLM = "system_llm"              # System — cheap-fast internal
    EMBEDDINGS = "embeddings"              # Embeddings — vectorization

    # Non-LLM operational categories
    SYSTEM_LOGGING = "system_logging"
    API_RATE_LIMITING = "api_rate_limiting"
    BACKEND_API_KEYS = "backend_api_keys"
    PERFORMANCE = "performance"
    SECURITY = "security"
    NOTIFICATIONS = "notifications"
    BACKUPS = "backups"
    MONITORING = "monitoring"
    LLM_COST_AUDIT = "llm_cost_audit"  # Cost tracking, budget alerts

    # Tool-loop limits (chatbot conversational + recipe step + heartbeat + coordinator)
    CHATBOT = "chatbot"
    RECIPE = "recipe"
    AGENT_HEARTBEAT = "agent_heartbeat"
    COORDINATOR = "coordinator"


class SystemSetting(Base):
    """System-wide configuration settings"""
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False, index=True)
    key = Column(String(100), nullable=False, index=True)
    value = Column(Text, nullable=True)
    value_type = Column(String(20), default="string")  # string, number, boolean, json
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False)  # For password-like fields
    is_required = Column(Boolean, default=False)
    default_value = Column(Text, nullable=True)
    validation_rules = Column(JSON, nullable=True)  # For form validation
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(String(100), default="system")
    
    # Unique constraint
    __table_args__ = (
        {"extend_existing": True},
    )


# Pydantic Models for API
class SystemSettingResponse(BaseModel):
    """System setting response model"""
    id: int
    category: str
    key: str
    value: Optional[str]
    value_type: str
    description: Optional[str]
    is_sensitive: bool
    is_required: bool
    default_value: Optional[str]
    validation_rules: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    class Config:
        from_attributes = True


class SystemSettingUpdate(BaseModel):
    """System setting update model"""
    value: str
    description: Optional[str] = None


class SystemSettingCreate(BaseModel):
    """System setting create model"""
    category: str
    key: str
    value: str
    value_type: str = "string"
    description: Optional[str] = None
    is_sensitive: bool = False
    is_required: bool = False
    default_value: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None


class SystemSettingsByCategory(BaseModel):
    """Settings grouped by category"""
    category: str
    settings: List[SystemSettingResponse]
    total_count: int


class SystemSettingsStats(BaseModel):
    """System settings statistics"""
    total_settings: int
    categories: Dict[str, int]
    sensitive_settings: int
    required_settings: int
