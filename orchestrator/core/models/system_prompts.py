"""
System Prompts Database Models (PRD-58)
=======================================

SQLAlchemy models for managing system-wide prompt templates.
Replaces hardcoded prompt strings with database-backed, versioned prompts.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float,
    ForeignKey, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4

from core.database.base import Base


# ===================================================================
# SQLAlchemy ORM Models
# ===================================================================

class SystemPrompt(Base):
    """Registry of system prompts used across the orchestrator."""
    __tablename__ = "system_prompts"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False, index=True)  # core | orchestration | domain | persona

    # Points to the currently active version
    active_version_id = Column(PG_UUID(as_uuid=True), nullable=True)

    # Metadata
    source_file = Column(String(300), nullable=True)
    variables = Column(JSONB, default=list)  # [{name, description}]
    impact_description = Column(String(500), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    versions = relationship("SystemPromptVersion", back_populates="prompt", order_by="SystemPromptVersion.version.desc()")

    __table_args__ = ({"extend_existing": True},)


class SystemPromptVersion(Base):
    """Versioned content for a system prompt."""
    __tablename__ = "system_prompt_versions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    prompt_id = Column(PG_UUID(as_uuid=True), ForeignKey("system_prompts.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False)

    # Content
    content = Column(Text, nullable=False)
    change_notes = Column(Text, nullable=True)

    # Status: draft | testing | active | archived
    status = Column(String(20), default="draft", nullable=False)

    # Evaluation scores (populated by FutureAGI in Phase 1B)
    eval_scores = Column(JSONB, nullable=True)
    eval_model = Column(String(100), nullable=True)
    eval_dataset_id = Column(String(200), nullable=True)
    eval_run_at = Column(DateTime(timezone=True), nullable=True)

    # A/B testing
    traffic_percentage = Column(Integer, default=0)

    # Authorship
    created_by = Column(String(100), default="system")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    prompt = relationship("SystemPrompt", back_populates="versions")
    eval_runs = relationship("SystemPromptEvalRun", back_populates="version", order_by="SystemPromptEvalRun.started_at.desc()")

    __table_args__ = (
        UniqueConstraint("prompt_id", "version", name="uq_prompt_version"),
        {"extend_existing": True},
    )


class SystemPromptEvalRun(Base):
    """Record of a FutureAGI evaluation run against a prompt version."""
    __tablename__ = "system_prompt_eval_runs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    version_id = Column(PG_UUID(as_uuid=True), ForeignKey("system_prompt_versions.id", ondelete="CASCADE"), nullable=False)

    # Evaluation details
    eval_type = Column(String(50), nullable=False)  # quality | safety | optimization | comparison
    metrics = Column(JSONB, nullable=True)
    model_used = Column(String(100), nullable=True)
    dataset_size = Column(Integer, nullable=True)
    algorithm = Column(String(50), nullable=True)

    # Results
    overall_score = Column(Float, nullable=True)
    passed = Column(Boolean, nullable=True)
    optimized_content = Column(Text, nullable=True)

    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="running")  # running | completed | failed

    # Relationships
    version = relationship("SystemPromptVersion", back_populates="eval_runs")

    __table_args__ = ({"extend_existing": True},)


# ===================================================================
# Pydantic Schemas (API request/response)
# ===================================================================

class PromptVariableSchema(BaseModel):
    name: str
    description: str = ""


class SystemPromptResponse(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    category: str
    source_file: Optional[str] = None
    variables: List[PromptVariableSchema] = []
    impact_description: Optional[str] = None
    active_version_id: Optional[str] = None
    # Denormalized from the active version for list views
    active_version_number: Optional[int] = None
    active_content: Optional[str] = None
    active_eval_scores: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SystemPromptVersionResponse(BaseModel):
    id: str
    prompt_id: str
    version: int
    content: str
    change_notes: Optional[str] = None
    status: str
    eval_scores: Optional[Dict[str, Any]] = None
    eval_model: Optional[str] = None
    eval_run_at: Optional[datetime] = None
    traffic_percentage: int = 0
    created_by: str = "system"
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class CreateVersionRequest(BaseModel):
    content: str = Field(..., min_length=1)
    change_notes: Optional[str] = None


class EvalRunResponse(BaseModel):
    id: str
    version_id: str
    eval_type: str
    metrics: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    dataset_size: Optional[int] = None
    algorithm: Optional[str] = None
    overall_score: Optional[float] = None
    passed: Optional[bool] = None
    optimized_content: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "running"

    class Config:
        from_attributes = True
