"""
PRD-58: System Prompt Management Models
========================================

Database models for versioned system prompts with evaluation tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from core.database.base import Base


class SystemPrompt(Base):
    """
    A named system prompt that can have multiple versions.
    slug is the stable identifier used by code (e.g. "chatbot-friendly").
    """

    __tablename__ = "system_prompts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(String(120), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    category = Column(
        String(50),
        nullable=False,
        default="general",
        index=True,
    )
    description = Column(Text, nullable=True)
    variables = Column(JSONB, nullable=True)  # e.g. {"agent_name": "str", "tools_list": "str[]"}
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    versions = relationship(
        "SystemPromptVersion",
        back_populates="prompt",
        cascade="all, delete-orphan",
        order_by="SystemPromptVersion.version_number.desc()",
    )
    eval_runs = relationship(
        "SystemPromptEvalRun",
        back_populates="prompt",
        cascade="all, delete-orphan",
        order_by="SystemPromptEvalRun.created_at.desc()",
    )


class SystemPromptVersion(Base):
    """
    An immutable snapshot of a prompt's content.
    Only one version per prompt can be status='active' at a time.
    """

    __tablename__ = "system_prompt_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(
        UUID(as_uuid=True),
        ForeignKey("system_prompts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    change_note = Column(String(500), nullable=True)
    status = Column(
        String(20),
        nullable=False,
        default="draft",
    )  # draft | active | archived
    created_by = Column(String(255), nullable=True)  # user id or "system"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Eval scores snapshot (populated after FutureAGI evaluation)
    eval_scores = Column(JSONB, nullable=True)

    # Relationships
    prompt = relationship("SystemPrompt", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("prompt_id", "version_number", name="uq_prompt_version"),
    )


class SystemPromptEvalRun(Base):
    """
    Tracks a FutureAGI evaluation run against a prompt version.
    """

    __tablename__ = "system_prompt_eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(
        UUID(as_uuid=True),
        ForeignKey("system_prompts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_id = Column(
        UUID(as_uuid=True),
        ForeignKey("system_prompt_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    run_type = Column(String(30), nullable=False, default="evaluate")  # evaluate | optimize | safety
    status = Column(String(20), nullable=False, default="pending")  # pending | running | completed | failed
    scores = Column(JSONB, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    prompt = relationship("SystemPrompt", back_populates="eval_runs")
    version = relationship("SystemPromptVersion")


# ========================================================================
# Pydantic schemas for API serialization
# ========================================================================

from pydantic import BaseModel, Field


class PromptVersionCreate(BaseModel):
    content: str = Field(..., min_length=1)
    change_note: Optional[str] = None


class PromptVersionResponse(BaseModel):
    id: str
    version_number: int
    content: str
    change_note: Optional[str] = None
    status: str
    created_by: Optional[str] = None
    created_at: str
    eval_scores: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}


class PromptResponse(BaseModel):
    id: str
    slug: str
    display_name: str
    category: str
    description: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    is_active: bool
    active_version_number: Optional[int] = None
    active_content: Optional[str] = None
    active_eval_scores: Optional[Dict[str, Any]] = None
    version_count: int = 0
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class EvalRunResponse(BaseModel):
    id: str
    prompt_id: str
    version_id: Optional[str] = None
    run_type: str
    status: str
    scores: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str

    model_config = {"from_attributes": True}


class EvalRunCreate(BaseModel):
    """Request body for triggering a FutureAGI evaluation."""
    version_id: Optional[str] = None  # If None, uses active version
    run_type: str = "evaluate"  # evaluate | optimize | safety
    config: Optional[Dict[str, Any]] = None  # Extra config (algorithm, metrics, etc.)
