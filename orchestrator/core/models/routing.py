"""
Routing models (PRD-50: Universal Orchestrator Router).

Pydantic models for the request envelope and routing decisions,
plus SQLAlchemy ORM models for persisting routing decisions, rules,
and unrouted events.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChannelSource(str, Enum):
    CHATBOT = "chatbot"
    JIRA_TRIGGER = "jira_trigger"
    SLACK = "slack"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    API = "api"
    WORKFLOW = "workflow"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    GOOGLE_CHAT = "google_chat"
    SIGNAL = "signal"
    IMESSAGE = "imessage"
    IRC = "irc"
    MATRIX = "matrix"
    LINE = "line"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RequestUser(BaseModel):
    id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    auth_type: str = "anonymous"
    clerk_user_id: Optional[str] = None


class RequestEnvelope(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: ChannelSource
    content: str
    raw_payload: Dict[str, Any] = Field(default_factory=dict)
    user: RequestUser = Field(default_factory=RequestUser)
    workspace_id: UUID
    metadata: Dict[str, Any] = Field(default_factory=dict)
    override_agent_id: Optional[int] = None
    override_workflow_id: Optional[int] = None
    conversation_id: Optional[UUID] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RoutingDecision(BaseModel):
    route_type: str  # "agent" | "workflow" | "orchestrate"
    agent_id: Optional[int] = None
    workflow_id: Optional[int] = None
    confidence: float = 0.0
    reasoning: str = ""
    cached: bool = False
    intent_category: str = ""


# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------

class RoutingDecisionRecord(Base):
    __tablename__ = "routing_decisions"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    request_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    envelope_hash = Column(String(64), nullable=False, index=True)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=True, index=True)
    source = Column(String(50), nullable=False)
    content = Column(Text, nullable=True)
    route_type = Column(String(50), nullable=False)
    agent_id = Column(Integer, nullable=True)
    workflow_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False, default=0.0)
    cached = Column(Boolean, nullable=False, default=False)
    was_corrected = Column(Boolean, nullable=False, default=False)
    corrected_agent_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class RoutingRule(Base):
    __tablename__ = "routing_rules"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    source_pattern = Column(String(100), nullable=True)
    source_channel = Column(String(50), nullable=True)  # PRD-55: channel-based routing
    intent_keywords = Column(JSONB, default=list)
    target_agent_id = Column(Integer, nullable=True)
    target_workflow_id = Column(Integer, nullable=True)
    priority = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class UnroutedEvent(Base):
    __tablename__ = "unrouted_events"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    source = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    raw_payload = Column(JSONB, default=dict)
    reason = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
