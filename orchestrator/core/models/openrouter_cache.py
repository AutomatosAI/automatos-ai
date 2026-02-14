"""
OpenRouter LLM model cache models.

These tables back the /api/openrouter/* endpoints:
- openrouter_models_cache   -- cached model catalog from OpenRouter API
- openrouter_sync_jobs      -- sync job tracking (separate from Composio)

Pattern mirrors composio_cache.py for consistency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB

from core.database.base import Base


class OpenRouterModelCache(Base):
    """Cached OpenRouter model catalog entry."""

    __tablename__ = "openrouter_models_cache"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)

    # Core identity (from OpenRouter API)
    model_id = Column(String(255), nullable=False, unique=True)      # e.g. "anthropic/claude-opus-4.6"
    slug = Column(String(255))                                        # canonical_slug if available
    display_name = Column(String(255), nullable=False)                # "Anthropic: Claude Opus 4.6"
    description = Column(Text)
    provider = Column(String(100), nullable=False, index=True)        # derived: first segment of model_id

    # Pricing (stored as cost per token, same as OpenRouter returns)
    prompt_cost = Column(Float, default=0)         # cost per token for input
    completion_cost = Column(Float, default=0)      # cost per token for output
    image_cost = Column(Float, default=0)           # cost per image (if applicable)
    request_cost = Column(Float, default=0)         # flat cost per request (if applicable)
    cache_read_cost = Column(Float, default=0)      # prompt cache read cost
    cache_write_cost = Column(Float, default=0)     # prompt cache write cost

    # Sizing
    context_length = Column(Integer, default=0)
    max_completion_tokens = Column(Integer, default=0)

    # Architecture
    modality = Column(String(100))                  # e.g. "text+image->text"
    input_modalities = Column(JSONB, default=list)  # ["text", "image"]
    output_modalities = Column(JSONB, default=list) # ["text"]
    tokenizer = Column(String(100))                 # "Claude", "GPT", etc.

    # Capabilities (derived from supported_parameters + modalities)
    supports_tools = Column(Boolean, default=False)
    supports_vision = Column(Boolean, default=False)
    supports_streaming = Column(Boolean, default=True)
    supports_json_mode = Column(Boolean, default=False)
    supports_reasoning = Column(Boolean, default=False)
    supported_parameters = Column(JSONB, default=list)

    # Provider info
    top_provider_context = Column(Integer, default=0)
    top_provider_max_tokens = Column(Integer, default=0)
    is_moderated = Column(Boolean, default=False)

    # Categorization (derived)
    category = Column(String(50), index=True)       # fast, balanced, premium, coding, reasoning, vision
    tier = Column(String(50))                        # free, budget, mid, premium
    tags = Column(JSONB, default=list)

    # Status
    status = Column(String(50), default="active")
    created_timestamp = Column(Integer)              # Unix timestamp from OpenRouter

    # Full raw data for future use
    raw_data = Column("metadata", JSONB, default=dict)

    # Sync timestamps
    last_synced_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "slug": self.slug,
            "display_name": self.display_name,
            "description": self.description,
            "provider": self.provider,
            "prompt_cost": self.prompt_cost or 0,
            "completion_cost": self.completion_cost or 0,
            "image_cost": self.image_cost or 0,
            "request_cost": self.request_cost or 0,
            "cache_read_cost": self.cache_read_cost or 0,
            "cache_write_cost": self.cache_write_cost or 0,
            "context_length": self.context_length or 0,
            "max_completion_tokens": self.max_completion_tokens or 0,
            "modality": self.modality,
            "input_modalities": self.input_modalities or [],
            "output_modalities": self.output_modalities or [],
            "tokenizer": self.tokenizer,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "supports_streaming": self.supports_streaming,
            "supports_json_mode": self.supports_json_mode,
            "supports_reasoning": self.supports_reasoning,
            "supported_parameters": self.supported_parameters or [],
            "top_provider_context": self.top_provider_context or 0,
            "top_provider_max_tokens": self.top_provider_max_tokens or 0,
            "is_moderated": self.is_moderated,
            "category": self.category,
            "tier": self.tier,
            "tags": self.tags or [],
            "status": self.status,
            "created_timestamp": self.created_timestamp,
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
        }


class OpenRouterSyncJob(Base):
    """Tracks OpenRouter sync jobs (separate from Composio sync jobs)."""

    __tablename__ = "openrouter_sync_jobs"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    job_type = Column(String(50), nullable=False)         # full_sync
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed
    models_synced = Column(Integer, default=0)
    models_added = Column(Integer, default=0)
    models_updated = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    error_details = Column(JSONB)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)

    # DB column named "metadata"
    job_metadata = Column("metadata", JSONB, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "models_synced": self.models_synced,
            "models_added": self.models_added,
            "models_updated": self.models_updated,
            "errors_count": self.errors_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }
