from __future__ import annotations

from typing import Optional, Any, Dict
from datetime import datetime
from sqlalchemy import String, Integer, JSON, DateTime, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from .models import Base


class ContextPolicyModel(Base):
    __tablename__ = "context_policies"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=text("gen_random_uuid()"))
    policy_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    domain: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)

    slots: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    max_total_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=12000)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))


