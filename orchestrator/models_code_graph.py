from __future__ import annotations

from typing import Optional
from datetime import datetime
from sqlalchemy import String, Integer, Text, DateTime, Index, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from .models import Base


class CodeSymbol(Base):
    __tablename__ = "code_symbols"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    project: Mapped[str] = mapped_column(String(128), index=True)
    file_path: Mapped[str] = mapped_column(Text)
    symbol_name: Mapped[str] = mapped_column(String(256), index=True)
    symbol_type: Mapped[str] = mapped_column(String(32))  # function|class|method|module
    signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))

    __table_args__ = (
        Index("ix_code_symbols_project_name", "project", "symbol_name"),
    )


class CodeEdge(Base):
    __tablename__ = "code_edges"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    project: Mapped[str] = mapped_column(String(128), index=True)
    src_symbol_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("code_symbols.id", ondelete="CASCADE"), index=True)
    dst_symbol_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("code_symbols.id", ondelete="CASCADE"), index=True)
    edge_type: Mapped[str] = mapped_column(String(32))  # calls|imports|inherits|references
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))

    __table_args__ = (
        Index("ix_code_edges_project_type", "project", "edge_type"),
    )


class Playbook(Base):
    __tablename__ = "playbooks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    name: Mapped[str] = mapped_column(String(128), index=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    pattern: Mapped[str] = mapped_column(Text)  # JSON string of steps
    support: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))


