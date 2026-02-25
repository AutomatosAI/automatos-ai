from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database.base import Base


class WidgetInstallation(Base):
    __tablename__ = "widget_installations"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    widget_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_widgets.id", ondelete="CASCADE"), nullable=False)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    last_used_at = Column(DateTime(timezone=True))
    use_count = Column(Integer, server_default="0")
    installed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    uninstalled_at = Column(DateTime(timezone=True))

    widget = relationship("MarketplaceWidget", back_populates="installations")
