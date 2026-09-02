from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database.base import Base


class WidgetReview(Base):
    __tablename__ = "widget_reviews"
    __table_args__ = (
        UniqueConstraint("widget_id", "user_id", name="uq_widget_reviews_widget_user"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    widget_id = Column(PGUUID(as_uuid=True), ForeignKey("marketplace_widgets.id", ondelete="CASCADE"), nullable=False)
    # Integer, matching users.id and the creating migration (sa.Integer) —
    # see marketplace_widget.developer_id for the fresh-DB crash this fixes.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer, nullable=False)
    title = Column(String(200))
    body = Column(Text)
    is_verified_purchase = Column(Boolean, server_default="false")
    status = Column(String(20), server_default="published")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    widget = relationship("MarketplaceWidget", back_populates="reviews")
