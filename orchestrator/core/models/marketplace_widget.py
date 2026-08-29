from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database.base import Base


class MarketplaceWidget(Base):
    __tablename__ = "marketplace_widgets"
    __table_args__ = {"extend_existing": True}

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    long_description = Column(Text)
    # Integer, matching users.id (Integer PK) and the creating migration
    # (20260225_create_marketplace_widgets: sa.Integer). The old PGUUID here
    # made the FK impossible — create_all crashed on any fresh database
    # (DatatypeMismatch); prod never noticed because its table already existed.
    developer_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    developer_name = Column(String(200))
    version = Column(String(20))
    changelog = Column(Text)
    pricing_type = Column(String(20), server_default="free")
    price_cents = Column(Integer)
    currency = Column(String(3), server_default="USD")
    icon_url = Column(Text)
    screenshots = Column(JSONB, server_default="[]")
    readme = Column(Text)
    keywords = Column(ARRAY(String))
    categories = Column(ARRAY(String))
    bundle_url = Column(Text)
    bundle_size = Column(Integer)
    permissions = Column(ARRAY(String))
    min_plan = Column(String(50))
    install_count = Column(Integer, server_default="0")
    rating_average = Column(Numeric(3, 2), server_default="0")
    rating_count = Column(Integer, server_default="0")
    status = Column(String(20), server_default="draft")
    published_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    installations = relationship("WidgetInstallation", back_populates="widget", cascade="all, delete-orphan")
    reviews = relationship("WidgetReview", back_populates="widget", cascade="all, delete-orphan")
