"""
SQLAlchemy Base (shared across all ORM models).

This module exists to avoid circular imports between:
- core.database.database (engine/session setup)
- core.models.* (ORM model declarations)

All ORM models should import `Base` from here:
    from core.database.base import Base
"""

from __future__ import annotations

from sqlalchemy.orm import declarative_base

Base = declarative_base()

