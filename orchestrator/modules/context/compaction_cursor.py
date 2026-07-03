"""Field-compaction resume cursor — PRD-178 S3 (F063).

Persists the Qdrant scroll cursor a workspace's field-compaction sweep should
resume from, so a subsequent run continues where the last one stopped instead
of re-scanning the whole collection each hour.

No new table: the cursor is a single ``system_settings`` row per workspace
(category ``field_compaction``, key ``cursor:<workspace_id>``). A ``None`` value
means "start a fresh full pass" and is stored as an empty value. The vector
adapter stays DB-free — it only produces/consumes the opaque cursor; this module
owns the persistence.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from core.models.system_settings import SystemSetting

logger = logging.getLogger(__name__)

_CATEGORY = "field_compaction"


def _key(workspace_id: str) -> str:
    return f"cursor:{workspace_id}"


def load_compaction_cursor(db: Session, workspace_id: str) -> Optional[str]:
    """Return the persisted resume cursor for a workspace, or ``None`` when
    absent/empty (→ the next sweep starts a fresh full pass)."""
    try:
        row = (
            db.query(SystemSetting)
            .filter(
                SystemSetting.category == _CATEGORY,
                SystemSetting.key == _key(workspace_id),
            )
            .first()
        )
    except Exception:
        logger.warning(
            "[FieldCompaction] cursor load failed for ws=%s", workspace_id,
            exc_info=True,
        )
        return None
    if row is None or not row.value:
        return None
    return row.value


def save_compaction_cursor(
    db: Session, workspace_id: str, cursor: Optional[str]
) -> None:
    """Persist the resume cursor for a workspace. ``cursor=None`` clears it (a
    full pass completed) by storing an empty value. Upserts the single row."""
    value = "" if cursor is None else str(cursor)
    try:
        row = (
            db.query(SystemSetting)
            .filter(
                SystemSetting.category == _CATEGORY,
                SystemSetting.key == _key(workspace_id),
            )
            .first()
        )
        if row is None:
            row = SystemSetting(
                category=_CATEGORY,
                key=_key(workspace_id),
                value=value,
                value_type="string",
                description="Field-compaction resume cursor (PRD-178 S3)",
            )
            db.add(row)
        else:
            row.value = value
        db.flush()
    except Exception:
        logger.warning(
            "[FieldCompaction] cursor save failed for ws=%s", workspace_id,
            exc_info=True,
        )
