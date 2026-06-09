"""Auto autonomy level — per-workspace control over how far Auto runs unsupervised.

Single canonical reader/writer for ``workspace.settings.autonomy``. Mirrors the
``auto_reporting`` service pattern (settings live on the Workspace JSON column;
the caller owns the transaction).

Settings shape (``workspace.settings.autonomy``):

    { "level": "standard" | "full" }

Levels and what they change in PlatformActionExecutor.execute():

    standard (default)
        - admin_only actions require workspace admin/owner role.
        - requires_confirmation actions stop and return {"requires_confirmation": True}.
        This is the historical behaviour — nothing is bypassed.

    full
        - Auto is treated as admin (admin_only observability tools unlock).
        - the confirmation gate is skipped: writes AND the 4 destructive deletes
          run without asking. (The destructive backstop is unaffected — the 4
          deletes keep requires_confirmation=True, so it never fires on them.)
        Everything stays workspace-scoped; rate limits and the PRD-140 agent
        hierarchy check are NOT bypassed.

Fail-safe: callers that can't read the setting fall back to ``standard`` — the
dial fails to the supervised behaviour, never to full.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

STANDARD = "standard"
FULL = "full"
VALID_LEVELS = frozenset({STANDARD, FULL})

DEFAULTS: Dict[str, Any] = {"level": STANDARD}


def load_autonomy(db: Session, workspace_id: UUID | str) -> Dict[str, Any]:
    """Return the workspace's ``autonomy`` settings merged onto defaults."""
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        return dict(DEFAULTS)

    settings = ws.settings or {}
    user_cfg = settings.get("autonomy") or {}
    level = user_cfg.get("level")
    # Guard isinstance first: a corrupt non-string level (dict/list) is
    # unhashable and would raise on the `in` test — fail safe to standard.
    if not isinstance(level, str) or level not in VALID_LEVELS:
        level = STANDARD
    return {"level": level}


def get_autonomy_level(db: Session, workspace_id: UUID | str) -> str:
    """Return just the level string (``standard`` | ``full``)."""
    return load_autonomy(db, workspace_id)["level"]


def is_full_autonomy(db: Session, workspace_id: UUID | str) -> bool:
    """True when the workspace is running at full autonomy."""
    return get_autonomy_level(db, workspace_id) == FULL


def set_autonomy_level(
    db: Session, workspace_id: UUID | str, level: str
) -> Dict[str, Any]:
    """Persist ``level`` to ``workspace.settings.autonomy``.

    Caller owns the transaction — this stages the change and flushes. Raises
    ValueError on an unknown level or missing workspace.
    """
    if level not in VALID_LEVELS:
        raise ValueError(
            f"invalid autonomy level {level!r}; expected one of {sorted(VALID_LEVELS)}"
        )

    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        raise ValueError(f"workspace {workspace_id} not found")

    # Reassign the whole dict so SQLAlchemy detects the JSON mutation.
    settings = dict(ws.settings or {})
    settings["autonomy"] = {"level": level}
    ws.settings = settings
    db.flush()

    logger.info("[autonomy] workspace=%s level set to %s", workspace_id, level)
    return {"level": level}
