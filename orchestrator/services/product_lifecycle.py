"""A product lists, closes and restores its customers' workspaces (PRD-274 Wave 5c, decision 25).

- **Close** soft-deletes the workspace (``deleted_at``, ``is_active`` false): it stops working at once. Its
  members' sessions and its board key are refused by ``core.auth.hybrid``, its webhooks by ``api/webhooks``
  (``is_active``), and every one of its keys by ``ApiKeyService.validate_api_key``, which refuses a key whose
  workspace is closed. Its schedules queue nothing (``core.jobs.schedules``). Closing again changes nothing.
- **Restore** undoes a close within ``PRODUCT_CLOSE_GRACE_DAYS`` (7). After that its purge is due, and it
  can't be restored (``PastGrace``).
- **The purge** of a workspace closed longer than the grace is a daily platform schedule
  (``services.product_purge``), through ``purge_workspace_sync``.

A product reaches only its own workspaces: one that doesn't carry its ``settings.product_id`` is not found.
Each close and restore is recorded in the product's owner workspace's audit log.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from config import config
from core.models.products import Product
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

LIST_MAX = 200
_THEIRS = text("settings->>'product_id' = :product")


class NotTheirs(LookupError):
    """No workspace of this product has that id."""


class NotClosed(Exception):
    """A restore of a workspace that is open."""


class PastGrace(Exception):
    """A restore after the grace: the workspace's purge is due."""


def grace() -> timedelta:
    return timedelta(days=max(1, int(config.PRODUCT_CLOSE_GRACE_DAYS)))


def _utc_now() -> datetime:
    return datetime.utcnow()  # deleted_at is naive UTC, as the admin close writes it


def view(workspace: Workspace) -> Dict[str, Any]:
    """A workspace as the product sees it."""
    settings = workspace.settings or {}
    closed = workspace.deleted_at
    return {
        "workspace_id": str(workspace.id), "external_id": settings.get("source_external_id"), "name": workspace.name,
        "status": "closed" if closed else "open", "created_at": _iso(workspace.created_at),
        "closed_at": _iso(closed), "purge_after": _iso(closed + grace()) if closed else None,
        "credit_cap": settings.get("product_credit_cap"),  # its own cap, when the product set one (caps.py)
    }


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.isoformat() + "Z" if moment else None


def theirs(db: Any, product: Product, workspace_id: Any) -> Workspace:
    """The product's workspace with that id; ``NotTheirs`` for any other."""
    workspace = (db.query(Workspace).filter(Workspace.id == workspace_id, _THEIRS)
                 .params(product=str(product.id)).first())
    if workspace is None:
        raise NotTheirs(str(workspace_id))
    return workspace


def list_theirs(db: Any, product: Product, *, include_closed: bool, limit: int, offset: int) -> List[Dict[str, Any]]:
    """The product's workspaces, oldest first; closed ones only when asked."""
    query = db.query(Workspace).filter(_THEIRS).params(product=str(product.id))
    if not include_closed:
        query = query.filter(Workspace.deleted_at.is_(None))
    rows = query.order_by(Workspace.created_at, Workspace.id).offset(offset).limit(min(limit, LIST_MAX)).all()
    return [view(workspace) for workspace in rows]


def close(db: Any, product: Product, workspace_id: Any) -> Dict[str, Any]:
    """Closes the product's workspace now; its purge is due after the grace. Closing a closed one changes
    nothing. Commits."""
    workspace = theirs(db, product, workspace_id)
    if workspace.deleted_at is None:
        workspace.deleted_at, workspace.is_active = _utc_now(), False
        db.commit()
        _audit(db, product, workspace, "closed")
    return view(workspace)


def restore(db: Any, product: Product, workspace_id: Any) -> Dict[str, Any]:
    """Opens the product's closed workspace again, within the grace. Commits."""
    workspace = theirs(db, product, workspace_id)
    if workspace.deleted_at is None:
        raise NotClosed(str(workspace_id))
    if workspace.deleted_at + grace() <= _utc_now():
        raise PastGrace(str(workspace_id))
    workspace.deleted_at, workspace.is_active = None, True
    db.commit()
    _audit(db, product, workspace, "restored")
    return view(workspace)


def _audit(db: Any, product: Product, workspace: Workspace, action: str) -> None:
    from core.workspaces.audit import AuditService

    logger.warning("[products] product %s %s workspace %s", product.slug, action, workspace.id)
    AuditService(db).log(workspace_id=str(product.owner_workspace_id), user_id=None, actor_type="system",
                         action=f"product.workspace_{action}", resource_type="workspace",
                         resource_id=str(workspace.id), resource_name=workspace.name,
                         details={"product": product.slug,
                                  "external_id": (workspace.settings or {}).get("source_external_id")})


def set_cap(db: Any, product: Product, workspace_id: Any, cap: Optional[int]) -> Dict[str, Any]:
    """Sets the workspace's own monthly credit cap (None: back to its product's default), and answers it with
    the cap in force and what it spent this month (``services/credits/caps``). Commits."""
    from services.credits.caps import SETTING, cap_changed, monthly_cap, spent_this_month

    workspace = theirs(db, product, workspace_id)
    settings = dict(workspace.settings or {})
    if cap is None:
        settings.pop(SETTING, None)
    else:
        settings[SETTING] = int(cap)
    if settings.get(SETTING) != (workspace.settings or {}).get(SETTING):
        cap_changed(db, workspace.id)
    workspace.settings = settings
    db.commit()
    return {**view(workspace), "monthly_cap": monthly_cap(db, workspace),
            "spent_this_month": spent_this_month(db, workspace.id)}
