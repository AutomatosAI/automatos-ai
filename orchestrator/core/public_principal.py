"""PRD-264 FR-15: the one place that decides the public name an AI notice gives for whom it acts.

A notice, or the platform rule at the end of an agent's prompt, names its principal only by a
public name:

* the widget, and an agent's prompt on a widget turn, use the Site's ``display_name``
  (:func:`public_site_name`, read by :func:`site_principal_name`);
* everything else uses, in order, the business profile's ``company_name``, then the workspace's
  name when it is not the auto-created "{name or email}'s Workspace" form
  (``core/auth/hybrid.py``) (:func:`public_principal_name`, read by
  :func:`workspace_principal_name`).

A name holding an "@" is never used: it may be the owner's email. Otherwise there is no name and
the notice says only that it is an AI.

It lives in the shared kernel so every surface that names a principal (the agent prompt's rule
in ``modules.context``, the channel bots, Auto Live voice, the widget API) imports this one copy
without crossing the feature-module independence contract (``orchestrator/.importlinter``).
The DB reads run inside a savepoint, so a failed read leaves the caller's session usable; a
failure is logged and names no one.
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

# FR-15: an auto-created workspace is named "{name or email}'s Workspace" (core/auth/hybrid.py),
# so that form, or any name with an "@", may carry the owner's name or email.
AUTO_WORKSPACE_NAME_SUFFIX = "'s workspace"
PERSONAL_DATA_MARK = "@"

# The newest business profile that has a company name.
COMPANY_NAME_SQL = sql_text(
    "SELECT company_name FROM business_profiles "
    "WHERE workspace_id = CAST(:ws AS uuid) AND company_name IS NOT NULL AND company_name <> '' "
    "ORDER BY updated_at DESC LIMIT 1"
)
WORKSPACE_NAME_SQL = sql_text("SELECT name FROM workspaces WHERE id = CAST(:ws AS uuid)")
# The workspace's default Site, oldest first: the one its widget speaks for (api/widgets/config.py).
SITE_NAME_SQL = sql_text(
    "SELECT display_name FROM sites WHERE workspace_id = CAST(:ws AS uuid) ORDER BY created_at ASC LIMIT 1"
)


def public_text(value: Any) -> Optional[str]:
    """``value`` stripped when it is text that carries no personal-data mark, else ``None``."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or PERSONAL_DATA_MARK in stripped:
        return None
    return stripped


def public_chosen_name(value: Any) -> Optional[str]:
    """``value`` stripped when it is public text and not the auto-created workspace form."""
    text = public_text(value)
    if text and not text.casefold().endswith(AUTO_WORKSPACE_NAME_SUFFIX):
        return text
    return None


def public_principal_name(company_name: Any, workspace_name: Any) -> Optional[str]:
    """The public name for whom the AI acts (FR-15), or ``None``.

    The business profile's ``company_name`` first; then the workspace name, unless it is the
    auto-created "{name or email}'s Workspace" form. A name holding an "@" is never used.
    """
    return public_text(company_name) or public_chosen_name(workspace_name)


def public_site_name(display_name: Any) -> Optional[str]:
    """A Site's ``display_name`` when public output may show it (FR-15), or ``None``.

    Sites created from a workspace (the PRD-008-A backfill, ``ensure_shopify_site``) copy the
    workspace's name, which may be the auto-created form; that form, and any name holding an
    "@", is never shown.
    """
    return public_chosen_name(display_name)


def _workspace_uuid(workspace_id: Any) -> Optional[str]:
    """The workspace id as a canonical UUID string, or ``None`` when it is not one."""
    if not workspace_id:
        return None
    try:
        return str(UUID(str(workspace_id)))
    except ValueError:
        return None


def _scalar(db: Any, statement: Any, workspace: str) -> Any:
    """The first column of the first row ``statement`` returns, or ``None``."""
    row = db.execute(statement, {"ws": workspace}).fetchone()
    return row[0] if row else None


def workspace_principal_name(db: Any, workspace_id: Any) -> Optional[str]:
    """The workspace's public name (FR-15): its company name, else its public workspace name.

    ``None`` when there is no session, the id is not a UUID, no public name exists or the read
    fails (logged).
    """
    workspace = _workspace_uuid(workspace_id)
    if db is None or workspace is None:
        return None
    try:
        with db.begin_nested():
            named = public_principal_name(_scalar(db, COMPANY_NAME_SQL, workspace), None)
            if named:
                return named
            return public_principal_name(None, _scalar(db, WORKSPACE_NAME_SQL, workspace))
    except Exception:
        logger.exception("[PRD-264] principal lookup failed for ws=%s; the notice names no one", workspace)
        return None


def site_principal_name(db: Any, workspace_id: Any) -> Optional[str]:
    """The public name of the workspace's default Site (FR-15), the one its widget speaks for.

    ``None`` when there is no session, the id is not a UUID, the workspace has no Site, its name
    is not public or the read fails (logged). The workspace's own name is never used.
    """
    workspace = _workspace_uuid(workspace_id)
    if db is None or workspace is None:
        return None
    try:
        with db.begin_nested():
            return public_site_name(_scalar(db, SITE_NAME_SQL, workspace))
    except Exception:
        logger.exception("[PRD-264] site name lookup failed for ws=%s; the notice names no one", workspace)
        return None
