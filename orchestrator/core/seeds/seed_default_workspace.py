"""PRD-277 US-405: the install's default workspace, made by the migration Job with no owner.

Where people sign in through the organisation's identity provider (``sign_in`` is ``oidc``:
enterprise), ``DEFAULT_WORKSPACE_ID`` names the workspace sign-ins are recorded in, that
``ENTERPRISE_DEFAULT_ROLE`` gives a role in, and that the first admins own
(``ENTERPRISE_BOOTSTRAP_ADMINS``, ``core.auth.first_admin``). ``seed_default_workspace`` writes its
``workspaces`` row as the local first-run seed writes the local one: inserted only when missing
(``ON CONFLICT (id) DO NOTHING``), never changed after, and with no ``owner_id``. The first admin
to sign in becomes its owner.

It runs in the full seed loader (``docker-entrypoint.sh migrate``, which the migration Job runs,
``core.database.load_seed_data``), and sign-in calls ``ensure_default_workspace`` before granting
the first admin a membership, so an install whose Job has not run yet still works. Every other
edition is left untouched: nothing is read or written.

The caller owns the commit.
"""
from __future__ import annotations

from typing import Any, Dict, Union
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core import editions

DEFAULT_WORKSPACE_NAME = "Default workspace"
# The onboarding state of a workspace nobody has opened yet (the local first-run seed's).
FRESH_ONBOARDING = '{"stage": "not_started", "stages": {}, "segment": {}}'

_INSERT = text(
    "INSERT INTO workspaces (id, name, is_personal, is_active, onboarding) "
    "VALUES (CAST(:id AS uuid), :name, FALSE, TRUE, CAST(:onboarding AS jsonb)) "
    "ON CONFLICT (id) DO NOTHING"
)


def ensure_default_workspace(db: Session, workspace_id: Union[UUID, str]) -> str:
    """The ``workspaces`` row ``workspace_id`` names, inserted with no owner when missing; not committed.

    ``created`` or ``present``.
    """
    result = db.execute(_INSERT, {"id": str(UUID(str(workspace_id))), "name": DEFAULT_WORKSPACE_NAME,
                                  "onboarding": FRESH_ONBOARDING})
    return "created" if result.rowcount == 1 else "present"


def seed_default_workspace(db: Session) -> Dict[str, Any]:
    """The default workspace, where sign-in is OIDC and ``DEFAULT_WORKSPACE_ID`` is set; else skipped."""
    if editions.current().sign_in != editions.SIGN_IN_OIDC:
        return {"skipped": "sign-in is not oidc"}
    raw = (config.DEFAULT_WORKSPACE_ID or "").strip()
    if not raw:
        return {"skipped": "no-default-workspace"}
    return {"workspace_id": str(UUID(raw)), "workspace": ensure_default_workspace(db, raw)}


__all__ = ["DEFAULT_WORKSPACE_NAME", "ensure_default_workspace", "seed_default_workspace"]
