"""PRD-261 A-08a — the workspaces a caller belongs to, for the routes that can't be workspace-scoped.

The theme files route and the boot route are called by the browser without an
``X-Workspace-ID`` header (a native ``import()``, a ``<video src>``, the root layout's server
fetch: ``core/auth/hybrid.py`` then resolves a default workspace). So they never read
``ctx.workspace_id``: they authorise by membership of the workspace that holds the install.

* A signed-in person (``auth_type`` ``clerk``): every usable workspace (not paused, not
  deleted) where they have an active ``workspace_members`` row or are the owner
  (``workspaces.owner_id``, the fallback ``core/auth/workspace_permission.py`` also reads).
* The local edition's one anonymous operator (``AUTH_EDITION=local``): the owner of every
  workspace on the instance (PRD-175, ``hybrid.py``'s local lane). Gated by the edition,
  never by role.
* A paired phone or browser (``auth_type`` ``device``, PRD-266): exactly the workspace its
  row is bound to (``core/auth/device.py`` checked that it is usable), never every
  workspace, the operator's own phone included.
* Anyone else (an API key, an SDK key, SaaS's anonymous fallback): no workspace.

The user id is always ``core.auth.principal.resolve_user_pk``, never ``ctx.user.id``.
"""

from __future__ import annotations

from typing import Any, FrozenSet, NamedTuple, Optional
from uuid import UUID

from sqlalchemy import text

from config import config
from core.auth.principal import resolve_user_pk

LOCAL_EDITION = "local"
ANONYMOUS_AUTH = "anonymous"
DEVICE_AUTH = "device"  # core.auth.device.DEVICE_AUTH_TYPE
PERSON_AUTH_TYPES = frozenset({"clerk", DEVICE_AUTH})

# Bound parameters only. A paused or deleted workspace serves nothing, as the request
# context refuses it (hybrid.py _assert_workspace_usable, _workspace_exists).
MEMBER_WORKSPACES_SQL = text(
    "SELECT w.id FROM workspaces w "
    "WHERE w.deleted_at IS NULL AND w.paused_at IS NULL "
    "AND (w.owner_id = :uid OR EXISTS ("
    "SELECT 1 FROM workspace_members wm "
    "WHERE wm.workspace_id = w.id AND wm.user_id = :uid AND wm.is_active = true))"
)


class Membership(NamedTuple):
    """Who the caller is (``users.id``) and which workspaces they may read theme files of."""

    user_pk: Optional[int]
    workspace_ids: FrozenSet[UUID]
    every_workspace: bool

    def includes(self, workspace_id: Any) -> bool:
        """True when the caller is a member or owner of ``workspace_id``."""
        if workspace_id is None:
            return False
        return self.every_workspace or UUID(str(workspace_id)) in self.workspace_ids

    @property
    def is_member_anywhere(self) -> bool:
        """True when the caller belongs to at least one workspace."""
        return self.every_workspace or bool(self.workspace_ids)


NO_MEMBERSHIP = Membership(user_pk=None, workspace_ids=frozenset(), every_workspace=False)


def is_local_operator(ctx: Any) -> bool:
    """The local edition's anonymous operator: the one owner of the instance."""
    return config.AUTH_EDITION == LOCAL_EDITION and getattr(ctx, "auth_type", None) == ANONYMOUS_AUTH


def member_workspace_ids(db: Any, user_pk: int) -> FrozenSet[UUID]:
    """Every usable workspace where ``user_pk`` is an active member or the owner."""
    rows = db.execute(MEMBER_WORKSPACES_SQL, {"uid": user_pk}).fetchall()
    return frozenset(UUID(str(row[0])) for row in rows)


def _bound_membership(user_pk: int, ctx: Any) -> Membership:
    """A paired device's membership: the one workspace its row is bound to (the lane's ``workspace_id``)."""
    bound = getattr(ctx, "workspace_id", None)
    workspace_ids = frozenset({UUID(str(bound))}) if bound is not None else frozenset()
    return Membership(user_pk=user_pk, workspace_ids=workspace_ids, every_workspace=False)


def caller_membership(db: Any, ctx: Any) -> Membership:
    """The caller's membership, read from the database, never from ``ctx.workspace_id``.

    The one exception is a paired device: the lane sets ``ctx.workspace_id`` from the
    device's row and refuses a request naming any other, so that IS its binding.
    """
    if is_local_operator(ctx):
        return Membership(user_pk=resolve_user_pk(db, ctx), workspace_ids=frozenset(), every_workspace=True)
    if getattr(ctx, "auth_type", None) not in PERSON_AUTH_TYPES:
        return NO_MEMBERSHIP
    user_pk = resolve_user_pk(db, ctx)
    if user_pk is None:
        return NO_MEMBERSHIP
    if ctx.auth_type == DEVICE_AUTH:
        return _bound_membership(user_pk, ctx)
    return Membership(user_pk=user_pk, workspace_ids=member_workspace_ids(db, user_pk), every_workspace=False)


__all__ = ["Membership", "NO_MEMBERSHIP", "caller_membership", "is_local_operator", "member_workspace_ids"]
