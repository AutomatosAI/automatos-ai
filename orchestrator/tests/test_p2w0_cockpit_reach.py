"""PRD-185 S12: operator cockpit reach + honest tiles.

The Command Center "is-it-working" strip used to 403 to blank for everyone but a
super-admin. S12 splits the analytics router so a workspace's own owner/admin can
read their OWN health tiles (all ctx.workspace_id-scoped), while platform/cross-
workspace analytics stay super-admin. These are pure tests: the gate decision core
takes a fake ctx + a MagicMock session; the new endpoints are driven directly with
a mocked db. No DB / network.
"""
import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


def _ctx(*, system_role="user", role="user", clerk_user_id=None, workspace_id=None):
    """A minimal RequestContext-shaped fake."""
    user = SimpleNamespace(
        id="u", email=None, role=role, system_role=system_role, clerk_user_id=clerk_user_id
    )
    return SimpleNamespace(user=user, workspace_id=workspace_id or uuid.uuid4())


# ---------------------------------------------------------------------------
# The gate: may_see_own_workspace_health / require_workspace_admin
# ---------------------------------------------------------------------------

def test_super_admin_sees_any_workspace_without_touching_db():
    from core.auth.workspace_admin import may_see_own_workspace_health

    db = MagicMock()
    db.execute.side_effect = AssertionError("super-admin must short-circuit before any DB read")
    assert may_see_own_workspace_health(db, _ctx(system_role="super_admin")) is True


def test_workspace_owner_admin_member_passes():
    from core.auth.workspace_admin import may_see_own_workspace_health

    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (1,)  # an owner/admin membership row
    assert may_see_own_workspace_health(db, _ctx(clerk_user_id="clerk_1")) is True


def test_plain_member_denied():
    from core.auth.workspace_admin import may_see_own_workspace_health

    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None  # not an owner/admin member
    assert may_see_own_workspace_health(db, _ctx(clerk_user_id="clerk_1")) is False


def test_api_key_admin_denied_without_membership():
    """API-key principals carry system_role='admin' but no clerk_user_id — they
    must NOT reach the observability tiles just for being 'admin' (PRD-143 posture)."""
    from core.auth.workspace_admin import may_see_own_workspace_health

    db = MagicMock()
    db.execute.side_effect = AssertionError("must not query without a clerk uid")
    assert may_see_own_workspace_health(db, _ctx(system_role="admin", role="admin")) is False


def test_anonymous_denied():
    from core.auth.workspace_admin import may_see_own_workspace_health

    ctx = SimpleNamespace(user=None, workspace_id=uuid.uuid4())
    assert may_see_own_workspace_health(MagicMock(), ctx) is False


def test_require_workspace_admin_allows_super_admin_returns_ctx():
    from core.auth.workspace_admin import require_workspace_admin

    ctx = _ctx(system_role="super_admin")
    assert asyncio.run(require_workspace_admin(ctx=ctx, db=MagicMock())) is ctx


def test_require_workspace_admin_403s_plain_member():
    from fastapi import HTTPException

    from core.auth.workspace_admin import require_workspace_admin

    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    with pytest.raises(HTTPException) as ei:
        asyncio.run(require_workspace_admin(ctx=_ctx(clerk_user_id="c"), db=db))
    assert ei.value.status_code == 403


# ---------------------------------------------------------------------------
# The router split: own-workspace tiles vs platform tiles
# ---------------------------------------------------------------------------

def test_analytics_router_split_is_correct():
    from api.analytics_real import router, ws_router
    from core.auth.super_admin import require_super_admin
    from core.auth.workspace_admin import require_workspace_admin

    def _deps(r):
        return [getattr(d, "dependency", None) for d in (r.dependencies or [])]

    # Each router carries exactly its own tier gate, router-wide.
    assert require_super_admin in _deps(router)
    assert require_workspace_admin in _deps(ws_router)
    assert require_super_admin not in _deps(ws_router)

    ws_paths = {rt.path for rt in ws_router.routes}
    plat_paths = {rt.path for rt in router.routes}

    # The strip's own-workspace tiles are reachable by a workspace admin...
    for p in (
        "/api/analytics/slos",
        "/api/analytics/primitive-health",
        "/api/analytics/errors/by-subsystem",
        "/api/analytics/dashboard/success-rate",
        "/api/analytics/widget-engagement",
        "/api/analytics/activation/workspace",
        "/api/analytics/deliverable-freshness",
    ):
        assert p in ws_paths, f"{p} must be on the workspace-admin router"
        assert p not in plat_paths, f"{p} must NOT be double-registered on the super-admin router"

    # ...while platform/cross-workspace analytics stay super-admin ONLY.
    for p in ("/api/analytics/activation", "/api/analytics/selection-health"):
        assert p in plat_paths, f"{p} must stay on the super-admin router"
        assert p not in ws_paths, f"{p} must never be exposed to a workspace admin"


# ---------------------------------------------------------------------------
# New tenant-scoped tiles
# ---------------------------------------------------------------------------

def test_workspace_activation_is_scoped_boolean_first_value():
    from api.analytics_real import get_workspace_activation

    db = MagicMock()
    db.query.return_value.filter.return_value.scalar.return_value = 3
    out = asyncio.run(get_workspace_activation(ctx=_ctx(), db=db))
    assert out["activated"] is True
    assert out["completed_missions"] == 3


def test_workspace_activation_zero_completed_is_not_activated():
    from api.analytics_real import get_workspace_activation

    db = MagicMock()
    db.query.return_value.filter.return_value.scalar.return_value = 0
    out = asyncio.run(get_workspace_activation(ctx=_ctx(), db=db))
    assert out["activated"] is False
    assert out["completed_missions"] == 0


def test_deliverable_freshness_honest_empty():
    """A workspace with no deliverables reports null, never a fabricated fresh zero."""
    from api.analytics_real import get_deliverable_freshness

    db = MagicMock()
    db.execute.return_value.fetchone.return_value = SimpleNamespace(last_at=None, total=0)
    out = asyncio.run(get_deliverable_freshness(ctx=_ctx(), db=db))
    assert out["last_produced_at"] is None
    assert out["age_seconds"] is None
    assert out["total"] == 0


def test_deliverable_freshness_reports_age_seconds():
    from api.analytics_real import get_deliverable_freshness

    db = MagicMock()
    two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
    db.execute.return_value.fetchone.return_value = SimpleNamespace(last_at=two_hours_ago, total=5)
    out = asyncio.run(get_deliverable_freshness(ctx=_ctx(), db=db))
    assert out["total"] == 5
    assert out["age_seconds"] is not None
    assert out["age_seconds"] >= 7000  # ~2h, allowing for clock drift during the call
