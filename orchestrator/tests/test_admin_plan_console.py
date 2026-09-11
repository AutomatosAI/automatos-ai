"""Operator console: the plan dropdown + "Last active" (2026-09-11).

Context — the defect these two features answer. PRD-222 W2b (#631) landed the
tier gating and, in the same commit, a migration backfilling every legacy
``plan = 'starter'`` row to ``'basic'``. ``basic`` declares
``families.team = False``, so ``exposure_for_plan`` reports ``nav.team = False``
and the rail drops Team Management. Every self-serve tenant was swept into that
tier; only ``Automatos Primary`` (``enterprise``, which declares NO families and
is therefore unrestricted) kept the full rail — so the platform owner never saw
it. The console now lets an operator retier a workspace directly.

Two properties are pinned here:

  * ``with_budget=False`` — the console applies a tier's seats/agents but mints
    NO spend ceiling. ``modules/policy/budget.py`` enforces
    ``plan_limits.budget`` via ``check_budget``, and a workspace with no budget
    key is ceiling-less, so minting one while retiering a live tenant would
    silently throttle it. The default (``True``) is unchanged — the onboarding
    path still writes tier ceilings.
  * the last-seen stamp is throttled and never fatal — it sits on the hottest
    path in the app (every authenticated request).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services import plan_tiers as pt


# --------------------------------------------------------------------------- #
# The regression that started it: basic hides Team, and the tiers that show it
# --------------------------------------------------------------------------- #


def test_basic_hides_team_and_analytics_nav():
    """The exact reason a tenant loses Team Management from the rail."""
    nav = pt.exposure_for_plan("basic")["nav"]
    assert nav["team"] is False
    assert nav["analytics"] is False


@pytest.mark.parametrize("plan", ["pro", "business"])
def test_paid_tiers_expose_team_nav(plan):
    assert pt.exposure_for_plan(plan)["nav"]["team"] is True


def test_enterprise_declares_no_families_so_nothing_is_gated():
    """Why the owner's own workspace never showed the symptom."""
    exposure = pt.exposure_for_plan("enterprise")
    assert exposure["families"] == {}
    assert exposure["nav"] == {"analytics": True, "team": True}


# --------------------------------------------------------------------------- #
# with_budget=False — the console's assignment never mints a ceiling
# --------------------------------------------------------------------------- #


def test_plan_limits_without_budget_keeps_every_other_limit():
    """Seats/agents apply exactly as normal; only the ceiling is withheld."""
    with_budget = pt.plan_limits_for_tier("pro")
    without = pt.plan_limits_for_tier("pro", with_budget=False)

    assert with_budget["budget"]["max_cost_usd"] == 100.0
    assert "budget" not in without
    # Every other key is identical — the switch withholds the ceiling, nothing else.
    assert {k: v for k, v in with_budget.items() if k != "budget"} == without


def test_plan_limits_with_budget_is_still_the_default():
    """Regression guard: the onboarding path (US-025) is untouched."""
    assert "budget" in pt.plan_limits_for_tier("basic")
    assert "budget" in pt.plan_limits_for_tier("pro")


def test_console_assignment_mints_no_ceiling_on_a_ceilingless_workspace():
    """A live tenant with no budget key stays ceiling-less after retiering.

    This is the InBuildUK shape: plan_limits carries generous legacy values and
    NO budget, so the workspace is ceiling-less today (``check_budget`` only
    binds on an explicit budget for a non-autonomy workspace).
    """
    ws = SimpleNamespace(
        plan="basic",
        plan_limits={"max_agents": 10, "max_members": 5, "max_documents": 100},
    )
    pt.assign_plan(None, ws, "pro", with_budget=False)

    assert ws.plan == "pro"
    assert "budget" not in ws.plan_limits  # no throttle appeared
    assert ws.plan_limits["max_members"] == 5  # pro seats
    assert ws.plan_limits["max_agents"] == 20
    assert ws.plan_limits["max_documents"] == 100  # unmanaged key survives


def test_console_assignment_clears_a_stale_tier_ceiling():
    """A ceiling a previous tier assignment wrote is cleared, not inherited."""
    ws = SimpleNamespace(plan="basic", plan_limits={})
    pt.assign_plan(None, ws, "pro")  # normal path writes the $100 tier ceiling
    assert ws.plan_limits["budget"]["max_cost_usd"] == 100.0

    # basic carries its OWN $25 ceiling on the normal path, so this proves the
    # switch both withholds the new ceiling and clears the stale one — a tier
    # with budget_usd=0 (business) would have cleared it either way.
    pt.assign_plan(None, ws, "basic", with_budget=False)
    assert ws.plan == "basic"
    assert "budget" not in ws.plan_limits


def test_console_assignment_preserves_an_admin_custom_budget():
    """An admin's own ceiling is the customer's, and survives any tier move."""
    ws = SimpleNamespace(
        plan="pro",
        plan_limits={"budget": {"window": "month", "max_cost_usd": 500.0}},
    )
    pt.assign_plan(None, ws, "basic", with_budget=False)
    assert ws.plan_limits["budget"] == {"window": "month", "max_cost_usd": 500.0}


def test_console_assignment_still_rejects_a_non_assignable_tier():
    """The endpoint maps this ValueError to a 400 — it must keep being raised."""
    ws = SimpleNamespace(plan="basic", plan_limits={})
    with pytest.raises(ValueError):
        pt.assign_plan(None, ws, "enterprise", with_budget=False)
    with pytest.raises(ValueError):
        pt.assign_plan(None, ws, "nope", with_budget=False)
    assert ws.plan == "basic"  # unchanged on rejection


def test_downgrade_seats_below_current_members_is_allowed_not_blocked():
    """The console warns; it does not refuse. The cap binds the NEXT invite
    (core/workspaces/invitations.py), never the members already in place."""
    ws = SimpleNamespace(plan="pro", plan_limits={"max_members": 5})
    pt.assign_plan(None, ws, "basic", with_budget=False)
    assert ws.plan == "basic"
    assert ws.plan_limits["max_members"] == 1  # the cap the console warns about


# --------------------------------------------------------------------------- #
# The last-seen stamp — throttled, single-statement, never fatal
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Records statements; optionally raises on execute to prove non-fatality."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.executed: list = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement, params=None):
        if self.fail:
            raise RuntimeError("database is having a day")
        self.executed.append((str(statement), params))

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def hybrid():
    mod = pytest.importorskip("core.auth.hybrid")
    mod._last_seen_touched.clear()
    yield mod
    mod._last_seen_touched.clear()


def test_last_seen_writes_once_then_throttles(hybrid):
    db = _FakeSession()
    hybrid._touch_last_seen(db, "user_abc")
    hybrid._touch_last_seen(db, "user_abc")
    hybrid._touch_last_seen(db, "user_abc")
    assert len(db.executed) == 1, "in-process throttle must suppress repeats"
    assert db.commits == 1


def test_last_seen_throttle_is_per_user(hybrid):
    db = _FakeSession()
    hybrid._touch_last_seen(db, "user_abc")
    hybrid._touch_last_seen(db, "user_xyz")
    assert len(db.executed) == 2


def test_last_seen_statement_carries_its_own_age_predicate(hybrid):
    """The UPDATE re-checks the age, so a second worker or a cold process
    cannot write more often than the interval either. No read-then-write."""
    db = _FakeSession()
    hybrid._touch_last_seen(db, "user_abc")
    sql, params = db.executed[0]
    assert "UPDATE users SET last_sign_in" in sql
    assert "last_sign_in IS NULL" in sql
    assert "make_interval" in sql
    assert params["cid"] == "user_abc"
    assert params["age"] == hybrid._LAST_SEEN_THROTTLE_SECONDS


def test_last_seen_ignores_a_missing_principal(hybrid):
    db = _FakeSession()
    hybrid._touch_last_seen(db, None)
    hybrid._touch_last_seen(db, "")
    assert db.executed == []
    assert db.commits == 0


def test_last_seen_failure_rolls_back_and_never_raises(hybrid):
    """Nobody loses a request because a bookkeeping write failed."""
    db = _FakeSession(fail=True)
    hybrid._touch_last_seen(db, "user_abc")  # must not raise
    assert db.rollbacks == 1
    assert db.commits == 0


def test_last_seen_failure_backs_off_instead_of_retrying_every_request(hybrid):
    """A PERSISTENT fault must not turn the hottest path into a write storm.

    The stamp sits on every authenticated request, so a failing UPDATE that is
    retried each time would hammer the database. Recording the attempt costs at
    most one interval of staleness and bounds the damage to one try per user
    per interval.
    """
    db = _FakeSession(fail=True)
    for _ in range(25):
        hybrid._touch_last_seen(db, "user_abc")
    assert db.rollbacks == 1, "a standing failure must be attempted once, not 25 times"
    assert hybrid._last_seen_touched.get("user_abc") is not None


def test_last_seen_cache_is_bounded(hybrid):
    """The map is per-process and must not grow without bound."""
    db = _FakeSession()
    for i in range(hybrid._LAST_SEEN_CACHE_MAX + 5):
        hybrid._touch_last_seen(db, f"user_{i}")
    assert len(hybrid._last_seen_touched) <= hybrid._LAST_SEEN_CACHE_MAX


# --------------------------------------------------------------------------- #
# The endpoints themselves — gate, validation, and the response the console reads
# --------------------------------------------------------------------------- #


class _FakeQuery:
    """Chainable stand-in: `.filter(...)` returns self, terminals return fixtures."""

    def __init__(self, result=None, scalar=None, one=None):
        self._result = result
        self._scalar = scalar
        self._one = one

    def filter(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def first(self):
        return self._result

    def scalar(self):
        return self._scalar

    def one(self):
        return self._one


def _client(workspace, members_count=0, system_role="super_admin"):
    """A TestClient over the admin router with ctx/db dependency overrides."""
    from unittest.mock import MagicMock
    from uuid import uuid4

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.admin_workspaces import router as admin_router
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db
    from core.models.workspaces import Workspace as WorkspaceModel

    db = MagicMock()

    def _query(*args):
        if args and args[0] is WorkspaceModel:
            return _FakeQuery(result=workspace)
        # The members count (and the resolver's users lookup, which must miss).
        return _FakeQuery(result=None, scalar=members_count)

    db.query.side_effect = _query

    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=uuid4(),
        user=UserContext(
            id="test-user",
            email="admin@example.com",
            role="owner",
            system_role=system_role,
        ),
        auth_type="clerk",
    )
    app.dependency_overrides[get_db] = lambda: (yield db)
    return TestClient(app)


def _workspace(plan="basic", limits=None, deleted_at=None):
    from uuid import uuid4

    return SimpleNamespace(
        id=uuid4(),
        name="InBuild UK",
        plan=plan,
        plan_limits=limits if limits is not None else {"max_members": 5, "max_agents": 10},
        deleted_at=deleted_at,
    )


def test_plans_catalogue_lists_every_tier_and_marks_assignability():
    client = _client(_workspace())
    res = client.get("/api/admin/workspaces/plans")
    assert res.status_code == 200
    tiers = {t["name"]: t for t in res.json()["tiers"]}
    assert {"basic", "pro", "business", "enterprise"} <= set(tiers)
    assert tiers["pro"]["assignable"] is True
    # enterprise is coming-soon: visible (a workspace sits on it) but not selectable.
    assert tiers["enterprise"]["assignable"] is False
    assert tiers["enterprise"]["coming_soon"] is True
    # The nav the dropdown explains its choice with.
    assert tiers["basic"]["nav"]["team"] is False
    assert tiers["pro"]["nav"]["team"] is True


def test_plans_catalogue_refuses_a_non_admin():
    client = _client(_workspace(), system_role="user")
    assert client.get("/api/admin/workspaces/plans").status_code == 403


def test_plan_change_moves_the_tier_and_writes_no_budget():
    """The InBuildUK fix: basic → pro, seats applied, ceiling NOT minted."""
    ws = _workspace(plan="basic", limits={"max_members": 5, "max_agents": 10})
    client = _client(ws, members_count=5)

    res = client.patch(f"/api/admin/workspaces/{ws.id}/plan", json={"plan": "pro"})
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["previous_plan"] == "basic"
    assert body["plan"] == "pro"
    assert ws.plan == "pro"
    assert "budget" not in body["plan_limits"], "the console must never mint a ceiling"
    assert body["plan_limits"]["max_agents"] == 20
    assert body["limits_changed"]["max_agents"] == {"from": 10, "to": 20}
    assert body["warnings"] == []  # pro seats (5) == members (5): no squeeze


def test_plan_change_warns_when_seats_fall_below_current_members():
    ws = _workspace(plan="pro", limits={"max_members": 5})
    client = _client(ws, members_count=5)

    res = client.patch(f"/api/admin/workspaces/{ws.id}/plan", json={"plan": "basic"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["plan"] == "basic"
    assert len(body["warnings"]) == 1
    assert "5 active members" in body["warnings"][0]
    # Warned, not refused — the cap binds the next invitation, not the members.
    assert body["plan_limits"]["max_members"] == 1


@pytest.mark.parametrize("plan", ["enterprise", "nope", "starter"])
def test_plan_change_rejects_a_non_assignable_tier_with_400(plan):
    ws = _workspace(plan="basic")
    client = _client(ws)
    res = client.patch(f"/api/admin/workspaces/{ws.id}/plan", json={"plan": plan})
    assert res.status_code == 400
    assert ws.plan == "basic"  # nothing written on rejection


def test_plan_change_refuses_a_non_admin():
    ws = _workspace(plan="basic")
    client = _client(ws, system_role="user")
    res = client.patch(f"/api/admin/workspaces/{ws.id}/plan", json={"plan": "pro"})
    assert res.status_code == 403
    assert ws.plan == "basic"


def test_plan_change_refuses_a_deleted_workspace():
    """Every sibling mutator guards this; the client-side disable is not enough."""
    from datetime import datetime

    ws = _workspace(plan="basic", deleted_at=datetime(2026, 9, 1))
    client = _client(ws)
    res = client.patch(f"/api/admin/workspaces/{ws.id}/plan", json={"plan": "pro"})
    assert res.status_code == 400
    assert ws.plan == "basic"


def test_plan_change_404s_for_an_unknown_workspace():
    from uuid import uuid4

    client = _client(None)
    res = client.patch(f"/api/admin/workspaces/{uuid4()}/plan", json={"plan": "pro"})
    assert res.status_code == 404


# --------------------------------------------------------------------------- #
# "Last active" is serialised as EXPLICIT UTC — a naive string would be read as
# local time by the browser and the elapsed-time column would be wrong by the
# viewer's offset.
# --------------------------------------------------------------------------- #


def test_last_active_is_serialised_with_an_explicit_utc_offset():
    from datetime import datetime, timezone

    from api.admin_workspaces import _utc_iso

    naive = datetime(2026, 9, 11, 9, 20, 20)  # what the DB column hands back
    out = _utc_iso(naive)
    assert out.endswith("+00:00"), f"no offset in {out!r} — a browser would read it local"
    # Round-trips to the same instant rather than shifting by the viewer's offset.
    assert datetime.fromisoformat(out) == naive.replace(tzinfo=timezone.utc)


def test_utc_iso_passes_through_none_and_keeps_an_aware_value():
    from datetime import datetime, timedelta, timezone

    from api.admin_workspaces import _utc_iso

    assert _utc_iso(None) is None
    aware = datetime(2026, 9, 11, 9, 20, 20, tzinfo=timezone(timedelta(hours=2)))
    assert _utc_iso(aware) == aware.isoformat()  # already explicit: untouched
