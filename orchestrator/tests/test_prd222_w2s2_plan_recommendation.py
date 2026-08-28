"""PRD-222 W2·S2 (US-025) — the proposal recommends a plan; acceptance sets it.

Pure tests pin the explainable recommendation rules, the proposal copy (display
pricing + enterprise-soon), the plan funnel events, the section injection, and
the platform_update_onboarding handler: an accepted plan writes plan + plan_limits
through the US-023 helper and stamps plan_accepted; a proposal advance stamps
plan_recommended; 'enterprise' is rejected with honest copy. Handler tests run
against a MagicMock session + fake workspace — no DB (mirrors the US-003 harness).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from modules.context.sections.onboarding import OnboardingSection
from modules.tools.discovery.handlers_onboarding import update_onboarding
from services import plan_tiers as pt
from services.onboarding_state import record_plan_event


def _run(coro):
    return asyncio.run(coro)


class _FakeWorkspace:
    def __init__(self, *, onboarding=None, plan="basic", plan_limits=None):
        self.id = "ws-test"
        self.onboarding = onboarding if onboarding is not None else {"stage": "not_started"}
        self.plan = plan
        self.plan_limits = plan_limits if plan_limits is not None else {}


def _db_returning(workspace):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    return db


# --------------------------------------------------------------------------- #
# Recommendation rules (pure, explainable)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "segment,team_size,expected",
    [
        ({"business": "solo barber shop", "comfort": "brand new"}, None, "basic"),
        ({}, None, "basic"),
        ({"comfort": "very technical"}, None, "pro"),
        ({"goal": "automate our data pipeline with sql"}, None, "pro"),
        ({"business": "small marketing team"}, 3, "pro"),
        ({"business": "a creative agency with several teams"}, None, "business"),
        ({"business": "consultancy"}, 8, "business"),
    ],
)
def test_recommend_plan_rules(segment, team_size, expected):
    plan, reason = pt.recommend_plan(segment, team_size)
    assert plan == expected
    assert isinstance(reason, str) and reason  # explainable, non-empty


def test_recommend_plan_only_returns_assignable_tiers():
    for seg in ({}, {"business": "agency"}, {"comfort": "technical"}):
        plan, _ = pt.recommend_plan(seg)
        assert pt.is_assignable(plan)


def test_proposal_copy_has_pricing_and_enterprise_soon():
    copy = pt.plan_proposal_copy({"business": "solo shop", "comfort": "brand new"})
    assert "Basic $19/mo" in copy and "Pro $49/mo" in copy and "Business $99/mo" in copy
    assert "early access" in copy.lower()
    assert "enterprise is coming soon" in copy.lower()
    # Tells Auto to set the accepted plan via the tool (basic recommended here).
    assert "platform_update_onboarding" in copy
    assert 'plan: "basic"' in copy


def test_section_proposal_injects_recommendation():
    sec = OnboardingSection()
    copy = sec._plan_recommendation({"segment": {"business": "solo barber", "comfort": "brand new"}})
    assert "Basic" in copy and "$19" in copy


# --------------------------------------------------------------------------- #
# Funnel events (pure — db=None escape hatch)
# --------------------------------------------------------------------------- #


def test_record_plan_event_stamps_funnel():
    ws = _FakeWorkspace()
    record_plan_event(None, ws, "plan_accepted", "pro")
    assert ws.onboarding["funnel"]["plan_accepted"]["plan"] == "pro"
    assert "at" in ws.onboarding["funnel"]["plan_accepted"]


def test_record_plan_event_rejects_unknown_event():
    with pytest.raises(ValueError):
        record_plan_event(None, _FakeWorkspace(), "plan_invented", "pro")


# --------------------------------------------------------------------------- #
# Handler — platform_update_onboarding with a plan
# --------------------------------------------------------------------------- #


def test_handler_accepts_plan_writes_limits_and_funnel():
    ws = _FakeWorkspace(plan="basic", plan_limits={"max_documents": 100})
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {"plan": "pro"}))
    assert res["success"] is True
    assert ws.plan == "pro"
    # US-023 helper wrote the tier limits (seat key = max_members) and preserved
    # the unmanaged key.
    assert ws.plan_limits["max_members"] == 5
    assert ws.plan_limits["max_agents"] == 20
    assert ws.plan_limits["max_documents"] == 100
    # plan_accepted funnel event stamped.
    assert ws.onboarding["funnel"]["plan_accepted"]["plan"] == "pro"


def test_handler_rejects_enterprise_with_honest_copy():
    ws = _FakeWorkspace(plan="basic")
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {"plan": "enterprise"}))
    assert res["success"] is False
    assert "coming soon" in res["error"].lower()
    assert ws.plan == "basic"  # unchanged — rejected before any write


def test_handler_rejects_unknown_plan():
    ws = _FakeWorkspace(plan="basic")
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {"plan": "galaxy"}))
    assert res["success"] is False
    assert ws.plan == "basic"


def test_handler_proposal_advance_records_recommendation():
    ws = _FakeWorkspace(onboarding={"stage": "teach"})
    res = _run(
        update_onboarding(
            _db_returning(ws),
            uuid4(),
            {"advance_to": "proposal", "segment": {"business": "a creative agency"}},
        )
    )
    assert res["success"] is True
    assert ws.onboarding["stage"] == "proposal"
    # The recommendation (from the stored segment) is stamped as a funnel event.
    assert ws.onboarding["funnel"]["plan_recommended"]["plan"] == "business"


def test_handler_at_least_one_rule_names_plan():
    ws = _FakeWorkspace()
    res = _run(update_onboarding(_db_returning(ws), uuid4(), {}))
    assert res["success"] is False
    assert "plan" in res["error"]


# --------------------------------------------------------------------------- #
# Atomicity — plan/stage + funnel land in ONE commit, or nothing (RVW-2, FR-4)
# --------------------------------------------------------------------------- #


def _raise(*_a, **_k):
    raise RuntimeError("funnel write failed")


def test_accept_path_commits_exactly_once():
    # plan + plan_limits + plan_accepted are ONE transaction, not two commits.
    ws = _FakeWorkspace(plan="basic", plan_limits={"max_documents": 100})
    db = _db_returning(ws)
    res = _run(update_onboarding(db, uuid4(), {"plan": "pro"}))
    assert res["success"] is True
    assert db.commit.call_count == 1


def test_accept_funnel_failure_commits_nothing_and_reports_failure(monkeypatch):
    # A funnel-write failure must NOT leave a durable plan change reported as a
    # failure: the single commit never runs and the tx is rolled back, so no
    # partial write reaches the DB while success is False (FR-4).
    import modules.tools.discovery.handlers_onboarding as h

    monkeypatch.setattr(h, "record_plan_event", _raise)
    ws = _FakeWorkspace(plan="basic", plan_limits={"max_documents": 100})
    db = _db_returning(ws)
    res = _run(update_onboarding(db, uuid4(), {"plan": "pro"}))
    assert res["success"] is False
    assert db.commit.call_count == 0  # nothing committed — no partial write
    assert db.rollback.call_count == 1


def test_proposal_advance_commits_exactly_once():
    # Stage advance + plan_recommended land in ONE commit.
    ws = _FakeWorkspace(onboarding={"stage": "teach"})
    db = _db_returning(ws)
    res = _run(
        update_onboarding(db, uuid4(), {"advance_to": "proposal", "segment": {"business": "an agency"}})
    )
    assert res["success"] is True
    assert db.commit.call_count == 1


def test_proposal_funnel_failure_leaves_stage_uncommitted(monkeypatch):
    # A funnel-write failure on the proposal path does not leave the stage
    # advanced without its recorded recommendation — nothing is committed.
    import modules.tools.discovery.handlers_onboarding as h

    monkeypatch.setattr(h, "record_plan_event", _raise)
    ws = _FakeWorkspace(onboarding={"stage": "teach"})
    db = _db_returning(ws)
    res = _run(
        update_onboarding(db, uuid4(), {"advance_to": "proposal", "segment": {"business": "an agency"}})
    )
    assert res["success"] is False
    assert db.commit.call_count == 0
    assert db.rollback.call_count == 1


# --------------------------------------------------------------------------- #
# @integration — real transaction: a funnel failure rolls back the plan write so
# workspace.plan is UNCHANGED in the DB (the literal RVW-2 AC). Skips w/o Postgres.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text

    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"plan-accept atomicity integration test needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.mark.integration
def test_accept_funnel_failure_does_not_change_plan_real_db(engine, new_session, monkeypatch):
    from sqlalchemy import text

    from core.models.workspaces import Workspace
    import modules.tools.discovery.handlers_onboarding as h

    s = new_session()
    wid = str(uuid4())
    try:
        s.execute(
            text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:i AS uuid), :n, 'basic')"),
            {"i": wid, "n": "w2s2-atomic"},
        )
        s.commit()

        # Inject a failure on the funnel write that follows assign_plan's flush.
        monkeypatch.setattr(h, "record_plan_event", _raise)
        res = _run(update_onboarding(s, wid, {"plan": "pro"}))
        assert res["success"] is False

        # The flushed-but-uncommitted plan write was rolled back — the row is
        # still 'basic' (no partial durable write; FR-4).
        row = s.execute(
            text("SELECT plan FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": wid}
        ).fetchone()
        assert row[0] == "basic"
    finally:
        s.rollback()
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": wid})
        s.commit()
