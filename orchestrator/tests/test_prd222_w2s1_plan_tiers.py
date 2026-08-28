"""PRD-222 W2·S1 (US-023) — tier config v1 + plan default rename & backfill.

Pure tests pin the approved strawman numbers (config.PLAN_TIERS), the
config-driven env override (repricing/re-gating with no redeploy), the
assignment helper's plan_limits mapping (seats→max_members, budget, enterprise
rejected), and the migration's chain position + single head + idempotent/scoped
backfill SQL. An @integration test proves the backfill on real rows (starter →
basic, other tiers spared, second run a no-op) and that assign_plan persists
plan + plan_limits — skips cleanly without Postgres; CI runs it.
"""
from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import PLAN_TIERS, load_plan_tiers
from services import plan_tiers as pt

_MIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic" / "versions" / "prd222_w2s1_plan_default_basic.py"
)
NEW_REVISION = "prd222_w2s1_plan_default_basic"
PRIOR_HEAD = "prd222_w2s5_drop_onboarding_agents"


def _load_migration():
    spec = importlib.util.spec_from_file_location("_prd222_w2s1_mig", _MIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _script_dir():
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(Path(__file__).resolve().parents[1] / "alembic"))


# --------------------------------------------------------------------------- #
# PLAN_TIERS matches the approved v1 strawman exactly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,price,seats,agents,concurrency,watchers,depth",
    [
        ("basic", 19, 1, 5, 1, 1, 1),
        ("pro", 49, 5, 20, 3, 5, 2),
        ("business", 99, 25, 0, 10, 0, 3),
    ],
)
def test_plan_tiers_numbers_match_strawman(name, price, seats, agents, concurrency, watchers, depth):
    t = PLAN_TIERS[name]
    assert t["display_price_usd"] == price
    assert t["seats"] == seats
    assert t["max_agents"] == agents  # 0 == unlimited
    assert t["mission_concurrency"] == concurrency
    assert t["watcher_limit"] == watchers  # 0 == unlimited
    assert t["marketplace_depth"] == depth
    assert t["assignable"] is True


@pytest.mark.parametrize(
    "name,codegraph,nl2sql,team,voice",
    [
        ("basic", False, False, False, False),
        ("pro", True, True, True, False),
        ("business", True, True, True, True),
    ],
)
def test_capability_families_match_strawman(name, codegraph, nl2sql, team, voice):
    fam = PLAN_TIERS[name]["families"]
    assert fam == {"codegraph": codegraph, "nl2sql": nl2sql, "team": team, "voice": voice}


def test_display_prices_are_early_access_labels_only():
    # Q5: display pricing only — every assignable tier carries an early-access
    # label so repricing stays free; there is no charge/amount field.
    for name in ("basic", "pro", "business"):
        assert PLAN_TIERS[name]["price_label"] == "early access"


def test_enterprise_is_coming_soon_and_not_assignable():
    ent = PLAN_TIERS["enterprise"]
    assert ent.get("coming_soon") is True
    assert ent.get("assignable") is not True
    assert pt.is_assignable("enterprise") is False
    # A coming-soon label carries NO limits (never sized).
    assert "seats" not in ent and "max_agents" not in ent


# --------------------------------------------------------------------------- #
# Config-driven: an env override reprices / re-gates without a code change
# --------------------------------------------------------------------------- #


def test_env_override_flips_price_and_family_without_redeploy():
    override = '{"basic": {"display_price_usd": 29, "families": {"codegraph": true}}}'
    tiers = load_plan_tiers(env_override=override)
    assert tiers["basic"]["display_price_usd"] == 29
    assert tiers["basic"]["families"]["codegraph"] is True
    # Untouched fields keep their defaults (deep-merge, not replace).
    assert tiers["basic"]["seats"] == 1
    assert tiers["basic"]["families"]["nl2sql"] is False
    # The module constant is not mutated by an override load.
    assert PLAN_TIERS["basic"]["display_price_usd"] == 19


def test_malformed_env_override_falls_back_to_defaults():
    tiers = load_plan_tiers(env_override="{not valid json")
    assert tiers["basic"]["display_price_usd"] == 19
    assert tiers["pro"]["seats"] == 5


# --------------------------------------------------------------------------- #
# Assignment helper — plan_limits mapping keyed for the LIVE consumers
# --------------------------------------------------------------------------- #


def test_assignable_tiers_excludes_enterprise():
    names = set(pt.assignable_tiers().keys())
    assert names == {"basic", "pro", "business"}


def test_plan_limits_for_tier_maps_seats_to_max_members_and_budget():
    # basic: seats→max_members, a positive budget lands under plan_limits.budget.
    basic = pt.plan_limits_for_tier("basic")
    assert basic["max_members"] == 1  # the LIVE seat key (invitations + checklist)
    assert basic["max_agents"] == 5
    assert basic["mission_concurrency"] == 1
    assert basic["watcher_limit"] == 1
    assert basic["marketplace_depth"] == 1
    assert basic["budget"] == {"window": "month", "max_cost_usd": 25.0}
    # business: seats 25, unlimited agents (0), custom budget (0) ⇒ no budget key
    biz = pt.plan_limits_for_tier("business")
    assert biz["max_members"] == 25
    assert biz["max_agents"] == 0
    assert "budget" not in biz


def test_plan_limits_for_tier_rejects_enterprise_and_unknown():
    with pytest.raises(ValueError):
        pt.plan_limits_for_tier("enterprise")
    with pytest.raises(ValueError):
        pt.plan_limits_for_tier("nope")


def test_assign_plan_pure_rebuilds_plan_limits_preserving_unmanaged_keys():
    # db=None escape hatch: in-memory reassignment only. An unmanaged key
    # (max_documents) survives; a NEW dict object is assigned (rebuild-not-mutate).
    ws = SimpleNamespace(plan="starter", plan_limits={"max_documents": 100, "max_members": 999})
    before = ws.plan_limits
    result = pt.assign_plan(None, ws, "pro")
    assert ws.plan == "pro"
    assert ws.plan_limits is not before  # rebuilt, not mutated in place
    assert ws.plan_limits["max_documents"] == 100  # unmanaged key preserved
    assert ws.plan_limits["max_members"] == 5  # tier seats overwrote the old value
    assert result is ws.plan_limits


def test_assign_plan_rejects_enterprise():
    ws = SimpleNamespace(plan="basic", plan_limits={})
    with pytest.raises(ValueError):
        pt.assign_plan(None, ws, "enterprise")
    assert ws.plan == "basic"  # unchanged on rejection


# --------------------------------------------------------------------------- #
# Migration — chain position, single head, idempotent + scoped backfill
# --------------------------------------------------------------------------- #


def test_migration_chains_onto_current_head():
    mod = _load_migration()
    assert mod.revision == NEW_REVISION
    assert mod.down_revision == PRIOR_HEAD


def test_exactly_one_head_after_this_migration():
    heads = _script_dir().get_heads()
    assert len(heads) == 1, f"expected exactly one alembic head, got {heads}"
    assert heads[0] == NEW_REVISION


def test_backfill_sql_is_scoped_to_starter():
    mod = _load_migration()
    sql = mod._BACKFILL_SQL.lower()
    assert "update workspaces set plan = 'basic'" in sql
    assert "where plan = 'starter'" in sql  # scoped — never a blanket rewrite


# --------------------------------------------------------------------------- #
# @integration — real rows: backfill migrates starter, spares other tiers,
# second run is a no-op; assign_plan persists plan + limits. Skips w/o Postgres.
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
        pytest.skip(f"plan-tiers integration test needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.mark.integration
def test_backfill_migrates_starter_spares_other_tiers(engine, new_session):
    from sqlalchemy import text

    mod = _load_migration()
    s = new_session()
    starter_id = str(uuid.uuid4())
    pro_id = str(uuid.uuid4())
    try:
        s.execute(
            text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:i AS uuid), :n, 'starter')"),
            {"i": starter_id, "n": "w2s1-starter"},
        )
        s.execute(
            text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:i AS uuid), :n, 'pro')"),
            {"i": pro_id, "n": "w2s1-pro"},
        )
        s.commit()

        def plan_of(wid):
            return s.execute(
                text("SELECT plan FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": wid}
            ).fetchone()[0]

        # First run: starter → basic, pro untouched.
        s.execute(text(mod._BACKFILL_SQL))
        s.commit()
        assert plan_of(starter_id) == "basic"
        assert plan_of(pro_id) == "pro"

        # Second run is idempotent (nothing left matching 'starter').
        s.execute(text(mod._BACKFILL_SQL))
        s.commit()
        assert plan_of(starter_id) == "basic"
        assert plan_of(pro_id) == "pro"
    finally:
        s.execute(
            text("DELETE FROM workspaces WHERE id IN (CAST(:a AS uuid), CAST(:b AS uuid))"),
            {"a": starter_id, "b": pro_id},
        )
        s.commit()


@pytest.mark.integration
def test_assign_plan_writes_plan_and_limits_real_db(engine, new_session):
    from sqlalchemy import text

    from core.models.workspaces import Workspace

    s = new_session()
    wid = str(uuid.uuid4())
    try:
        s.execute(
            text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:i AS uuid), :n, 'basic')"),
            {"i": wid, "n": "w2s1-assign"},
        )
        s.commit()

        ws = s.query(Workspace).get(wid)
        pt.assign_plan(s, ws, "pro")

        row = s.execute(
            text("SELECT plan, plan_limits FROM workspaces WHERE id = CAST(:i AS uuid)"),
            {"i": wid},
        ).fetchone()
        assert row[0] == "pro"
        assert row[1]["max_members"] == 5  # the seat key the checklist reads
        assert row[1]["max_agents"] == 20
        assert row[1]["budget"] == {"window": "month", "max_cost_usd": 100.0}
    finally:
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": wid})
        s.commit()
