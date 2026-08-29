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
    # A tier-derived ceiling is tagged source="tier" so a later move to a
    # no-ceiling tier can clear it without wiping an admin custom budget (RVW-4).
    assert basic["budget"] == {"window": "month", "max_cost_usd": 25.0, "source": "tier"}
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
# RVW-4 — a tier-owned budget is re-derived on every assignment; upgrading to a
# no-ceiling tier (business) clears the prior tier's stale ceiling, but an admin
# custom budget survives untouched. Cross-assignment on the SAME workspace —
# the merge path the isolated plan_limits_for_tier tests never exercised.
# --------------------------------------------------------------------------- #


def test_assign_business_after_pro_clears_stale_tier_ceiling():
    # pro writes a tier-owned $100 ceiling; upgrading to business (custom / no
    # ceiling) must not leave that $100 lingering on plan_limits.
    ws = SimpleNamespace(plan="basic", plan_limits={})
    pt.assign_plan(None, ws, "pro")
    assert ws.plan_limits["budget"]["max_cost_usd"] == 100.0  # tier ceiling landed
    pt.assign_plan(None, ws, "business")
    assert ws.plan == "business"
    assert "budget" not in ws.plan_limits  # no stale ceiling survives the upgrade
    assert ws.plan_limits["max_members"] == 25  # business's other limits still land
    assert ws.plan_limits["max_agents"] == 0


def test_assign_business_after_basic_clears_stale_tier_ceiling():
    ws = SimpleNamespace(plan="basic", plan_limits={})
    pt.assign_plan(None, ws, "basic")
    assert ws.plan_limits["budget"]["max_cost_usd"] == 25.0
    pt.assign_plan(None, ws, "business")
    assert "budget" not in ws.plan_limits  # basic's $25 ceiling gone too


def test_assign_business_preserves_admin_custom_budget():
    # An admin-set custom budget (no tier provenance marker) is the customer's
    # own ceiling — valid on business (custom) and NEVER cleared by a tier change.
    ws = SimpleNamespace(
        plan="pro",
        plan_limits={"budget": {"window": "month", "max_cost_usd": 500.0}},
    )
    pt.assign_plan(None, ws, "business")
    assert ws.plan == "business"
    assert ws.plan_limits["budget"] == {"window": "month", "max_cost_usd": 500.0}


def test_assign_positive_tier_after_business_sets_fresh_tier_ceiling():
    # business (no ceiling) → pro re-derives the tier-owned $100 ceiling.
    ws = SimpleNamespace(plan="business", plan_limits={"max_members": 25})
    pt.assign_plan(None, ws, "pro")
    assert ws.plan_limits["budget"] == {
        "window": "month", "max_cost_usd": 100.0, "source": "tier",
    }


def test_load_budget_shows_no_ceiling_after_upgrade_to_business(monkeypatch):
    # RVW-4 AC3: the ceiling GET /budget surfaces (modules.policy.budget.load_budget,
    # governance.py:257) reads NO stale tier ceiling for a workspace upgraded
    # pro → business — proving the fix at the exact read the customer/admin sees.
    pytest.importorskip("core.models.workspaces")
    pytest.importorskip("core.services.auto_autonomy")
    from unittest.mock import MagicMock

    from modules.policy.budget import load_budget

    # A supervised (non-autonomy) workspace stays ceiling-less when no budget key.
    monkeypatch.setattr("core.services.auto_autonomy.is_full_autonomy", lambda db, ws: False)

    ws = SimpleNamespace(plan="basic", plan_limits={})
    pt.assign_plan(None, ws, "pro")       # tier-owned $100 ceiling written
    pt.assign_plan(None, ws, "business")  # business = no ceiling — clears the $100

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ws
    assert load_budget(db, "ws-rvw4") == {}  # no ceiling surfaced, not a stale $100


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
    # FIX (2026-08-29): pinning heads[0] to THIS revision broke the moment the
    # next migration (prd222_veteran_skip_backfill, PR #633) chained on top —
    # a guard that fails on every future migration guards nothing. The intent
    # ("our revision is properly chained into the single-headed graph") is the
    # ancestry property:
    sd = _script_dir()
    chain = set()
    cursor = [heads[0]]
    while cursor:
        rev_id = cursor.pop()
        if rev_id in chain:
            continue
        chain.add(rev_id)
        rev = sd.get_revision(rev_id)
        down = rev.down_revision
        if down is None:
            continue
        cursor.extend([down] if isinstance(down, str) else list(down))
    assert NEW_REVISION in chain, (
        f"{NEW_REVISION} is not an ancestor of the single head {heads[0]}"
    )


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
        assert row[1]["budget"] == {"window": "month", "max_cost_usd": 100.0, "source": "tier"}
    finally:
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": wid})
        s.commit()


@pytest.mark.integration
def test_load_budget_no_ceiling_after_upgrade_to_business_real_db(engine, new_session):
    # RVW-4 AC3 on real rows: pro (tier $100) → business (no ceiling) leaves GET
    # /budget's load_budget with NO max_cost_usd — the stale ceiling is gone at
    # the exact read governance.py:257 serves. Skips w/o Postgres; CI runs it.
    from sqlalchemy import text

    from core.models.workspaces import Workspace
    from modules.policy.budget import load_budget

    s = new_session()
    wid = str(uuid.uuid4())
    try:
        s.execute(
            text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:i AS uuid), :n, 'basic')"),
            {"i": wid, "n": "w2s1-rvw4-budget"},
        )
        s.commit()

        ws = s.query(Workspace).get(wid)
        pt.assign_plan(s, ws, "pro")
        assert load_budget(s, wid).get("max_cost_usd") == 100.0  # tier ceiling live

        pt.assign_plan(s, ws, "business")
        # business = custom / no ceiling — the $100 must not linger.
        assert "max_cost_usd" not in load_budget(s, wid)
    finally:
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:i AS uuid)"), {"i": wid})
        s.commit()
