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
        # PRD-265 US-001: Own keys (basic) is repriced to $9 and carries EXACTLY
        # Pro's limits — only who pays for the AI differs (facts sheet §4).
        ("basic", 9, 5, 20, 3, 5, 2),
        ("pro", 29, 5, 20, 3, 5, 2),
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
        # Own keys (basic) now matches Pro's families exactly (PRD-265 US-001).
        ("basic", True, True, True, False),
        ("pro", True, True, True, False),
        ("business", True, True, True, True),
    ],
)
def test_capability_families_match_strawman(name, codegraph, nl2sql, team, voice):
    fam = PLAN_TIERS[name]["families"]
    assert fam == {"codegraph": codegraph, "nl2sql": nl2sql, "team": team, "voice": voice}


def test_plan_tiers_carry_no_price_label():
    # PRD-265: price_label is no longer a static tier field — it's derived at
    # read time by services.plan_tiers.price_label() from config.BILLING_OPEN,
    # so one flag flips every tier's label with no redeploy or data migration.
    for name in ("basic", "pro", "business", "enterprise"):
        assert "price_label" not in PLAN_TIERS[name]


def test_price_label_is_beta_while_billing_closed_then_none_when_open(monkeypatch):
    # PRD-265 US-001: beta pricing until config.BILLING_OPEN flips (facts sheet §4).
    from config import config as app_config

    monkeypatch.setattr(app_config, "BILLING_OPEN", False)
    assert pt.price_label() == pt.BETA_PRICE_LABEL == "beta"
    monkeypatch.setattr(app_config, "BILLING_OPEN", True)
    assert pt.price_label() is None


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
    # basic now carries Pro's families (all True except voice), so flip a
    # family the override can actually move: turn codegraph OFF and prove the
    # deep-merge still leaves nl2sql (untouched by the override) at its
    # default True — a field flip, not a tier-wide replace.
    override = '{"basic": {"display_price_usd": 15, "families": {"codegraph": false}}}'
    tiers = load_plan_tiers(env_override=override)
    assert tiers["basic"]["display_price_usd"] == 15
    assert tiers["basic"]["families"]["codegraph"] is False
    # Untouched fields keep their defaults (deep-merge, not replace).
    assert tiers["basic"]["seats"] == 5
    assert tiers["basic"]["families"]["nl2sql"] is True
    # The module constant is not mutated by an override load.
    assert PLAN_TIERS["basic"]["display_price_usd"] == 9


def test_malformed_env_override_falls_back_to_defaults():
    tiers = load_plan_tiers(env_override="{not valid json")
    assert tiers["basic"]["display_price_usd"] == 9
    assert tiers["pro"]["seats"] == 5


# --------------------------------------------------------------------------- #
# Assignment helper — plan_limits mapping keyed for the LIVE consumers
# --------------------------------------------------------------------------- #


def test_assignable_tiers_excludes_enterprise():
    names = set(pt.assignable_tiers().keys())
    assert names == {"basic", "pro", "business"}


def test_plan_limits_for_tier_maps_seats_to_max_members_and_budget():
    # basic (Own keys): seats→max_members, a positive budget lands under
    # plan_limits.budget. PRD-265: basic now carries EXACTLY Pro's limits.
    basic = pt.plan_limits_for_tier("basic")
    assert basic["max_members"] == 5  # the LIVE seat key (invitations + checklist)
    assert basic["max_agents"] == 20
    assert basic["mission_concurrency"] == 3
    assert basic["watcher_limit"] == 5
    assert basic["marketplace_depth"] == 2
    # A tier-derived ceiling is tagged source="tier" so a later move to a
    # no-ceiling tier can clear it without wiping an admin custom budget (RVW-4).
    assert basic["budget"] == {"window": "month", "max_cost_usd": 100.0, "source": "tier"}
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
    assert ws.plan_limits["budget"]["max_cost_usd"] == 100.0  # PRD-265: basic = Pro's $100
    pt.assign_plan(None, ws, "business")
    assert "budget" not in ws.plan_limits  # basic's $100 ceiling gone too


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
# PRD-265 US-001 — new commerce helpers: platform_ai_allowed, is_metered,
# monthly_credits, is_self_serve, purchase_path. Pure reads off PLAN_TIERS.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "plan,allowed",
    [("basic", False), ("pro", True), ("business", True), ("enterprise", True)],
)
def test_platform_ai_allowed_by_tier(plan, allowed):
    # Own keys runs on the customer's own AI keys only (PRD-265 FR-3); every
    # other tier may spend platform credits.
    assert pt.platform_ai_allowed(plan) is allowed


def test_platform_ai_allowed_unknown_plan_resolves_to_basic_false():
    # An unknown/stale plan string resolves to the entry tier (basic), which
    # is platform_ai=False — so an unresolvable plan never gets platform AI.
    assert pt.platform_ai_allowed("nope") is False


@pytest.mark.parametrize(
    "plan,metered",
    [("basic", True), ("pro", True), ("business", True), ("enterprise", False)],
)
def test_is_metered_only_enterprise_false(plan, metered):
    # enterprise runs on an explicit platform allowance: calls are recorded but
    # never gated by credits. Every other tier is metered.
    assert pt.is_metered(plan) is metered


@pytest.mark.parametrize(
    "plan,credits",
    [("basic", 0), ("pro", 2000), ("business", 8000), ("enterprise", 0)],
)
def test_monthly_credits_by_tier(plan, credits):
    assert pt.monthly_credits(plan) == credits


@pytest.mark.parametrize(
    "plan,self_serve",
    [("basic", True), ("pro", True), ("business", False), ("enterprise", False)],
)
def test_is_self_serve_by_tier(plan, self_serve):
    # Business is by-invite only; Enterprise is coming-soon (not assignable at all).
    assert pt.is_self_serve(plan) is self_serve


def test_is_self_serve_unknown_plan_is_false():
    # Unlike platform_ai_allowed/monthly_credits, is_self_serve does NOT fall
    # back to the entry tier — an unassignable plan is never self-serve.
    assert pt.is_self_serve("nope") is False


def test_purchase_path_self_serve_plans_only():
    assert pt.purchase_path("basic") == "/settings?tab=billing&plan=basic"
    assert pt.purchase_path("pro") == "/settings?tab=billing&plan=pro"
    assert pt.purchase_path("business") is None  # by invite — no self-serve checkout
    assert pt.purchase_path("enterprise") is None
    assert pt.purchase_path("nope") is None


def test_own_keys_matches_pro_limits_and_families():
    """PRD-265 facts sheet §4: Own keys (basic) = Pro, only credits differ —
    basic runs on the customer's own AI keys (no platform credits granted),
    Pro's are included. Every seat/agent/concurrency/watcher/depth/budget/
    render limit and every capability family must be identical."""
    basic = PLAN_TIERS["basic"]
    pro = PLAN_TIERS["pro"]
    shared_limit_keys = (
        "seats", "max_agents", "mission_concurrency", "watcher_limit",
        "marketplace_depth", "budget_usd", "render_minutes_month", "families",
    )
    for key in shared_limit_keys:
        assert basic[key] == pro[key], f"Own keys and Pro must match on {key!r}"
    # ...and only the commerce facts the facts sheet calls out actually differ.
    assert basic["platform_ai"] is False and pro["platform_ai"] is True
    assert basic["monthly_credits"] == 0 and pro["monthly_credits"] == 2000


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
