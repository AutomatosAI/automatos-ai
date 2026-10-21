"""PRD-265 US-006 (FR-4) — Auto records the plan a person chooses; it never assigns it.

Before billing, ``platform_update_onboarding`` with ``plan`` wrote the plan and its
limits onto the workspace, so Auto could put anyone on Business for free
(``handlers_onboarding.py:369-371``, leak 3). Every plan is paid now, and only a
verified payment or a platform admin moves a workspace's plan. Pinned here:

- a chosen plan is stamped as ``plan_accepted`` and the workspace's plan and limits
  stay exactly as they were;
- the answer says how the owner buys it: nothing while billing is closed (beta
  terms), the Billing tab once it opens, an invite for Business;
- "Own keys" names the ``basic`` tier; enterprise and non-plans are refused;
- a call that also advances the stage still advances it.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

from config import config
from modules.tools.discovery.handlers_onboarding import update_onboarding
from modules.tools.discovery.onboarding_plan_choice import how_to_buy, record_plan_choice
from services import plan_tiers as pt


class _Workspace:
    def __init__(self, plan="basic", stage="proposal"):
        self.id = "ws-plan"
        self.plan = plan
        self.plan_limits = {"max_members": 5, "max_agents": 20}
        self.onboarding = {"stage": stage}


def _db(workspace):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    return db


def _run(coro):
    return asyncio.run(coro)


def test_a_chosen_plan_is_recorded_and_never_assigned(monkeypatch):
    monkeypatch.setattr(config, "BILLING_OPEN", False)
    ws = _Workspace(plan="basic")
    result = _run(update_onboarding(_db(ws), uuid4(), {"plan": "business"}))
    assert result["success"] is True
    assert ws.plan == "basic"  # nothing assigned — Business stays an invite
    assert ws.plan_limits == {"max_members": 5, "max_agents": 20}
    assert ws.onboarding["funnel"]["plan_accepted"]["plan"] == "business"
    assert result["plan"]["buy"] is None and "invite" in result["plan"]["message"].lower()


def test_while_billing_is_closed_nothing_is_bought(monkeypatch):
    monkeypatch.setattr(config, "BILLING_OPEN", False)
    answer = how_to_buy("pro")
    assert answer["buy"] is None
    assert "nothing is charged" in answer["message"]


def test_once_billing_opens_a_self_serve_plan_is_bought_on_the_billing_tab(monkeypatch):
    monkeypatch.setattr(config, "BILLING_OPEN", True)
    assert how_to_buy("pro")["buy"] == "/settings?tab=billing&plan=pro"
    assert how_to_buy("basic")["name"] == "Own keys"
    assert how_to_buy("business")["buy"] is None  # by invite, even when open


def test_own_keys_by_its_display_name_is_the_basic_tier(monkeypatch):
    monkeypatch.setattr(config, "BILLING_OPEN", False)
    ws = _Workspace(plan="pro")
    result = record_plan_choice(_db(ws), uuid4(), " Own keys ")
    assert result["success"] is True and result["plan"]["plan"] == "basic"
    assert ws.plan == "pro"
    assert pt.tier_key("PRO") == "pro" and pt.tier_key("galaxy") == "galaxy"


def test_enterprise_and_non_plans_are_refused_without_a_write():
    ws = _Workspace()
    for plan in ("enterprise", "galaxy"):
        result = _run(update_onboarding(_db(ws), uuid4(), {"plan": plan}))
        assert result["success"] is False
        assert "funnel" not in ws.onboarding
    assert "coming soon" in record_plan_choice(_db(ws), uuid4(), "enterprise")["error"].lower()


def test_a_plan_beside_an_advance_still_advances(monkeypatch):
    monkeypatch.setattr(config, "BILLING_OPEN", False)
    seen = {}

    async def _handler(db, workspace_id, params):
        seen.update(params)
        return {"success": True, "data": {"stage": "building"}}

    from modules.tools.discovery.onboarding_plan_choice import plan_is_chosen_never_assigned

    wrapped = plan_is_chosen_never_assigned(_handler)
    ws = _Workspace(plan="basic", stage="building")
    result = _run(wrapped(_db(ws), uuid4(), {"advance_to": "building", "plan": "pro"}))
    assert seen == {"advance_to": "building"}  # the plan never reaches the stage writer
    assert result["success"] is True and result["plan"]["plan"] == "pro"
    assert ws.plan == "basic"
