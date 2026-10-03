"""F244 (night 7): the AI credit ran out with no warning, and what was said about it was wrong.

- The Analytics credit card said "No OpenRouter API key configured" beside $22.65 of
  OpenRouter spend: it looked for fewer keys than the calls use.
- Auto read "The AI provider's account ran out of credit" on a card as a café over
  its credit limit, and told the owner to call the café.
- Every failed run's report was a "published" deliverable: #0177 had nine, each only
  the credit message.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace as NS

import httpx
import openai
import pytest


# ── the credit card reads the key the calls use ─────────────────────────────

def test_the_local_credit_card_reads_the_key_the_calls_use(monkeypatch):
    from api import llm_analytics
    from config import config
    from core.llm import key_resolver

    monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", True)
    monkeypatch.setattr(key_resolver, "resolve_provider_key", lambda db, provider, workspace_id=None, agent_name="":
                        key_resolver.ResolvedKey("sk-or-operator", "platform_workspace", False, provider))
    assert llm_analytics._resolve_openrouter_key("ws-1", db=None) == "sk-or-operator"


class _NoKeyRows:
    def query(self, *a):
        return self

    def filter(self, *a):
        return self

    def order_by(self, *a):
        return self

    def first(self):
        return None


def test_the_hosted_card_never_reads_the_operators_key(monkeypatch):
    from fastapi import HTTPException

    from api import llm_analytics
    from config import config
    from core.llm import key_resolver

    monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", False)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(key_resolver, "resolve_provider_key", lambda *a, **k: pytest.fail("the operator's key"))
    with pytest.raises(HTTPException) as refused:
        llm_analytics._resolve_openrouter_key("ws-1", db=_NoKeyRows())
    assert refused.value.status_code == 404


@pytest.fixture
def tiers(monkeypatch):
    """The platform's three key tiers, each settable."""
    found = {"workspace": None, "credential": None}
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: found["workspace"]))
    monkeypatch.setitem(sys.modules, "core.credentials.resolver", types.SimpleNamespace(
        get_credential_resolver=lambda: NS(get_credential_field=lambda name, field:
                                           found["credential"] if (name, field) == ("openrouter", "api_key") else None)))
    return found


def test_the_resolver_follows_the_calls_order(monkeypatch, tiers):
    from config import config
    from core.llm.key_resolver import resolve_provider_key

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-env")
    tiers.update(workspace="sk-operator", credential="sk-cred")
    assert resolve_provider_key(None, "openrouter").source == "platform_workspace"     # night 7's key
    tiers.update(workspace=None)
    assert (resolve_provider_key(None, "openrouter").api_key, resolve_provider_key(None, "openrouter").source) == (
        "sk-cred", "platform")
    tiers.update(credential=None)
    assert resolve_provider_key(None, "openrouter").source == "env"


# ── the words say whose credit it is ────────────────────────────────────────

def test_the_sentence_says_it_is_the_ai_credit_and_not_a_customers():
    from core.llm.credit import OUT_OF_CREDIT_TEXT, OUTAGE_NOTICE, is_out_of_credit

    assert OUT_OF_CREDIT_TEXT.startswith("Your AI credit ran out: the AI provider account that pays for the agents'")
    assert "It is not a customer's or a supplier's credit." in OUT_OF_CREDIT_TEXT
    assert OUTAGE_NOTICE.startswith("Your AI credit ran out")
    assert is_out_of_credit(OUT_OF_CREDIT_TEXT)
    assert is_out_of_credit("The AI provider's account ran out of credit, so this stopped")   # rows from before


def test_autos_own_reply_says_your_ai_credit_ran_out():
    from consumers.chatbot.turn_errors import describe_turn_error

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    refused = openai.APIStatusError("Error code: 402 - {'error': {'message': 'requires more credits'}}",
                                    response=httpx.Response(402, request=request), body=None)
    message = describe_turn_error(refused, agent_name="Auto").message
    assert message.startswith("Auto could not finish this reply: your AI credit ran out")
    assert "out of credits" in message


# ── a failed run is not a published deliverable ────────────────────────────

def test_a_failed_runs_report_is_failed_not_published():
    from services.deliverable_service import shown_status

    failed = NS(artifact_type="report", extra={"task_status": "failed", "trigger": "task"}, status="published")
    done = NS(artifact_type="report", extra={"task_status": "done", "trigger": "task"}, status="published")
    post = NS(artifact_type="blog_post", extra={}, status="draft")
    assert (shown_status(failed), shown_status(done), shown_status(post)) == ("failed", "published", "draft")


# ── an agent that can't run says so ─────────────────────────────────────────

@pytest.fixture
def shop(db_session, seed_workspace):
    """Night 7 at 07:30: an API agent, a CLI session agent with no host, and the credit out."""
    from datetime import datetime, timedelta, timezone
    from uuid import UUID

    from sqlalchemy import text

    ws = UUID(seed_workspace())

    def agent(name, configuration):
        return db_session.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
            "VALUES (:n, 'custom', :w, 'active', CAST(:c AS json), 'workspace') RETURNING id"),
            {"n": name, "w": str(ws), "c": json.dumps(configuration)}).scalar()

    now = datetime.now(timezone.utc)
    analyst = agent("Shopify Business Analyst", {"model": "anthropic/claude-sonnet-4"})
    mac = agent("Numbers (on my Mac)", {"runtime": "cli", "provider": "claude"})
    db_session.execute(text(
        "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, request_type, input_tokens, output_tokens, "
        "total_tokens, input_cost, output_cost, total_cost, created_at) VALUES (CAST(:w AS uuid), 'm', 'openrouter', "
        "'direct', 'board_task', 10, 5, 15, 0, 0, 0, :at)"), {"w": str(ws), "at": now - timedelta(minutes=40)})

    def failed_for_credit(minutes_ago, words):
        db_session.execute(text(
            "INSERT INTO board_tasks (workspace_id, title, status, error_message, completed_at) "
            "VALUES (CAST(:w AS uuid), 'Break-even on the gift box', 'failed', :e, :at)"),
            {"w": str(ws), "e": words, "at": now - timedelta(minutes=minutes_ago)})
    return NS(db=db_session, ws=ws, analyst=analyst, mac=mac, now=now, failed_for_credit=failed_for_credit)


def test_the_credit_is_out_after_a_credit_failure_newer_than_the_last_call_that_went_through(shop):
    from core.llm.credit import OUT_OF_CREDIT_TEXT
    from services.agent_availability import ai_credit_out

    assert ai_credit_out(shop.db, shop.ws) is False
    shop.failed_for_credit(50, OUT_OF_CREDIT_TEXT)               # before the last call that went through
    assert ai_credit_out(shop.db, shop.ws) is False
    shop.failed_for_credit(10, "The AI provider's account ran out of credit, so this stopped before it finished.")
    assert ai_credit_out(shop.db, shop.ws) is True


def test_each_agent_says_why_it_cannot_run(shop):
    from core.llm.credit import OUT_OF_CREDIT_TEXT
    from services.agent_availability import CREDIT_OUT, NO_HOST, why_unavailable
    from core.models import Agent

    shop.failed_for_credit(5, OUT_OF_CREDIT_TEXT)
    agents = shop.db.query(Agent).filter(Agent.id.in_([shop.analyst, shop.mac])).all()
    assert why_unavailable(shop.db, shop.ws, agents) == {shop.analyst: CREDIT_OUT, shop.mac: NO_HOST}


def test_autos_agent_list_says_which_can_run(shop):
    from core.llm.credit import OUT_OF_CREDIT_TEXT
    from modules.tools.discovery.handlers_agents import list_agents
    from services.agent_availability import CREDIT_OUT

    shop.failed_for_credit(5, OUT_OF_CREDIT_TEXT)
    listed = {a["name"]: a for a in asyncio.run(list_agents(shop.db, shop.ws, {}))["agents"]}
    assert (listed["Shopify Business Analyst"]["can_run"], listed["Shopify Business Analyst"]["why"]) == (False,
                                                                                                         CREDIT_OUT)
    assert listed["Numbers (on my Mac)"]["can_run"] is False


def test_the_agents_page_list_carries_why_each_agent_cannot_run(shop):
    from pydantic import BaseModel

    from core.llm.credit import OUT_OF_CREDIT_TEXT
    from services.agent_availability import CREDIT_OUT, with_unavailable

    class Row(BaseModel):
        id: int
        configuration: dict
        unavailable: str | None = None

    @with_unavailable
    async def endpoint(ctx=None, db=None):
        return [Row(id=shop.analyst, configuration={}), Row(id=999999, configuration={"runtime": "cli"})]

    shop.failed_for_credit(5, OUT_OF_CREDIT_TEXT)
    rows = asyncio.run(endpoint(ctx=NS(workspace_id=shop.ws), db=shop.db))
    assert rows[0].unavailable == CREDIT_OUT and rows[1].unavailable is not None


# ── the credit warns before a night's work would run past it ────────────────

@pytest.fixture
def watch(monkeypatch):
    """The credit watch with the provider's balance and the bell stubbed."""
    from services import credit_watch

    rang = []
    seen = {"credits": {"total_credits": 18.69, "total_usage": 17.49}}

    class _Analytics:
        async def get_credits(self, api_key):
            return seen["credits"]

    async def ring(workspace_id, left, day):
        rang.append((workspace_id, round(left, 2), day))

    monkeypatch.setattr(credit_watch, "_key_and_last_day", lambda ws: ("sk-or-operator", 9.10))
    monkeypatch.setattr("core.llm.openrouter_analytics.OpenRouterAnalyticsService", _Analytics)
    monkeypatch.setattr(credit_watch, "_ring", ring)
    monkeypatch.setattr(credit_watch, "_noticed_on", {})
    monkeypatch.setattr(credit_watch, "_last_check", {})
    return NS(module=credit_watch, rang=rang, seen=seen)


def test_the_bell_says_the_credit_is_low_once_a_day(watch):
    """Night 7: $18.69 at the start, $18.24 spent, and no warning before 06:53."""
    assert round(asyncio.run(watch.module.check_credit("ws-7")), 2) == 1.20
    asyncio.run(watch.module.check_credit("ws-7"))
    assert watch.rang == [("ws-7", 1.20, 9.10)]
    assert watch.module.LOW_NOTICE.format(left=1.20, day=9.10).startswith("Your AI credit has $1.20 left")


def test_enough_credit_for_a_day_of_work_rings_nothing(watch):
    watch.seen["credits"] = {"total_credits": 60.0, "total_usage": 22.65}
    assert asyncio.run(watch.module.check_credit("ws-7")) == pytest.approx(37.35)
    assert watch.rang == []


def test_new_work_has_the_credit_looked_at_at_most_every_quarter_hour(watch, monkeypatch):
    from config import config

    checked = []

    async def check(workspace_id):
        checked.append(workspace_id)

    monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", True)
    monkeypatch.setattr(watch.module, "check_credit", check)
    guard = watch.module.watches_the_credit(lambda db, ws, what: "over the day's ceiling")

    async def starts():
        answers = [guard(None, "ws-7", "board task 1"), guard(None, "ws-7", "board task 2")]
        await asyncio.sleep(0)
        return answers

    assert asyncio.run(starts()) == ["over the day's ceiling"] * 2       # the guard's answer is unchanged
    assert checked == ["ws-7"]
    monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", False)
    monkeypatch.setattr(watch.module, "_last_check", {})
    asyncio.run(starts())
    assert checked == ["ws-7"]                                            # the hosted edition never looks
