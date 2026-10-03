"""F244 (night 7): the AI credit ran out with no warning, and what was said about it was wrong.

- The Analytics credit card said "No OpenRouter API key configured" beside $22.65 of
  OpenRouter spend: it looked for fewer keys than the calls use.
- Auto read "The AI provider's account ran out of credit" on a card as a café over
  its credit limit, and told the owner to call the café.
- Every failed run's report was a "published" deliverable: #0177 had nine, each only
  the credit message.
"""
from __future__ import annotations

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
