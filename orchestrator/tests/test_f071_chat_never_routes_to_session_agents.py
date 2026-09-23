"""F071 — a chat turn is never routed to a session agent.

Night 3: 15 of 34 Auto replies (44 %) failed. Tier 3 offered the LLM every
active agent, session (runtime: cli) agents included; a session agent is never
activated in the LLM runtime by design (PRD-234 S1a — its work is a board
ticket its CLI host claims), so chat raised "Failed to activate agent N" and
told the owner to check a provider key that was fine.

The rule these tests pin: for a CHAT envelope, whatever tier decides — the
cache included — the result is an api agent or ``orchestrate`` (Auto), never a
session agent. Other sources share the router and are left alone.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

from core.cli_runtime import RuntimeMismatchError
from core.models.routing import ChannelSource, RequestEnvelope, RoutingDecision
from core.routing import engine
from core.routing.engine import UniversalRouter

API, CLI = 11, 22
ROSTER = {
    API: NS(id=API, name="WRITER-API", configuration={"runtime": "api"}, description="writes"),
    CLI: NS(id=CLI, name="NEWSROOM", configuration={"runtime": "cli"}, description="news"),
}


class _Query:
    def __init__(self, db, what):
        self.db, self.what, self.wanted = db, what, None

    def filter(self, *conds):
        for cond in conds:                       # Agent.id == N  ->  capture N
            right = getattr(cond, "right", None)
            value = getattr(right, "value", None)
            if isinstance(value, int):
                self.wanted = value
        return self

    def first(self):
        agent = ROSTER.get(self.wanted)
        if agent is None:
            return None
        return (agent.configuration, agent.name)

    def all(self):                                # Tier 3's roster read
        return list(ROSTER.values())


class _Db:
    def query(self, *what):
        return _Query(self, what)

    def add(self, *_a):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass


def _envelope(source=ChannelSource.CHATBOT):
    return RequestEnvelope(source=source, content="what's on the roast schedule?", workspace_id=uuid4())


def _to(agent_id):
    return RoutingDecision(route_type="agent", agent_id=agent_id, confidence=0.9, reasoning="picked")


def _router():
    return UniversalRouter(_Db(), cache=None)


# ── the choke point ──────────────────────────────────────────────────────────

def test_a_chat_turn_routed_to_a_session_agent_becomes_orchestrate():
    out = _router()._chat_never_routes_to_a_session(_envelope(), _to(CLI))
    assert out.route_type == "orchestrate" and out.agent_id is None
    assert "session agent" in out.reasoning and "NEWSROOM" in out.reasoning


def test_a_chat_turn_routed_to_an_api_agent_is_untouched():
    decision = _to(API)
    assert _router()._chat_never_routes_to_a_session(_envelope(), decision) is decision


def test_other_sources_keep_their_session_agent_routes():
    """Webhooks, triggers and channels share this router; their consumers may
    legitimately file a ticket for a session agent."""
    decision = _to(CLI)
    for source in (ChannelSource.WEBHOOK,) if hasattr(ChannelSource, "WEBHOOK") else ():
        assert _router()._chat_never_routes_to_a_session(_envelope(source), decision) is decision


def test_every_tier_including_the_cache_leaves_through_the_guard(monkeypatch):
    """A route cached before this fix — or to an agent switched to cli after it
    was cached — must not replay into the failure."""
    router = _router()

    async def cached_route_to_a_session(_env):
        return _to(CLI)                           # what a stale cache hit would return

    monkeypatch.setattr(router, "_route_through_tiers", cached_route_to_a_session)
    out = asyncio.run(router.route(_envelope()))
    assert out.route_type == "orchestrate" and out.agent_id is None


# ── Tier 3: a mixed roster only ever yields an api agent or orchestrate ──────

def _run_tier3(monkeypatch, llm_reply):
    class _Llm:
        async def generate_response(self, _messages):
            return NS(content=llm_reply)

    class _Ctx:
        def __init__(self, _db):
            pass

        async def build_context(self, **_kw):
            return NS(system_prompt="route this")

    monkeypatch.setattr(engine, "create_llm_manager", lambda **_kw: _Llm())
    monkeypatch.setattr(engine, "ContextService", _Ctx)
    router = _router()
    monkeypatch.setattr(router, "_build_agent_descriptions",
                        lambda agents: [{"agent_id": a.id, "name": a.name, "apps": [],
                                         "description": a.description} for a in agents])
    seen = {}

    def _prompt(content, descriptions, semantic_candidates=None):
        seen["offered"] = [d["agent_id"] for d in descriptions]
        return "classify"

    monkeypatch.setattr(router, "_build_classification_prompt", _prompt)
    decision = asyncio.run(router._classify_with_llm(_envelope(), None))
    return decision, seen.get("offered", [])


def test_tier3_never_offers_a_session_agent_to_the_llm_for_chat(monkeypatch):
    _, offered = _run_tier3(monkeypatch, f'{{"agent_id": {API}, "confidence": 0.95}}')
    assert offered == [API]


def test_tier3_rejects_a_session_agent_the_llm_names_anyway(monkeypatch):
    """Tier 2.5 hints can still put a session agent's name in front of the LLM;
    the pick is validated against the filtered roster and refused."""
    decision, _ = _run_tier3(monkeypatch, f'{{"agent_id": {CLI}, "confidence": 0.95}}')
    assert decision is None or decision.agent_id != CLI


def test_tier3_routes_to_the_api_agent_it_was_offered(monkeypatch):
    decision, _ = _run_tier3(monkeypatch, f'{{"agent_id": {API}, "confidence": 0.95}}')
    assert decision.route_type == "agent" and decision.agent_id == API


# ── the error the owner reads ────────────────────────────────────────────────

def test_a_session_agent_that_reaches_chat_gets_the_true_reason_not_the_key_advice():
    from consumers.chatbot.service import _session_agent_mismatch
    from consumers.chatbot.turn_errors import describe_turn_error

    exc = _session_agent_mismatch(_Db(), CLI)
    assert isinstance(exc, RuntimeMismatchError)
    message = describe_turn_error(exc, agent_name="NEWSROOM").message
    assert "session" in message.lower() and "ticket" in message.lower()
    assert "provider key" not in message.lower()


def test_an_api_agent_that_fails_to_activate_keeps_the_generic_path():
    from consumers.chatbot.service import _session_agent_mismatch

    assert _session_agent_mismatch(_Db(), API) is None
