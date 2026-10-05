"""#837 — mission dispatch never stands the event loop still.

A local install's loop watchdog caught the event loop standing still for 9-17.5 s
during mission dispatch: the coordinator's async tick called the synchronous
dispatcher on the loop, and matching a step's agent waited there, on ``.result()``,
for an embedding call run in a helper thread's own loop
(``match_signals.compute_semantic_signals_sync``). It paid for that call even for a
step a person had given to one agent ("override=True ... Explicitly assigned"). And
each dispatch left "Task exception was never retrieved ... AsyncClient.aclose()": the
embedding client, made on the helper loop, closed itself later on a loop it never
ran on.

These tests hold the three fixes: a pinned step costs no embedding call; the
dispatch runs on a worker thread, in the caller's context, while the loop keeps
serving; and the embedding clients a match makes are closed on its own loop before
it returns. No database: the coordinator runs over a mocked session.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys as _sys

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

# CI collection-order guard (as tests/test_prd164_agent_match.py): drop origin-less
# modules.* stubs an earlier test left behind, so the real packages import fresh.
for _name in [n for n, m in list(_sys.modules.items())
              if (n == "modules" or n.startswith("modules."))
              and getattr(m, "__spec__", None) is None]:
    _sys.modules.pop(_name, None)

import asyncio  # noqa: E402
import contextvars  # noqa: E402
import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402

import modules.coordination.agent_matcher as agent_matcher  # noqa: E402
import modules.coordination.match_signals as match_signals  # noqa: E402
from core.llm import embedding_manager  # noqa: E402
from core.llm.embedding_manager import EmbeddingManager  # noqa: E402
from modules.coordination.agent_matcher import AgentMatcher  # noqa: E402
from services import coordinator_service as csmod  # noqa: E402

# Only reached when the dispatch blocks the loop (the bug): then the loop cannot run
# the dispatch's callback, and the wait gives up after this long instead of hanging.
LOOP_WAIT_S = 10.0

_MARK: contextvars.ContextVar = contextvars.ContextVar("issue_837_mark", default=None)


def _agent(agent_id, *, status="active", card=(1.0, 0.0, 0.0)):
    return SimpleNamespace(id=agent_id, name=f"agent-{agent_id}", status=status,
                           semantic_embedding=list(card), workspace_id=None)


def _step(context=None):
    return SimpleNamespace(id=uuid4(), agent_role="writer", title="Write the launch post",
                           description="Two paragraphs for the shop.", input_context=context or {})


ROSTER = [_agent(1), _agent(2, card=(0.0, 1.0, 0.0)), _agent(3, status="paused")]


# ---------------------------------------------------------------------------
# 1. A step a person gave to one agent costs no embedding call
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_calls(monkeypatch):
    """Record each call that would make the embedding call (the sync bridge)."""
    calls = []

    def _bridge(**kwargs):
        calls.append(kwargs["task"].id)
        return None

    monkeypatch.setattr(agent_matcher, "compute_semantic_signals_sync", _bridge)
    return calls


def test_a_pinned_step_is_matched_without_an_embedding_call(bridge_calls):
    step = _step({"pinned_agent_id": 2, "required_tools": []})

    signals = AgentMatcher.compute_semantic_signals_sync(task=step, agents=ROSTER, workspace_id=uuid4())

    assert signals is None
    assert bridge_calls == [], "the pin picks the agent: no embedding call for an explicitly assigned step"


@pytest.mark.parametrize("context", [
    {},                               # routed by capability
    {"pinned_agent_id": 3},           # pinned to a paused agent: not an override, so it is scored
    {"pinned_agent_id": 999},         # pinned to no agent of the roster
    {"pinned_agent_id": "writer"},    # a role word never pins
])
def test_a_step_without_a_live_pin_is_still_scored(bridge_calls, context):
    step = _step(context)

    AgentMatcher.compute_semantic_signals_sync(task=step, agents=ROSTER, workspace_id=uuid4())

    assert bridge_calls == [step.id]


# ---------------------------------------------------------------------------
# 2. The dispatch runs off the event loop, in the caller's context
# ---------------------------------------------------------------------------


def _coordinator(monkeypatch):
    svc = csmod.CoordinatorService.__new__(csmod.CoordinatorService)
    svc._get_field = MagicMock(return_value=None)
    svc._create_mission_field = AsyncMock(return_value="field-1")
    svc._joiner_checkpoint = AsyncMock()
    monkeypatch.setattr(csmod.MissionReconciler, "reconcile", AsyncMock())
    return svc


def test_the_loop_keeps_running_while_a_mission_step_is_dispatched(monkeypatch):
    svc = _coordinator(monkeypatch)
    run = SimpleNamespace(id=uuid4(), workspace_id=uuid4(), config={"field_id": "field-1"},
                          state=csmod.RunState.RUNNING.value, stop_reason=None, max_concurrent=1)
    seen = {}

    async def _tick():
        loop = asyncio.get_running_loop()
        loop_ran = threading.Event()

        def _dispatch_ready(db, dispatched_run, agents):
            seen["thread"] = threading.get_ident()
            seen["mark"] = _MARK.get()
            loop.call_soon_threadsafe(loop_ran.set)       # runs only if the loop is free
            seen["loop_ran_meanwhile"] = loop_ran.wait(LOOP_WAIT_S)
            return []

        monkeypatch.setattr(csmod.MissionDispatcher, "dispatch_ready", staticmethod(_dispatch_ready))
        _MARK.set("the tick's context")
        await svc._process_run(MagicMock(), run)
        return threading.get_ident()

    loop_thread = asyncio.run(_tick())

    assert seen["loop_ran_meanwhile"] is True, "the dispatch held the event loop (#837)"
    assert seen["thread"] != loop_thread
    assert seen["mark"] == "the tick's context", "the dispatch thread runs in a copy of the tick's context (F196)"
    csmod.MissionReconciler.reconcile.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. The embedding client a match makes is closed on that match's own loop
# ---------------------------------------------------------------------------


class _Client:
    """The provider's async HTTP client: records the loops it is used and closed on."""

    def __init__(self, record):
        self._record = record

    async def close(self):
        self._record["closed_on"] = asyncio.get_running_loop()


class _Provider:
    def __init__(self, record):
        self._record = record
        self.client = _Client(record)

    async def generate_embedding(self, text):
        self._record["used_on"] = asyncio.get_running_loop()
        return [1.0, 0.0, 0.0]


def test_a_matchs_embedding_client_is_closed_on_its_own_loop_before_it_returns(monkeypatch):
    record = {}

    def _load_provider(manager):
        manager.provider = _Provider(record)

    monkeypatch.setattr(EmbeddingManager, "_load_provider", _load_provider)

    async def _dispatch_on_the_loop():
        # As before the fix: the bridge is called on a running loop and ships the
        # match to a helper thread's loop.
        signals = match_signals.compute_semantic_signals_sync(task=_step(), agents=ROSTER, workspace_id=None)
        record["closed_when_it_returned"] = "closed_on" in record
        return signals, asyncio.get_running_loop()

    signals, outer_loop = asyncio.run(_dispatch_on_the_loop())

    assert record["closed_when_it_returned"], "the embedding client outlived its call (#837)"
    assert record.get("closed_on") is record["used_on"], "closed on the loop it ran on, while that loop was alive"
    assert record["used_on"] is not outer_loop
    assert signals is not None and signals.similarity_by_agent[1] > signals.similarity_by_agent[2]


def test_a_timed_out_match_still_closes_its_client(monkeypatch):
    record = {}

    class _SlowProvider(_Provider):
        async def generate_embedding(self, text):
            self._record["used_on"] = asyncio.get_running_loop()
            await asyncio.sleep(LOOP_WAIT_S)

    monkeypatch.setattr(EmbeddingManager, "_load_provider",
                        lambda manager: setattr(manager, "provider", _SlowProvider(record)))
    monkeypatch.setattr(match_signals.Config, "AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS", 0.05, raising=False)

    signals = match_signals.compute_semantic_signals_sync(task=_step(), agents=ROSTER, workspace_id=None)

    assert signals is None                                    # fail-open: lexical-only dispatch
    assert record.get("closed_on") is record["used_on"], "a cancelled match closes its client too"


def test_aclose_closes_a_real_openai_client():
    openai = pytest.importorskip("openai")
    manager = EmbeddingManager()
    manager.provider = SimpleNamespace(client=openai.AsyncOpenAI(api_key="test"))

    asyncio.run(manager.aclose())

    assert manager.provider.client.is_closed()


def test_the_shared_manager_is_never_closed(monkeypatch):
    shared = EmbeddingManager()
    client = SimpleNamespace(close=AsyncMock())
    shared.provider = SimpleNamespace(client=client)
    monkeypatch.setattr(embedding_manager, "_embedding_manager", shared)

    asyncio.run(shared.aclose())

    client.close.assert_not_awaited()
