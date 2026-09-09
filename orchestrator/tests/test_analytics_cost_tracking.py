"""Analytics cost tracking (2026-09-09) — every provider tagged, every lane
attributed, nothing silently booked at $0.

What the Analytics page found wrong on the local edition:

- Claude Code sessions (the user's own subscription) never reached
  ``llm_usage`` — their tokens lived only in ``board_tasks.runtime_ref``;
- ``request_type`` said ``orchestrator`` for chat, tickets, missions and
  heartbeats alike; ``execution_id`` was NULL on 99% of rows;
- a metered route with no catalogue row booked $0 (285 gemini calls);
- ``tier`` mixed three vocabularies; cache tokens were not recorded at all;
- the OpenRouter activity sync double-counted every dollar it copied.

Pure unit tests (fake sessions, no DB) — the endpoint queries run in
``test_analytics_cost_tracking_realdb.py``.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from core.llm import usage_context as uc  # noqa: E402
from core.llm import usage_tracker as ut  # noqa: E402
from core.llm.providers import describe_usage_provider  # noqa: E402

WS = UUID("00000000-0000-0000-0000-0000000000c1")


# ── the scope ────────────────────────────────────────────────────────────────

def test_scope_is_task_local_and_nests_by_inheritance():
    assert uc.current_usage_scope() == {}
    with uc.usage_scope(request_type="mission", execution_id="mission:7", agent_id=3):
        assert uc.current_usage_scope()["execution_id"] == "mission:7"
        with uc.usage_scope(request_type="verifier"):
            inner = uc.current_usage_scope()
            assert inner["request_type"] == "verifier" and inner["execution_id"] == "mission:7" and inner["agent_id"] == 3
        assert uc.current_usage_scope()["request_type"] == "mission"
    assert uc.current_usage_scope() == {}


def test_two_concurrent_tasks_never_see_each_others_scope():
    seen = {}

    async def run(name, delay):
        with uc.usage_scope(request_type=name, execution_id=f"board_task:{name}"):
            await asyncio.sleep(delay)
            seen[name] = dict(uc.current_usage_scope())

    async def main():
        await asyncio.gather(run("a", 0.02), run("b", 0.01))

    asyncio.run(main())
    assert seen["a"]["execution_id"] == "board_task:a" and seen["b"]["execution_id"] == "board_task:b"


@pytest.mark.parametrize(
    "context, lane, ref",
    [
        ({"source": "board_task", "task_id": 97}, "board_task", "board_task:97"),
        ({"run_id": "r1", "task_id": "t1", "mission_id": "r1", "agent_id": 5}, "mission", "mission:r1"),
        ({"source": "heartbeat", "heartbeat_id": 12}, "heartbeat", "heartbeat:12"),
        ({"source": "manual"}, "manual_run", None),
        ({"source": "cli", "task_id": 3}, "session", "task:3"),
        (None, None, None),
    ],
)
def test_lane_and_execution_ref_follow_the_context(context, lane, ref):
    assert uc.lane_for_context(context) == lane
    assert uc.execution_ref_for_context(context) == ref


# ── the tracker ──────────────────────────────────────────────────────────────

class _Query:
    """A query over in-memory rows that honours ``column == value`` clauses —
    the route lookup (model_id AND serving_provider) must not match any row."""

    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *clauses):
        rows = self._rows
        for c in clauses:
            try:
                col, val = c.left.name, c.right.value
                rows = [r for r in rows if getattr(r, col, None) == val]
            except Exception:  # noqa: BLE001 — a clause the fake cannot read keeps the rows
                pass
        return _Query(rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    added: list = []

    def __init__(self, tables):
        self._tables = tables

    def query(self, model):
        return _Query(self._tables.get(model.__name__, []))

    def add(self, row):
        _FakeSession.added.append(row)

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_db(monkeypatch):
    def install(tables=None):
        _FakeSession.added = []
        monkeypatch.setitem(
            sys.modules, "core.database.database",
            types.SimpleNamespace(SessionLocal=lambda: _FakeSession(tables or {})),
        )
        return _FakeSession
    return install


def _route_row(**kw):
    base = dict(input_cost_per_1k_tokens=0.003, output_cost_per_1k_tokens=0.015,
                sourcing="aggregator", serving_provider="openrouter", model_id="moonshotai/kimi-k3")
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_route_row_prices_the_call_and_names_the_tier(fake_db):
    session = fake_db({"LLMModel": [_route_row()]})
    ut.UsageTracker.track(workspace_id=WS, model_id="moonshotai/kimi-k3", provider="openrouter",
                          input_tokens=1000, output_tokens=1000, request_type="chat")
    row = session.added[0]
    assert row.tier == "aggregator" and row.total_cost == pytest.approx(0.018)
    assert row.cache_read_tokens == 0 and row.request_type == "chat"


def test_a_metered_route_with_no_row_falls_back_to_the_catalogue_then_the_estimate(fake_db):
    cache = SimpleNamespace(model_id="google/gemini-2.5-flash", prompt_cost=0.0000003, completion_cost=0.0000025)
    session = fake_db({"OpenRouterModelCache": [cache]})
    ut.UsageTracker.track(workspace_id=WS, model_id="google/gemini-2.5-flash", provider="openrouter",
                          input_tokens=1_000_000, output_tokens=0)
    row = session.added[0]
    assert row.total_cost == pytest.approx(0.30)            # never $0 for a paid route
    assert row.tier == "aggregator"                          # the registry kind, not the caller's default
    session = fake_db({})
    ut.UsageTracker.track(workspace_id=WS, model_id="openai/gpt-4o-mini", provider="openrouter",
                          input_tokens=1000, output_tokens=0)
    assert session.added[0].total_cost > 0                   # the static estimate map matched "gpt-4o"
    session = fake_db({})
    ut.UsageTracker.track(workspace_id=WS, model_id="vendor/never-heard-of-it", provider="openrouter",
                          input_tokens=1000, output_tokens=0)
    assert session.added[0].total_cost == 0                  # nothing matched: $0, logged — never a made-up rate


def test_a_free_route_books_zero_whatever_the_fallback_says(fake_db):
    session = fake_db({"LLMModel": [_route_row()]})  # only the OpenRouter row exists
    ut.UsageTracker.track(workspace_id=WS, model_id="moonshotai/kimi-k3", provider="nvidia",
                          input_tokens=5000, output_tokens=500)
    row = session.added[0]
    assert row.total_cost == 0.0 and row.tier == "hosted_open" and row.provider == "nvidia"


def test_cache_tokens_are_recorded_and_repriced_at_the_vendors_rates(fake_db):
    session = fake_db({"LLMModel": [_route_row(serving_provider="anthropic", sourcing="direct",
                                                model_id="claude-sonnet-4-5", input_cost_per_1k_tokens=0.003)]})
    ut.UsageTracker.track(workspace_id=WS, model_id="claude-sonnet-4-5", provider="anthropic",
                          input_tokens=10_000, output_tokens=0, cache_read_tokens=8_000, cache_write_tokens=1_000)
    row = session.added[0]
    assert row.cache_read_tokens == 8_000 and row.cache_write_tokens == 1_000 and row.input_tokens == 10_000
    # 1,000 fresh × 1.0 + 8,000 read × 0.1 + 1,000 write × 1.25 = 3,050 token-equivalents × $0.003/1k
    assert row.input_cost == pytest.approx(0.00915)


def test_a_provider_reported_cost_beats_the_estimate(fake_db):
    session = fake_db({"LLMModel": [_route_row()]})
    ut.UsageTracker.track(workspace_id=WS, model_id="moonshotai/kimi-k3", provider="openrouter",
                          input_tokens=1000, output_tokens=1000, reported_cost=0.0123)
    assert session.added[0].total_cost == pytest.approx(0.0123)


def test_a_subscription_session_books_one_zero_cost_row_per_model(fake_db):
    session = fake_db({"LLMModel": [_route_row(serving_provider="anthropic", model_id="claude-fable-5")]})
    written = ut.UsageTracker.track_session(
        WS, cli_provider="claude", agent_id=15, execution_id="board_task:97", request_type="board_task",
        usage={"model": "claude-fable-5", "per_model": {
            "claude-fable-5": {"input_tokens": 6, "output_tokens": 666, "cache_read_input_tokens": 117716, "cache_creation_input_tokens": 119012},
            "claude-haiku-4-5": {"input_tokens": 50, "output_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        }},
        latency_ms=41_000,
    )
    assert written == 2
    fable = next(r for r in session.added if r.model_id == "claude-fable-5")
    assert fable.provider == "claude_code" and fable.tier == "subscription" and fable.is_byok is True
    assert fable.total_cost == 0.0                             # the plan pays; an anthropic price row must not leak in
    assert fable.input_tokens == 6 + 117716 + 119012 and fable.cache_read_tokens == 117716
    assert fable.output_tokens == 666 and fable.execution_id == "board_task:97" and fable.agent_id == 15


def test_an_empty_session_usage_writes_nothing():
    from services.cli_host_service import book_session_usage
    task = SimpleNamespace(id=1, workspace_id=WS, assigned_agent_id=15)
    assert book_session_usage(task, {"provider": "claude"}, {}, status="success",
                              request_type="board_task", execution_id="board_task:1") == 0
    assert book_session_usage(task, {"provider": "claude"}, None, status="success",
                              request_type="board_task", execution_id="board_task:1") == 0


def test_book_session_usage_hands_the_ticket_facts_to_the_tracker(monkeypatch):
    from services import cli_host_service as svc
    calls = []
    monkeypatch.setattr(svc.UsageTracker if hasattr(svc, "UsageTracker") else ut.UsageTracker, "track_session",
                        staticmethod(lambda *a, **k: calls.append((a, k)) or 1))
    task = SimpleNamespace(id=97, workspace_id=WS, assigned_agent_id=15)
    ref = {"provider": "claude", "model": "fable", "claimed_at": "2026-09-09T10:00:00+00:00"}
    n = svc.book_session_usage(task, ref, {"model": "claude-fable-5", "input_tokens": 6, "output_tokens": 1},
                               status="success", request_type="board_task", execution_id="board_task:97")
    assert n == 1
    (args, kwargs), = calls
    assert args == (WS,) and kwargs["cli_provider"] == "claude" and kwargs["agent_id"] == 15
    assert kwargs["execution_id"] == "board_task:97" and kwargs["fallback_model"] == "fable"
    assert isinstance(kwargs["latency_ms"], int)


def test_workspace_falls_back_to_the_local_edition_default(monkeypatch):
    from config import config as cfg
    monkeypatch.setattr(cfg, "AUTH_EDITION", "local", raising=False)
    monkeypatch.setattr(cfg, "DEFAULT_WORKSPACE_ID", str(WS), raising=False)
    assert ut.resolve_workspace_id(None) == str(WS)
    assert ut.resolve_workspace_id(uuid4()) != str(WS)
    monkeypatch.setattr(cfg, "AUTH_EDITION", "saas", raising=False)
    assert ut.resolve_workspace_id(None) is None


def test_no_workspace_means_no_row_not_an_error(fake_db, monkeypatch):
    from config import config as cfg
    monkeypatch.setattr(cfg, "AUTH_EDITION", "saas", raising=False)
    session = fake_db({})
    ut.UsageTracker.track(workspace_id=None, model_id="x", provider="openrouter", input_tokens=1, output_tokens=1)
    assert session.added == []


def test_rerank_is_priced_per_search_unit_never_per_token(fake_db):
    session = fake_db({})
    ut.UsageTracker.track_rerank(provider="cohere", model_id="rerank-v3.5", search_units=3, usd_per_1k_units=2.0)
    row = session.added[0]
    assert row.request_type == "rerank" and row.input_tokens == 3 and row.total_cost == pytest.approx(0.006)


def test_embedding_calls_inherit_the_enclosing_lane(fake_db):
    session = fake_db({})
    with uc.usage_scope(request_type="chat", execution_id="chat:abc", agent_id=1, workspace_id=WS):
        ut.UsageTracker.track_embedding(provider="openrouter", model_id="qwen/qwen3-embedding-8b", prompt_tokens=512)
    row = session.added[0]
    assert row.request_type == "embedding" and row.execution_id == "chat:abc" and row.agent_id == 1
    assert row.output_tokens == 0 and row.input_tokens == 512


# ── the providers as the page reads them ─────────────────────────────────────

@pytest.mark.parametrize(
    "slug, label, billing",
    [
        ("openrouter", "OpenRouter", "metered"),
        ("nvidia", "NVIDIA", "free"),
        ("claude_code", "Claude Code", "subscription"),
        ("codex", "Codex", "subscription"),
        ("aws_bedrock", "AWS Bedrock", "metered"),
        ("mystery", "mystery", "unknown"),
    ],
)
def test_describe_usage_provider_covers_apis_and_runtimes(slug, label, billing):
    facts = describe_usage_provider(slug)
    assert facts["label"] == label and facts["billing"] == billing


def test_route_keys_keep_free_and_paid_routes_apart():
    from api.llm_analytics import route_facts, route_key
    free = route_facts("moonshotai/kimi-k3", "nvidia")
    paid = route_facts("moonshotai/kimi-k3", "openrouter")
    assert free["key"] != paid["key"] and free["billing"] == "free" and paid["billing"] == "metered"
    assert route_key("m", "p") == "m@p" and free["label"] == "moonshotai/kimi-k3 · NVIDIA"


# ── the clients' usage dicts ─────────────────────────────────────────────────

def test_openai_compatible_usage_dict_carries_cache_and_openrouter_cost():
    from core.llm.clients.openai_compatible_client import usage_dict
    usage = SimpleNamespace(
        prompt_tokens=1200, completion_tokens=30, total_tokens=1230,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1000),
        model_extra={"cost": 0.0021, "cost_details": {"upstream_inference_cost": 0.0}},
    )
    out = usage_dict(usage)
    assert out["prompt_tokens"] == 1200 and out["cache_read_tokens"] == 1000 and out["cost"] == pytest.approx(0.0021)
    assert usage_dict(None) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_anthropic_usage_dict_reports_the_full_prompt():
    from core.llm.clients.anthropic_client import _usage_dict
    out = _usage_dict(SimpleNamespace(input_tokens=6, output_tokens=666, cache_read_input_tokens=117716,
                                      cache_creation_input_tokens=119012))
    assert out["prompt_tokens"] == 6 + 117716 + 119012 and out["cache_read_tokens"] == 117716
    assert out["cache_write_tokens"] == 119012 and out["total_tokens"] == out["prompt_tokens"] + 666


def test_openrouter_requests_ask_for_the_cost():
    from core.llm import providers as reg
    assert reg.get_spec("openrouter").reports_cost is True
    assert reg.get_spec("nvidia").reports_cost is False
