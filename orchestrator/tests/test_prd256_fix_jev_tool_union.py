"""P256-FIX-N14-8 — Jev's tool shortlist, an experiment behind a flag.

``TOOL_SHORTLIST_JEV_UNION`` off: the router offers the platform's list exactly
as before (the ``tool_rerank_mode`` dial alone decides). On, with Jev answering
in time: Jev's kept top few, then the platform's embedding top few, each name
once, Jev's first. On, with Jev late, silent or failing: the platform's list
unchanged. Every union turn writes one shadow row naming the list offered.

PURE tests: no index, no registry, no HTTP (the runner is injected).
"""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from config import config
from core.llm.decisions import rerank as rr
from core.llm.decisions.engine import DecisionEngine
from core.llm.decisions.questions import DecisionAnswer, DecisionResult
from modules.tools.discovery import decision_rerank as runner
from modules.tools.discovery import jev_union

PLATFORM = [f"p{i}" for i in range(15)]  # the platform's embedding top-15, in rank order
WIDE = [(f"j{i}", 0.4) for i in range(5)] + [(name, 0.9 - i * 0.01) for i, name in enumerate(PLATFORM)]
# Jev's kept cut, by probability: j0, p3, j1, j2, p12, j3 (six clear the 0.5 floor).
PROBS = {name: 0.1 for name, _ in WIDE}
PROBS.update(j0=0.95, p3=0.9, j1=0.85, j2=0.8, p12=0.75, j3=0.7)
JEV_TOP5 = ["j0", "p3", "j1", "j2", "p12"]
UNION = JEV_TOP5 + ["p0", "p1", "p2", "p4", "p5", "p6", "p7", "p8", "p9"]


def _result(probs: Dict[str, float]) -> DecisionResult:
    top = len(rr.HELP_LEVELS) - 1
    answers = {n: DecisionAnswer(type="score", score=p * top, confidence=0.8) for n, p in probs.items()}
    return DecisionResult(answers=answers, provider="fake", model="jev-1", latency_ms=410, input_tokens=900)


async def _rank_wide(n: int):
    return WIDE[:n]


class _Recorder:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def __call__(self, row) -> None:
        self.rows.append(dict(row))


def _decider(probs: Optional[Dict[str, float]] = None, *, delay: float = 0.0, exc: Optional[Exception] = None):
    calls: List[Dict[str, Any]] = []

    async def decide(**kw):
        calls.append(kw)
        if delay:
            await asyncio.sleep(delay)
        if exc:
            raise exc
        return _result(probs) if probs is not None else None

    decide.calls = calls  # type: ignore[attr-defined]
    return decide


@pytest.fixture
def union_on(monkeypatch):
    monkeypatch.setattr(config, "TOOL_SHORTLIST_JEV_UNION", True, raising=False)
    monkeypatch.setattr(config, "TOOL_SHORTLIST_JEV_TIMEOUT_S", 1.0, raising=False)
    monkeypatch.setattr(config, "TOOL_SHORTLIST_JEV_TOP", 5, raising=False)
    monkeypatch.setattr(config, "TOOL_SHORTLIST_PLATFORM_TOP", 10, raising=False)


def _kwargs(decide, recorder, **over):
    base = dict(
        query="draft the supplier email", allowed=list(PLATFORM), mode=jev_union.offered_mode("off"),
        rank_wide=_rank_wide, describe=lambda n: f"{n} does things", decide=decide, record_shadow=recorder,
        top_k=15, candidates_n=30, min_probability=0.5, min_keep=5, workspace_id="ws",
    )
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# The flag and its default
# --------------------------------------------------------------------------- #


def test_the_flag_is_off_by_default_and_the_dial_decides_while_it_is():
    assert config.TOOL_SHORTLIST_JEV_UNION is False
    assert config.TOOL_SHORTLIST_JEV_TIMEOUT_S == 1.0
    assert (config.TOOL_SHORTLIST_JEV_TOP, config.TOOL_SHORTLIST_PLATFORM_TOP) == (5, 10)
    for dial in ("off", "shadow", "live"):
        assert jev_union.offered_mode(dial) == dial


def test_the_flag_on_selects_the_union_whatever_the_dial(union_on):
    for dial in ("off", "shadow", "live"):
        assert jev_union.offered_mode(dial) == runner.MODE_UNION


def test_union_of_is_jevs_first_then_the_platforms_each_name_once():
    assert jev_union.union_of(["b", "x", ""], ["a", "b", "c"]) == ["b", "x", "a", "c"]
    assert jev_union.union_of([], ["a", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Flag off: the platform list, unchanged, at the router
# --------------------------------------------------------------------------- #


def _no_index(monkeypatch) -> None:
    def never():
        raise AssertionError("the index must not be consulted while the flag and the dial are off")

    fake = types.ModuleType("modules.tools.discovery.action_semantic_index")
    fake.get_action_semantic_index = never
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_semantic_index", fake)


@pytest.mark.asyncio
async def test_flag_off_the_router_offers_the_platform_list_unchanged(monkeypatch):
    import core.llm.decisions as decisions_pkg
    from modules.tools import tool_router

    _no_index(monkeypatch)
    monkeypatch.setattr(config, "TOOL_SHORTLIST_JEV_UNION", False, raising=False)
    off = DecisionEngine(settings_reader=lambda c, k, d: "off" if k == "tool_rerank_mode" else None,
                         backend_factory=lambda d: None)
    monkeypatch.setattr(decisions_pkg, "get_decision_engine", lambda: off)
    allowed = list(PLATFORM)
    assert await tool_router._apply_decision_rerank("draft the supplier email", allowed, True, False, "ws") is allowed


@pytest.mark.asyncio
async def test_flag_on_the_router_runs_the_union_even_with_the_dial_off(monkeypatch, union_on):
    import core.llm.decisions as decisions_pkg
    from modules.tools import tool_router

    fake = types.ModuleType("modules.tools.discovery.action_semantic_index")
    fake.get_action_semantic_index = lambda: object()
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_semantic_index", fake)
    off = DecisionEngine(settings_reader=lambda c, k, d: "off" if k == "tool_rerank_mode" else None,
                         backend_factory=lambda d: None)
    monkeypatch.setattr(decisions_pkg, "get_decision_engine", lambda: off)
    seen: Dict[str, Any] = {}

    async def capture(**kw):
        seen.update(kw)
        return kw["allowed"]

    monkeypatch.setattr(runner, "narrow_with_decisions", capture)
    await tool_router._apply_decision_rerank("draft the supplier email", list(PLATFORM), True, False, "ws")
    assert seen["mode"] == runner.MODE_UNION


# --------------------------------------------------------------------------- #
# Flag on: the union when Jev answers, the platform's list when it doesn't
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_flag_on_with_a_jev_answer_offers_the_union_jevs_first(union_on):
    rec = _Recorder()
    out = await runner.narrow_with_decisions(**_kwargs(_decider(PROBS), rec))
    assert out == UNION  # Jev's five, then the platform's ten less p3, each once
    assert len(rec.rows) == 1
    row = rec.rows[0]
    assert row["purpose"] == "tool_rerank" and row["mode"] == "union"
    assert row["offered"] == "jev_union" and row["offered_reason"] is None and row["offered_list"] == UNION
    assert row["embedding_top"] == PLATFORM and row["kept"][:5] == JEV_TOP5
    assert row["provider"] == "fake" and row["latency_ms"] == 410 and "shadow_ms" in row


@pytest.mark.asyncio
async def test_flag_on_with_a_jev_timeout_offers_the_platform_list_and_the_log_says_so(monkeypatch, union_on):
    monkeypatch.setattr(config, "TOOL_SHORTLIST_JEV_TIMEOUT_S", 0.05, raising=False)
    rec = _Recorder()
    out = await runner.narrow_with_decisions(**_kwargs(_decider(PROBS, delay=0.5), rec))
    assert out == PLATFORM
    row = rec.rows[-1]
    assert row["offered"] == "platform" and row["offered_reason"] == "timeout" and row["error"] == "timeout"
    assert row["offered_list"] == PLATFORM and row["mode"] == "union"


@pytest.mark.asyncio
async def test_flag_on_with_no_answer_or_an_error_offers_the_platform_list(union_on):
    rec = _Recorder()
    assert await runner.narrow_with_decisions(**_kwargs(_decider(None), rec)) == PLATFORM
    assert rec.rows[-1]["offered"] == "platform" and rec.rows[-1]["offered_reason"] == "no_answer"
    assert rec.rows[-1]["error"] == "no_result"
    assert await runner.narrow_with_decisions(**_kwargs(_decider(exc=RuntimeError("down")), rec)) == PLATFORM
    assert rec.rows[-1]["offered"] == "platform"

    async def broken_rank(n):
        raise RuntimeError("index down")

    out = await runner.narrow_with_decisions(**_kwargs(_decider(PROBS), rec, rank_wide=broken_rank))
    assert out == PLATFORM
    assert rec.rows[-1]["offered_reason"] == "error" and "index down" in rec.rows[-1]["error"]


@pytest.mark.asyncio
async def test_a_shadow_writer_that_fails_never_costs_the_turn(union_on):
    def broken(row):
        raise OSError("disk full")

    assert await runner.narrow_with_decisions(**_kwargs(_decider(PROBS), broken)) == UNION
