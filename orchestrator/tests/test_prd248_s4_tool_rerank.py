"""PRD-248 S4 — the tool-surface rerank.

The pure cut (floor, order, minimum kept, cap, "nothing fits", too-few-answers
= miss); the production runner in off / shadow / live with everything
injected (never changes a surface in shadow, never turns a list into None in
live); the engine's new dials; the eval harness's ``jev_rerank`` mode falling
back to the plain top-K on a miss; and the uplift harness's challenger scored
beside — never inside — the gate.

PURE tests: no index, no registry, no HTTP.
"""
from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from core.llm.decisions import rerank as rr
from core.llm.decisions.engine import DecisionEngine
from core.llm.decisions.questions import DecisionAnswer, DecisionResult
from evals.operating_graph_uplift import (
    EvalCase,
    _build_parser,
    _jev_challenger_factory,
    render_report,
    run_uplift_eval,
)
from modules.tools.discovery import decision_rerank as runner

_ORCH = Path(__file__).resolve().parent.parent


def _result(**probs: float) -> DecisionResult:
    answers = {name: DecisionAnswer(type="noul", noul=p) for name, p in probs.items()}
    return DecisionResult(answers=answers, provider="fake", model="fake-1", latency_ms=280, input_tokens=1200)


# --------------------------------------------------------------------------- #
# The pure cut
# --------------------------------------------------------------------------- #


def test_questions_are_one_noul_per_candidate_with_the_description_folded_in():
    q = rr.build_questions([("platform_list_agents", "List agents"), ("platform_x", "x" * 400), ("", "skip")])
    assert list(q) == ["platform_list_agents", "platform_x"]
    wire = q["platform_list_agents"].to_wire()
    assert wire["type"] == "noul" and "`platform_list_agents` (List agents) would help" in wire["instructions"]
    assert len(q["platform_x"].instructions) < 400 + 80
    with pytest.raises(ValueError):
        rr.build_questions([])


def test_cut_keeps_above_the_floor_ordered_by_probability_and_capped():
    result = _result(a=0.9, b=0.2, c=0.8, d=0.7, e=0.1, f=0.05, g=0.6, h=0.3)
    cut = rr.apply_rerank(result, list("abcdefgh"), top_k=3, min_probability=0.5, min_keep=5)
    assert cut is not None
    assert cut.kept == ["a", "c", "d"]  # four clear the floor, capped at three
    assert cut.nothing_fits is False and cut.answered == 8
    assert cut.probabilities["a"] == 0.9


def test_cut_tops_up_to_the_minimum_and_flags_nothing_fits():
    result = _result(a=0.4, b=0.1, c=0.3, d=0.2, e=0.05, f=0.45)
    cut = rr.apply_rerank(result, list("abcdef"), top_k=15, min_probability=0.5, min_keep=3)
    assert cut is not None
    assert cut.nothing_fits is True
    assert cut.kept == ["f", "a", "c"]  # the three most likely, by probability
    assert rr.apply_rerank(result, list("abcdef"), top_k=15, min_probability=0.5, min_keep=0).kept == []


def test_too_few_answers_is_a_miss_not_a_narrower_surface():
    six = list("abcdef")
    assert rr.apply_rerank(_result(a=0.9, b=0.9), six, top_k=15) is None
    assert rr.apply_rerank(_result(a=0.9, b=0.9, c=0.9), six, top_k=15) is not None
    assert rr.apply_rerank(_result(a=0.9), [], top_k=15) is None


def test_compare_surfaces_names_what_moved():
    cmp = rr.compare_surfaces(["a", "b", "c"], ["a", "c", "d"])
    assert cmp == {
        "embedding_size": 3, "rerank_size": 3, "overlap": 2,
        "dropped": ["b"], "added": ["d"], "same_top": True,
    }


@pytest.mark.asyncio
async def test_rerank_candidates_passes_the_purpose_and_reports_misses():
    seen: Dict[str, Any] = {}

    async def decide(**kw):
        seen.update(kw)
        return None

    cut, result = await rr.rerank_candidates(
        query="list my agents", candidates=[("a", "A"), ("b", "B")], decide=decide, top_k=5, workspace_id="ws",
    )
    assert (cut, result) == (None, None)
    assert seen["purpose"] == "tool_rerank" and seen["workspace_id"] == "ws"
    assert seen["state"] == {"request": "list my agents"} and set(seen["questions"]) == {"a", "b"}
    assert await rr.rerank_candidates(query="q", candidates=[], decide=decide, top_k=5) == (None, None)


# --------------------------------------------------------------------------- #
# The production runner: off / shadow / live
# --------------------------------------------------------------------------- #

WIDE = [(f"a{i}", round(1 - i * 0.01, 2)) for i in range(8)]  # a0..a7


async def _rank_wide(n: int):
    return WIDE[:n]


def _describe(name: str) -> str:
    return f"{name} does things"


class _Recorder:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def __call__(self, row) -> None:
        self.rows.append(dict(row))


def _decider(probs: Optional[Dict[str, float]] = None, exc: Optional[Exception] = None):
    calls: List[Dict[str, Any]] = []

    async def decide(**kw):
        calls.append(kw)
        if exc:
            raise exc
        return _result(**probs) if probs is not None else None

    decide.calls = calls  # type: ignore[attr-defined]
    return decide


PROBS = {"a0": 0.9, "a1": 0.2, "a2": 0.8, "a3": 0.7, "a4": 0.1, "a5": 0.05, "a6": 0.6, "a7": 0.3}


async def _drain():
    pending = list(runner._SHADOW_TASKS)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for _ in range(3):
        await asyncio.sleep(0)


def _kwargs(mode, decide, recorder, **over):
    base = dict(
        query="list my agents", allowed=["a0", "a1", "a2"], mode=mode, rank_wide=_rank_wide,
        describe=_describe, decide=decide, record_shadow=recorder, top_k=3,
        candidates_n=8, min_probability=0.5, min_keep=5, workspace_id="ws",
    )
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_off_returns_the_allow_list_unchanged_and_never_decides():
    decide, rec = _decider(PROBS), _Recorder()
    out = await runner.narrow_with_decisions(**_kwargs("off", decide, rec))
    await _drain()
    assert out == ["a0", "a1", "a2"] and decide.calls == [] and rec.rows == []
    # no query / no list → untouched whatever the mode
    assert await runner.narrow_with_decisions(**_kwargs("live", decide, rec, query="")) == ["a0", "a1", "a2"]
    assert await runner.narrow_with_decisions(**_kwargs("live", decide, rec, allowed=None)) is None
    assert decide.calls == []


@pytest.mark.asyncio
async def test_shadow_keeps_the_embedding_cut_and_records_the_comparison():
    decide, rec = _decider(PROBS), _Recorder()
    out = await runner.narrow_with_decisions(**_kwargs("shadow", decide, rec))
    assert out == ["a0", "a1", "a2"]  # unchanged, and returned before the judge answered
    await _drain()
    assert len(rec.rows) == 1
    row = rec.rows[0]
    assert row["purpose"] == "tool_rerank" and row["workspace_id"] == "ws" and row["query_sha"]
    assert row["embedding_top"] == ["a0", "a1", "a2"] and row["wide"] == [n for n, _ in WIDE]
    assert row["kept"] == ["a0", "a2", "a3"] and row["nothing_fits"] is False
    assert row["candidate_source"] == "embedding"
    assert row["compare"]["dropped"] == ["a1"] and row["compare"]["added"] == ["a3"]
    assert row["provider"] == "fake" and row["latency_ms"] == 280 and "shadow_ms" in row
    # the judge saw the wide list with descriptions, not the narrow one
    assert set(decide.calls[0]["questions"]) == {n for n, _ in WIDE}
    assert "a7 does things" in decide.calls[0]["questions"]["a7"].instructions


@pytest.mark.asyncio
async def test_shadow_labels_no_candidates_and_judges_a_lexical_shortlist():
    """An empty candidate list never reaches the engine and is labelled
    'no_candidates', not 'no_result' (2026-09-22: an embedding timeout read as an
    engine miss); lexical candidates (None scores) are judged and marked."""
    rec = _Recorder()
    decide = _decider(PROBS)

    async def empty(n):
        return []

    await runner.narrow_with_decisions(**_kwargs("shadow", decide, rec, rank_wide=empty))
    await _drain()
    assert rec.rows[-1]["error"] == "no_candidates" and rec.rows[-1]["candidate_source"] == "none"
    assert decide.calls == []

    async def lexical(n):
        return [(name, None) for name, _score in WIDE[:n]]

    await runner.narrow_with_decisions(**_kwargs("shadow", decide, rec, rank_wide=lexical))
    await _drain()
    assert rec.rows[-1]["candidate_source"] == "lexical" and rec.rows[-1]["kept"] == ["a0", "a2", "a3"]


@pytest.mark.asyncio
async def test_router_rank_wide_falls_back_to_the_lexical_shortlist(monkeypatch):
    import core.llm.decisions as decisions_pkg
    from modules.tools import tool_router
    from modules.tools.discovery import decision_rerank as dr

    class _Index:
        async def rank_actions(self, query, **kw):
            return []

        def lexical_rank(self, query, **kw):
            return ["platform_list_agents", "platform_list_tasks"]

    fake_index_mod = types.ModuleType("modules.tools.discovery.action_semantic_index")
    fake_index_mod.get_action_semantic_index = lambda: _Index()
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_semantic_index", fake_index_mod)
    shadow = DecisionEngine(settings_reader=lambda c, k, d: "shadow" if k == "tool_rerank_mode" else None,
                            backend_factory=lambda d: None)
    monkeypatch.setattr(decisions_pkg, "get_decision_engine", lambda: shadow)
    captured: Dict[str, Any] = {}

    async def capture(**kw):
        captured.update(kw)
        return kw["allowed"]

    monkeypatch.setattr(dr, "narrow_with_decisions", capture)
    assert await tool_router._apply_decision_rerank("list my agents", ["a"], True, False, "ws") == ["a"]
    assert await captured["rank_wide"](30) == [("platform_list_agents", None), ("platform_list_tasks", None)]


@pytest.mark.asyncio
async def test_shadow_records_a_miss_and_an_error_without_raising():
    rec = _Recorder()
    await runner.narrow_with_decisions(**_kwargs("shadow", _decider(None), rec))
    await _drain()
    assert rec.rows[-1]["error"] == "no_result"
    await runner.narrow_with_decisions(**_kwargs("shadow", _decider(exc=RuntimeError("down")), rec))
    await _drain()
    assert "RuntimeError" in rec.rows[-1]["error"]


@pytest.mark.asyncio
async def test_live_returns_the_cut_in_canonical_order_for_the_prompt_cache():
    """The same set of kept actions must be the same bytes every turn: the tool
    block heads Auto's cached prefix, so probability order would turn every
    turn into a cache miss."""
    rec = _Recorder()
    probs = {"a0": 0.6, "a1": 0.9, "a2": 0.2, "a3": 0.8, "a4": 0.1, "a5": 0.05, "a6": 0.7, "a7": 0.3}
    out = await runner.narrow_with_decisions(**_kwargs("live", _decider(probs), rec, top_k=4))
    assert out == ["a0", "a1", "a3", "a6"]  # by probability it would be a1, a3, a6, a0


@pytest.mark.asyncio
async def test_live_replaces_the_cut_and_falls_open_on_every_miss():
    rec = _Recorder()
    assert await runner.narrow_with_decisions(**_kwargs("live", _decider(PROBS), rec)) == ["a0", "a2", "a3"]
    assert await runner.narrow_with_decisions(**_kwargs("live", _decider(None), rec)) == ["a0", "a1", "a2"]
    assert await runner.narrow_with_decisions(
        **_kwargs("live", _decider(exc=RuntimeError("down")), rec)
    ) == ["a0", "a1", "a2"]

    async def broken_rank(n):
        raise RuntimeError("index down")

    assert await runner.narrow_with_decisions(
        **_kwargs("live", _decider(PROBS), rec, rank_wide=broken_rank)
    ) == ["a0", "a1", "a2"]
    assert rec.rows == []  # live mode writes no shadow rows


@pytest.mark.asyncio
async def test_router_call_site_is_byte_identical_when_off(monkeypatch):
    """The claim nights 3–8 rest on, at the production call site: with the dial
    off (or unreadable) the router returns the very list it was given and never
    consults the index, the registry or the engine's backend."""
    import core.llm.decisions as decisions_pkg
    from modules.tools import tool_router

    def never():
        raise AssertionError("the index must not be consulted while the dial is off")

    fake_index_mod = types.ModuleType("modules.tools.discovery.action_semantic_index")
    fake_index_mod.get_action_semantic_index = never
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_semantic_index", fake_index_mod)

    allowed = ["platform_list_agents", "platform_create_agent"]
    off = DecisionEngine(settings_reader=lambda c, k, d: "off" if k == "tool_rerank_mode" else None,
                         backend_factory=lambda d: never())
    monkeypatch.setattr(decisions_pkg, "get_decision_engine", lambda: off)
    assert await tool_router._apply_decision_rerank("list my agents", allowed, True, False, "ws") is allowed

    def broken(category, key, default):
        raise RuntimeError("db down")

    unreadable = DecisionEngine(settings_reader=broken, backend_factory=lambda d: never())
    monkeypatch.setattr(decisions_pkg, "get_decision_engine", lambda: unreadable)
    assert await tool_router._apply_decision_rerank("list my agents", allowed, True, False, "ws") is allowed
    assert await tool_router._apply_decision_rerank("list my agents", None, True, False, "ws") is None
    assert await tool_router._apply_decision_rerank("", allowed, True, False, "ws") is allowed


def test_night_briefs_are_the_same_evening_word_for_word():
    """PRD-247 nights 8 and 9 must be the same evening; only the supervisor's
    dials differ. A drift between the two briefs would silently confound the comparison."""
    nights = _ORCH.parent / "scripts" / "ralph" / "customer-night" / "nights"
    baseline = (nights / "auto-brain.md").read_bytes()
    jev = (nights / "auto-brain-jev.md").read_bytes()
    assert baseline == jev and baseline.startswith(b"## The evening you have in mind")
    assert (nights / "auto-brain-jev.SUPERVISOR.md").exists()


def test_engine_dials_parse_and_clamp_the_rerank_knobs():
    values = {"tool_rerank_mode": "shadow", "rerank_candidates": "500", "rerank_min_probability": "0.6", "rerank_min_keep": "abc"}
    d = DecisionEngine(settings_reader=lambda c, k, default: values.get(k)).dials()
    assert d.tool_rerank_mode == "shadow" and d.rerank_candidates == 120
    assert d.rerank_min_probability == 0.6 and d.rerank_min_keep == 5


# --------------------------------------------------------------------------- #
# The eval harness: jev_rerank mode
# --------------------------------------------------------------------------- #


@dataclass
class _FakeAction:
    name: str
    description: str
    category: str = "agents"
    parameters: Optional[Dict[str, Any]] = None


FAKE_ACTIONS = [_FakeAction(f"platform_{n}", f"Does {n}") for n in ("alpha", "beta", "gamma", "delta")]


@pytest.fixture
def harness(monkeypatch):
    """The lazy ``modules.tools.discovery.action_registry`` import inside the
    builder resolves to a fake, scoped to the test (the graph-mode suite's
    pattern, via monkeypatch so nothing leaks)."""
    registry = MagicMock()
    registry.build_filtered_prompt_summary.side_effect = (
        lambda names, **kw: "\n## Available Platform Actions\n\n" + "\n".join(f"- `{n}`" for n in names) + "\n"
    )
    fake_mod = types.ModuleType("modules.tools.discovery.action_registry")
    fake_mod.get_action_registry = lambda: registry
    fake_mod.ActionDefinition = MagicMock()
    fake_mod.ActionRegistry = MagicMock()
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_registry", fake_mod)
    from scripts.eval.tool_routing.prompt_builder import PromptBuilder, _PREAMBLE

    monkeypatch.setattr(
        PromptBuilder, "_rank_action_names",
        lambda self, query, top_k: [a.name for a in FAKE_ACTIONS][:top_k],
    )
    return PromptBuilder, _PREAMBLE, registry


def test_jev_rerank_mode_surfaces_the_judged_cut(harness):
    PromptBuilder, preamble, registry = harness
    seen: Dict[str, Any] = {}

    def decider(state, questions):
        seen["state"], seen["questions"] = state, questions
        return _result(platform_alpha=0.3, platform_beta=0.9, platform_gamma=0.8, platform_delta=0.1)

    builder = PromptBuilder(actions=FAKE_ACTIONS, decider=decider, rerank_candidates=4, rerank_min_keep=1)
    prompt, surfaced = builder.build("run beta please", mode="jev_rerank", top_k=2)
    assert surfaced == ["platform_beta", "platform_gamma"]
    assert prompt.startswith(preamble) and "`platform_beta`" in prompt and "`platform_alpha`" not in prompt
    assert seen["state"] == {"request": "run beta please"}
    assert "(Does delta)" in seen["questions"]["platform_delta"].instructions
    registry.build_filtered_prompt_summary.assert_called_once()
    assert registry.build_filtered_prompt_summary.call_args.args[0] == ["platform_beta", "platform_gamma"]


def test_jev_rerank_falls_back_to_the_filtered_top_k_on_a_miss(harness):
    PromptBuilder, _preamble, _registry = harness
    builder = PromptBuilder(actions=FAKE_ACTIONS, decider=lambda s, q: None)
    _prompt, surfaced = builder.build("anything", mode="jev_rerank", top_k=2)
    assert surfaced == ["platform_alpha", "platform_beta"]

    def boom(s, q):
        raise RuntimeError("judge down")

    _prompt, surfaced = PromptBuilder(actions=FAKE_ACTIONS, decider=boom).build("x", mode="jev_rerank", top_k=3)
    assert surfaced == ["platform_alpha", "platform_beta", "platform_gamma"]


def test_build_tools_narrows_the_schema_for_jev_rerank(harness):
    PromptBuilder, _p, _r = harness
    tools = [{"type": "function", "function": {"name": "platform_execute", "parameters": {"properties": {"action": {"type": "string"}}}}}]
    narrowed = PromptBuilder(actions=FAKE_ACTIONS).build_tools(
        tools, "q", mode="jev_rerank", ranked_names=["platform_beta"],
    )
    assert narrowed[0]["function"]["parameters"]["properties"]["action"]["enum"] == ["platform_beta"]
    assert "enum" not in tools[0]["function"]["parameters"]["properties"]["action"]


def test_runner_and_scorer_know_the_mode():
    run_eval = (_ORCH / "scripts" / "eval" / "tool_routing" / "run_eval.py").read_text()
    assert '"jev_rerank"' in run_eval.split("choices=", 1)[1].split("]", 1)[0]
    assert '"jev_rerank"' in run_eval.split("needs_embedding = ", 1)[1].split("\n", 1)[0]
    score = (_ORCH / "scripts" / "eval" / "tool_routing" / "score.py").read_text()
    assert '"jev_rerank": 3' in score


# --------------------------------------------------------------------------- #
# The uplift harness: a challenger beside the gate
# --------------------------------------------------------------------------- #

CASES = [
    EvalCase("handle the thing", "platform_do_alpha", "ambiguous", "ws"),
    EvalCase("handle the thing please", "platform_do_alpha", "ambiguous", "ws"),
    EvalCase("deal with the thing", "platform_do_alpha", "ambiguous", "ws"),
    EvalCase("handle the thing now", "platform_do_alpha", "ambiguous", "ws"),
    EvalCase("show the cost breakdown", "platform_get_cost_breakdown", "analytics", "ws"),
    EvalCase("cost breakdown please", "platform_get_cost_breakdown", "analytics", "ws"),
]


def test_challenger_is_scored_beside_the_gate_and_never_moves_it():
    plain = run_uplift_eval(CASES)
    always_alpha = run_uplift_eval(
        CASES, challenger_factory=lambda actions, cats: (lambda q: "platform_do_alpha"),
    )
    assert plain.has_challenger is False and always_alpha.has_challenger is True
    assert always_alpha.mean_uplift_points == plain.mean_uplift_points  # the gate is untouched
    tenant = always_alpha.tenants[0]
    assert tenant.challenger_acc is not None and tenant.challenger_uplift_points is not None
    always_alpha.meta["challenger"] = "jev"
    doc = always_alpha.to_dict()
    assert doc["challenger"] == "jev" and "challenger_top1" in doc["tenants"][0]
    assert "mean_challenger_uplift_points" in doc and "challenger" not in doc["loader"]
    text = render_report(always_alpha)
    assert "challenger top-1" in text and "Challenger `jev`" in text
    assert "challenger" not in render_report(plain)


def test_jev_challenger_takes_the_choice_and_falls_back_on_none_or_a_miss():
    actions = ["platform_do_alpha", "platform_get_cost_breakdown"]
    cats = {"platform_do_alpha": "ambiguous", "platform_get_cost_breakdown": "analytics"}
    asked: List[Dict[str, Any]] = []

    def decide_with(choice):
        def decide(state, questions):
            asked.append({"state": state, "questions": questions})
            return DecisionResult(
                answers={"action": DecisionAnswer(type="choice", choice=choice, confidence=0.9)},
                provider="fake", model="fake-1", latency_ms=200,
            )
        return decide

    pick = _jev_challenger_factory(actions, cats, candidates=2, decide=decide_with("platform_get_cost_breakdown"))
    assert pick("handle the thing") == "platform_get_cost_breakdown"
    options = asked[-1]["questions"]["action"].options
    assert set(options) == {"platform_do_alpha", "platform_get_cost_breakdown", "none"}
    assert asked[-1]["state"] == {"request": "handle the thing"}

    none = _jev_challenger_factory(actions, cats, candidates=2, decide=decide_with("none"))
    assert none("show the cost breakdown") == "platform_get_cost_breakdown"  # the proxy's own top-1

    def boom(state, questions):
        raise RuntimeError("down")

    miss = _jev_challenger_factory(actions, cats, candidates=2, decide=boom)
    assert miss("show the cost breakdown") == "platform_get_cost_breakdown"

    described = _jev_challenger_factory(
        actions, cats, candidates=2, decide=decide_with("none"), describe=lambda a: f"desc of {a}",
    )
    described("x")
    assert asked[-1]["questions"]["action"].criteria["platform_do_alpha"] == "desc of platform_do_alpha"


def test_cli_exposes_the_challenger_flags():
    args = _build_parser().parse_args(["--ranker", "jev", "--rerank-candidates", "12"])
    assert args.ranker == "jev" and args.rerank_candidates == 12
    assert _build_parser().parse_args([]).ranker == "none"
