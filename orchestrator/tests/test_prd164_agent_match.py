"""PRD-164 S2 — semantic agent matching with reasons (Q21 blend).

Pure, DB-free proof of the extended AgentMatcher:

  * golden matrix — 10 task fixtures → expected top agent (ties allowed);
  * explicit agent_overrides (PRD-163 S4 approval-edited ``agent_role`` naming a
    roster agent) ALWAYS win, regardless of score and even below threshold;
  * every ranked agent carries a human-readable reason string;
  * the Q21 blend (capability-card similarity + live field signal) renormalizes
    away cleanly when signals are absent — byte-identical to the legacy
    five-component score, so missing embeddings never change dispatch behavior;
  * the field-signal aggregation and the persisted match annotation are pure.

The scoring core is exercised through ``AgentMatcher._rank_with_context`` (the
prefetched, DB-free seam ``rank()``/``match()`` delegate to).
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys as _sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern, see
# tests/test_harness_self_management.py) — the port points at nothing so the
# modules.tools import chain's fail-soft DB connect refuses instantly instead
# of hanging on a wedged local proxy. CI exports real POSTGRES_* so these
# setdefaults no-op there. Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which
# pulls modules.rag's ingestion chain (camelot at module top). Stub the missing
# *leaf* only when truly absent — never the modules.rag package.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import modules.coordination.agent_matcher as agent_matcher  # noqa: E402
from modules.coordination.agent_matcher import (  # noqa: E402
    AgentMatcher,
    SemanticSignals,
    build_match_annotation,
)
from modules.coordination.match_signals import (  # noqa: E402
    _field_signal_from_patterns,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal Agent shape the matcher reads (mirrors
# test_planner_capability_routing.py).
# ---------------------------------------------------------------------------


def _skill(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=None)


def _agent(agent_id, name, description, *, skills=(), tags=(), status="active",
           model_id="gpt-4o") -> SimpleNamespace:
    return SimpleNamespace(
        id=agent_id,
        name=name,
        description=description,
        skills=[_skill(s) for s in skills],
        tags=list(tags),
        status=status,
        model_config={"model_id": model_id},
        slug=None,
    )


SCOUT = _agent(1, "SCOUT", "Research specialist that investigates topics and gathers sources via web search",
               skills=["research", "web_search"], tags=["research"])
SCRIBE = _agent(2, "SCRIBE", "Writer that drafts long-form blog content and articles",
                skills=["writing", "copywriting", "blog"], tags=["writer"])
VECTOR = _agent(3, "VECTOR", "Growth strategist focused on marketing funnels and conversion optimisation",
                skills=["growth_strategy", "marketing"], tags=["growth"])
FORGE = _agent(4, "FORGE", "Software engineer that writes and reviews code",
               skills=["coding", "development"], tags=["coder"])
ATLAS = _agent(5, "ATLAS", "Data analyst focused on metrics and reporting",
               skills=["analysis", "reporting"], tags=["analyst"])
RECON = _agent(6, "RECON", "Research specialist for deep investigations",
               skills=["research"], tags=["research"])

ROSTER = [SCOUT, SCRIBE, VECTOR, FORGE, ATLAS, RECON]


def _rank(*, agent_role=None, required_tools=None, preferred_model=None,
          has_upstream=False, tool_map=None, busy_ids=frozenset(),
          history_map=None, semantic=None, agents=ROSTER):
    return AgentMatcher._rank_with_context(
        agents=agents,
        agent_role=agent_role,
        required_tools=required_tools or [],
        preferred_model=preferred_model,
        has_upstream=has_upstream,
        tool_map=tool_map or {},
        busy_agent_ids=busy_ids,
        history_map=history_map or {},
        semantic=semantic,
    )


def _sig(sim=None, fld=None) -> SemanticSignals:
    return SemanticSignals(similarity_by_agent=sim or {}, field_by_agent=fld or {})


# ---------------------------------------------------------------------------
# AC1: golden matrix — 10 task fixtures → expected top agent (ties allowed)
# ---------------------------------------------------------------------------


GOLDEN_MATRIX = [
    # (label, kwargs, allowed_top_agent_ids)
    ("role research, no signals (lexical tie SCOUT/RECON, id-stable)",
     dict(agent_role="research"), {SCOUT.id, RECON.id}),
    ("role writer → SCRIBE",
     dict(agent_role="writer"), {SCRIBE.id}),
    ("role coder → FORGE",
     dict(agent_role="coder"), {FORGE.id}),
    ("semantic card similarity lifts ATLAS for analyst work",
     dict(agent_role="analyst",
          semantic=_sig(sim={ATLAS.id: 0.9, SCOUT.id: 0.4, SCRIBE.id: 0.2,
                             VECTOR.id: 0.2, FORGE.id: 0.2, RECON.id: 0.2})),
     {ATLAS.id}),
    ("small semantic edge does NOT flip a strong lexical role match",
     dict(agent_role="research",
          semantic=_sig(sim={SCOUT.id: 0.7, ATLAS.id: 0.8, RECON.id: 0.2,
                             SCRIBE.id: 0.1, VECTOR.id: 0.1, FORGE.id: 0.1})),
     {SCOUT.id}),
    ("required tools dominate when no role is set",
     dict(required_tools=["github"], tool_map={FORGE.id: {"github"}}),
     {FORGE.id}),
    ("verified-task history splits equal lexical candidates",
     dict(agent_role="research", history_map={RECON.id: 0.95, SCOUT.id: 0.2}),
     {RECON.id}),
    ("live field signal tips an exact tie",
     dict(agent_role="research", semantic=_sig(fld={SCOUT.id: 1.0})),
     {SCOUT.id}),
    ("busy agent loses the tie",
     dict(agent_role="research", busy_ids=frozenset({RECON.id})),
     {SCOUT.id}),
    ("explicit agent_role naming VECTOR overrides a higher-scoring SCOUT",
     dict(agent_role="vector",
          busy_ids=frozenset({VECTOR.id}),
          history_map={VECTOR.id: 0.1, SCOUT.id: 0.9},
          semantic=_sig(sim={SCOUT.id: 0.95, VECTOR.id: 0.0})),
     {VECTOR.id}),
]


@pytest.mark.parametrize("label,kwargs,expected_ids",
                         GOLDEN_MATRIX, ids=[g[0] for g in GOLDEN_MATRIX])
def test_golden_matrix_top_agent(label, kwargs, expected_ids):
    ranked = _rank(**kwargs)
    assert ranked, f"{label}: no candidates ranked"
    assert ranked[0].agent_id in expected_ids, (
        f"{label}: expected top in {expected_ids}, got "
        f"{[(r.agent_name, r.total_score) for r in ranked[:3]]}"
    )


def test_golden_matrix_every_ranked_agent_has_a_reason():
    """Q21: ranked agents EACH carry a human-readable reason string."""
    for label, kwargs, _expected in GOLDEN_MATRIX:
        for r in _rank(**kwargs):
            assert isinstance(r.reason, str) and r.reason.strip(), (
                f"{label}: agent {r.agent_name} ranked without a reason"
            )


def test_ranking_is_deterministic_on_ties():
    """Equal scores order by agent_id so the golden matrix can't flap."""
    ranked = _rank(agent_role="research")
    scout = next(r for r in ranked if r.agent_id == SCOUT.id)
    recon = next(r for r in ranked if r.agent_id == RECON.id)
    assert scout.total_score == recon.total_score
    assert ranked[0].agent_id == SCOUT.id  # lower id wins the tie


# ---------------------------------------------------------------------------
# AC2: override always wins
# ---------------------------------------------------------------------------


def test_override_outranks_higher_scorer_and_is_flagged():
    ranked = _rank(
        agent_role="vector",
        busy_ids=frozenset({VECTOR.id}),
        history_map={VECTOR.id: 0.1, SCOUT.id: 0.9},
        semantic=_sig(sim={SCOUT.id: 0.95, VECTOR.id: 0.0}),
    )
    assert ranked[0].agent_id == VECTOR.id
    assert ranked[0].is_override is True
    # "regardless of score": the override agent genuinely scores LOWER than #2.
    assert ranked[0].total_score < ranked[1].total_score
    assert "explicit" in ranked[0].reason.lower()


def test_override_wins_even_below_match_threshold(monkeypatch):
    """match() returns the override agent even when its blended score is below
    MATCH_THRESHOLD — an explicit PRD-163 S4 assignment is never discarded."""
    monkeypatch.setattr(agent_matcher, "_build_tool_map", lambda db, ids: {})
    monkeypatch.setattr(agent_matcher, "_get_busy_agent_ids",
                        lambda db, ids: frozenset({VECTOR.id}))
    monkeypatch.setattr(agent_matcher, "_build_history_map",
                        lambda db, ids, **kw: {VECTOR.id: 0.0})

    task = SimpleNamespace(id=uuid4(), agent_role="vector",
                           input_context={"required_tools": ["github"]})
    spec = {"agent_role": "vector", "required_tools": ["github"]}
    semantic = _sig(sim={VECTOR.id: 0.0, SCOUT.id: 0.0})

    result = AgentMatcher.match(db=None, task=task, agents=[VECTOR, SCOUT],
                                task_spec=spec, semantic=semantic)
    assert result is not None and result.agent_id == VECTOR.id
    assert result.is_override is True
    assert result.total_score < agent_matcher.MATCH_THRESHOLD


def test_no_override_below_threshold_still_returns_none(monkeypatch):
    """Regression: without an explicit override the threshold still gates."""
    monkeypatch.setattr(agent_matcher, "_build_tool_map", lambda db, ids: {})
    monkeypatch.setattr(agent_matcher, "_get_busy_agent_ids",
                        lambda db, ids: frozenset())
    monkeypatch.setattr(agent_matcher, "_build_history_map",
                        lambda db, ids, **kw: {})

    task = SimpleNamespace(id=uuid4(), agent_role="research", input_context={})
    result = AgentMatcher.match(db=None, task=task, agents=[VECTOR],
                                task_spec={"agent_role": "research"})
    assert result is None


def test_inactive_agent_is_never_an_override_target():
    sleeper = _agent(7, "SLEEPER", "Disabled twin", status="inactive")
    ranked = _rank(agent_role="sleeper", agents=ROSTER + [sleeper])
    assert all(r.agent_id != sleeper.id for r in ranked)


# ---------------------------------------------------------------------------
# Q21 blend mechanics — renormalization keeps legacy behavior when signals
# are absent (missing embeddings must never change dispatch decisions).
# ---------------------------------------------------------------------------


def test_absent_signals_reproduce_legacy_score_exactly():
    ranked = _rank(agent_role="research")
    scout = next(r for r in ranked if r.agent_id == SCOUT.id)
    # skill 1.0*.40 + tools 0.5*.25 + model 0.5*.15 + avail 1.0*.10 + hist 0.5*.10
    assert scout.total_score == pytest.approx(0.75)
    assert scout.semantic is None or scout.semantic == pytest.approx(0.5)


def test_blended_score_is_renormalized_weighted_mean():
    ranked = _rank(agent_role="research",
                   semantic=_sig(sim={SCOUT.id: 0.7}, fld={SCOUT.id: 1.0}))
    scout = next(r for r in ranked if r.agent_id == SCOUT.id)
    expected = (0.75 + agent_matcher.WEIGHT_SEMANTIC * 0.7
                + agent_matcher.WEIGHT_FIELD_SIGNAL * 1.0) / (
        1.0 + agent_matcher.WEIGHT_SEMANTIC + agent_matcher.WEIGHT_FIELD_SIGNAL)
    assert scout.total_score == pytest.approx(expected, abs=1e-4)


def test_agent_without_capability_card_gets_neutral_semantic():
    """When SOME agents have cards, card-less agents score neutral 0.5 — having
    no embedding is not evidence of a bad fit."""
    ranked = _rank(agent_role="research", semantic=_sig(sim={ATLAS.id: 0.2}))
    scout = next(r for r in ranked if r.agent_id == SCOUT.id)
    assert scout.semantic == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Reasons — content sanity
# ---------------------------------------------------------------------------


def test_reason_mentions_role_and_capability_alignment():
    ranked = _rank(agent_role="analyst",
                   semantic=_sig(sim={ATLAS.id: 0.9, SCOUT.id: 0.4, SCRIBE.id: 0.2,
                                      VECTOR.id: 0.2, FORGE.id: 0.2, RECON.id: 0.2}))
    top = ranked[0]
    assert top.agent_id == ATLAS.id
    assert "analyst" in top.reason.lower()
    assert "capability" in top.reason.lower()


def test_reason_reports_tool_coverage_fraction():
    ranked = _rank(required_tools=["github"], tool_map={FORGE.id: {"github"}})
    top = ranked[0]
    assert top.agent_id == FORGE.id
    assert "1/1" in top.reason


def test_reason_flags_busy_agent():
    ranked = _rank(agent_role="research", busy_ids=frozenset({RECON.id}))
    recon = next(r for r in ranked if r.agent_id == RECON.id)
    assert "busy" in recon.reason.lower()


# ---------------------------------------------------------------------------
# Field-signal aggregation (pure)
# ---------------------------------------------------------------------------


def test_field_signal_aggregates_and_normalizes_per_agent():
    patterns = [
        {"agent_id": 1, "score": 0.5},
        {"agent_id": 1, "score": 0.3},
        {"agent_id": 2, "score": 0.4},
        {"agent_id": 0, "score": 9.0},   # seeder injections never credit an agent
        {"agent_id": None, "score": 1.0},
    ]
    out = _field_signal_from_patterns(patterns)
    assert out == {1: 1.0, 2: 0.5}


def test_field_signal_empty_inputs():
    assert _field_signal_from_patterns([]) == {}
    assert _field_signal_from_patterns(None) == {}
    assert _field_signal_from_patterns([{"agent_id": 1, "score": 0.0}]) == {}


# ---------------------------------------------------------------------------
# Persisted annotation (consumed by the task row + approval card payload)
# ---------------------------------------------------------------------------


def test_build_match_annotation_shape():
    ranked = _rank(agent_role="research", history_map={RECON.id: 0.95, SCOUT.id: 0.2})
    ann = build_match_annotation(ranked)
    assert ann["agent_id"] == RECON.id
    assert ann["agent_name"] == "RECON"
    assert isinstance(ann["score"], float)
    assert isinstance(ann["reason"], str) and ann["reason"]
    assert ann["is_override"] is False
    assert 1 <= len(ann["ranked"]) <= 3
    assert ann["ranked"][0]["agent_id"] == RECON.id
    for entry in ann["ranked"]:
        assert set(entry) == {"agent_id", "agent_name", "score", "reason"}


def test_build_match_annotation_marks_override():
    ranked = _rank(
        agent_role="vector",
        busy_ids=frozenset({VECTOR.id}),
        history_map={VECTOR.id: 0.1, SCOUT.id: 0.9},
        semantic=_sig(sim={SCOUT.id: 0.95, VECTOR.id: 0.0}),
    )
    ann = build_match_annotation(ranked)
    assert ann["agent_id"] == VECTOR.id and ann["is_override"] is True
