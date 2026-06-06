"""Regression net for capability-based agent routing.

The mission planner used to present each agent to the LLM as
``**NAME** (role: name.lower())`` — teaching it to bind a task to a specific
agent NAME. The matcher scores an exact name match at 1.0, so a blog/research
task tagged ``agent_role="vector"`` would land on VECTOR (a growth strategist)
instead of a research agent, then loop and time out.

These tests pin the fix:

1. The matcher routes a *capability* role ("research") to the research-capable
   agent, never the growth agent — proving capability roles work.
2. ``_render_agent_roster`` no longer teaches name-as-role and instead lists
   the canonical capability vocabulary.
3. ``_best_role_fit`` (the validation seam) accepts a fillable capability and
   rejects a role no active agent can satisfy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.coordination.agent_matcher import (  # noqa: E402
    CANONICAL_ROLES,
    _compute_skill_match,
)
from modules.coordination.planner import (  # noqa: E402
    _best_role_fit,
    _render_agent_roster,
)


# ---------------------------------------------------------------------------
# Fakes — minimal Agent shape the matcher/planner read.
# ---------------------------------------------------------------------------


def _skill(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _agent(name, description, *, skills=(), tags=(), status="active",
           model_id="gpt-4o") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        skills=[_skill(s) for s in skills],
        tags=list(tags),
        status=status,
        model_config={"model_id": model_id},
    )


_SCOUT = _agent(
    "SCOUT",
    "Research specialist that investigates topics and gathers sources via web search",
    skills=["research", "web_search", "source_analysis"],
    tags=["research"],
)
_VECTOR = _agent(
    "VECTOR",
    "Growth strategist focused on marketing funnels and conversion optimisation",
    skills=["growth_strategy", "marketing"],
    tags=["growth"],
)
_SCRIBE = _agent(
    "SCRIBE",
    "Writer that drafts long-form blog content and articles",
    skills=["writing", "copywriting", "blog"],
    tags=["writer"],
)


# ---------------------------------------------------------------------------
# 1. Capability routing — the core "stop picking by name" proof.
# ---------------------------------------------------------------------------


def test_research_role_routes_to_research_agent_over_growth_agent():
    """A capability role 'research' must score the research agent far above
    the growth agent — the growth agent should not be a candidate at all."""
    scout_score = _compute_skill_match(_SCOUT, "research")
    vector_score = _compute_skill_match(_VECTOR, "research")

    assert scout_score >= 0.6, f"research agent should match 'research', got {scout_score}"
    assert vector_score == 0.0, f"growth agent must NOT match 'research', got {vector_score}"
    assert scout_score > vector_score


def test_writer_role_routes_to_writer_not_research_agent():
    assert _compute_skill_match(_SCRIBE, "writer") >= 0.6
    assert _compute_skill_match(_SCRIBE, "writer") > _compute_skill_match(_SCOUT, "writer")


def test_exact_name_match_is_the_mechanism_we_route_around():
    """Documents WHY name-binding mis-routed: an agent's own name scores 1.0,
    which beats any capability fit. The roster fix stops the LLM emitting names
    so this 1.0 path is never reached for a capability task."""
    assert _compute_skill_match(_VECTOR, "vector") == 1.0
    # ...and that same growth agent is a 0.0 fit for the research work it wrongly got.
    assert _compute_skill_match(_VECTOR, "research") == 0.0


# ---------------------------------------------------------------------------
# 2. Roster no longer teaches name-as-role.
# ---------------------------------------------------------------------------


def test_roster_does_not_present_name_as_role():
    roster = _render_agent_roster([_SCOUT, _VECTOR, _SCRIBE])
    assert "(role:" not in roster, "roster must not map agent name to a role"
    assert "scout" not in roster.lower().split("agent_role")[0] or "(role:" not in roster


def test_roster_lists_canonical_capability_vocabulary_and_steers_off_names():
    roster = _render_agent_roster([_SCOUT, _VECTOR, _SCRIBE])
    assert "agent_role" in roster
    # The canonical capabilities the planner should choose from are surfaced.
    assert "research" in roster and "writer" in roster
    # Explicitly steered away from binding to a name.
    assert "name" in roster.lower()


def test_roster_skips_inactive_agents():
    inactive = _agent("GHOST", "inactive", status="inactive")
    roster = _render_agent_roster([_SCOUT, inactive])
    assert "SCOUT" in roster
    assert "GHOST" not in roster


# ---------------------------------------------------------------------------
# 3. Validation seam — capability coverage, not name matching.
# ---------------------------------------------------------------------------


def test_best_role_fit_accepts_fillable_capability():
    assert _best_role_fit("research", [_SCOUT, _VECTOR]) >= 0.6
    assert _best_role_fit("writer", [_SCOUT, _SCRIBE]) >= 0.6


def test_best_role_fit_rejects_unfillable_role():
    assert _best_role_fit("underwater_basket_weaving", [_SCOUT, _VECTOR, _SCRIBE]) == 0.0


def test_best_role_fit_ignores_inactive_agents():
    only_inactive = _agent("SCOUT", "research", skills=["research"], status="inactive")
    assert _best_role_fit("research", [only_inactive]) == 0.0


def test_canonical_roles_cover_research_and_writer():
    assert "research" in CANONICAL_ROLES
    assert "writer" in CANONICAL_ROLES
