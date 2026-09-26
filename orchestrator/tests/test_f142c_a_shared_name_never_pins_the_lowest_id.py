"""F142 (c) — agents that share a name are told apart, and only a person pins one.

Night 4: c1 had three active WRITERs (58/306/309), COUNTINGHOUSEs and OPSes. The
planner's roster showed three identical "- WRITER: …" lines. An agent_role that
spelled a name pinned the LOWEST id, silently; the role WORD "writer" pinned
WRITER 58 too; and the card then said "Explicitly assigned … approval override"
although nobody had chosen. The tool result named the agent but not its id.

Now:
- the roster lines carry each agent's id (and slug);
- only an agent a person pinned (``input_context.pinned_agent_id``) overrides
  the ranking; a role string never does, so a role word pins nobody;
- an approval edit that names ONE agent (by id, slug or name) pins it; a name
  several active agents share is refused with their ids;
- "Explicitly assigned" appears only for a pin, and says a person chose;
- the tool result carries the matched agent's id.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest


def _agent(agent_id, name, *, slug=None, status="active", description="Writes the café notes"):
    return NS(id=agent_id, name=name, slug=slug, status=status, description=description, skills=[],
              tags=[], model_config={}, capabilities=[])


WRITERS = [_agent(58, "WRITER", slug="writer"), _agent(306, "WRITER", slug="writer-2"),
           _agent(309, "WRITER", slug="writer-3")]
OPS = _agent(267, "OPS", slug="ops", description="Books the visit slots")
ROSTER = WRITERS + [OPS]


def _rank(agents=ROSTER, **kwargs):
    from modules.coordination.agent_matcher import AgentMatcher

    return AgentMatcher._rank_with_context(
        agents=agents, agent_role=kwargs.pop("agent_role", None), required_tools=[], preferred_model=None,
        has_upstream=False, tool_map={}, busy_agent_ids=frozenset(), history_map={}, **kwargs)


# ── the roster ──────────────────────────────────────────────────────────────

def test_the_planners_roster_tells_same_named_agents_apart():
    from modules.coordination.planner import _render_agent_roster

    roster = _render_agent_roster(WRITERS)
    for agent in WRITERS:
        assert f"- WRITER (id {agent.id}, slug {agent.slug}): " in roster


# ── only a pin overrides ────────────────────────────────────────────────────

@pytest.mark.parametrize("role", ["writer", "WRITER", "OPS"])
def test_a_role_string_never_pins_an_agent(role):
    ranked = _rank(agent_role=role)
    assert not any(r.is_override for r in ranked)
    assert not any(r.reason.startswith("Explicitly assigned") for r in ranked)


def test_a_pin_overrides_and_says_a_person_chose():
    ranked = _rank(agent_role="writer", pinned_agent_id=306)
    assert (ranked[0].agent_id, ranked[0].is_override) == (306, True)
    assert ranked[0].reason.startswith("Explicitly assigned: a person chose agent 'WRITER'")
    assert [r.agent_id for r in ranked if r.is_override] == [306]


def test_a_pin_to_an_agent_that_is_off_or_elsewhere_is_ignored():
    asleep = _agent(310, "WRITER", status="inactive")
    assert not any(r.is_override for r in _rank(agents=ROSTER + [asleep], pinned_agent_id=310))
    assert not any(r.is_override for r in _rank(pinned_agent_id=999))


# ── naming an agent in an approval edit ─────────────────────────────────────

def test_a_shared_name_is_refused_with_its_ids():
    from modules.coordination.agent_matcher import resolve_named_agent

    agent, why = resolve_named_agent("Writer-team", ROSTER + [_agent(400, "Writer-team"), _agent(401, "writer-team")])
    assert agent is None
    assert why == "2 active agents are called 'Writer-team' (ids 400, 401). Name the one you mean by its id."


@pytest.mark.parametrize("named, expected", [
    ("306", 306), ("agent:309", 309), ("writer-2", 306), ("ops", None), ("OPS", None), ("writer", None),
    ("Countinghouse", None),
])
def test_one_named_agent_is_resolved_and_a_role_word_is_not(named, expected):
    from modules.coordination.agent_matcher import resolve_named_agent

    ops_named = [_agent(267, "Opsdesk", slug="ops-desk")] + WRITERS
    agent, why = resolve_named_agent(named, ops_named)
    assert why is None
    assert (agent.id if agent else None) == expected


def test_an_edit_naming_one_agent_pins_it_and_a_shared_name_is_refused():
    from services.coordinator_service import _pin_the_named_agent

    assert _pin_the_named_agent({"sequence_number": 2, "agent_id": 306}, ROSTER) == {
        "sequence_number": 2, "agent_role": "WRITER", "pinned_agent_id": 306}
    assert _pin_the_named_agent({"sequence_number": 1, "agent_role": "OPS"}, ROSTER) == {
        "sequence_number": 1, "agent_role": "OPS", "pinned_agent_id": 267}
    assert _pin_the_named_agent({"sequence_number": 1, "agent_role": "writer"}, ROSTER) == {
        "sequence_number": 1, "agent_role": "writer", "pinned_agent_id": None}
    twins = ROSTER + [_agent(311, "OPS")]
    with pytest.raises(ValueError, match=r"2 active agents are called 'OPS' \(ids 267, 311\)"):
        _pin_the_named_agent({"sequence_number": 1, "agent_role": "OPS"}, twins)
    with pytest.raises(ValueError, match="No active agent with id 999"):
        _pin_the_named_agent({"sequence_number": 1, "agent_id": 999}, ROSTER)


def test_the_pin_rides_the_task_and_the_plan_and_a_role_word_clears_it():
    from services.coordinator_service import apply_plan_task_edits

    task = NS(id="t1", sequence_number=1, agent_role="writer", title="Notes", description="d",
              input_context={"required_tools": []})
    plan = {"tasks": [{"sequence_number": 1, "agent_role": "writer", "title": "Notes", "description": "d"}]}
    plan, changed = apply_plan_task_edits([task], plan, [{"sequence_number": 1, "agent_role": "WRITER",
                                                          "pinned_agent_id": 306}])
    assert task.input_context == {"required_tools": [], "pinned_agent_id": 306}
    assert plan["tasks"][0]["pinned_agent_id"] == 306 and changed == 2
    plan, _ = apply_plan_task_edits([task], plan, [{"sequence_number": 1, "agent_role": "writer",
                                                    "pinned_agent_id": None}])
    assert task.input_context == {"required_tools": []} and plan["tasks"][0]["pinned_agent_id"] is None


# ── the tool result ─────────────────────────────────────────────────────────

def test_the_tool_result_carries_the_matched_agents_id():
    from modules.tools.discovery.handlers_missions import _plan_task_summary

    summary = _plan_task_summary([{"title": "Notes", "agent_role": "writer", "sequence_number": 1,
                                   "match_agent": "WRITER", "match_agent_id": 306}])
    assert summary[0]["match_agent_id"] == 306
