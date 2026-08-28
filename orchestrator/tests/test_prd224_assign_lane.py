"""PRD-224 US-004 -- Action.ASSIGN in AutoBrain (the middle lane).

Pure, LLM-free coverage of the ASSIGN routing lane:

- the enum + action parser accept ASSIGN; WORKFLOW stays the deprecated alias
  for MISSION and is otherwise untouched;
- the Tier-3 assessment prompt carries the three-lane rubric with the named-agent
  and defer-phrasing signals + the target_agent JSON field;
- roster name-matching resolves target_agent_id (exact, then contains);
- api/chat.py's ASSIGN dispatch (apply_assign_bias) steers to the three ticket
  tools and attaches the manager directive -- start-now vs queued vs ask-in-thread
  -- and the ASSIGN branch is checked BEFORE the platform-hint reroute;
- the ASSIGN directive is injected into the system prompt via the existing seam.

No live model is called anywhere (the classifier LLM is never invoked here).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from consumers.chatbot.auto import (
    ASSIGN_TOOL_HINTS,
    Action,
    AutoBrain,
    Complexity,
    ComplexityAssessment,
    apply_assign_bias,
    build_assessment_prompt,
    build_assign_directive,
    is_deferred_phrasing,
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _assessment(**over):
    base = dict(complexity=Complexity.MOLECULE, action=Action.ASSIGN, reasoning="r")
    base.update(over)
    return ComplexityAssessment(**base)


# ---------------------------------------------------------------------------
# AC1 -- enum + parser
# ---------------------------------------------------------------------------


def test_action_assign_exists():
    assert Action.ASSIGN.value == "assign"


def test_workflow_untouched():
    assert Action.WORKFLOW.value == "workflow"


@pytest.mark.parametrize("raw, expected", [
    ("assign", Action.ASSIGN),
    ("respond", Action.RESPOND),
    ("delegate", Action.DELEGATE),
    ("mission", Action.MISSION),
    ("workflow", Action.MISSION),   # deprecated alias → mission (PRD-125)
    ("ASSIGN", Action.ASSIGN),      # case-insensitive
    (" assign ", Action.ASSIGN),    # whitespace-tolerant
])
def test_normalize_action_accepts(raw, expected):
    assert AutoBrain._normalize_action(raw) == expected


def test_normalize_action_unknown_raises():
    """An unknown action raises (the caller's LLM-failure fallback owns it) —
    the previous behaviour is preserved, not silently coerced."""
    with pytest.raises(ValueError):
        AutoBrain._normalize_action("banana")


def test_assign_survives_cache_round_trip():
    """to_dict carries the resolved ASSIGN target so a cached 'have <agent> do X'
    keeps its routing (the _cache_lookup reconstruction path)."""
    a = _assessment(target_agent_id=7, target_agent_name="Jim")
    d = a.to_dict()
    assert d["action"] == "assign"
    assert d["target_agent_id"] == 7 and d["target_agent_name"] == "Jim"
    rebuilt = ComplexityAssessment(
        complexity=Complexity(d["complexity"]),
        action=AutoBrain._normalize_action(d["action"]),
        reasoning="cached",
        target_agent_id=d.get("target_agent_id"),
        target_agent_name=d.get("target_agent_name"),
    )
    assert rebuilt.action == Action.ASSIGN
    assert rebuilt.target_agent_id == 7 and rebuilt.target_agent_name == "Jim"


# ---------------------------------------------------------------------------
# AC2 -- the assessment prompt's three-lane rubric
# ---------------------------------------------------------------------------


def test_prompt_has_three_lane_rubric():
    p = build_assessment_prompt("do a thing", 0, "")
    # The three lanes are named and described.
    assert "delegate" in p and "assign" in p and "mission" in p
    assert "Routing lanes" in p
    assert "answers THIS conversation" in p
    assert "OFF-THREAD" in p or "off-thread" in p


def test_prompt_has_named_agent_signal():
    p = build_assessment_prompt("x", 0, "")
    assert "my accountant agent" in p
    assert "role possessive" in p


def test_prompt_has_defer_phrasing_signals():
    p = build_assessment_prompt("x", 0, "")
    for phrase in ("queue it", "later", "when free"):
        assert phrase in p, phrase


def test_prompt_json_schema_offers_assign_and_target_agent():
    p = build_assessment_prompt("x", 0, "")
    assert '"action": "respond|delegate|assign|mission"' in p
    assert "target_agent" in p


def test_prompt_embeds_platform_context():
    p = build_assessment_prompt("x", 0, "\n## Available agents (for routing)\n- Jim (dev)\n")
    assert "Jim (dev)" in p


# ---------------------------------------------------------------------------
# roster name-matching → target_agent_id (assessment stubbed)
# ---------------------------------------------------------------------------


def _brain():
    return AutoBrain(None, "00000000-0000-0000-0000-000000000001")


def _agent(id, name):
    return SimpleNamespace(id=id, name=name)


def test_match_roster_exact_name():
    roster = [_agent(3, "Researcher"), _agent(7, "Accountant")]
    assert _brain()._match_roster_agent("accountant", roster) == (7, "Accountant")


def test_match_roster_contains_possessive():
    roster = [_agent(7, "Accountant")]
    # "my accountant agent" contains the roster name → resolves.
    assert _brain()._match_roster_agent("my accountant agent", roster) == (7, "Accountant")


def test_match_roster_no_match_is_ask_signal():
    roster = [_agent(7, "Accountant")]
    assert _brain()._match_roster_agent("zephyr", roster) == (None, None)


def test_match_roster_empty_name():
    assert _brain()._match_roster_agent("", [_agent(7, "Accountant")]) == (None, None)


# ---------------------------------------------------------------------------
# P224-RVW-1 -- an ambiguous within-workspace contains-match must ASK, never
# silently pick the first row (which was Postgres-order-dependent).
# ---------------------------------------------------------------------------


def test_match_roster_ambiguous_contains_asks_not_guesses():
    """'Jim' contains-matches BOTH 'Jim Whitfield' and 'Jimmy Cross'
    ('jim' is a prefix of 'jimmy'), so it is ambiguous → (None, None)."""
    roster = [_agent(3, "Jim Whitfield"), _agent(7, "Jimmy Cross")]
    assert _brain()._match_roster_agent("Jim", roster) == (None, None)


def test_match_roster_ambiguous_is_order_independent():
    """The ambiguous verdict must not depend on roster (Postgres row) order."""
    a, b = _agent(3, "Jim Whitfield"), _agent(7, "Jimmy Cross")
    assert _brain()._match_roster_agent("Jim", [a, b]) == (None, None)
    assert _brain()._match_roster_agent("Jim", [b, a]) == (None, None)


def test_match_roster_two_short_names_are_ambiguous():
    """Short names ('Ops','AI') substring-match a longer target; 2+ hits ask."""
    roster = [_agent(3, "Ops"), _agent(7, "AI")]
    # target 'AI Ops' contains both 'ops' and 'ai' → ambiguous.
    assert _brain()._match_roster_agent("AI Ops", roster) == (None, None)


def test_match_roster_single_contains_still_resolves_amid_others():
    """With only ONE distinct contains-match in a multi-agent roster, resolve —
    the ambiguity guard must not block a genuinely unique match."""
    roster = [_agent(3, "Researcher"), _agent(7, "Accountant")]
    assert _brain()._match_roster_agent("my accountant agent", roster) == (7, "Accountant")


def test_ambiguous_match_drives_ask_in_thread_directive():
    """End-to-end: an ambiguous name leaves target_agent_id None, so
    apply_assign_bias emits the ask-in-thread directive (never auto-pick)."""
    roster = [_agent(3, "Jim Whitfield"), _agent(7, "Jimmy Cross")]
    agent_id, agent_name = _brain()._match_roster_agent("Jim", roster)
    a = _assessment(target_agent_id=agent_id, target_agent_name=agent_name)
    apply_assign_bias(a, "have Jim send the invoice reminder now")
    assert "platform_list_agents" in a.context_directive
    assert "Do NOT guess or auto-pick" in a.context_directive


# ---------------------------------------------------------------------------
# AC3 -- api/chat.py ASSIGN dispatch (apply_assign_bias, assessment stubbed)
# ---------------------------------------------------------------------------


def test_assign_bias_named_agent_starts_now():
    a = _assessment(target_agent_id=7, target_agent_name="Jim")
    deferred = apply_assign_bias(a, "have Jim chase the overdue invoices")
    assert deferred is False
    for tool in ASSIGN_TOOL_HINTS:
        assert tool in a.tool_hints
    assert a.context_directive is not None
    assert "Jim" in a.context_directive
    assert "in_progress" in a.context_directive           # start immediately
    assert "file this as a board ticket" in a.context_directive


def test_assign_bias_no_name_takes_ask_path():
    a = _assessment(target_agent_id=None, target_agent_name="ghost")
    apply_assign_bias(a, "get someone to chase the invoices")
    assert "platform_list_agents" in a.context_directive   # ask, never auto-pick
    assert "confirm the agent first" in a.context_directive
    assert "Do NOT guess or auto-pick" in a.context_directive


def test_assign_bias_deferred_phrasing_queues():
    a = _assessment(target_agent_id=7, target_agent_name="Jim")
    deferred = apply_assign_bias(a, "get Jim to draft the board pack, no rush")
    assert deferred is True
    assert "queued" in a.context_directive
    assert "in_progress" not in a.context_directive        # NOT started


def test_assign_bias_merges_existing_hints():
    a = _assessment(target_agent_id=7, target_agent_name="Jim", tool_hints=["platform"])
    apply_assign_bias(a, "have Jim do it")
    assert "platform" in a.tool_hints                       # existing hint kept
    for tool in ASSIGN_TOOL_HINTS:
        assert tool in a.tool_hints


# ---------------------------------------------------------------------------
# P224-RVW-3 -- the directive must not SCRIPT supervision. AUTO_TICKET_WATCH can
# be off (or the watch fails to attach), so create_board_task may return
# supervised=False; the confirmation reports the result's field, never a
# hardcoded 'supervised — you'll report back here'.
# ---------------------------------------------------------------------------


def test_assign_directive_defers_supervision_to_create_result():
    """The resolved-agent directive no longer asserts supervision unconditionally;
    it tells Auto to report the platform_create_task result's own supervision
    field, so an AUTO_TICKET_WATCH-off ticket is not described as supervised."""
    d = build_assign_directive(target_agent_name="Jim", resolved=True, deferred=False)
    # The old unconditional claim is gone.
    assert "and supervised — you" not in d
    assert "supervised — you'll report back here" not in d
    # It defers to the create result's supervision field, honestly.
    assert "supervision" in d
    assert "platform_create_task result" in d
    assert "unless that field says so" in d


def test_assign_directive_queued_also_defers_supervision():
    """A deferred (queued) ASSIGN ticket is supervised-per-result too — the same
    honesty applies, and the queued path still carries no unconditional claim."""
    d = build_assign_directive(target_agent_name="Jim", resolved=True, deferred=True)
    assert "queued" in d
    assert "supervised — you'll report back here" not in d
    assert "'supervision' field" in d


@pytest.mark.parametrize("phrase, deferred", [
    ("do it now", False),
    ("start it immediately", False),
    ("queue it for tomorrow", True),
    ("when you're free", True),
    ("no rush at all", True),
])
def test_is_deferred_phrasing(phrase, deferred):
    assert is_deferred_phrasing(phrase) is deferred


# ---------------------------------------------------------------------------
# AC3/AC4 -- source guards: the branch is wired and ordered correctly
# ---------------------------------------------------------------------------


def test_chat_py_wires_assign_before_platform_reroute():
    """api/chat.py handles ASSIGN, biases via apply_assign_bias, routes to Auto,
    and the ASSIGN check precedes the RESPOND/_platform_hints branch so a
    'platform' tool_hint can't collapse an ASSIGN into RESPOND."""
    with open(os.path.join(_HERE, "api", "chat.py")) as f:
        src = f.read()
    assert "apply_assign_bias(complexity_assessment, message_text)" in src
    assign_at = src.index("complexity_assessment.action == Action.ASSIGN")
    respond_at = src.index("complexity_assessment.action == Action.RESPOND or _platform_hints")
    assert assign_at < respond_at, "ASSIGN must be checked before the platform-hint reroute"
    # ASSIGN is NOT routed to the Universal Router (that is DELEGATE/MISSION only).
    router_line = src.index("complexity_assessment.action in (Action.DELEGATE, Action.MISSION)")
    assert "Action.ASSIGN" not in src[router_line:router_line + 120]


def test_service_py_injects_the_assign_directive():
    """The ASSIGN directive reaches Auto's system prompt through the existing
    per-turn injection seam (same as the mission suggestion)."""
    with open(os.path.join(_HERE, "consumers", "chatbot", "service.py")) as f:
        src = f.read()
    assert 'getattr(complexity_assessment, "context_directive", None)' in src
    assert "_assign_directive" in src
