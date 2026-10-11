"""PRD-256 FX-014 (M3, F387): the Tier-3 verdict is read field by field; a bad field loses only itself.

Night 12: the classifier wrote ``"complexity": "assign"``; ``Complexity("assign")`` raised inside
``AutoBrain._llm_classify`` and the whole verdict (action, target agent, tool hints) was dropped for
MOLECULE/RESPOND, while the log said "falling back to ATOM". 33 hand-offs ("Get OPS to…", "Ask
RESEARCHER…") lost their lane before the credit ran out.

These run ``_llm_classify`` with the classifier's reply faked (no model is called) over a roster read
from a faked session, and ``verdict_parser`` on its own.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from consumers.chatbot import auto
from consumers.chatbot.auto import Action, AutoBrain, Complexity, build_assessment_prompt
from consumers.chatbot.verdict_parser import (
    VerdictUnreadable, lost_the_lane, read_action, read_complexity, read_verdict,
)

OPS = NS(id=267, name="OPS", job_title="Operations Manager")
RESEARCHER = NS(id=57, name="RESEARCHER", job_title="Market researcher")
GET_OPS = "Get OPS to reorder the green stock under 50 kg."
NIGHT_12 = {"complexity": "assign", "action": "assign", "target_agent": "OPS"}
AUTO_LOG = "consumers.chatbot.auto"


def _classified(reply, message=GET_OPS, roster=(OPS, RESEARCHER)):
    """AutoBrain's Tier 3 over ``reply`` (a dict is sent as JSON; an exception is raised by the call)."""
    db = MagicMock()
    db.query.return_value.filter.return_value.limit.return_value.all.return_value = list(roster)
    brain = AutoBrain(db=db, workspace_id="ws-c1")
    brain._redis = None

    class _Classifier:
        async def generate_response(self, messages):
            if isinstance(reply, Exception):
                raise reply
            return NS(content=json.dumps(reply) if isinstance(reply, dict) else reply)

    with patch("core.llm.create_llm_manager", return_value=_Classifier()):
        return asyncio.run(brain._llm_classify(message, 3))


# ── night 12's verdict ────────────────────────────────────────────────────────

def test_night_12s_lane_word_in_the_complexity_field_is_an_assign_verdict_for_ops():
    verdict = _classified(NIGHT_12)

    assert verdict.action == Action.ASSIGN
    assert (verdict.target_agent_id, verdict.target_agent_name) == (267, "OPS")
    assert verdict.complexity == Complexity.MOLECULE and verdict.confidence == 0.85


def test_the_lane_word_sent_only_as_the_complexity_is_read_as_the_action():
    verdict = _classified({"complexity": "assign", "target_agent": "RESEARCHER"},
                          message="Ask RESEARCHER to price the wholesale offer.")

    assert (verdict.action, verdict.target_agent_id) == (Action.ASSIGN, 57)


@pytest.mark.parametrize("garbage", ["assign", "medium", "", None, 7])
def test_a_garbage_complexity_keeps_the_action_the_agent_and_the_tools(garbage):
    verdict = _classified({**NIGHT_12, "complexity": garbage, "tool_hints": ["platform"], "needs_memory": True})

    assert (verdict.action, verdict.target_agent_name) == (Action.ASSIGN, "OPS")
    assert verdict.tool_hints == ["platform"] and verdict.needs_memory is True
    assert verdict.complexity == Complexity.MOLECULE


def test_a_mission_with_no_level_is_an_organ():
    verdict = _classified({"complexity": "big", "action": "mission", "needs_multi_agent": True})

    assert (verdict.action, verdict.complexity, verdict.needs_multi_agent) == (Action.MISSION, Complexity.ORGAN, True)


def test_a_respond_with_no_level_keeps_the_tools():
    verdict = _classified({"complexity": "respond", "action": "respond", "tool_hints": ["email"]})

    assert (verdict.action, verdict.complexity, verdict.tool_hints) == (Action.RESPOND, Complexity.MOLECULE, ["email"])


def test_an_unknown_action_is_auto_answering_with_the_rest_of_the_verdict():
    verdict = _classified({"complexity": "cell", "action": "escalate", "tool_hints": ["platform"]})

    assert (verdict.action, verdict.complexity, verdict.tool_hints) == (Action.RESPOND, Complexity.CELL, ["platform"])


def test_a_good_verdict_reads_as_before():
    verdict = _classified({"complexity": "atom", "action": "respond", "reasoning": "a greeting"}, message="Morning")

    assert (verdict.complexity, verdict.action, verdict.reasoning) == (Complexity.ATOM, Action.RESPOND, "a greeting")
    assert verdict.target_agent_id is None


# ── a verdict that can't be read: the tools are kept and the log says so ──────

@pytest.mark.parametrize("reply", ["", "I think this is an assign.", '{"complexity": "assign", "action":',
                                   RuntimeError("402: credit ran out")])
def test_no_verdict_is_molecule_respond_and_the_log_says_tools_kept_lane_lost(reply, caplog):
    with caplog.at_level(logging.WARNING, logger=AUTO_LOG):
        verdict = _classified(reply)

    assert (verdict.complexity, verdict.action, verdict.confidence) == (Complexity.MOLECULE, Action.RESPOND, 0.5)
    assert verdict.target_agent_id is None and verdict.tool_hints == []
    assert "kept tools, lost the lane" in caplog.text
    assert "ATOM" not in caplog.text


def test_the_fallback_never_says_atom_anywhere_in_autobrain():
    source = inspect.getsource(auto)

    assert "falling back to ATOM" not in source
    assert "Complexity(data.get(" not in source


# ── the parser on its own ─────────────────────────────────────────────────────

def test_the_action_is_read_before_the_complexity():
    assert read_action({"complexity": "assign", "action": "respond"}) == Action.RESPOND
    assert read_action({"complexity": "mission"}) == Action.MISSION
    assert read_action({"action": "workflow"}) == Action.MISSION       # the deprecated alias (PRD-125)
    assert read_action({"action": "Delegate"}) == Action.DELEGATE       # the lane converts it (US-010)


def test_a_complexity_defaults_to_its_lanes_own():
    assert read_complexity({"complexity": "Organ"}, Action.RESPOND) == Complexity.ORGAN
    assert read_complexity({"complexity": "assign"}, Action.ASSIGN) == Complexity.MOLECULE
    assert read_complexity({}, Action.MISSION) == Complexity.ORGAN


def test_the_verdict_is_read_from_prose_around_the_json_and_bad_hints_are_dropped():
    verdict = read_verdict('Here you go: {"action": "assign", "target_agent": " OPS ", "tool_hints": "platform"}')

    assert (verdict.action, verdict.target_agent, verdict.tool_hints) == (Action.ASSIGN, "OPS", [])


@pytest.mark.parametrize("content", ["", "no json", "{not json}", "[1, 2]"])
def test_unreadable_replies_raise_for_the_caller_to_log(content):
    with pytest.raises(VerdictUnreadable):
        read_verdict(content)


def test_the_lost_lane_keeps_the_tools():
    lost = lost_the_lane()

    assert (lost.complexity, lost.action, lost.confidence) == (Complexity.MOLECULE, Action.RESPOND, 0.5)


# ── the rubric names the two fields' values once more ─────────────────────────

def test_the_output_format_names_each_fields_allowed_values():
    prompt = build_assessment_prompt(GET_OPS, 3)

    assert "complexity is ONLY one of atom|molecule|cell|organ|organism" in prompt
    assert "action ONLY one of respond|assign|mission" in prompt
    assert "a lane word never goes in complexity" in prompt
