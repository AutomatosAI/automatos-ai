"""PRD-248 tuning (6 Oct): Auto's classifier questions, split and reworded.

* The roster in the state is each agent's name and role only: the full descriptions
  drew the model away from the complexity and action questions.
* ``target_agent`` was one Choice over 9 to 31 agents with an instruction that said
  "Pick none unless ...". It is one Score per agent now, worded as what the message
  does ("names this agent", "names this agent's role"), never as what it lacks.
* The tool domain was one 8-way Choice; it is one yes/no per domain, so a message can
  need two (email and calendar).

The verdict and the field-by-field comparison keep their shapes, so the scorer and
AutoBrain read them as before.
"""
from __future__ import annotations

from types import SimpleNamespace

from consumers.chatbot import auto_decisions as AD
from core.llm.decisions.questions import DecisionAnswer, DecisionResult


def _result(**answers: DecisionAnswer) -> DecisionResult:
    return DecisionResult(answers=answers, provider="fake", model="fake-1", latency_ms=200)


def _choice(option: str, p: float = 0.9) -> DecisionAnswer:
    return DecisionAnswer(type="choice", choice=option, probabilities={option: p}, confidence=p)


def _noul(p: float) -> DecisionAnswer:
    return DecisionAnswer(type="noul", noul=p)


def _points(level: float) -> DecisionAnswer:
    return DecisionAnswer(type="score", score=level, confidence=0.8)


def test_the_roster_is_names_and_roles_only():
    agents = [
        SimpleNamespace(name="Jim", role="writer", description="Drafts board packs and long reports", configuration={"runtime": "cli"}),
        SimpleNamespace(name="Atlas", role=None, description="Plans"),
        SimpleNamespace(name="", role="x"),
    ]
    assert AD.roster_entries(agents) == [{"name": "Jim", "role": "writer"}, {"name": "Atlas"}]


def test_each_agent_gets_its_own_positively_worded_score():
    questions = AD.target_questions([{"name": "Jim", "role": "writer"}, {"name": "Atlas"}])
    assert list(questions) == ["agent:Jim", "agent:Atlas"]
    jim = questions["agent:Jim"]
    assert jim.to_wire()["type"] == "score" and list(jim.criteria) == list(AD.TARGET_LEVELS)
    assert "Its role: writer" in jim.instructions
    words = " ".join([jim.instructions, *AD.TARGET_LEVELS]).lower().split()
    assert "unless" not in words and "not" not in words and "no" not in words


def test_the_pick_needs_the_message_to_name_the_agent_or_its_role():
    assert AD.target_pick(_result(**{"agent:Jim": _points(2.9), "agent:Atlas": _points(1.0)})) == "Jim"
    assert AD.target_pick(_result(**{"agent:Jim": _points(1.4)})) == "none"  # suits its role, never named
    assert AD.target_pick(_result(complexity=_choice("atom"))) is None


def test_a_message_can_need_two_tool_domains():
    assert set(AD.domain_questions()) == {f"domain:{d}" for d in AD.DOMAIN_CRITERIA if d != AD.NO_DOMAIN}
    both = _result(**{"domain:email": _noul(0.8), "domain:calendar": _noul(0.7), "domain:web": _noul(0.2)})
    assert AD.domain_hints(both) == ["email", "calendar"]
    assert AD.domain_hints(_result(**{"domain:web": _noul(0.1)})) == []
    assert AD.domain_hints(_result()) is None


def test_the_verdict_and_the_comparison_keep_their_shapes():
    result = _result(
        complexity=_choice("molecule"), action=_choice("assign"), needs_memory=_noul(0.2), needs_multi_agent=_noul(0.1),
        **{"domain:email": _noul(0.8), "domain:calendar": _noul(0.7), "agent:Jim": _points(3.0)},
    )
    verdict = AD.verdict_from_result(result, min_confidence=0.5)
    assert verdict["tool_hints"] == ["email", "calendar"] and verdict["target_agent_name"] == "Jim"
    tier = {"complexity": "molecule", "action": "assign", "tool_hints": ["calendar"], "needs_memory": False,
            "needs_multi_agent": False, "target_agent_name": "jim"}
    assert AD.compare(tier, result) == {
        "complexity": True, "action": True, "needs_memory": True, "needs_multi_agent": True,
        "tool_domain": True, "target_agent": True,
    }
    no_tools = {**tier, "tool_hints": [], "target_agent_name": None}
    assert AD.compare(no_tools, result)["tool_domain"] is False and AD.compare(no_tools, result)["target_agent"] is False
    assert AD.compare(tier, _result())["tool_domain"] is None and AD.compare(tier, _result())["target_agent"] is None
