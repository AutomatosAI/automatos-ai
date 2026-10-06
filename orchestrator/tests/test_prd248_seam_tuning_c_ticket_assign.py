"""PRD-248 tuning (6 Oct): ticket_assign reads the owner's words and scores each candidate.

Night 5: 15 of 44 mission steps went wrong for both the matcher and Jev, because
both read the planner's ``role_wanted: writer`` literally. The state now carries the
owner's own words for the step (the mission's goal; the step text when there is
none) beside the role. And the 9-to-30-way Choice over the roster is one four-level
fit Score per candidate: the vendor's figures are 40% right for a 12-way choice and
91% decomposed. The pick is the best fit; below "its role suits this step" it is
``none``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from core.llm.decisions import judgements as J
from core.llm.decisions.questions import DecisionAnswer, DecisionResult

WS = "11111111-1111-1111-1111-111111111111"
GOAL = "Write a warm, plain-English email to our 40 café customers announcing the Christmas blend."


def _result(**answers: DecisionAnswer) -> DecisionResult:
    return DecisionResult(answers=answers, provider="fake", model="fake-1", latency_ms=200)


def _fit(level: float) -> DecisionAnswer:
    return DecisionAnswer(type="score", score=level, confidence=0.8)


def test_the_owners_words_sit_beside_the_role_wanted():
    state = J.assignment_state(title="Draft the email", description="Write the announcement copy.",
                               role="writer", required_tools=[], mission_brief=GOAL)
    assert state["owners_words"] == GOAL and state["role_wanted"] == "writer"
    assert state["step"] == "Write the announcement copy."
    fallback = J.assignment_state(title="t", description="Write the copy.", role="writer", required_tools=[])
    assert fallback["owners_words"] == "Write the copy."


def test_a_thirty_agent_roster_is_thirty_small_questions_not_one_big_choice():
    roster = [(f"Agent {i}", f"Role {i}") for i in range(30)]
    questions = J.assignment_questions(roster)
    assert len(questions) == 30
    assert {q.to_wire()["type"] for q in questions.values()} == {"score"}
    assert all(len(q.criteria) == 4 for q in questions.values())
    assert not any("not" in level.lower().split() or "unless" in level.lower() for level in J.FIT_LEVELS)


def test_the_pick_is_the_best_fit_and_the_roster_order_breaks_a_tie():
    names = ["Jim", "Atlas", "Nova"]
    pick, top, fits = J.assignment_pick(_result(Jim=_fit(2.0), Atlas=_fit(2.6), Nova=_fit(2.6)), names)
    assert pick == "Atlas" and round(top, 4) == round(2.6 / 3, 4) and fits["Nova"] == 2.6
    assert J.assignment_pick(_result(Jim=_fit(1.4)), names)[0] == "none"
    assert J.assignment_pick(_result(Jim=_fit(1.5)), names)[0] == "Jim"
    assert J.assignment_pick(_result(), names) is None


def test_the_matcher_hands_the_missions_goal_to_the_shadow(monkeypatch):
    from modules.coordination import agent_matcher as am

    seen: List[Any] = []

    class _On:
        def dials(self):
            return SimpleNamespace(ticket_assign_mode="shadow")

        def shadow(self, coro, purpose="shadow"):
            seen.append(coro.cr_frame.f_locals)
            coro.close()
            return True

    class _Db:
        def __init__(self):
            self.asked = []

        def query(self, column):
            self.asked.append(column)
            return self

        def filter(self, *criteria):
            return self

        def scalar(self):
            return GOAL

    import core.llm.decisions as pkg

    monkeypatch.setattr(pkg, "get_decision_engine", lambda: _On())
    ranked = [am.MatchResult(agent_id=7, agent_name="Jim", total_score=0.9, tool_coverage=1, skill_match=1,
                             model_fit=1, availability=1, history=0)]
    task = SimpleNamespace(id=41, run_id="run-1", title="Draft the email", description="Write the copy.")
    agents = [SimpleNamespace(id=7, name="Jim", description="Writes", workspace_id=WS)]

    am._shadow_assignment(task, agents, ranked, "writer", [], db=_Db())

    (frame,) = seen
    assert frame["mission_brief"] == GOAL
    assert am._mission_goal(None, task) is None
    assert am._mission_goal(_Db(), SimpleNamespace(run_id=None)) is None
