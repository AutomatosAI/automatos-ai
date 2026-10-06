"""PRD-248 tuning (6 Oct): tool_rerank asks one four-level Score per candidate.

A rerank orders candidates against each other, and a Noul's P(yes) is calibrated
for its own proposition only (the invariants note): 0.7 on one action and 0.6 on
another say nothing about which helps more. A Score puts every candidate on the
same four described levels. The cut reads the score scaled to 0..1, so the
``rerank_min_probability`` dial keeps its meaning (0.5 sits between "related, for
other work" and "helps with part of the request"); the row keeps its
``probabilities`` field.
"""
from __future__ import annotations

from core.llm.decisions import rerank as rr
from core.llm.decisions.questions import DecisionAnswer, DecisionResult


def _result(**answers: DecisionAnswer) -> DecisionResult:
    return DecisionResult(answers=answers, provider="fake", model="fake-1", latency_ms=200)


def _level(score: float) -> DecisionAnswer:
    return DecisionAnswer(type="score", score=score, confidence=0.8)


def test_every_candidate_is_scored_on_the_same_four_levels():
    questions = rr.build_questions([("send_email", "Send an email"), ("list_events", "")])
    assert {q.to_wire()["type"] for q in questions.values()} == {"score"}
    assert all(list(q.criteria) == list(rr.HELP_LEVELS) for q in questions.values())
    assert len(rr.HELP_LEVELS) == 4


def test_a_score_scales_to_a_probability_and_a_noul_still_reads():
    assert rr.help_probability(_level(3.0)) == 1.0
    assert rr.help_probability(_level(1.5)) == 0.5
    assert rr.help_probability(_level(0.0)) == 0.0
    assert rr.help_probability(DecisionAnswer(type="noul", noul=0.7)) == 0.7
    assert rr.help_probability(None) is None


def test_the_cut_orders_by_score_and_the_floor_keeps_its_meaning():
    result = _result(send_email=_level(2.9), list_events=_level(1.2), search_docs=_level(2.0), web=_level(0.3))
    cut = rr.apply_rerank(result, ["web", "list_events", "search_docs", "send_email"], top_k=10,
                          min_probability=0.5, min_keep=0)
    assert cut.kept == ["send_email", "search_docs"]
    assert cut.probabilities["list_events"] == 0.4 and cut.nothing_fits is False
