"""F037 (night 1) — the owner's answer reaches the agent whole, or says it was cut.

The answer to a session's question was cut at exactly 2,000 characters, mid-word,
with success reported to the owner and nothing said to the agent. Three of 28
answers lost ~1,050 characters (#174: the agent quoted the cut at character
1,994 of 2,349). An answer is now kept to 16,000 characters, and past that the
text the next session reads ends with a note saying where the rest is.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from services import cli_host_service as svc


def _asked(grant_id=42, question="Which payment term?"):
    return svc.record_session_ask({}, grant_id=grant_id, question=question)


def test_night_ones_2349_character_answer_is_kept_whole():
    answer = ("one rule out of tonight that I want you to apply to anything you write: a comp" + "x" * 2300)[:2349]
    ref = svc.record_session_answer(_asked(), grant_id=42, answer=answer)
    assert svc.session_asks(ref)[0]["answer"] == answer


def test_an_answer_past_the_limit_is_cut_with_a_note_the_session_reads():
    answer = "a" * (svc.MAX_ASK_ANSWER_KEPT + 500)
    ref = svc.record_session_answer(_asked(), grant_id=42, answer=answer)
    kept = svc.session_asks(ref)[0]["answer"]
    assert kept.startswith("a" * svc.MAX_ASK_ANSWER_KEPT)
    assert kept.endswith("[The owner's answer is cut here at 16,000 characters; 500 more are on question #42.]")
    fold_in = svc._answers_fold_in(NS(runtime_ref=ref))
    assert "500 more are on question #42" in fold_in


def test_a_long_question_says_so_too():
    ref = _asked(question="q" * (svc.MAX_ASK_QUESTION_KEPT + 7))
    assert svc.session_asks(ref)[0]["question"].endswith(
        "[The question is cut here at 1,000 characters; 7 more are on question #42.]")
