"""F091-A2 (night 3) — a card that refuses a verb says exactly what to call instead.

The persona sent `grant` to a CLI hold (a question) and `answer "deny"` to an
approval; both came back 422 with text that said "approval-grants/{id}/answer"
literally. The kinds keep their own verbs (PRD-225), and each refusal now names
the real id and the call to make — a question's options included, and for an
approval the refusal the caller was reaching for first.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from api import approval_grants as ag


class _Db:
    def __init__(self, grant):
        self.grant = grant

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def first(self):
        return self.grant


def _refusal(call):
    with pytest.raises(HTTPException) as caught:
        asyncio.run(call)
    assert caught.value.status_code == 422
    assert "{id}" not in caught.value.detail
    return caught.value.detail


def test_grant_on_a_hold_names_the_answer_call_and_its_options():
    hold = NS(id=612, kind="question", status="pending", options=["Allow", "Deny"])
    detail = _refusal(ag.grant_approval(612, ctx=NS(workspace_id="ws"), db=_Db(hold)))
    assert detail.startswith("Grant 612 is a question, not an approval — answer it: "
                             'POST /api/v1/approval-grants/612/answer with {"option": "<one of: Allow, Deny>"}')


def test_a_free_text_question_asks_for_answer_text():
    assert ag.question_not_approval(NS(id=7, options=None)).startswith(
        'Grant 7 is a question, not an approval — answer it: POST /api/v1/approval-grants/7/answer '
        'with {"answer_text": "<your answer>"}')


def test_answer_deny_on_an_approval_leads_with_the_deny_call():
    delete = NS(id=600, kind="approval", status="pending", options=None)
    detail = _refusal(ag.answer_question(600, body=ag.AnswerRequest(answer_text="deny"),
                                         ctx=NS(workspace_id="ws"), db=_Db(delete)))
    assert detail.startswith("Grant 600 is an approval, not a question — "
                             "to refuse it: POST /api/v1/approval-grants/600/deny; "
                             "to approve it: POST /api/v1/approval-grants/600/grant.")
    assert ag.approval_not_question(delete, "yes, go ahead").split(" — ")[1].startswith("to approve it")
