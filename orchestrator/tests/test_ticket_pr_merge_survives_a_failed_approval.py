"""Ticket PR fields: a merge that happened is never lost, whatever fails after it.

A person's Approve merges the ticket's pull request after the approval's own action and before its
result is written. If a later step then failed, the approval's failure path rolled everything back
and reopened Review: the pull request stayed merged on GitHub while the ticket said nothing of it.
Now "Merged <url>" and the merge's mark are committed in a transaction of their own the moment
GitHub says it merged (``ticket_pr_merge._keep_the_merge``), and the failure path keeps the ticket
Done when this approval merged it (``api/board_card_moves.its_merge_stands``).

These run on the real schema in a session whose commits are savepoints, so a rollback undoes only
what came after the last commit, as it does in production.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

from config import config
from core.models.core import BoardTask
from services import ticket_pr_merge as tpm
from services import ticket_pull_request as tpr
from tests.test_prd256_done_needs_artifact import QUESTION, _body, _person
from tests.test_prd256_done_needs_artifact import board, quiet  # noqa: F401 — the fixtures, as that suite builds them
from tests.test_ticket_pr_merge_on_approve import BRANCH, GREEN_RUNS, MERGE_SHA, REPO, URL, _GitHub

LATER_STEP_FAILED = "the database went away while the approval was written"


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, with each commit a savepoint (``create_savepoint``): a
    rollback goes back to the last commit, not to the start of the test."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    outer = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    outer.rollback()
    connection.close()


@pytest.fixture
def approving(db_session, board, quiet, monkeypatch):
    """The setting on, the repository listed, GitHub connected and scripted; the merge's own short
    transaction is this session, committed as ``core.database.database.get_db_session`` commits."""
    from core.database import database

    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR", True)
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR_REPOS", REPO)
    monkeypatch.setattr(tpm, "github_connected", lambda db, ws: True)
    state = NS(github=_GitHub())
    monkeypatch.setattr(tpm, "_executor", lambda db: state.github)

    @contextmanager
    def its_own_transaction():
        yield db_session
        db_session.commit()

    monkeypatch.setattr(database, "get_db_session", its_own_transaction)

    def a_later_step_fails(*args, **kwargs):
        raise RuntimeError(LATER_STEP_FAILED)

    monkeypatch.setattr("api.board_tasks.record_approval", a_later_step_fails)
    task = board.card(QUESTION, status="in_progress")
    tpr.write_pull_request(db_session, task_id=task.id, workspace_id=board.ws, by="CODER", note="",
                           fields=tpr.pull_request_fields({"pull_request_url": URL, "branch": BRANCH}))
    db_session.refresh(task)
    task.status = "review"
    db_session.commit()
    return NS(state=state, task=task, ws=board.ws)


def _approve_and_fail(db, approving):
    from api.board_tasks import approve_task

    with pytest.raises(RuntimeError, match=LATER_STEP_FAILED):
        asyncio.run(approve_task(approving.task.id, _body({}), ctx=_person(approving.ws), db=db))
    db.expire_all()
    return db.get(BoardTask, approving.task.id)


def test_a_merge_stands_and_the_ticket_stays_done_when_a_later_step_of_the_approval_fails(db_session, approving):
    task = _approve_and_fail(db_session, approving)
    assert approving.state.github.actions()[-1] == tpm.MERGE_PULL_REQUEST
    assert task.status == "done"                                                # not reopened
    notes = [(n["by"], n["note"]) for n in task.runtime_ref["session_notes"]]
    assert ("you", f"Merged {URL}") in notes                                    # the note survived the rollback
    assert task.runtime_ref["pull_request"]["merged"]["sha"] == MERGE_SHA


def test_without_a_merge_a_failed_approval_reopens_review_as_before(db_session, approving):
    approving.state.github = _GitHub(runs={**GREEN_RUNS, "check_runs": [
        {"name": "backend", "status": "completed", "conclusion": "failure"}], "total_count": 1})
    task = _approve_and_fail(db_session, approving)
    assert tpm.MERGE_PULL_REQUEST not in approving.state.github.actions()
    assert task.status == "review" and "merged" not in task.runtime_ref["pull_request"]


def test_an_earlier_approvals_merge_does_not_keep_this_one_done(db_session, approving):
    """The mark names the approval that merged: another approval's failure path is not kept Done by it."""
    from api.board_card_moves import its_merge_stands
    from datetime import datetime, timezone

    _approve_and_fail(db_session, approving)
    later = datetime.now(timezone.utc)
    assert its_merge_stands(db_session, approving.task.id, approving.ws, decided_at=later) is False
