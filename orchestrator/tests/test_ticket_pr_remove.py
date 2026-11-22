"""Ticket PR fields: the owner takes a wrong pull request off a ticket.

``DELETE /api/v1/tasks/{task_id}/pull-request`` (``api/board_task_pull_request.py``) is a person's
only (the Done gate's rule), scoped to the caller's workspace, removes the key with one statement,
notes it on the ticket, and nothing puts it back: a stale whole-document write after it keeps it
removed (``core/models/kept_runtime_keys.py``). It is in the route manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

from api import board_task_pull_request as route
from core.models.core import BoardTask
from core.models.kept_runtime_keys import PULL_REQUEST_KEY
from services import ticket_pull_request as tpr
from tests.test_prd256_done_needs_artifact import QUESTION, _not_a_person, _person
from tests import test_prd256_done_needs_artifact as _gate_suite

board = _gate_suite.board  # the fixtures, as that suite builds them (assigned, not imported: tests take them by name)
quiet = _gate_suite.quiet

URL = "https://github.com/automatos-ai/automatos-ai/pull/1559"
ROUTE = {"method": "DELETE", "path": "/api/v1/tasks/{task_id}/pull-request"}


def _with_a_pull_request(db, board):
    task = board.card(QUESTION, status="in_progress")
    tpr.write_pull_request(db, task_id=task.id, workspace_id=board.ws, by="CODER", note="",
                           fields={"pull_request_url": URL})
    db.refresh(task)
    task.status = "review"
    db.flush()
    return task


def _ref(db, task):
    db.expire_all()
    return db.get(BoardTask, task.id).runtime_ref or {}


@pytest.mark.parametrize("lane", ["clerk", "device"])
def test_a_person_takes_the_pull_request_off_and_the_ticket_says_so(db_session, board, quiet, lane):
    task = _with_a_pull_request(db_session, board)
    out = route.remove_ticket_pull_request(task.id, ctx=_person(board.ws, lane), db=db_session)
    assert out == {"task_id": task.id, "removed": True, "url": URL, "message": route.REMOVED}
    ref = _ref(db_session, task)
    assert PULL_REQUEST_KEY not in ref
    assert [(n["by"], n["note"]) for n in ref["session_notes"]][-1] == ("you", tpr.CLEARED_NOTE.format(url=URL))


def test_a_stale_whole_document_write_after_it_never_puts_it_back(db_session, board, quiet):
    task = _with_a_pull_request(db_session, board)
    stale = dict(task.runtime_ref)                                # read before the owner took it off
    assert stale[PULL_REQUEST_KEY]["url"] == URL
    route.remove_ticket_pull_request(task.id, ctx=_person(board.ws), db=db_session)
    task = db_session.get(BoardTask, task.id)
    task.runtime_ref = {**stale, "events_seen": 7}                # the host's flush, from its old copy
    db_session.commit()
    ref = _ref(db_session, task)
    assert PULL_REQUEST_KEY not in ref and ref["events_seen"] == 7


@pytest.mark.parametrize("ctx", [
    lambda ws: _not_a_person(ws),                                         # no lane: Auto's in-process call
    lambda ws: NS(workspace_id=ws, user_id="2", auth_type="anonymous"),   # an agent's http_request to localhost
    lambda ws: NS(workspace_id=ws, user_id=None, auth_type="api_key"),
])
def test_only_a_person_may_take_it_off(db_session, board, quiet, ctx):
    task = _with_a_pull_request(db_session, board)
    with pytest.raises(HTTPException) as refused:
        route.remove_ticket_pull_request(task.id, ctx=ctx(board.ws), db=db_session)
    assert refused.value.status_code == 403 and refused.value.detail == route.ONLY_A_PERSON
    assert _ref(db_session, task)[PULL_REQUEST_KEY]["url"] == URL


def test_a_ticket_with_none_or_in_another_workspace_is_a_404_and_nothing_changes(db_session, seed_workspace, board,
                                                                                 quiet):
    task = _with_a_pull_request(db_session, board)
    elsewhere = UUID(seed_workspace())
    with pytest.raises(HTTPException) as other:
        route.remove_ticket_pull_request(task.id, ctx=_person(elsewhere), db=db_session)
    assert other.value.status_code == 404 and other.value.detail == route.NO_PULL_REQUEST
    assert _ref(db_session, task)[PULL_REQUEST_KEY]["url"] == URL

    bare = board.card(QUESTION)
    with pytest.raises(HTTPException) as none:
        route.remove_ticket_pull_request(bare.id, ctx=_person(board.ws), db=db_session)
    assert none.value.status_code == 404


def test_the_route_is_in_the_manifest_and_mounted():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json").read_text())
    assert ROUTE in manifest["routes"] and manifest["route_count"] == len(manifest["routes"])
    from router_manifest import MANIFEST_ROUTERS

    assert any(spec.module == "api.board_task_pull_request" for spec in MANIFEST_ROUTERS)
    paths = {(m, r.path) for r in route.router.routes for m in r.methods}
    assert (ROUTE["method"], ROUTE["path"]) in paths
