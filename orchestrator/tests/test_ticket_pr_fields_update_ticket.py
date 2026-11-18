"""Ticket PR fields: a session's update_ticket reports the pull request it opened, and the ticket keeps it.

A Claude Code session on the owner's development machine commits, pushes a branch and opens a
pull request itself, then reports it through ``update_ticket`` with four optional fields
(``pull_request_url``, ``branch``, ``commits``, ``tested``). This pins:

* the fields are checked at the boundary, and anything that does not pass refuses the call in
  words the session can act on (a non-URL, a ``javascript:`` URL, a ref with spaces, a bad sha,
  an over-long list);
* a note-only call is exactly what it was;
* the ticket keeps them under ``runtime_ref.pull_request``, written with ONE ``jsonb_set`` of that
  key inside a row lock, with the note "Pull request: <url>"; a second call updates the object;
* on the real schema, the object survives an event flush that read the ticket before it landed,
  and a claim that builds a fresh ``runtime_ref`` (``core/models/kept_runtime_keys.py``); once the
  owner takes it off, a stale copy never puts it back;
* it is frozen once the ticket leaves In progress, and a re-run may replace it;
* the board's serializer carries it, re-checked, with its stamp and whether Approve would merge it.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from config import config
from core.models.core import BoardTask
from core.models.kept_runtime_keys import PULL_REQUEST_KEY, with_stored_keys
from services import cli_host_service as svc
from services import session_tools as st
from services import ticket_pull_request as tpr
from services.board_task_view import board_dict
from tests.helpers_mission_lane import quiet_board, session_working

CTX = st.SessionContext(task_id=119, agent_id=268, agent_name="CODER", workspace_id="ws-c1")
URL = "https://github.com/automatos-ai/automatos-ai/pull/1559"
OTHER_URL = "https://github.com/automatos-ai/automatos-ai/pull/1560"
SHA = "9a9cb36d9e1f4c2b7a0d3e5f6a7b8c9d0e1f2a3b"
COMMITS = [{"sha": SHA, "subject": "feat(board): the card carries its pull request"},
           {"sha": "114d30a", "subject": "test(board): the card's pull request"}]
TESTED = "py_compile on the changed files; nothing else could be run here."
REPORTED = {"pull_request_url": URL, "branch": "feat/ticket-pr-fields", "commits": COMMITS, "tested": TESTED}


@pytest.fixture
def quiet(monkeypatch):
    return quiet_board(monkeypatch)


def _scope(arguments):
    return st.resolve_parameters(st.get_tool("update_ticket"), arguments, CTX)


# ── the boundary ─────────────────────────────────────────────────────────────

def test_update_ticket_takes_the_four_fields_and_says_what_they_are_for():
    tool = st.get_tool("update_ticket")
    assert set(tool.input_schema["properties"]) == {"note", *tpr.FIELDS}
    assert tool.input_schema["required"] == ["note"]
    assert tool.input_schema["properties"]["commits"]["maxItems"] == tpr.MAX_COMMITS
    for field in tpr.FIELDS:
        assert field in tool.description


def test_the_four_fields_are_checked_and_kept_as_given():
    scoped = _scope({"note": "Opened the pull request.", **REPORTED, "status": "in_progress"})
    assert scoped == {"note": "Opened the pull request.", "pull_request_url": URL,
                      "branch": "feat/ticket-pr-fields", "commits": COMMITS, "tested": TESTED}


def test_a_note_only_call_is_exactly_what_it_was():
    assert _scope({"note": "reading the brief"}) == {"note": "reading the brief"}
    assert _scope({"note": "x", "pull_request_url": None, "branch": ""}) == {"note": "x"}


def test_a_call_that_reports_a_pull_request_keeps_it_without_a_note():
    assert _scope({"pull_request_url": URL}) == {"note": "", "pull_request_url": URL}
    with pytest.raises(st.SessionToolRefused, match="progress note"):
        _scope({})


@pytest.mark.parametrize("url", [
    "https://gitlab.com/automatos/platform/core/-/merge_requests/42",     # GitLab's shape, nested groups
    "https://git.example.org:8443/team/app/pull/7",                      # any host, a port
    "https://github.com/automatos-ai/.github/pull/3/",                   # a trailing slash is dropped
])
def test_a_pull_requests_own_address_on_any_host_passes(url):
    assert _scope({"note": "n", "pull_request_url": url})["pull_request_url"] == url.rstrip("/")


@pytest.mark.parametrize("url", [
    "not a url", "javascript:alert(1)", "http://github.com/o/r/pull/1", "https://github.com/o/r/issues/1",
    "https://github.com/o/r/pull/0", "https://github.com/o/r/pull/1/files", "https://github.com/o/r/pull/1?x=1",
    "https://github.com/o/r/pull/1#discussion", "https://someone@github.com/o/r/pull/1", "https://github.com/../r/pull/1",
    "https://github.com/o/r/pull/1 extra", "https://github.com/o/r%2F/pull/1", "https:///o/r/pull/1",
    "https://gitlab.com/group/-/-/merge_requests/1", 1559, ["https://github.com/o/r/pull/1"],
    "https://github.com/o/r/pull/" + "1" * 11, "https://github.com/" + "o" * 600 + "/r/pull/1",
])
def test_anything_else_as_the_address_refuses_the_call(url):
    with pytest.raises(st.SessionToolRefused) as refused:
        _scope({"note": "n", "pull_request_url": url})
    assert str(refused.value) == tpr.BAD_URL


@pytest.mark.parametrize("branch", [
    "feat/ticket pr", "feat/..", "-delete", "/main", ".hidden", "main.lock", "feat//x", "a@{1}", "x:y", "x~1",
    "feat/\tx", "@", "b" * (tpr.MAX_BRANCH_CHARS + 1), 12,
])
def test_a_branch_git_would_not_name_refuses_the_call(branch):
    with pytest.raises(st.SessionToolRefused) as refused:
        _scope({"note": "n", "branch": branch})
    assert str(refused.value) == tpr.BAD_BRANCH


@pytest.mark.parametrize("commits", [
    [{"sha": "9a9cb3", "subject": "too short"}], [{"sha": "z" * 40, "subject": "not hex"}],
    [{"sha": "a" * 41, "subject": "too long"}], [{"subject": "no sha"}], ["9a9cb36d9"],
    [{"sha": SHA, "subject": 3}], [{"sha": SHA, "subject": "s"}] * (tpr.MAX_COMMITS + 1), {"sha": SHA},
])
def test_a_bad_sha_or_an_over_long_list_refuses_the_call(commits):
    with pytest.raises(st.SessionToolRefused) as refused:
        _scope({"note": "n", "commits": commits})
    assert str(refused.value) == tpr.BAD_COMMITS


def test_fifty_commits_pass_and_a_subject_is_one_capped_line():
    commits = [{"sha": SHA.upper(), "subject": "line one\nline two " + "x" * 400}] * tpr.MAX_COMMITS
    kept = _scope({"note": "n", "commits": commits})["commits"]
    assert len(kept) == tpr.MAX_COMMITS and kept[0]["sha"] == SHA
    assert "\n" not in kept[0]["subject"] and len(kept[0]["subject"]) == tpr.MAX_SUBJECT_CHARS


def test_tested_is_text_and_capped():
    assert len(_scope({"note": "n", "tested": "t" * 5000})["tested"]) == tpr.MAX_TESTED_CHARS
    with pytest.raises(st.SessionToolRefused) as refused:
        _scope({"note": "n", "tested": {"ran": "pytest"}})
    assert str(refused.value) == tpr.BAD_TESTED


def test_a_session_still_cannot_move_its_ticket_whatever_it_reports():
    with pytest.raises(st.SessionToolRefused, match="ask_human"):
        _scope({"note": "n", "status": "done", **REPORTED})


# ── the object a call leaves (pure) ──────────────────────────────────────────

def test_a_second_call_on_the_same_pull_request_updates_the_object():
    first = tpr.updated_pull_request({}, {k: v for k, v in _scope(REPORTED).items() if k != "note"}, by="CODER", at="1")
    second = tpr.updated_pull_request(first, {"tested": "CI went green."}, by="CODER", at="2")
    assert second == {**first, "tested": "CI went green.", "at": "2"}


def test_a_new_pull_request_replaces_it_and_the_old_address_is_kept_a_while():
    stored = {"url": URL, "branch": "feat/a", "commits": COMMITS, "previous_urls": [f"{URL}0"]}
    replaced = tpr.updated_pull_request(stored, {"pull_request_url": OTHER_URL}, by="CODER", at="3")
    assert replaced == {"url": OTHER_URL, "previous_urls": [URL, f"{URL}0"], "by": "CODER", "at": "3"}
    many = {"url": URL, "previous_urls": [f"{URL}{n}" for n in range(10)]}
    assert len(tpr.updated_pull_request(many, {"pull_request_url": OTHER_URL}, by="b", at="4")["previous_urls"]) \
        == tpr.PREVIOUS_URLS_KEPT


# ── the writer (its statements) ──────────────────────────────────────────────

class _Recording:
    """A session that records each statement; the locked read answers the ticket's ``status`` and ``stored``."""

    def __init__(self, stored=None, status="in_progress"):
        self.calls, self.commits, self.stored, self.status = [], 0, stored, status

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), dict(params or {})))
        return NS(first=lambda: (self.status, self.stored), rowcount=1)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _notes_written(db):
    return [json.loads(p["entry"])[0]["note"] for s, p in db.calls if p.get("key") == "session_notes"]


def test_the_pull_request_is_written_with_one_targeted_statement_under_a_row_lock(monkeypatch):
    published = []
    monkeypatch.setattr(svc, "publish_canvas_events", lambda ws, evs: published.extend(evs))
    db = _Recording()
    out = tpr.record_ticket_update(db, task_id=116, workspace_id="ws-c1", agent_name="CODER",
                                   params=_scope({"note": "Opened it.", **REPORTED}))

    assert out["success"] is True and out["result"]["pull_request"] == sorted(tpr.FIELDS)
    locked, written = db.calls[0], db.calls[1]
    assert "FOR NO KEY UPDATE" in locked[0] and locked[1]["key"] == PULL_REQUEST_KEY   # the UPDATE's own lock
    assert "jsonb_set" in written[0] and written[1]["path"] == "{pull_request}"
    stored = json.loads(written[1]["value"])
    assert (stored["url"], stored["branch"], stored["commits"], stored["tested"], stored["by"]) == (
        URL, "feat/ticket-pr-fields", COMMITS, TESTED, "CODER")
    assert all("jsonb_set" in stmt for stmt, _ in db.calls if "UPDATE board_tasks" in stmt)   # never the whole document
    assert all("workspace_id = :ws" in stmt and p["ws"] == "ws-c1" for stmt, p in db.calls)    # this workspace's ticket
    assert not any(re.search(r":\w+::", stmt) for stmt, _ in db.calls)           # this repo's bind trap
    assert _notes_written(db) == ["Opened it.", tpr.PULL_REQUEST_NOTE.format(url=URL)]
    assert db.commits == 1
    assert [e["data"]["note"] for e in published] == ["Opened it.", f"Pull request: {URL}"]   # live, too


def test_the_same_address_again_leaves_no_second_pull_request_note(monkeypatch):
    monkeypatch.setattr(svc, "publish_canvas_events", lambda ws, evs: None)
    db = _Recording(stored={"url": URL, "branch": "feat/a"})
    tpr.record_ticket_update(db, task_id=116, workspace_id="ws-c1", agent_name="CODER",
                             params={"note": "", "pull_request_url": URL, "tested": "CI green."})
    assert json.loads(db.calls[1][1]["value"])["branch"] == "feat/a"                # kept from the first call
    assert _notes_written(db) == []


def test_a_ticket_that_left_in_progress_keeps_the_pull_request_the_owner_is_reviewing():
    db = _Recording(stored={"url": URL}, status="review")
    out = tpr.record_ticket_update(db, task_id=116, workspace_id="ws-c1", agent_name="CODER",
                                   params={"note": "", "pull_request_url": OTHER_URL})
    assert out == {"success": False, "error": tpr.NOT_OPEN_FOR_CHANGES}
    assert len(db.calls) == 1 and db.commits == 0                       # the locked read, and nothing written


def test_a_note_only_call_goes_the_way_it_always_went(monkeypatch):
    seen = []
    monkeypatch.setattr(svc, "record_session_note", lambda db, **kw: seen.append(kw) or {"success": True})
    assert tpr.record_ticket_update(_Recording(), task_id=1, workspace_id="w", agent_name="A",
                                    params={"note": "reading"}) == {"success": True}
    assert seen == [{"task_id": 1, "workspace_id": "w", "agent_name": "A", "note": "reading"}]


def test_a_failed_write_is_told_to_the_session_not_raised():
    class _Broken(_Recording):
        def execute(self, *_a, **_k):
            raise RuntimeError("db down")

    out = tpr.record_ticket_update(_Broken(), task_id=1, workspace_id="w", agent_name=None,
                                   params={"note": "x", "pull_request_url": URL})
    assert out == {"success": False, "error": tpr.NOT_SAVED.format(kind="RuntimeError")}


# ── on the real schema: it survives the host's whole-document writes ─────────

def _stored(db, task_id):
    db.expire_all()
    return db.get(BoardTask, task_id).runtime_ref or {}


def test_the_pull_request_survives_an_event_flush_that_read_the_ticket_before_it_landed(
        db_session, seed_workspace, quiet, monkeypatch):
    """``_absorb_events`` reads the ticket, renews the lease, then writes the whole document back.
    The pull request lands in between, the way a session's tool call races its own hook events."""
    ws = UUID(seed_workspace())
    _run, _step, card, host, _claim = session_working(db_session, ws)
    renew = svc.renew_lease

    def renew_while_the_session_reports(db, task_id, **kwargs):
        renewed = renew(db, task_id, **kwargs)
        tpr.write_pull_request(db, task_id=task_id, workspace_id=ws, by="NEWSROOM", note="",
                               fields=tpr.pull_request_fields(REPORTED))
        return renewed

    monkeypatch.setattr(svc, "renew_lease", renew_while_the_session_reports)
    svc._absorb_events(db_session, host, card.id, [{"event": "PreToolUse", "tool_name": "Bash", "subject": "git push"}])

    ref = _stored(db_session, card.id)
    assert ref["pull_request"]["url"] == URL and ref["pull_request"]["commits"] == COMMITS
    assert ref["events_seen"] >= 1 and ref["recent_tools"][-1]["tool"] == "Bash"   # the flush's own write landed too


def test_a_claim_that_builds_a_fresh_runtime_ref_keeps_the_pull_request(db_session, seed_workspace, quiet):
    """A Reject's redo, or an answered question, is claimed again: ``_claim_ref`` builds a fresh
    document and carries only its own record. The ticket's pull request stays on it."""
    ws = UUID(seed_workspace())
    _run, _step, card, host, first = session_working(db_session, ws)
    tpr.write_pull_request(db_session, task_id=card.id, workspace_id=ws, by="NEWSROOM", note="",
                           fields=tpr.pull_request_fields(REPORTED))
    db_session.refresh(card)
    card.status = "assigned"
    db_session.flush()

    (again,) = svc.claim_for_host(db_session, host, 1)["tasks"]
    ref = _stored(db_session, card.id)
    assert again["task_id"] == card.id and ref["session_id"] != first["session_id"]   # a fresh document…
    assert ref["pull_request"]["url"] == URL                                           # …that kept it


def test_record_ticket_update_stores_it_on_the_ticket_and_a_second_call_updates_it(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    _run, _step, card, _host, _claim = session_working(db_session, ws)
    first = tpr.record_ticket_update(db_session, task_id=card.id, workspace_id=ws, agent_name="NEWSROOM",
                                     params=_scope({"note": "Pushed and opened the pull request.", **REPORTED}))
    second = tpr.record_ticket_update(db_session, task_id=card.id, workspace_id=ws, agent_name="NEWSROOM",
                                      params=_scope({"note": "CI is green.", "tested": "CI went green."}))

    ref = _stored(db_session, card.id)
    assert first["success"] and second["success"]
    assert ref["pull_request"]["url"] == URL and ref["pull_request"]["tested"] == "CI went green."
    assert ref["pull_request"]["branch"] == "feat/ticket-pr-fields"
    assert [n["note"] for n in ref["session_notes"]][-3:] == [
        "Pushed and opened the pull request.", f"Pull request: {URL}", "CI is green."]


def test_once_the_ticket_leaves_in_progress_its_pull_request_is_frozen_and_a_rerun_may_replace_it(
        db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    _run, _step, card, _host, _claim = session_working(db_session, ws)
    tpr.write_pull_request(db_session, task_id=card.id, workspace_id=ws, by="NEWSROOM", note="",
                           fields=tpr.pull_request_fields(REPORTED))
    db_session.refresh(card)
    card.status = "review"                                   # the owner is looking at it now
    db_session.flush()
    with pytest.raises(tpr.TicketNotOpen):
        tpr.write_pull_request(db_session, task_id=card.id, workspace_id=ws, by="NEWSROOM", note="",
                               fields={"pull_request_url": OTHER_URL})
    assert _stored(db_session, card.id)["pull_request"]["url"] == URL

    card = db_session.get(BoardTask, card.id)
    card.status = "in_progress"                              # a re-run
    db_session.flush()
    tpr.write_pull_request(db_session, task_id=card.id, workspace_id=ws, by="NEWSROOM", note="",
                           fields={"pull_request_url": OTHER_URL})
    replaced = _stored(db_session, card.id)["pull_request"]
    assert replaced["url"] == OTHER_URL and replaced["previous_urls"] == [URL]


def test_a_pull_request_the_owner_took_off_is_never_put_back_by_a_flush_that_read_it_before(
        db_session, seed_workspace, quiet, monkeypatch):
    ws = UUID(seed_workspace())
    _run, _step, card, host, _claim = session_working(db_session, ws)
    tpr.write_pull_request(db_session, task_id=card.id, workspace_id=ws, by="NEWSROOM", note="",
                           fields=tpr.pull_request_fields(REPORTED))
    db_session.refresh(card)                                 # the flush below reads it with the pull request
    renew = svc.renew_lease

    def renew_while_the_owner_removes_it(db, task_id, **kwargs):
        renewed = renew(db, task_id, **kwargs)
        assert tpr.clear_pull_request(db, task_id=task_id, workspace_id=ws) == URL
        return renewed

    monkeypatch.setattr(svc, "renew_lease", renew_while_the_owner_removes_it)
    svc._absorb_events(db_session, host, card.id, [{"event": "PreToolUse", "tool_name": "Bash"}])
    ref = _stored(db_session, card.id)
    assert PULL_REQUEST_KEY not in ref and ref["events_seen"] >= 1


def test_a_ticket_that_is_not_in_the_workspace_is_never_written():
    """The locked read is scoped to the caller's workspace; no row there, nothing is written."""
    class _NoRow(_Recording):
        def execute(self, stmt, params=None):
            self.calls.append((str(stmt), dict(params or {})))
            return NS(first=lambda: None, rowcount=0)

    db = _NoRow()
    out = tpr.record_ticket_update(db, task_id=5, workspace_id="another-ws", agent_name="X",
                                   params={"note": "", "pull_request_url": URL})
    assert out == {"success": False, "error": tpr.NOT_SAVED.format(kind="LookupError")}
    assert len(db.calls) == 1 and db.calls[0][1]["ws"] == "another-ws" and db.commits == 0


def test_a_whole_document_write_keeps_the_key_as_the_row_holds_it():
    assert with_stored_keys({"a": 1}, {PULL_REQUEST_KEY: {"url": URL}}) == {"a": 1, PULL_REQUEST_KEY: {"url": URL}}
    same = {"a": 1, PULL_REQUEST_KEY: {"url": URL}}
    assert with_stored_keys(same, {PULL_REQUEST_KEY: {"url": URL}}) is same       # nothing to do
    assert with_stored_keys({"a": 1}, {PULL_REQUEST_KEY: None}) == {"a": 1}        # the row holds none
    assert with_stored_keys(None, {PULL_REQUEST_KEY: {"url": URL}}) == {PULL_REQUEST_KEY: {"url": URL}}
    stale = {"a": 1, PULL_REQUEST_KEY: {"url": URL}}                                # read before the owner took it off
    assert with_stored_keys(stale, {PULL_REQUEST_KEY: None}) == {"a": 1}


# ── the board's serializer ───────────────────────────────────────────────────

def _served(runtime_ref):
    task = BoardTask(id=7, status="review", source_type="user", review_mode="human", tags=[], attempts=0,
                     runtime_ref=runtime_ref)
    return board_dict(task, "#0007")["pull_request"]


AT = "2026-11-18T09:30:00+00:00"


def test_the_board_serves_the_pull_request_re_checked():
    stored = {"url": URL, "branch": "feat/ticket-pr-fields", "commits": COMMITS, "tested": TESTED, "by": "CODER",
              "at": AT, "previous_urls": [OTHER_URL, "javascript:alert(1)"]}
    assert _served({PULL_REQUEST_KEY: stored}) == {
        "url": URL, "host": "github.com", "repository": "automatos-ai/automatos-ai", "number": 1559,
        "branch": "feat/ticket-pr-fields", "commits": COMMITS, "tested": TESTED, "previous_urls": [OTHER_URL],
        "at": AT, "merges_on_approve": False}


@pytest.mark.parametrize("on, repos, url, merges", [
    (True, "automatos-ai/automatos-ai", URL, True),
    (True, " Other/Repo , AUTOMATOS-AI/Automatos-AI ", URL, True),            # listed, case and spaces aside
    (False, "automatos-ai/automatos-ai", URL, False),                         # the setting off
    (True, "", URL, False),                                                   # nothing listed: nothing merges
    (True, "automatos-ai/other", URL, False),                                 # another repository
    (True, "automatos-ai/automatos-ai", "https://git.example.org/automatos-ai/automatos-ai/pull/3", False),
])
def test_the_board_says_whether_approve_would_merge_it(monkeypatch, on, repos, url, merges):
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR", on)
    monkeypatch.setattr(config, "TICKET_APPROVE_MERGES_PR_REPOS", repos)
    assert _served({PULL_REQUEST_KEY: {"url": url, "at": AT}})["merges_on_approve"] is merges


def test_a_ticket_without_one_or_with_a_value_that_does_not_pass_serves_none_of_it():
    assert _served(None) is None and _served({"session_notes": []}) is None
    assert _served({PULL_REQUEST_KEY: {"url": "javascript:alert(1)"}}) is None
    tampered = _served({PULL_REQUEST_KEY: {"url": "javascript:alert(1)", "branch": "feat/x"}})
    assert tampered["url"] is None and tampered["number"] is None and tampered["branch"] == "feat/x"
