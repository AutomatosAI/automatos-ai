"""PRD-263 US-062: the production skill signal compares skill bodies with counts only.

``scripts/skill_signal_report.py`` reads, across workspaces, the Auto replies whose
``context_trace`` names the body that served them. Per body hash and per day it
counts the NOT_DONE, NOTHING_DONE_SAID and NO_SUCH_ID lines in those replies and the
grants their turns raised and the owner denied. Gerard runs it against prod, so its
output must hold counts and rates only: no reply text, no workspace or chat id.

The seeded tests run on the real test database (root conftest ``db_session``, rolled
back); the template tests are pure.
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import text

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from consumers.chatbot.claim_check import NO_SUCH_ID, NOT_DONE, NOTHING_DONE_SAID  # noqa: E402
from scripts import skill_signal_report as report  # noqa: E402

CORE = "platform-management"
SECRET = "Kiambu AA ships to Mrs Achieng at 14 Rose Lane"   # tenant text that must never print
NOT_DONE_LINE = NOT_DONE.format(said=NOTHING_DONE_SAID)
NO_SUCH_ID_LINE = NO_SUCH_ID.format(ids="Task #1100", verb="does", it="it")


# --- the templates, matched with any words in their placeholders ---


@pytest.mark.parametrize("template, filled", [
    (NOT_DONE, NOT_DONE.format(said="I haven't sent anything in this reply")),
    (NOT_DONE, NOT_DONE_LINE),
    (NOTHING_DONE_SAID, NOT_DONE_LINE),
    (NO_SUCH_ID, NO_SUCH_ID_LINE),
    (NO_SUCH_ID, NO_SUCH_ID.format(ids="Task #1100 and agent ID 102", verb="do", it="them")),
])
def test_each_template_matches_a_saved_reply_that_carries_it(template, filled):
    assert re.search(report.template_pattern(template), f"Here is the plan.\n\n{filled}\nAnything else?")


@pytest.mark.parametrize("reply", [
    "Just to be clear: the post went out at nine.",
    "Just to be clear: nothing\n. Ask me again if you want it done.",   # not one line
    "I haven't done that yet.",
])
def test_a_reply_that_only_resembles_a_template_does_not_match(reply):
    assert not any(re.search(report.template_pattern(t), reply) for _, t in report.TEMPLATES)


# --- the window starts at a UTC midnight ---


def test_the_window_starts_at_midnight_so_only_today_is_a_part_day():
    now = datetime(2026, 10, 18, 15, 30, 12, tzinfo=timezone.utc)
    assert report.window_start(1, now) == datetime(2026, 10, 18)
    assert report.window_start(3, now) == datetime(2026, 10, 16)
    assert report.window_start(1, datetime(2026, 10, 18, 0, 30, tzinfo=timezone(timedelta(hours=2)))) == \
        datetime(2026, 10, 17)


# --- the seeded database ---

# One clock read for the module: a run that crosses midnight cannot split a day's rows.
_TODAY = datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)


def _at(days_ago, hour, minute, second=0):
    return _TODAY - timedelta(days=days_ago) + timedelta(hours=hour, minutes=minute, seconds=second)


@pytest.fixture
def prod(db_session, seed_workspace):
    """Two workspaces' chats with Auto: an old body on one day, an edited body the next."""
    db = db_session
    tag = uuid.uuid4().hex[:8]
    user = db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"prd263-{tag}@example.test", "u": f"prd263-{tag}"}).scalar()
    seeded = NS(db=db, old=uuid.uuid4().hex * 2, new=uuid.uuid4().hex * 2, ids=[])
    for name in ("cafe", "bakery"):
        ws, chat = seed_workspace(), uuid.uuid4()
        db.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility) "
                        "VALUES (CAST(:c AS uuid), :u, CAST(:w AS uuid), 'prd-263', 'private')"),
                   {"c": str(chat), "u": user, "w": ws})
        setattr(seeded, name, NS(ws=ws, chat=str(chat)))
        seeded.ids += [ws, str(chat)]
    return seeded


def _say(seeded, place, role, words, at, bodies=None):
    trace = None if bodies is None else {"mode": "chatbot", "skill_bodies": bodies}
    seeded.db.execute(text(
        "INSERT INTO messages (id, chat_id, workspace_id, role, parts, context_trace, created_at) VALUES "
        "(CAST(:i AS uuid), CAST(:c AS uuid), CAST(:w AS uuid), :r, CAST(:p AS jsonb), CAST(:t AS jsonb), :at)"),
        {"i": str(uuid.uuid4()), "c": place.chat, "w": place.ws, "r": role,
         "p": json.dumps([{"type": "reasoning", "reasoning": SECRET}, {"type": "text", "text": words}]),
         "t": None if trace is None else json.dumps(trace), "at": at})


def _grant(seeded, place, status, at, kind="approval", conversation=True):
    details = {"action": "platform_send_email", "lane": "chat"}
    if conversation:
        details["conversation_id"] = place.chat
    seeded.db.execute(text(
        "INSERT INTO approval_grants (workspace_id, subject_type, subject_id, tool_name, status, kind, "
        "details, requested_at) VALUES (CAST(:w AS uuid), 'tool_call', :s, 'platform_send_email', :st, :k, "
        "CAST(:d AS jsonb), :at)"),
        {"w": place.ws, "s": uuid.uuid4().hex, "st": status, "k": kind, "d": json.dumps(details),
         "at": at.replace(tzinfo=timezone.utc)})


@pytest.fixture
def signal(prod):
    old = [{"name": CORE, "content_hash": prod.old}]
    new = [{"name": CORE, "content_hash": prod.new}]
    # Day one, the old body. The café's owner asks; Auto's first reply says the work is not done.
    _say(prod, prod.cafe, "user", f"Email the order to {SECRET}", _at(2, 10, 0))
    _grant(prod, prod.cafe, "denied", _at(2, 10, 0, 5))              # raised in that turn, refused
    _say(prod, prod.cafe, "assistant", f"I drafted it for {SECRET}.\n\n{NOT_DONE_LINE}", _at(2, 10, 0, 10), old)
    _grant(prod, prod.cafe, "granted", _at(2, 10, 4))                # the next turn's, approved
    _say(prod, prod.cafe, "assistant", f"{SECRET}\n{NO_SUCH_ID_LINE}", _at(2, 10, 5), old)
    _say(prod, prod.bakery, "assistant", f"Here is the rota for {SECRET}.", _at(2, 11, 0), old)
    _say(prod, prod.bakery, "assistant", NOT_DONE_LINE, _at(2, 11, 30))   # no trace: not counted
    # Resembles NOT_DONE, but its placeholder crosses a line: matched by Postgres, not counted.
    _say(prod, prod.bakery, "assistant", "Just to be clear: nothing\n. Ask me again if you want it done.",
         _at(2, 12, 0), old)
    # Day two, the edited body, in the other workspace.
    _grant(prod, prod.bakery, "denied", _at(1, 9, 0, 5))
    _grant(prod, prod.bakery, "denied", _at(1, 9, 0, 6), kind="question")    # a dismissed ask
    _grant(prod, prod.bakery, "denied", _at(1, 9, 0, 7), conversation=False)  # not raised in a chat
    _say(prod, prod.bakery, "assistant", NOT_DONE.format(said="I haven't sent anything in this reply"),
         _at(1, 9, 0, 10), new)
    _say(prod, prod.bakery, "assistant", NOT_DONE_LINE, _at(1, 9, 30), [])    # no core body served
    rows = report.fetch_signal(prod.db, since=_at(7, 0, 0))
    return {row["content_hash"]: row for row in rows if row["content_hash"] in (prod.old, prod.new)}


def test_each_body_gets_its_day_and_counts(prod, signal):
    old, new = signal[prod.old], signal[prod.new]
    assert (old["day"], old["skill"], new["day"], new["skill"]) == (
        _at(2, 0, 0).date().isoformat(), CORE, _at(1, 0, 0).date().isoformat(), CORE)
    # Four old-body replies across both workspaces; one says NOT_DONE (inside it, NOTHING_DONE_SAID),
    # one names an id that does not exist, one only resembles NOT_DONE. The reply with no trace and
    # the one with no body are not Auto's.
    assert {k: old[k] for k in report.COUNTS} == {
        "turns": 4, "not_done": 1, "nothing_done_said": 1, "no_such_id": 1, "grants": 2, "denied_grants": 1}
    assert {k: new[k] for k in report.COUNTS} == {
        "turns": 1, "not_done": 1, "nothing_done_said": 0, "no_such_id": 0, "grants": 1, "denied_grants": 1}


def test_rates_are_per_reply_and_per_grant(prod, signal):
    old = signal[prod.old]
    assert old["not_done_rate"] == pytest.approx(1 / 4) and old["no_such_id_rate"] == pytest.approx(1 / 4)
    assert old["denied_grant_rate"] == pytest.approx(0.5)
    assert signal[prod.new]["denied_grant_rate"] == pytest.approx(1.0)


def test_the_output_holds_counts_only(prod, signal):
    rows = list(signal.values())
    printed = report.render_text(rows, days=7) + json.dumps(rows)
    for leaked in (SECRET, "Just to be clear", NOTHING_DONE_SAID, *prod.ids):
        assert leaked not in printed
    for row in rows:
        assert set(row) == {"day", "skill", "content_hash", *report.COUNTS,
                            "not_done_rate", "nothing_done_said_rate", "no_such_id_rate", "denied_grant_rate"}
        assert all(isinstance(row[k], int) for k in report.COUNTS)


# --- the run itself is read-only ---


class _ReadOnlySession:
    def __init__(self):
        self.statements, self.closed = [], False

    def execute(self, statement, params=None):
        self.statements.append(str(statement).strip())
        return NS(mappings=lambda: [])

    def rollback(self):
        pass

    def close(self):
        self.closed = True


def test_a_run_reads_in_a_read_only_transaction_and_prints_no_rows(monkeypatch, capsys):
    import core.database.database as database

    session = _ReadOnlySession()
    monkeypatch.setattr(database, "SessionLocal", lambda: session)
    assert report.main(["--days", "7"]) == 0
    assert session.statements[0] == "SET TRANSACTION READ ONLY"
    assert all(s.startswith(("SET TRANSACTION", "WITH")) for s in session.statements) and session.closed
    assert "no Auto turn recorded a skill body" in capsys.readouterr().out
