"""PRD-268 W2 US-C08b (§6.5.2, K20) — the ``suggestion_changed`` live event, and discovery's
suggestions and readiness.

``services.board_events.notify_suggestion_event`` fires ``suggestion_changed`` on the
``board_events`` channel; ``services.proactive.live.suggestion_changed`` sends it after a change
has committed. Proven here:

* ``frame_for_payload`` forwards ``suggestion_changed`` as an SSE frame of its own, carrying the
  workspace, the suggestion's id, its addressee and its status; another workspace's is dropped,
  and an event name that is not a string stays the board refresh;
* the payload names ids and the status only, never the suggestion's words, and a NOTIFY that
  fails never raises;
* against Postgres, with committing sessions and a spy on ``notify_suggestion_event`` that reads
  the suggestion from another session when it is called: delivery (a run's suggestion and a
  failing streak's first card) is told once as ``open``, Start once as ``started``, Hide once as
  ``hidden``, and the streak's "works again" expiry and the 7-day lapse each once as ``expired``,
  each after its change committed; doing it again tells no one, and shadow delivers and tells
  nothing;
* a NOTIFY that fails in Postgres changes nothing: the post, the Start and its audit row, the
  Hide and the expiry all stand;
* discovery lists ``suggestions`` and ``readiness`` on their live routes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from services import board_events
from services.board_events import frame_for_payload, notify_suggestion_event

ORCH = Path(__file__).resolve().parents[1]
MANIFEST = ORCH / "reports" / "route-manifest.json"
TITLE = "Ticket #0912 failed twice overnight"
AGENT_ID = 4242


def _parsed(frame: str) -> tuple:
    """An SSE frame's event name and its data."""
    head, data = frame.strip().split("\n")
    return head.removeprefix("event: "), json.loads(data.removeprefix("data: "))


# --------------------------------------------------------------------------- the frame and the payload


def test_frame_for_payload_forwards_suggestion_changed_under_its_own_name():
    ws, sid = str(uuid.uuid4()), str(uuid.uuid4())
    payload = json.dumps({"workspace_id": ws, "suggestion_id": sid, "user_id": 7, "status": "started",
                          "event": "suggestion_changed"})

    event, data = _parsed(frame_for_payload(payload, ws))

    assert event == "suggestion_changed"
    assert data == {"workspace_id": ws, "suggestion_id": sid, "user_id": 7, "status": "started",
                    "event": "suggestion_changed"}


def test_frame_for_payload_drops_another_workspaces_suggestion_event():
    payload = json.dumps({"workspace_id": str(uuid.uuid4()), "suggestion_id": str(uuid.uuid4()), "user_id": 7,
                          "status": "open", "event": "suggestion_changed"})

    assert frame_for_payload(payload, str(uuid.uuid4())) is None


def test_chat_and_board_events_route_as_before_and_an_odd_event_name_is_the_board_refresh():
    ws = str(uuid.uuid4())
    chat = json.dumps({"workspace_id": ws, "chat_id": "c-1", "user_id": 7, "event": "chat_changed"})
    board = json.dumps({"workspace_id": ws, "task_id": 9, "status": "done", "event": "task_changed"})
    odd = json.dumps({"workspace_id": ws, "event": ["suggestion_changed"]})

    assert _parsed(frame_for_payload(chat, ws))[0] == "chat_changed"
    assert _parsed(frame_for_payload(board, ws))[0] == "board_changed"
    assert _parsed(frame_for_payload(odd, ws))[0] == "board_changed"


def test_notify_suggestion_event_sends_ids_and_the_status_on_the_board_events_channel():
    db, ws, sid = MagicMock(), uuid.uuid4(), uuid.uuid4()

    notify_suggestion_event(db, workspace_id=ws, suggestion_id=sid, user_id=3, status="hidden")

    params = db.execute.call_args.args[1]
    assert params["chan"] == "board_events"
    assert json.loads(params["payload"]) == {"workspace_id": str(ws), "suggestion_id": str(sid), "user_id": 3,
                                             "status": "hidden", "event": "suggestion_changed"}
    db.begin_nested.assert_called_once()  # a NOTIFY that fails leaves the caller's transaction usable


def test_a_notify_that_fails_never_raises(caplog):
    db = MagicMock()
    db.execute.side_effect = RuntimeError("the connection went away")

    with caplog.at_level(logging.ERROR, logger=board_events.__name__):
        notify_suggestion_event(db, workspace_id=uuid.uuid4(), suggestion_id="s-1", user_id=3, status="open")

    assert "pg_notify failed for suggestion s-1" in caplog.text


# --------------------------------------------------------------------------- discovery


def test_discovery_lists_suggestions_and_readiness_on_their_live_routes():
    from services.mobile import discovery

    assert discovery.CAPABILITY_ROUTES["suggestions"] == ("GET", "/api/proactive/suggestions")
    assert discovery.CAPABILITY_ROUTES["readiness"] == ("GET", "/api/readiness")
    assert {"suggestions", "readiness"} <= set(discovery.offered_capabilities())
    assert not {"suggestions", "readiness"} & set(discovery.CONFIGURED_CAPABILITIES)  # listed whenever this runs
    routes = {(r["method"], r["path"]) for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["routes"]}
    assert {("GET", "/api/proactive/suggestions"), ("GET", "/api/readiness")} <= routes


# --------------------------------------------------------------------------- against Postgres


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT delivered_at FROM proactive_suggestions LIMIT 1"))
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
            c.execute(text("SELECT locale FROM mobile_devices LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the suggestion event tests need Postgres with the PRD-205 and PRD-268 tables: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def cafe(engine, new_session, monkeypatch):
    """Ana owns the café's workspace with check-ins on, the platform mode is ``cafe.mode``, and
    every ``notify_suggestion_event`` is recorded with the suggestion as another session reads it
    at that moment, then sent as it would be."""
    from config import config
    from core.llm import manager
    from services.notification_events import auto_message

    s = new_session()
    clerk = f"user_prd268w2se_{uuid.uuid4().hex[:10]}"
    ana = int(s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar())
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) VALUES (CAST(:id AS uuid), "
                   "'prd268w2-suggestion-events', :o, CAST(:s AS jsonb))"),
              {"id": ws, "o": ana, "s": json.dumps({"proactive": {"enabled": True}})})
    s.commit()
    state = NS(ws=ws, ana=ana, clerk=clerk, mode="live", told=[])
    real = board_events.notify_suggestion_event

    def spy(db, **kwargs):
        state.told.append(NS(sent={"workspace_id": str(kwargs["workspace_id"]),
                                   "suggestion_id": str(kwargs["suggestion_id"]),
                                   "user_id": kwargs["user_id"], "status": kwargs["status"]},
                             committed=_stored(new_session, kwargs["suggestion_id"])))
        real(db, **kwargs)

    def announced(coro, *, what):
        coro.close()

    monkeypatch.setattr(board_events, "notify_suggestion_event", spy)
    monkeypatch.setattr(manager, "read_system_setting",
                        lambda category, key: state.mode if (category, key) == ("proactive", "mode") else None)
    monkeypatch.setattr(auto_message, "send_soon", announced)
    monkeypatch.setattr(config, "SPEND_CEILING_USD", 0)
    yield state
    s = new_session.sweep()
    for table in ("messages", "chats", "audit_logs", "proactive_suggestions", "proactive_runs",
                  "proactive_person_settings", "mobile_devices"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) = :w"), {"w": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = :w"), {"w": ws})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = :c"), {"c": clerk})
    s.commit()


def _stored(new_session, suggestion_id) -> NS:
    """The suggestion as another session reads it: what has committed."""
    with new_session() as s:
        row = s.execute(text("SELECT status, delivered_at FROM proactive_suggestions WHERE id = CAST(:id AS uuid)"),
                        {"id": str(suggestion_id)}).one()
    return NS(status=row.status, delivered=row.delivered_at is not None)


def _run_row(new_session, ws) -> UUID:
    from core.models.proactive import ProactiveRun

    run_id = uuid.uuid4()
    with new_session() as s:
        s.add(ProactiveRun(id=run_id, workspace_id=UUID(ws), kind="workspace_check_in", outcome="suggested",
                           window_until=datetime.now(timezone.utc)))
        s.commit()
    return run_id


def _suggestion(new_session, cafe, *, run_id=None, source_type="check_in", subject_key=None, expires_at=None) -> str:
    """An open suggestion for Ana that Start opens a chat for, can wait (so delivery rings no bell)."""
    from core.models.proactive import ProactiveSuggestion

    suggestion_id = uuid.uuid4()
    with new_session() as s:
        s.add(ProactiveSuggestion(
            id=suggestion_id, workspace_id=UUID(cafe.ws), user_id=cafe.ana, source_type=source_type,
            source_agent_id=AGENT_ID if source_type == "streak" else None, run_id=run_id, title=TITLE,
            detail="VECTOR's provider refused it.", evidence=[{"type": "ticket", "id": "0912"}],
            proposed_action={"kind": "chat", "payload": {}}, subject_key=subject_key or f"title:{uuid.uuid4().hex}",
            severity="can_wait", status="open", occurrences=3, created_at=datetime.now(timezone.utc),
            expires_at=expires_at))
        s.commit()
    return str(suggestion_id)


def _decider(cafe):
    from services.proactive.decisions import Decider

    return Decider(user_pk=cafe.ana, created_by=cafe.clerk, actor=f"user:{cafe.ana}", via="web")


def _told(cafe, status: str, sid: str) -> dict:
    return {"workspace_id": cafe.ws, "suggestion_id": sid, "user_id": cafe.ana, "status": status}


def _deliver(new_session, run_id):
    from services.proactive import delivery

    return asyncio.run(delivery.deliver_run(new_session(), run_id))


def _works_again(new_session, cafe) -> None:
    from services.proactive.failing_streaks import note_run

    note_run(new_session(), NS(outcome="silent"), UUID(cafe.ws), AGENT_ID, {}, datetime.now(timezone.utc))


def _streak_card(new_session, cafe) -> str:
    from modules.proactive.dedupe import agent_error_key

    return _suggestion(new_session, cafe, source_type="streak", subject_key=agent_error_key(AGENT_ID, "http_401"))


@pytest.mark.parametrize("source_type", ["check_in", "streak"])
def test_a_delivered_suggestion_is_told_once_as_open_after_its_post_commits(cafe, new_session, source_type):
    run = _run_row(new_session, cafe.ws)
    sid = _suggestion(new_session, cafe, run_id=run, source_type=source_type)

    assert _deliver(new_session, run).posted == 1
    assert _deliver(new_session, run).posted == 0  # delivered once, told once

    [told] = cafe.told
    assert told.sent == _told(cafe, "open", sid)
    assert told.committed == NS(status="open", delivered=True)


def test_shadow_delivers_nothing_and_tells_no_one(cafe, new_session):
    cafe.mode = "shadow"
    run = _run_row(new_session, cafe.ws)
    _suggestion(new_session, cafe, run_id=run)

    assert _deliver(new_session, run).posted == 0
    assert cafe.told == []


def test_start_is_told_once_as_started_after_its_commit(cafe, new_session):
    from services.proactive import decisions

    sid = _suggestion(new_session, cafe)

    started = decisions.start(new_session(), UUID(cafe.ws), UUID(sid), _decider(cafe), "en")
    with pytest.raises(decisions.Refused):
        decisions.start(new_session(), UUID(cafe.ws), UUID(sid), _decider(cafe), "en")

    assert started.suggestion["status"] == "started"
    [told] = cafe.told
    assert told.sent == _told(cafe, "started", sid)
    assert told.committed.status == "started"


def test_hide_is_told_once_as_hidden_after_its_commit_and_hidden_again_tells_no_one(cafe, new_session):
    from services.proactive import decisions

    sid = _suggestion(new_session, cafe)

    hidden = decisions.hide(new_session(), UUID(cafe.ws), UUID(sid), _decider(cafe), "not_now", any_row=False)
    again = decisions.hide(new_session(), UUID(cafe.ws), UUID(sid), _decider(cafe), None, any_row=False)

    assert hidden["status"] == again["status"] == "hidden"
    [told] = cafe.told
    assert told.sent == _told(cafe, "hidden", sid)
    assert told.committed.status == "hidden"


def test_the_streaks_works_again_expiry_is_told_once_as_expired_after_its_commit(cafe, new_session):
    sid = _streak_card(new_session, cafe)

    _works_again(new_session, cafe)
    _works_again(new_session, cafe)  # nothing open is left to expire

    [told] = cafe.told
    assert told.sent == _told(cafe, "expired", sid)
    assert told.committed.status == "expired"


def test_a_lapse_is_told_once_as_expired_after_its_commit(cafe, new_session):
    from services.proactive import decisions, listing

    sid = _suggestion(new_session, cafe, expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))

    with pytest.raises(decisions.Refused) as refused:
        decisions.start(new_session(), UUID(cafe.ws), UUID(sid), _decider(cafe), "en")
    listing.suggestions(new_session(), UUID(cafe.ws), status="open", limit=10, everyone=True, user_id=None)

    assert refused.value.code == "expired"
    [told] = cafe.told
    assert told.sent == _told(cafe, "expired", sid)
    assert told.committed.status == "expired"


def test_a_notify_that_fails_in_postgres_changes_nothing(cafe, new_session, monkeypatch, caplog):
    """An empty channel name is refused by Postgres's pg_notify: each NOTIFY really fails."""
    from services.proactive import decisions

    monkeypatch.setattr(board_events, "NOTIFY_CHANNEL", "")
    run = _run_row(new_session, cafe.ws)
    posted = _suggestion(new_session, cafe, run_id=run)
    to_start, to_hide, card = _suggestion(new_session, cafe), _suggestion(new_session, cafe), _streak_card(
        new_session, cafe)

    with caplog.at_level(logging.ERROR, logger=board_events.__name__):
        delivered = _deliver(new_session, run)
        started = decisions.start(new_session(), UUID(cafe.ws), UUID(to_start), _decider(cafe), "en")
        hidden = decisions.hide(new_session(), UUID(cafe.ws), UUID(to_hide), _decider(cafe), None, any_row=True)
        _works_again(new_session, cafe)

    assert caplog.text.count("pg_notify failed for suggestion") == 4
    assert delivered.posted == 1 and _stored(new_session, posted) == NS(status="open", delivered=True)
    assert started.suggestion["status"] == "started" and _stored(new_session, to_start).status == "started"
    assert hidden["status"] == "hidden" and _stored(new_session, to_hide).status == "hidden"
    assert _stored(new_session, card).status == "expired"
    with new_session() as s:
        audited = s.execute(text("SELECT action FROM audit_logs WHERE CAST(workspace_id AS text) = :w "
                                 "AND resource_id = :id"), {"w": cafe.ws, "id": to_start}).scalars().all()
    assert audited == ["proactive:suggestion_started"]  # the session was still usable after the failed NOTIFY
