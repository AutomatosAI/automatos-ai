"""PRD-268 W2 US-C09b (§6.5.3, §6.9, K9, K20) — who a suggestion push reaches, and when.

Proven here, on Postgres with committing sessions:

* ``POST /api/v1/mobile/devices`` takes ``push_categories`` (the known optional push types:
  ``suggestion`` only), stored on the caller's own row; left out, the stored list stays; an
  unknown type, a null or a non-list is 422 and changes nothing; another person registering the
  same install writes their own row, never the first person's categories.
* ``services.push.suggestion_rules.may_push_suggestion``: only while ``SUGGESTION_PUSH_ENABLED`` is
  on, and only for the event's own person's suggestion in its workspace, of severity ``today``,
  still open (not hidden, not past ``expires_at``). ``eligible_devices`` sends a suggestion only
  to phones whose app registered the category; an ``auto_message`` still reaches a phone that
  registered none.
* End to end, from ``deliver_run`` through the real ``NotificationDispatcher`` to the fan-out
  (handed over inline, to a fake sender): a phone without the category gets nothing; with the
  category and the switch on, one push per run however many suggestions it made; the third
  pushing run of the day pushes nothing; the person's quiet hours and the workspace's push
  nothing (the dispatcher alone holds the push in the workspace's); can_wait pushes nothing.
* Discovery lists ``suggestion_push`` on the registration route only while the switch is on.

CI's test database has no ``notifications`` or ``notification_preferences`` (no model), so the
fixture runs ``prd128_notifications``' own DDL, committed, and drops what it made.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from config import config
from services.mobile import discovery
from services.proactive import delivery
from services.push import fanout, suggestion_rules
from services.push.fanout import PushEvent
from services.push.sender import TICKET_OK, PushTicket

ORCH = Path(__file__).resolve().parents[1]
PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
NOTIFICATION_TABLES = ("notifications", "notification_preferences")
DEVICES = "/api/v1/mobile/devices"
EVENT = "suggestion_ready"
TODAY, CAN_WAIT = "today", "can_wait"
PUSH_KEY = "pk-c09b"  # under 10 characters: gitleaks' generic-api-key never matches it


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT push_categories FROM mobile_devices LIMIT 1"))
            c.execute(text("SELECT delivered_at, expires_at FROM proactive_suggestions LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the suggestion push rules need Postgres with the PRD-266 and PRD-268 tables: {exc}")
    yield eng
    eng.dispose()


class _FakeSender:
    """Accepts every message with a fresh ticket id, and remembers what was sent."""

    def __init__(self):
        self.sent = []

    def send(self, messages, /):
        self.sent.extend(messages)
        return [PushTicket(to=m.to, status=TICKET_OK, ticket_id=str(uuid.uuid4())) for m in messages]

    def receipts(self, ticket_ids, /):
        return {}


@pytest.fixture
def rig(monkeypatch, new_session):
    """SaaS, the platform mode live, the switch on, ``auto_message`` notices recorded, and the
    fan-out handed over inline to a fake sender on a session of its own (``rig.sender.sent``)."""
    from core.llm import manager
    from services.notification_events import auto_message

    state = NS(sender=_FakeSender(), now=datetime.now(timezone.utc))
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", True)
    monkeypatch.setattr(manager, "read_system_setting",
                        lambda category, key: "live" if (category, key) == ("proactive", "mode") else None)
    monkeypatch.setattr(auto_message, "send_soon", lambda coro, *, what: coro.close())
    monkeypatch.setattr(delivery, "_now", lambda: state.now)
    monkeypatch.setattr(fanout, "_deliver_off_loop",
                        lambda event: fanout.deliver(event, sender=state.sender, session_factory=new_session))
    return state


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = NS(execute=statements.append)
    revision.upgrade()
    return statements


def _user(s, tag):
    clerk = f"user_prd268w2pr_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text("INSERT INTO users (username, email, is_active, clerk_user_id) "
                        "VALUES (:u, :e, true, :c) RETURNING id"),
                   {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return NS(pk=int(pk), clerk=clerk, email=f"{clerk}@t.test")


def _workspace(s, owner, settings=None):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) VALUES (CAST(:id AS uuid), "
                   "'prd268w2-suggestion-push-rules', :o, CAST(:s AS jsonb))"),
              {"id": ws, "o": owner, "s": json.dumps(settings or {"proactive": {"enabled": True}})})
    return ws


@pytest.fixture
def cafe(engine, new_session, rig):
    """Ana owns the café's workspace (check-ins on) and Bea's; the bell's tables made when missing."""
    s = new_session()
    built = [t for t in NOTIFICATION_TABLES if not s.execute(text("SELECT to_regclass(:t) IS NOT NULL"),
                                                             {"t": t}).scalar()]
    if built:
        for statement in _revision_statements(PREFERENCES_REVISION):
            s.execute(text(statement))
    ana, bea = _user(s, "ana"), _user(s, "bea")
    ws, other = _workspace(s, ana.pk), _workspace(s, bea.pk)
    s.commit()
    yield NS(ws=ws, other=other, ana=ana, bea=bea)
    s = new_session.sweep()
    scope = {"a": ws, "b": other}
    for table in ("mobile_push_tickets", "mobile_devices", "messages", "chats", "proactive_suggestions",
                  "proactive_runs", "proactive_person_settings", *NOTIFICATION_TABLES):
        if table not in built:
            s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) IN (:a, :b)"),  # noqa: S608 — fixed names
                      scope)
    for table in built:
        s.execute(text(f"DROP TABLE IF EXISTS {table}"))  # noqa: S608 — fixed names
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) IN (:a, :b)"), scope)
    s.execute(text("DELETE FROM users WHERE id IN (:a, :b)"), {"a": ana.pk, "b": bea.pk})
    s.commit()


# --------------------------------------------------------------------------- seeds


def _phone(new_session, ws, user_id, categories=("suggestion",)) -> str:
    """An active iPhone with push on; ``categories`` is what its app registered. Its Expo token."""
    from core.models.mobile_devices import MobileDevice

    tag = uuid.uuid4().hex[:12]
    token = f"ExponentPushToken[{tag}]"
    with new_session() as s:
        s.add(MobileDevice(workspace_id=UUID(ws), user_id=user_id, kind="clerk", status="active",
                           install_id=f"prd268w2pr-{tag}", platform="ios", locale="en", expo_push_token=token,
                           push_enabled=True, push_key=PUSH_KEY, events=[], push_categories=list(categories)))
        s.commit()
    return token


def _run_row(new_session, ws) -> UUID:
    from core.models.proactive import ProactiveRun

    run_id = uuid.uuid4()
    with new_session() as s:
        s.add(ProactiveRun(id=run_id, workspace_id=UUID(ws), kind="workspace_check_in", outcome="suggested",
                           window_until=datetime.now(timezone.utc)))
        s.commit()
    return run_id


def _suggestion(new_session, ws, user_id, run_id=None, *, severity=TODAY, status="open", expires_at=None,
                minutes_ago=0) -> str:
    from core.models.proactive import ProactiveSuggestion

    suggestion_id = uuid.uuid4()
    with new_session() as s:
        s.add(ProactiveSuggestion(
            id=suggestion_id, workspace_id=UUID(ws), user_id=user_id, source_type="check_in", run_id=run_id,
            title="Ticket #0912 failed", proposed_action={"kind": "chat", "payload": {}},
            subject_key=f"title:ticket #0912 failed:{suggestion_id}", severity=severity, status=status,
            expires_at=expires_at, created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)))
        s.commit()
    return str(suggestion_id)


def _event(ws, user_id, suggestion_id, event_type=EVENT, link_type="suggestion") -> PushEvent:
    return PushEvent(workspace_id=ws, event_type=event_type, link_type=link_type, link_id=suggestion_id,
                     user_id=user_id, agent_name="Auto")


def _run(new_session, ws, user_id, *severities) -> str:
    """A run with one suggestion of each severity for ``user_id``, delivered; its newest today one."""
    run = _run_row(new_session, ws)
    made = [(severity, _suggestion(new_session, ws, user_id, run, severity=severity, minutes_ago=len(severities) - i))
            for i, severity in enumerate(severities)]
    asyncio.run(delivery.deliver_run(new_session(), run))
    return next((made_id for severity, made_id in reversed(made) if severity == TODAY), "")


def _bells(new_session, ws) -> int:
    with new_session() as s:
        return int(s.execute(text("SELECT count(*) FROM notifications WHERE workspace_id = CAST(:w AS uuid) "
                                  "AND event_type = :e"), {"w": ws, "e": EVENT}).scalar())


def _quiet_window(now):
    """A window from an hour before ``now`` to an hour after, in UTC, as ``HH:MM``."""
    return (now - timedelta(hours=1)).strftime("%H:%M"), (now + timedelta(hours=1)).strftime("%H:%M")


# --------------------------------------------------------------------------- registration


@pytest.fixture
def client(new_session, monkeypatch):
    """W1's device router; ``client(person)`` calls as that person's Clerk sign-in in the café."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_devices
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    monkeypatch.setattr(mobile_devices, "_rate_store", NS(check=lambda *a, **k: (True, 30, 29, 600)))

    def _make(ws, person):
        app = FastAPI()
        app.include_router(mobile_devices.router)
        app.dependency_overrides[get_db] = lambda: new_session()
        user = UserContext(id=person.clerk, email=person.email, clerk_user_id=person.clerk)
        app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
            workspace_id=UUID(ws), user=user, auth_type="clerk")
        return TestClient(app)

    return _make


def _register(client, install_id, **extra):
    body = {"install_id": install_id, "platform": "ios", "app_version": "1.4.0", "locale": "en-IE",
            "timezone": "Europe/Dublin", **extra}
    return client.post(DEVICES, json=body)


def _categories(new_session, ws, user_id, install_id):
    with new_session() as s:
        return s.execute(text("SELECT push_categories FROM mobile_devices WHERE workspace_id = CAST(:w AS uuid) "
                              "AND user_id = :u AND install_id = :i"),
                         {"w": ws, "u": user_id, "i": install_id}).scalar()


def test_registration_stores_the_categories_and_a_registration_without_them_keeps_them(cafe, new_session, client):
    phone, install = client(cafe.ws, cafe.ana), f"prd268w2pr-{uuid.uuid4().hex[:12]}"

    assert _register(phone, install, push_categories=["suggestion"]).status_code == 200
    stored = _categories(new_session, cafe.ws, cafe.ana.pk, install)
    assert _register(phone, install, app_version="1.4.1").status_code == 200
    kept = _categories(new_session, cafe.ws, cafe.ana.pk, install)
    assert _register(phone, install, push_categories=[]).status_code == 200

    assert (stored, kept, _categories(new_session, cafe.ws, cafe.ana.pk, install)) == (["suggestion"],
                                                                                      ["suggestion"], [])


@pytest.mark.parametrize("categories", [["review"], ["Suggestion"], [""], ["suggestion", "suggestion"], None,
                                        "suggestion"])
def test_registration_refuses_anything_but_the_known_optional_types(cafe, new_session, client, categories):
    phone, install = client(cafe.ws, cafe.ana), f"prd268w2pr-{uuid.uuid4().hex[:12]}"
    assert _register(phone, install, push_categories=["suggestion"]).status_code == 200

    assert _register(phone, install, push_categories=categories).status_code == 422
    assert _categories(new_session, cafe.ws, cafe.ana.pk, install) == ["suggestion"]


def test_a_registration_never_sets_another_persons_categories(cafe, new_session, client):
    install = f"prd268w2pr-{uuid.uuid4().hex[:12]}"
    assert _register(client(cafe.ws, cafe.ana), install, push_categories=[]).status_code == 200

    assert _register(client(cafe.ws, cafe.bea), install, push_categories=["suggestion"]).status_code == 200

    assert _categories(new_session, cafe.ws, cafe.bea.pk, install) == ["suggestion"]  # Bea's own row
    assert _categories(new_session, cafe.ws, cafe.ana.pk, install) == []  # the same install, Ana's row untouched


# --------------------------------------------------------------------------- the rules


def test_may_push_suggestion_holds_only_for_the_persons_own_open_today_suggestion(cafe, new_session, rig,
                                                                                    monkeypatch):
    now, ws, ana = datetime.now(timezone.utc), cafe.ws, cafe.ana.pk
    mine = _suggestion(new_session, ws, ana)

    def may(event):
        with new_session() as s:
            return suggestion_rules.may_push_suggestion(s, event, now)

    assert may(_event(ws, ana, mine)) is True
    assert may(_event(ws, ana, _suggestion(new_session, ws, ana, expires_at=now + timedelta(days=1)))) is True
    refused = [
        _event(ws, ana, _suggestion(new_session, ws, ana, severity=CAN_WAIT)),
        _event(ws, ana, _suggestion(new_session, ws, ana, status="hidden")),
        _event(ws, ana, _suggestion(new_session, ws, ana, expires_at=now - timedelta(minutes=1))),
        _event(ws, cafe.bea.pk, mine),  # another person's suggestion
        _event(cafe.other, ana, mine),  # another workspace's event
        _event(ws, None, mine), _event(ws, ana, mine, link_type="chat"), _event(ws, ana, "0912"),
    ]
    assert [may(event) for event in refused] == [False] * len(refused)
    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", False)
    assert may(_event(ws, ana, mine)) is False


def test_a_suggestion_goes_only_to_phones_that_registered_the_category(cafe, new_session, rig):
    ws, ana = cafe.ws, cafe.ana.pk
    bare, registered = _phone(new_session, ws, ana, categories=()), _phone(new_session, ws, ana)
    suggestion = _suggestion(new_session, ws, ana)

    with new_session() as s:
        pushed = [d.expo_push_token for d in fanout.eligible_devices(s, _event(ws, ana, suggestion), [ana])]
        chat = _event(ws, ana, str(uuid.uuid4()), "auto_message", "chat")
        messaged = {d.expo_push_token for d in fanout.eligible_devices(s, chat, [ana])}

    assert pushed == [registered] and messaged == {bare, registered}  # the other three types are unchanged


# --------------------------------------------------------------------------- end to end


def test_a_phone_without_the_category_gets_nothing_though_the_bell_rings(cafe, new_session, rig):
    _phone(new_session, cafe.ws, cafe.ana.pk, categories=())

    _run(new_session, cafe.ws, cafe.ana.pk, TODAY)

    assert _bells(new_session, cafe.ws) == 1 and rig.sender.sent == []


def test_one_push_per_run_however_many_suggestions_it_made(cafe, new_session, rig):
    token = _phone(new_session, cafe.ws, cafe.ana.pk)

    newest = _run(new_session, cafe.ws, cafe.ana.pk, TODAY, CAN_WAIT, TODAY, TODAY)

    [message] = rig.sender.sent
    assert message.to == token and message.data["type"] == "suggestion"
    assert (message.data["link_type"], message.data["link_id"]) == ("suggestion", newest)
    assert message.category_id == "automatos.suggestion" and message.body == "Auto has a suggestion"


def test_the_third_pushing_run_of_the_day_pushes_nothing(cafe, new_session, rig):
    assert config.PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY == 2
    _phone(new_session, cafe.ws, cafe.ana.pk)

    for _ in range(3):
        _run(new_session, cafe.ws, cafe.ana.pk, TODAY)

    assert len(rig.sender.sent) == 2 and _bells(new_session, cafe.ws) == 2


def test_the_persons_quiet_hours_push_nothing(cafe, new_session, rig):
    _phone(new_session, cafe.ws, cafe.ana.pk)
    start, end = _quiet_window(rig.now)
    with new_session() as s:
        s.execute(text("INSERT INTO proactive_person_settings (workspace_id, user_id, paused, quiet_start, "
                       "quiet_end, timezone) VALUES (CAST(:w AS uuid), :u, false, :a, :b, 'UTC')"),
                  {"w": cafe.ws, "u": cafe.ana.pk, "a": start, "b": end})
        s.commit()

    _run(new_session, cafe.ws, cafe.ana.pk, TODAY)

    assert rig.sender.sent == [] and _bells(new_session, cafe.ws) == 0


def _workspace_quiet(new_session, ws, now) -> None:
    start, end = _quiet_window(now)
    window = {"enabled": True, "quiet_hours": {"enabled": True, "start": start, "end": end, "timezone": "UTC"}}
    with new_session() as s:
        s.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:w AS uuid)"),
                  {"s": json.dumps({"proactive": {"enabled": True}, "auto_reporting": window}), "w": ws})
        s.commit()


def test_the_workspaces_quiet_hours_push_nothing(cafe, new_session, rig):
    _phone(new_session, cafe.ws, cafe.ana.pk)
    _workspace_quiet(new_session, cafe.ws, rig.now)

    _run(new_session, cafe.ws, cafe.ana.pk, TODAY)

    assert rig.sender.sent == [] and _bells(new_session, cafe.ws) == 0


def test_the_dispatcher_alone_holds_the_push_in_the_workspaces_quiet_hours(cafe, new_session, rig):
    from core.services.notification_dispatcher import NotificationDispatcher

    _phone(new_session, cafe.ws, cafe.ana.pk)
    suggestion = _suggestion(new_session, cafe.ws, cafe.ana.pk)
    _workspace_quiet(new_session, cafe.ws, datetime.now(timezone.utc))

    with new_session() as s:
        result = asyncio.run(NotificationDispatcher(s, cafe.ws).dispatch(
            event_type=EVENT, title="Auto has a suggestion", link_type="suggestion", link_id=suggestion,
            agent_name="Auto", user_id=cafe.ana.pk))
        s.commit()

    assert result == {"dispatched_to": ["in_app"]} and rig.sender.sent == []


def test_a_run_of_can_wait_suggestions_pushes_nothing(cafe, new_session, rig):
    _phone(new_session, cafe.ws, cafe.ana.pk)

    _run(new_session, cafe.ws, cafe.ana.pk, CAN_WAIT, CAN_WAIT)

    assert rig.sender.sent == [] and _bells(new_session, cafe.ws) == 0


def test_with_the_switch_off_a_run_rings_the_bell_and_pushes_nothing(cafe, new_session, rig, monkeypatch):
    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", False)
    _phone(new_session, cafe.ws, cafe.ana.pk)

    _run(new_session, cafe.ws, cafe.ana.pk, TODAY)

    assert _bells(new_session, cafe.ws) == 1 and rig.sender.sent == []


# --------------------------------------------------------------------------- discovery


def test_discovery_lists_suggestion_push_only_while_the_switch_is_on(monkeypatch):
    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", False)
    assert "suggestion_push" not in discovery.offered_capabilities()

    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", True)
    assert "suggestion_push" in discovery.offered_capabilities()
    assert discovery.CAPABILITY_ROUTES["suggestion_push"] == ("POST", DEVICES)
    assert "suggestion_push" in discovery.CONFIGURED_CAPABILITIES
