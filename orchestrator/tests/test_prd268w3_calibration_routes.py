"""PRD-268 US-C22b (§6.11.5, §12-Q22, K15, K16, K21, K22) — the calibration routes, against Postgres.

``GET`` and ``POST /api/auto/calibration``, ``POST …/{calibration_id}/lines/{line_id}/reject`` and
``POST …/{calibration_id}/accept`` (``api/auto_calibration.py``) over ``services/calibration``; the
model and the memory service are fakes (``tests/helpers_calibration_fakes.py``). Proven here:

* off (as shipped): GET answers ``available`` false and every POST is 404 ``not_available``,
  with nothing read, built, posted or stored; ``CALIBRATION_ENABLED`` defaults to false;
* GET offers it once, to a person who may answer it (a viewer may not), in their language;
* consent posts ONE Auto message ``[text, calibration@1]`` into the caller's own chat, never
  announced, its source {origin calibration, calibration_id}; the record keeps the
  calibration, the chat and the message; one model call; one audit row, which quotes no line;
* another member's chat, a widget visitor's chat (its ``user_id`` the caller's), a shared chat,
  another workspace's chat and no chat at all each land in the caller's own Auto thread;
* decline records ``declined`` with no model call and no message;
* once only: a second POST, either answer, is 409 ``already_offered`` with the record, also
  under two concurrent POSTs (one model call, one message);
* a failed build (the model's error, an answer with no JSON) or a post that does not land is
  503 and leaves no claim, so the next POST tries again;
* a claim whose claimant died before its post is taken over once stale (never while fresh), and
  one whose message did land is recorded, not posted again; a claimant whose claim was taken
  over keeps nothing and posts nothing;
* reject writes one correction and accept one memory per line not rejected (K16), each
  idempotent; a line that names a secret is answered but never stored, and the answer says so;
  once answered, a line can no longer be rejected; a failed write is 503 and changes nothing,
  also when it is the second of an accept's writes;
* another person's calibration, another workspace's (even the caller's own there), a declined
  one, an unknown line and an unknown id are 404, never 403, with nothing written;
* a viewer reads but cannot answer; no person is 403, no workspace 404, a non-member 403; a bad
  id or body is 422; a refusal speaks pt-BR; a database that fails is 503 and says nothing of it;
* discovery lists ``calibration`` only while it is on;
* the four routes are plain ``def``s, the POSTs gated by ``require_workspace_permission``,
  registered through ``RouterSpec`` and in the route manifest.

The ``v_workspace_outputs`` view is migration-managed and CI's create_all database has none: each
request's session makes a TEMP view over ``deliverables`` (as ``test_prd268w3_calibration_build.py``
does), which lives on its pooled connection until the module's engine is disposed. Every request
runs on its own committing session, and each test sweeps its own rows.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.auto_calibration as routes  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services.calibration import build as build_mod  # noqa: E402
from services.calibration import claim, memory, offer  # noqa: E402
from services.calibration.strings import STRINGS  # noqa: E402
from services.calibration.switch import calibration_enabled  # noqa: E402
from services.mobile import discovery  # noqa: E402
from services.notification_events import auto_message  # noqa: E402
from tests.helpers_calibration_fakes import LINE, FakeMemory, FakeModel  # noqa: E402

CALIBRATION = "/api/auto/calibration"
REJECT = "/api/auto/calibration/{}/lines/{}/reject"
ACCEPT = "/api/auto/calibration/{}/accept"
MANIFEST = ORCH / "reports" / "route-manifest.json"
OUTPUTS_VIEW = """
CREATE OR REPLACE TEMP VIEW v_workspace_outputs AS
SELECT d.id, d.workspace_id, d.source_type, d.artifact_type, d.title, d.summary, d.file_path, d.status,
       d.deleted_at, d.created_at
FROM deliverables d
"""
LINES = (("line-1", LINE), ("line-2", "You plan product launches"),
         ("line-3", "You keep the wallet seed phrase in a launch note"))
SOURCE = {"type": "deliverable", "id": "1", "label": "Q3 pricing review"}
SEEDED = {"v": 1, "lines": [{"id": i, "text": t, "sources": [SOURCE]} for i, t in LINES],
          "offers": ["Draft a document from a short brief", "Plan this week's work with you",
                     "Research a topic and summarise it"], "empty": False}


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind, widget_key_id FROM chats LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
            c.execute(text("SELECT rejected_lines FROM proactive_calibrations LIMIT 1"))
            c.execute(text("SELECT actor_type FROM audit_logs LIMIT 1"))
            c.execute(text("SELECT title, deleted_at FROM deliverables LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the calibration route tests need Postgres with the chats, messages and calibrations: {exc}")
    yield eng
    eng.dispose()


def _person(s, tag):
    clerk = f"user_prd268w3cr_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                   {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return NS(clerk=clerk, pk=int(pk))


def _workspace(s, owner):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) "
                   "VALUES (CAST(:id AS uuid), 'prd268w3 calibration routes', :o, CAST('{}' AS jsonb))"),
              {"id": ws, "o": owner})
    return ws


def _member(s, ws, user, role):
    s.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                   "VALUES (CAST(:w AS uuid), :u, :r, true)"), {"w": ws, "u": user, "r": role})


def _chat(s, ws, user, title, *, widget=False, visibility="private"):
    chat = str(uuid.uuid4())
    s.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility, kind, widget_key_id) "
                   "VALUES (CAST(:id AS uuid), :u, CAST(:w AS uuid), :t, :v, 'user', CAST(:wk AS uuid))"),
              {"id": chat, "u": user, "w": ws, "t": title, "v": visibility,
               "wk": str(uuid.uuid4()) if widget else None})
    return chat


def _sweep(new_session, workspaces, clerks):
    s = new_session.sweep()
    for table in ("audit_logs", "messages", "proactive_calibrations", "chats", "deliverables", "workspace_members"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = ANY(:c)"), {"c": clerks})
    s.commit()


@pytest.fixture
def world(engine, new_session):
    """Pia owns workspace A, with a Deliverable and her chat about Lisbon; Oto (an editor), Vic (a
    viewer) and Kim (an editor) are members. Pia also has a widget visitor's chat and a shared
    chat there. Kim owns workspace B, with a chat of her own. Zed belongs to neither."""
    s = new_session()
    pia, oto, vic, kim, zed = (_person(s, tag) for tag in ("pia", "oto", "vic", "kim", "zed"))
    ws, other = _workspace(s, pia.pk), _workspace(s, kim.pk)
    for person, role in ((oto, "editor"), (vic, "viewer"), (kim, "editor")):
        _member(s, ws, person.pk, role)
    s.execute(text("INSERT INTO deliverables (workspace_id, source_type, artifact_type, title, file_path, status) "
                   "VALUES (CAST(:w AS uuid), 'chat', 'report', 'Q3 pricing review', :p, 'ready')"),
              {"w": ws, "p": f"outputs/{uuid.uuid4().hex}.md"})
    t = NS(ws=ws, other=other, pia=pia, oto=oto, vic=vic, kim=kim, zed=zed, new_session=new_session,
           pia_chat=_chat(s, ws, pia.pk, "Pricing for Lisbon"), oto_chat=_chat(s, ws, oto.pk, "Hiring plan"),
           widget_chat=_chat(s, ws, pia.pk, "Widget visitor question", widget=True),
           public_chat=_chat(s, ws, pia.pk, "Shared notes", visibility="public"),
           elsewhere_chat=_chat(s, other, kim.pk, "Kim's own"), unknown_chat=str(uuid.uuid4()), no_chat=None)
    s.commit()
    yield t
    _sweep(new_session, [ws, other], [p.clerk for p in (pia, oto, vic, kim, zed)])


@pytest.fixture
def switched(monkeypatch):
    """Calibration switched on (it ships off)."""
    monkeypatch.setattr(config, "CALIBRATION_ENABLED", True)


@pytest.fixture
def model(monkeypatch):
    """Install a model; called again, it installs another."""

    def install(**kwargs):
        fake = FakeModel(**kwargs)
        monkeypatch.setattr(build_mod, "calibration_llm", lambda workspace_id: fake)
        return fake

    return install


@pytest.fixture
def memories(monkeypatch):
    """Install a memory service; called again, it installs another."""

    def install(**kwargs):
        fake = FakeMemory(**kwargs)
        monkeypatch.setattr(memory, "memory_service", lambda: fake)
        return fake

    return install


@pytest.fixture
def announced(monkeypatch):
    """Every auto_message notice the posts send."""
    sent = []
    monkeypatch.setattr(auto_message, "send_soon", lambda work, **kwargs: (sent.append(kwargs), work.close()))
    return sent


# --------------------------------------------------------------------------- helpers


def _ctx(person, ws, role="user") -> RequestContext:
    user = UserContext(id=person.clerk, clerk_user_id=person.clerk, system_role=role)
    return RequestContext(workspace_id=UUID(ws) if ws else None, user=user, auth_type="clerk")


def _session(world):
    s = world.new_session()
    s.execute(text(OUTPUTS_VIEW))
    return s


def _client(world, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: _session(world)
    return TestClient(app)


def _consent(world, person, consent=True, chat=None, ws=None, headers=None, **more):
    body = {"consent": consent, **({"chat_id": chat} if chat else {}), **more}
    return _client(world, _ctx(person, ws or world.ws)).post(CALIBRATION, json=body, headers=headers or {})


def _read(world, person, ws=None, headers=None):
    return _client(world, _ctx(person, ws or world.ws)).get(CALIBRATION, headers=headers or {})


def _reject(world, person, calibration_id, line_id, ws=None, headers=None):
    return _client(world, _ctx(person, ws or world.ws)).post(REJECT.format(calibration_id, line_id),
                                                             headers=headers or {})


def _accept(world, person, calibration_id, ws=None):
    return _client(world, _ctx(person, ws or world.ws)).post(ACCEPT.format(calibration_id))


def _seed(world, person, ws=None, status="open", calibration=SEEDED, rejected=(), consented=None):
    """A calibration record of ``person``'s, as an offer would have left it (``consented`` its
    ``consented_at``, now by default)."""
    ident = str(uuid.uuid4())
    with world.new_session() as s:
        s.execute(text(
            "INSERT INTO proactive_calibrations (id, workspace_id, user_id, status, calibration, rejected_lines, "
            "consented_at) VALUES (CAST(:id AS uuid), CAST(:w AS uuid), :u, :st, CAST(:cal AS jsonb), "
            "CAST(:rej AS jsonb), :at)"),
            {"id": ident, "w": ws or world.ws, "u": person.pk, "st": status,
             "cal": json.dumps({**calibration, "id": ident} if calibration else {}), "rej": json.dumps(list(rejected)),
             "at": consented or datetime.now(timezone.utc)})
        s.commit()
    return ident


def _landed_message(world, calibration_id):
    """The calibration's message in Pia's Lisbon chat, as a claimant that died recording it left it."""
    ident = str(uuid.uuid4())
    with world.new_session() as s:
        s.execute(text(
            "INSERT INTO messages (id, chat_id, workspace_id, role, parts, source) VALUES (CAST(:id AS uuid), "
            "CAST(:c AS uuid), CAST(:w AS uuid), 'assistant', CAST(:p AS jsonb), CAST(:src AS jsonb))"),
            {"id": ident, "c": world.pia_chat, "w": world.ws, "p": json.dumps([{"type": "text", "text": LINE}]),
             "src": json.dumps({"origin": "calibration", "calibration_id": calibration_id})})
        s.commit()
    return ident


def _records(world, ws=None):
    with world.new_session() as s:
        return s.execute(text(
            "SELECT CAST(id AS text) AS id, user_id, status, calibration, rejected_lines, accepted_lines, "
            "CAST(chat_id AS text) AS chat_id, CAST(message_id AS text) AS message_id, consented_at, decided_at "
            "FROM proactive_calibrations WHERE workspace_id = CAST(:w AS uuid) ORDER BY created_at"),
            {"w": ws or world.ws}).fetchall()


def _messages(world):
    with world.new_session() as s:
        return s.execute(text(
            "SELECT CAST(m.id AS text) AS id, CAST(m.chat_id AS text) AS chat_id, c.kind, c.user_id, m.role, "
            "m.parts, m.source FROM messages m JOIN chats c ON c.id = m.chat_id "
            "WHERE m.workspace_id = CAST(:w AS uuid) ORDER BY m.created_at"), {"w": world.ws}).fetchall()


def _audits(world, ws=None):
    with world.new_session() as s:
        return s.execute(text(
            "SELECT action, user_id, actor_type, resource_id, details FROM audit_logs "
            "WHERE workspace_id = CAST(:w AS uuid) AND resource_type = 'calibration' ORDER BY created_at"),
            {"w": ws or world.ws}).fetchall()


def _refusal(response, status_code, code, locale="en"):
    assert response.status_code == status_code, response.text
    body = response.json()
    assert (body["code"], body["detail"]) == (code, STRINGS[locale][code])
    return body


# --------------------------------------------------------------------------- off, as shipped


def test_calibration_ships_off():
    declared = [line for line in (ORCH / "config.py").read_text(encoding="utf-8").splitlines()
                if re.match(r"\s+CALIBRATION_ENABLED\b", line)]

    assert len(declared) == 1
    assert 'os.getenv("CALIBRATION_ENABLED", "false")' in declared[0]


def test_while_off_get_answers_unavailable_and_every_post_is_404(world, model, memories, monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_ENABLED", False)
    fake, stored = model(), memories()
    otos = _seed(world, world.oto)  # a record exists: the 404 is the switch's, not the lookup's

    read = _read(world, world.pia)

    assert read.status_code == 200, read.text
    assert read.json() == {"available": False, "offer": False, "consent_line": STRINGS["en"]["consent"],
                           "status": None, "calibration": None, "rejected_lines": [], "accepted_lines": []}
    _refusal(_consent(world, world.pia), 404, "not_available")
    _refusal(_consent(world, world.pia, consent=False), 404, "not_available")
    _refusal(_reject(world, world.oto, otos, "line-1"), 404, "not_available")
    _refusal(_accept(world, world.oto, otos), 404, "not_available")
    assert fake.calls == [] and stored.calls == [] and _messages(world) == [] and _audits(world) == []
    assert [(r.user_id, r.status, r.rejected_lines) for r in _records(world)] == [(world.oto.pk, "open", [])]


# --------------------------------------------------------------------------- the offer


def test_get_offers_it_once_to_a_person_who_may_answer_it_in_their_language(world, switched):
    pias = _read(world, world.pia)
    in_portuguese = _read(world, world.pia, headers={"Accept-Language": "pt-BR,pt;q=0.9"})
    vics = _read(world, world.vic)

    assert pias.status_code == 200, pias.text
    assert pias.json() == {"available": True, "offer": True, "consent_line": STRINGS["en"]["consent"],
                           "status": None, "calibration": None, "rejected_lines": [], "accepted_lines": []}
    assert in_portuguese.json()["consent_line"] == STRINGS["pt-BR"]["consent"]
    assert vics.status_code == 200 and vics.json()["offer"] is False  # a viewer cannot answer it


def test_consent_posts_one_message_with_the_part_into_the_callers_own_chat(world, switched, model, announced):
    fake = model()

    posted = _consent(world, world.pia, chat=world.pia_chat)

    assert posted.status_code == 200, posted.text
    body = posted.json()
    calibration = body["calibration"]
    assert (body["status"], body["chat_id"], calibration["v"], calibration["empty"]) == (
        "open", world.pia_chat, 1, False)
    assert [line["text"] for line in calibration["lines"]] == [LINE] and len(fake.calls) == 1
    (message,) = _messages(world)
    assert (message.id, message.chat_id, message.role) == (body["message_id"], world.pia_chat, "assistant")
    words, part = message.parts
    assert words["type"] == "text" and f"{LINE} (from 1 Deliverable and 1 Conversation)" in words["text"]
    assert part == {"type": "calibration@1", "calibration": calibration}
    assert (message.source["origin"], message.source["calibration_id"]) == ("calibration", calibration["id"])
    assert announced == []  # it answers the person's own click
    (record,) = _records(world)
    assert (record.id, record.status, record.calibration, record.chat_id, record.message_id) == (
        calibration["id"], "open", calibration, world.pia_chat, body["message_id"])
    assert record.consented_at is not None and record.decided_at is None
    (audit,) = _audits(world)
    assert (audit.action, audit.user_id, audit.actor_type, audit.resource_id, audit.details) == (
        "proactive:calibration_consented", world.pia.pk, "user", calibration["id"], {"lines": 1, "empty": False})
    read = _read(world, world.pia).json()
    assert (read["offer"], read["status"], read["calibration"]) == (False, "open", calibration)


@pytest.mark.parametrize("chat", ["oto_chat", "widget_chat", "public_chat", "elsewhere_chat", "unknown_chat",
                                  "no_chat"])
def test_a_chat_not_the_callers_own_falls_back_to_their_auto_thread(world, switched, model, chat):
    model()

    posted = _consent(world, world.pia, chat=getattr(world, chat))

    assert posted.status_code == 200, posted.text
    (message,) = _messages(world)
    assert (message.kind, message.user_id, message.chat_id) == ("auto", world.pia.pk, posted.json()["chat_id"])
    assert message.chat_id != getattr(world, chat)


def test_decline_records_declined_with_no_model_call_and_no_message(world, switched, model):
    fake = model()

    declined = _consent(world, world.pia, consent=False)

    assert declined.status_code == 200, declined.text
    assert declined.json() == {"status": "declined", "calibration": None, "chat_id": None, "message_id": None}
    assert fake.calls == [] and _messages(world) == []
    (record,) = _records(world)
    assert (record.status, record.calibration, record.consented_at) == ("declined", {}, None)
    assert record.decided_at is not None
    assert [(a.action, a.user_id, a.details) for a in _audits(world)] == [
        ("proactive:calibration_declined", world.pia.pk, {})]
    assert (_read(world, world.pia).json()["offer"], _read(world, world.pia).json()["status"]) == (False, "declined")


def test_a_second_post_either_answer_is_409_already_offered_with_the_record(world, switched, model):
    fake = model()
    first = _consent(world, world.pia).json()

    for again in (_consent(world, world.pia), _consent(world, world.pia, consent=False)):
        record = _refusal(again, 409, "already_offered")["record"]
        assert (record["status"], record["calibration"], record["message_id"]) == (
            "open", first["calibration"], first["message_id"])
    _consent(world, world.oto, consent=False)
    record = _refusal(_consent(world, world.oto), 409, "already_offered")["record"]
    assert (record["status"], record["calibration"]) == ("declined", None)
    assert len(fake.calls) == 1 and len(_messages(world)) == 1 and len(_records(world)) == 2


def test_two_concurrent_posts_offer_it_once(world, switched, model):
    fake = model(delay=0.5)
    barrier, answers = threading.Barrier(2), []

    def post():
        client = _client(world, _ctx(world.pia, world.ws))
        barrier.wait()
        answers.append(client.post(CALIBRATION, json={"consent": True}))

    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert sorted(answer.status_code for answer in answers) == [200, 409]
    assert len(fake.calls) == 1 and len(_messages(world)) == 1 and len(_records(world)) == 1


@pytest.mark.parametrize("failing", [{"failure": RuntimeError("provider at 10.0.0.7 refused")},
                                     {"content": "I would rather not say"}])
def test_a_failed_build_leaves_no_claim_and_the_next_post_tries_again(world, switched, model, failing):
    model(**failing)

    failed = _consent(world, world.pia)

    _refusal(failed, 503, "calibration_unavailable")
    assert "10.0.0.7" not in failed.text and "rather" not in failed.text
    assert _records(world) == [] and _messages(world) == [] and _audits(world) == []
    model()
    assert _consent(world, world.pia).status_code == 200
    assert len(_records(world)) == 1 and len(_messages(world)) == 1


def test_a_post_that_does_not_land_gives_the_claim_back(world, switched, model, monkeypatch):
    fake = model()
    monkeypatch.setattr(offer, "deliver_background_message", lambda db, **kwargs: None)

    _refusal(_consent(world, world.pia), 503, "calibration_unavailable")

    assert len(fake.calls) == 1 and _records(world) == [] and _messages(world) == []
    assert _read(world, world.pia).json()["offer"] is True


def test_a_claim_whose_claimant_died_is_taken_over_once_stale(world, switched, model):
    fake = model()
    stale = _seed(world, world.pia, calibration=None, consented=datetime.now(timezone.utc) - timedelta(minutes=11))

    posted = _consent(world, world.pia, chat=world.pia_chat)

    assert posted.status_code == 200, posted.text
    assert posted.json()["calibration"]["id"] == stale and len(fake.calls) == 1
    (record,) = _records(world)
    assert (record.id, record.status, record.message_id) == (stale, "open", posted.json()["message_id"])
    assert len(_messages(world)) == 1


def test_a_fresh_claim_is_never_taken_over(world, switched, model):
    fake = model()
    _seed(world, world.pia, calibration=None, consented=datetime.now(timezone.utc) - timedelta(minutes=1))

    record = _refusal(_consent(world, world.pia), 409, "already_offered")["record"]

    assert (record["status"], record["calibration"], record["message_id"]) == ("open", None, None)
    assert fake.calls == [] and _messages(world) == []
    assert _read(world, world.pia).json()["offer"] is False


def test_get_offers_it_again_once_a_claim_is_abandoned(world, switched):
    _seed(world, world.pia, calibration=None, consented=datetime.now(timezone.utc) - timedelta(minutes=11))
    _seed(world, world.oto, consented=datetime.now(timezone.utc) - timedelta(minutes=11))  # built, never posted
    _seed(world, world.kim, status="answered", consented=datetime.now(timezone.utc) - timedelta(days=2))

    pias, otos, kims = (_read(world, person).json() for person in (world.pia, world.oto, world.kim))

    assert (pias["offer"], pias["status"], pias["calibration"]) == (True, "open", None)
    assert otos["offer"] is True
    assert kims["offer"] is False  # answered long ago: never offered again


def test_a_stale_claim_whose_message_landed_is_recorded_not_posted_again(world, switched, model):
    fake = model()
    stale = _seed(world, world.pia, consented=datetime.now(timezone.utc) - timedelta(minutes=11))
    landed = _landed_message(world, stale)

    record = _refusal(_consent(world, world.pia), 409, "already_offered")["record"]

    assert (record["status"], record["chat_id"], record["message_id"]) == ("open", world.pia_chat, landed)
    assert fake.calls == [] and [m.id for m in _messages(world)] == [landed]


def test_a_claimant_whose_claim_was_taken_over_keeps_and_posts_nothing(world):
    held = datetime.now(timezone.utc) - timedelta(minutes=11)
    ident = _seed(world, world.pia, calibration=None, consented=held)
    later = datetime.now(timezone.utc)

    with world.new_session() as s:
        taken = claim.take_over(s, world.ws, world.pia.pk, ident, later)
        assert taken == claim.Lease(UUID(ident), later)
        assert claim.take_over(s, world.ws, world.pia.pk, ident, later) is None  # fresh again: one taker
        with pytest.raises(claim.LeaseLost):
            claim.hold(s, claim.Lease(UUID(ident), held), dict(SEEDED, id=ident), later)
        claim.release(s, claim.Lease(UUID(ident), held))  # not the caller's any more: nothing goes

    (record,) = _records(world)
    assert (record.id, record.calibration, record.consented_at) == (ident, {}, later)


# --------------------------------------------------------------------------- the answers


def test_reject_writes_one_correction_and_is_idempotent(world, switched, memories):
    stored = memories()
    pias = _seed(world, world.pia)

    answered = [_reject(world, world.pia, pias, "line-1") for _ in range(2)]

    for response in answered:
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["status"], body["rejected_lines"], body["accepted_lines"]) == ("open", ["line-1"], [])
        assert (body["not_stored"], body["note"]) == ([], None)
    (call,) = stored.calls
    assert (call["content"], call["subject_id"]) == (f"Not so: {LINE}", f"user:{world.pia.pk}")
    meta = call["metadata"]
    assert (meta["owner"], meta["scope"], meta["source_type"], meta["correction"], meta["line_id"]) == (
        f"user:{world.pia.pk}", "private", "platform_verified", True, "line-1")
    assert [(a.action, a.details) for a in _audits(world)] == [
        ("proactive:calibration_line_rejected", {"line_id": "line-1", "stored": True})]


def test_accept_remembers_each_line_not_rejected_once(world, switched, memories):
    stored = memories()
    pias = _seed(world, world.pia, rejected=["line-1"])

    first, again = _accept(world, world.pia, pias), _accept(world, world.pia, pias)

    for response in (first, again):
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["status"], body["rejected_lines"], body["accepted_lines"]) == (
            "answered", ["line-1"], ["line-2", "line-3"])
    assert (first.json()["not_stored"], first.json()["note"]) == (["line-3"], STRINGS["en"]["not_stored"])
    assert again.json()["not_stored"] == []
    assert [(c["content"], c["metadata"]["line_id"]) for c in stored.calls] == [("You plan product launches", "line-2")]
    assert "correction" not in stored.calls[0]["metadata"]
    assert [(a.action, a.details) for a in _audits(world)] == [
        ("proactive:calibration_accepted", {"accepted": 2, "stored": 1})]
    assert _records(world)[0].decided_at is not None
    _refusal(_reject(world, world.pia, pias, "line-2"), 409, "calibration_answered")
    assert _reject(world, world.pia, pias, "line-1").status_code == 200  # already rejected: as it is
    assert len(stored.calls) == 1


def test_a_line_that_names_a_secret_is_rejected_but_never_stored(world, switched, memories):
    stored = memories()
    pias = _seed(world, world.pia)

    rejected = _reject(world, world.pia, pias, "line-3")

    assert rejected.status_code == 200, rejected.text
    body = rejected.json()
    assert (body["rejected_lines"], body["not_stored"], body["note"]) == (
        ["line-3"], ["line-3"], STRINGS["en"]["not_stored"])
    assert stored.calls == []
    (audit,) = _audits(world)
    assert audit.details == {"line_id": "line-3", "stored": False} and "seed" not in json.dumps(audit.details)


def test_a_failed_memory_write_is_503_and_changes_nothing(world, switched, memories):
    memories(answer={"success": False, "error": "durable memory not configured (QDRANT_URL empty)"})
    pias = _seed(world, world.pia)

    rejected, accepted = _reject(world, world.pia, pias, "line-1"), _accept(world, world.pia, pias)

    for response in (rejected, accepted):
        _refusal(response, 503, "memory_unavailable")
        assert "QDRANT" not in response.text
    (record,) = _records(world)
    assert (record.status, record.rejected_lines, record.accepted_lines, record.decided_at) == ("open", [], [], None)
    assert _audits(world) == []


def test_an_accept_whose_second_write_fails_is_given_back_and_a_retry_answers_it(world, switched, memories):
    seeded = dict(SEEDED, lines=[{"id": i, "text": t, "sources": [SOURCE]} for i, t in LINES[:2]])
    pias = _seed(world, world.pia, calibration=seeded)
    first = memories(answers=[{"success": True, "id": "point-1"}, {"success": False, "error": "timed out"}])

    _refusal(_accept(world, world.pia, pias), 503, "memory_unavailable")

    assert [c["metadata"]["line_id"] for c in first.calls] == ["line-1", "line-2"]
    (record,) = _records(world)
    assert (record.status, record.accepted_lines, record.decided_at) == ("open", [], None)
    again = memories()
    answered = _accept(world, world.pia, pias)
    assert answered.status_code == 200 and answered.json()["accepted_lines"] == ["line-1", "line-2"]
    assert [c["metadata"]["line_id"] for c in again.calls] == ["line-1", "line-2"]


def test_another_person_another_workspace_and_a_declined_one_are_404(world, switched, memories):
    stored = memories()
    pias, kims_there = _seed(world, world.pia), _seed(world, world.kim, ws=world.other)
    otos_declined = _seed(world, world.oto, status="declined", calibration=None)
    kims_unbuilt = _seed(world, world.kim, calibration=None)  # her own, claimed but never built: nothing to answer

    for response in (_reject(world, world.oto, pias, "line-1"), _accept(world, world.oto, pias),
                     _accept(world, world.kim, kims_there),  # her own, asked from another workspace
                     _accept(world, world.kim, pias, ws=world.other),  # Pia's, asked from Kim's workspace
                     _accept(world, world.oto, otos_declined), _reject(world, world.oto, otos_declined, "line-1"),
                     _accept(world, world.pia, uuid.uuid4()), _accept(world, world.kim, kims_unbuilt)):
        _refusal(response, 404, "calibration_not_found")
    _refusal(_reject(world, world.pia, pias, "line-9"), 404, "line_not_found")
    assert stored.calls == [] and _audits(world) == [] and _audits(world, world.other) == []
    assert [(r.status, r.rejected_lines) for r in _records(world)] == [("open", []), ("declined", []), ("open", [])]
    assert [(r.status, r.accepted_lines) for r in _records(world, world.other)] == [("open", [])]


# --------------------------------------------------------------------------- who may, and what is refused


def test_a_viewer_reads_but_cannot_answer(world, switched, model, memories):
    fake, stored = model(), memories()
    vics = _seed(world, world.vic)

    assert _read(world, world.vic).status_code == 200
    assert _consent(world, world.vic).status_code == 403
    assert _accept(world, world.vic, vics).status_code == 403
    assert _reject(world, world.vic, vics, "line-1").status_code == 403
    assert fake.calls == [] and stored.calls == [] and _records(world)[0].status == "open"


def test_no_person_is_403_no_workspace_404_and_a_non_member_403(world, switched, model):
    fake = model()
    nobody = _client(world, _ctx(NS(clerk=f"user_prd268w3cr_nobody_{uuid.uuid4().hex[:10]}"), world.ws,
                                 "super_admin"))
    nowhere = _client(world, _ctx(world.pia, None, "super_admin"))
    outsider = _client(world, _ctx(world.zed, world.ws, "super_admin"))

    _refusal(nobody.get(CALIBRATION), 403, "calibration_no_person")
    _refusal(nobody.post(CALIBRATION, json={"consent": True}), 403, "calibration_no_person")
    _refusal(nowhere.get(CALIBRATION), 404, "calibration_no_workspace")
    _refusal(nowhere.post(ACCEPT.format(uuid.uuid4())), 404, "calibration_no_workspace")
    _refusal(outsider.post(CALIBRATION, json={"consent": True}), 403, "calibration_not_a_member")
    assert fake.calls == [] and _records(world) == []


def test_a_bad_id_or_body_is_422(world, switched, model):
    fake = model()
    pias = _seed(world, world.pia)
    client = _client(world, _ctx(world.pia, world.ws))

    assert client.post(ACCEPT.format("not-a-uuid")).status_code == 422
    for line_id in ("line 1", "x" * (routes.LINE_ID_MAX + 1)):
        assert client.post(REJECT.format(pias, line_id)).status_code == 422, line_id
    for body in ({"consent": "yes"}, {"consent": 1}, {}, {"consent": True, "person": 7},
                 {"consent": True, "chat_id": "lisbon"}):
        assert client.post(CALIBRATION, json=body).status_code == 422, body
    assert fake.calls == [] and _records(world)[0].rejected_lines == []


def test_a_refusal_speaks_the_callers_language(world, switched, model):
    model()
    _consent(world, world.pia, consent=False)

    _refusal(_consent(world, world.pia, locale="pt-BR"), 409, "already_offered", "pt-BR")
    _refusal(_reject(world, world.pia, uuid.uuid4(), "line-1", headers={"Accept-Language": "pt-BR"}), 404,
             "calibration_not_found", "pt-BR")
    codes = {routes.NOT_AVAILABLE, routes.NO_WORKSPACE, routes.NO_PERSON, routes.NOT_A_MEMBER, routes.ALREADY_OFFERED,
             routes.ANSWERED, routes.UNAVAILABLE, routes.MEMORY_UNAVAILABLE, routes.NOT_STORED,
             "calibration_not_found", "line_not_found"}
    assert all(code in STRINGS["en"] and code in STRINGS["pt-BR"] for code in codes)


def test_a_database_that_fails_is_503_and_says_nothing_of_it(world, switched, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise OperationalError("SELECT secret_column", {}, Exception("connection to 10.0.0.7 lost"))

    for name in ("record_of", "claim", "own_record"):
        monkeypatch.setattr(claim, name, broken)

    with caplog.at_level(logging.ERROR, logger=routes.__name__):
        answered = [_read(world, world.pia), _consent(world, world.pia, consent=False),
                    _accept(world, world.pia, uuid.uuid4())]

    for response in answered:
        _refusal(response, 503, "calibration_unavailable")
        assert "10.0.0.7" not in response.text and "secret_column" not in response.text
    records = [r for r in caplog.records if r.name == routes.__name__ and r.levelno == logging.ERROR]
    assert len(records) == 3 and all(isinstance(r.exc_info[1], OperationalError) for r in records)


# --------------------------------------------------------------------------- discovery and the routes


def test_discovery_lists_calibration_only_while_it_is_on(monkeypatch):
    assert discovery.CAPABILITY_ROUTES["calibration"] == ("POST", CALIBRATION)
    assert discovery.CONFIGURED_CAPABILITIES["calibration"] is calibration_enabled

    monkeypatch.setattr(config, "CALIBRATION_ENABLED", False)
    assert "calibration" not in discovery.offered_capabilities()
    monkeypatch.setattr(config, "CALIBRATION_ENABLED", True)
    assert "calibration" in discovery.offered_capabilities()


def test_the_routes_are_plain_defs_gated_registered_and_in_the_manifest():
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    handlers = (routes.read_calibration, routes.offer_calibration, routes.reject_calibration_line,
                routes.accept_calibration)
    assert not any(inspect.iscoroutinefunction(handler) for handler in handlers)
    assert RouterSpec("api.auto_calibration") in MANIFEST_ROUTERS
    gates = {}
    for route in routes.router.routes:
        (method,) = route.methods
        gates[(method, route.path)] = [getattr(d.call, PERMISSION_MARKER_ATTR, None)
                                       for d in route.dependant.dependencies]
    reject, accept = REJECT.format("{calibration_id}", "{line_id}"), ACCEPT.format("{calibration_id}")
    assert gates[("GET", CALIBRATION)].count("members:read") == 1
    for key in (("POST", CALIBRATION), ("POST", reject), ("POST", accept)):
        assert gates[key].count("agents:execute") == 1, key  # the authz sweep's class (a)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for method, path in (("GET", CALIBRATION), ("POST", CALIBRATION), ("POST", reject), ("POST", accept)):
        assert {"method": method, "path": path} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
