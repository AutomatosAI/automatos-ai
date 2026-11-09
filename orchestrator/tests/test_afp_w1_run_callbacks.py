"""PRD-274 Wave 1 (§8-8): a run an API caller starts reports its ending to a signed callback.

The workspace's signing secret is made once, kept encrypted and rotatable. ``callback_url`` must be
https and resolve outside blocked ranges, checked when the run is asked for and again before every
attempt. When the board's completion writer ends the ticket, the ending is POSTed with the run id,
signed; a 2xx delivers, a 5xx or 429 is retried, any other 4xx is not. Real database.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from api.agent_execute import AgentRunRequest, execute_agent
from core.security.web_access import OutboundTarget
from services import run_callbacks as rc

HOOK = "https://hooks.example.com/runs"
PUBLIC = OutboundTarget(ok=True, reason="", host="hooks.example.com", ip="93.184.216.34", port=443, scheme="https")


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.run_callbacks import RunCallback, WorkspaceSigningSecret

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    for model in (WorkspaceSigningSecret, RunCallback):  # a test database built before the tables
        model.__table__.create(eng, checkfirst=True)
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(new_session):
    from sqlalchemy import text

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1-callbacks')"), {"id": ws})
    agent_id = s.execute(text("INSERT INTO agents (name, agent_type, workspace_id, status) "
                              "VALUES ('Billing', 'custom', CAST(:w AS uuid), 'active') RETURNING id"), {"w": ws}).scalar()
    s.commit()
    s.close()
    yield ws, agent_id
    s = new_session.sweep()
    for table in ("run_callbacks", "workspace_signing_secrets", "approval_grants", "board_tasks", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


@pytest.fixture
def reachable(monkeypatch):
    """hooks.example.com resolves to a public address, at request time and at send time."""
    async def resolve_async(url, *, enforce_switch=True):
        return PUBLIC

    monkeypatch.setattr("core.security.web_access.resolve_outbound", lambda url, *, enforce_switch=True: PUBLIC)
    monkeypatch.setattr("core.security.web_access.resolve_outbound_async", resolve_async)
    monkeypatch.setattr(rc, "BACKOFF_SECONDS", (0, 0, 0))


def _receiver(monkeypatch, *codes):
    """Answers each POST with the next of ``codes``; returns the requests it got."""
    got, answers = [], list(codes)

    def handle(request):
        got.append(request)
        return httpx.Response(answers.pop(0) if len(answers) > 1 else answers[0])

    monkeypatch.setattr(rc, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle),
                                                                  follow_redirects=False))
    return got


def _ctx(ws):
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk",
                           user=SimpleNamespace(id="u-1", clerk_user_id="clerk_owner", email=None))


def _run(new_session, ws, agent_id, url=HOOK):
    s = new_session()
    try:
        return execute_agent(agent_id, AgentRunRequest(input="Draft Acme's renewal reminder.", callback_url=url),
                             ctx=_ctx(ws), db=s)
    finally:
        s.close()


def _end(new_session, task_id, status="done", result="Drafted the reminder.", run_id="run-abc"):
    from datetime import datetime, timezone

    from core.models import BoardTask

    s = new_session()
    task = s.get(BoardTask, task_id)
    task.status, task.result, task.runtime_ref = status, result, {"run_id": run_id}
    task.started_at = task.completed_at = datetime.now(timezone.utc)
    s.commit()
    s.close()


def _callback(new_session, task_id):
    from core.models.run_callbacks import RunCallback

    s = new_session()
    row = s.query(RunCallback).filter(RunCallback.task_id == task_id).first()
    s.close()
    return row


# --- the secret and the signature ------------------------------------------------------------------

def test_the_signing_secret_is_made_once_kept_encrypted_and_rotates(seeded, new_session):
    from core.models.run_callbacks import WorkspaceSigningSecret

    ws, _ = seeded
    s = new_session()
    first = rc.signing_secret(s, ws)
    assert first.startswith("whsec_") and rc.signing_secret(s, ws) == first
    assert first not in s.get(WorkspaceSigningSecret, uuid.UUID(ws)).secret_encrypted
    rotated = rc.rotate_signing_secret(s, ws)
    assert rotated != first and rc.signing_secret(s, ws) == rotated
    assert s.get(WorkspaceSigningSecret, uuid.UUID(ws)).rotated_at is not None
    s.close()


def test_a_receiver_can_verify_a_signature_and_refuses_a_forged_or_stale_one():
    body, now = b'{"task_id":7}', int(time.time())
    header = rc.signature("whsec_k", now, body)
    assert header.startswith(f"t={now},v1=") and rc.verify_signature("whsec_k", header, body)
    assert not rc.verify_signature("whsec_k", header, b'{"task_id":8}')
    assert not rc.verify_signature("whsec_other", header, body)
    assert not rc.verify_signature("whsec_k", rc.signature("whsec_k", now - 301, body), body)
    assert not rc.verify_signature("whsec_k", "v1=abc", body)


# --- the URL -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["https://127.0.0.1/hook", "https://169.254.169.254/latest/meta-data",
                                 "https://10.0.0.5/hook", "file:///etc/passwd"])
def test_a_callback_to_a_blocked_address_is_refused(url):
    with pytest.raises(rc.CallbackUrlRefused):
        rc.check_callback_url(url)


def test_a_callback_must_be_https(monkeypatch):
    plain = OutboundTarget(ok=True, reason="", host="hooks.example.com", ip="93.184.216.34", port=80, scheme="http")
    monkeypatch.setattr("core.security.web_access.resolve_outbound", lambda url, *, enforce_switch=True: plain)
    with pytest.raises(rc.CallbackUrlRefused, match="https"):
        rc.check_callback_url("http://hooks.example.com/runs")


def test_a_refused_callback_files_nothing_and_an_accepted_one_rides_the_tickets_commit(seeded, new_session,
                                                                                       reachable, monkeypatch):
    from sqlalchemy import text

    ws, agent_id = seeded
    monkeypatch.setattr("core.security.web_access.resolve_outbound",
                        lambda url, *, enforce_switch=True: OutboundTarget(ok=False, reason="blocked range"))
    with pytest.raises(HTTPException) as refused:
        _run(new_session, ws, agent_id, "https://intranet.example.com/hook")
    s = new_session()
    filed = s.execute(text("SELECT count(*) FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}).scalar()
    s.close()
    assert refused.value.status_code == 422 and "blocked range" in refused.value.detail and filed == 0

    monkeypatch.setattr("core.security.web_access.resolve_outbound", lambda url, *, enforce_switch=True: PUBLIC)
    answer = _run(new_session, ws, agent_id)
    row = _callback(new_session, answer["task_id"])
    assert answer["callback_url"] == HOOK and (row.url, row.status, row.attempts) == (HOOK, "pending", 0)


# --- delivery ---------------------------------------------------------------------------------------

def test_the_ending_is_posted_signed_with_the_run_id(seeded, new_session, reachable, monkeypatch):
    ws, agent_id = seeded
    got = _receiver(monkeypatch, 200)
    task_id = _run(new_session, ws, agent_id)["task_id"]
    _end(new_session, task_id)
    assert asyncio.run(rc.deliver_run_callback(task_id)) == "delivered"

    (request,) = got
    s = new_session()
    secret = rc.signing_secret(s, ws)
    s.close()
    sent = json.loads(request.content)
    assert rc.verify_signature(secret, request.headers[rc.SIGNATURE_HEADER], request.content)
    assert request.headers[rc.EVENT_HEADER] == "agent_run.finished"
    assert request.headers[rc.DELIVERY_HEADER] == sent["delivery_id"]
    assert request.headers["host"] == "hooks.example.com" and request.url.host == "93.184.216.34"
    assert (sent["task_id"], sent["run_id"], sent["agent_id"], sent["status"], sent["result"]) == (
        task_id, "run-abc", agent_id, "done", "Drafted the reminder.")
    assert "cost" in sent and sent["workspace_id"] == ws
    row = _callback(new_session, task_id)
    assert (row.status, row.attempts, row.last_status_code, row.delivered_at is not None) == ("delivered", 1, 200, True)
    assert asyncio.run(rc.deliver_run_callback(task_id)) is None  # delivered once, never again


def test_a_5xx_or_429_is_retried_and_a_4xx_is_the_receivers_answer(seeded, new_session, reachable, monkeypatch):
    ws, agent_id = seeded
    retried = _receiver(monkeypatch, 503, 429, 200)
    first = _run(new_session, ws, agent_id)["task_id"]
    _end(new_session, first)
    assert asyncio.run(rc.deliver_run_callback(first)) == "delivered" and len(retried) == 3

    refused = _receiver(monkeypatch, 410)
    second = _run(new_session, ws, agent_id)["task_id"]
    _end(new_session, second, status="failed", result=None)
    assert asyncio.run(rc.deliver_run_callback(second)) == "failed" and len(refused) == 1
    row = _callback(new_session, second)
    assert (row.status, row.attempts, row.last_status_code, row.last_error) == ("failed", 1, 410, "HTTP 410")


def test_an_address_that_turns_private_by_send_time_is_never_posted_to(seeded, new_session, reachable, monkeypatch):
    ws, agent_id = seeded
    got = _receiver(monkeypatch, 200)
    task_id = _run(new_session, ws, agent_id)["task_id"]
    _end(new_session, task_id)

    async def rebound(url, *, enforce_switch=True):
        return OutboundTarget(ok=False, reason="resolves to a blocked range")

    monkeypatch.setattr("core.security.web_access.resolve_outbound_async", rebound)
    assert asyncio.run(rc.deliver_run_callback(task_id)) == "failed" and got == []
    assert _callback(new_session, task_id).last_error.startswith("refused")


def test_the_completion_writer_sends_a_pending_callback_and_nothing_else(seeded, new_session, reachable, monkeypatch):
    ws, agent_id = seeded
    launched = []
    monkeypatch.setattr("core.utils.background_tasks.launch_guarded",
                        lambda coro, **kw: launched.append(kw) or coro.close())

    async def finalize(db, **kwargs):
        return kwargs.pop("ended")

    wrapped = rc.sends_run_callback(finalize)
    with_callback = _run(new_session, ws, agent_id)["task_id"]
    s = new_session()
    try:
        assert asyncio.run(wrapped(s, task_id=with_callback, agent_id=agent_id, run_id="r", ended="done")) == "done"
        assert asyncio.run(wrapped(s, task_id=with_callback, agent_id=agent_id, ended=None)) is None
        assert asyncio.run(wrapped(s, task_id=-1, agent_id=agent_id, ended="done")) == "done"
    finally:
        s.close()
    assert [k["operation"] for k in launched] == ["deliver"]


def test_the_boards_completion_writer_is_wrapped():
    from api import board_tasks

    layer, names = board_tasks.finalize_board_task_run, []
    while hasattr(layer, "__wrapped__"):
        names.append(layer.__code__.co_name)
        layer = layer.__wrapped__
    assert names[-1] == "finalize_then_call_back" and layer.__name__ == "finalize_board_task_run"


def test_the_secret_routes_are_for_managers_with_a_recent_sign_in():
    from fastapi.routing import APIRoute, iter_route_contexts

    from core.auth.step_up import STEP_UP_MARKER_ATTR
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app

    routes = {(m, r.path): r for r in iter_route_contexts(app.routes)
              if isinstance(r.original_route, APIRoute) for m in r.methods}
    for key in (("GET", "/api/v1/callbacks/signing-secret"), ("POST", "/api/v1/callbacks/signing-secret/rotate")):
        calls = [d.call for d in routes[key].dependant.dependencies]
        assert "workspace:manage" in [getattr(c, PERMISSION_MARKER_ATTR, None) for c in calls], key
        assert any(getattr(c, STEP_UP_MARKER_ATTR, None) for c in calls), key


def test_the_migration_declares_the_models_shape():
    from pathlib import Path

    from core.models.run_callbacks import RunCallback, WorkspaceSigningSecret

    sql = (Path(__file__).resolve().parents[1] / "alembic" / "versions" / "run_callbacks_20261109.py").read_text()
    for model in (WorkspaceSigningSecret, RunCallback):
        assert f"CREATE TABLE IF NOT EXISTS {model.__tablename__}" in sql
        for column in model.__table__.columns:
            assert f"\n    {column.name} " in sql, (model.__tablename__, column.name)
