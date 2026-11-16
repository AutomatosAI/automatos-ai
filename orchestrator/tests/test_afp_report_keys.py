"""A product's server key reports an agent's reply (``POST /api/v1/reports``, ``reports:write``).

The route takes a member with ``members:read`` as before, or a server key listing ``reports:write``
(``core/auth/server_key_gate.py``). A key reports as its own service user, the ``users`` row its
chat turns run as, and reaches only an assistant message in a chat that user holds, tagged with the
end user it names (``chats.end_user_id``), or tagged with none when it names none
(``services/content_reports/reporter.py``). Everything else is not found and writes nothing:
another end user's chat, the key's chats for no one, another key's chat, a member's chat (public
ones too), a person's own message, a chat in another workspace, an end user whose erase has started,
a Deliverable. A key that has never chatted has no service user, so whatever it reports is not found
and no service user is made. Its audit row names the key, never the end user; it counts against one
bucket per end user.

Real keys, users, chats and messages on the real database (CI); the owner's notice dispatcher, the
rate limiter and the Deliverable reader are fakes that record what they are given.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import bindparam, text
from starlette.requests import Request

from core.auth import server_key_gate as gate
from core.auth.scopes import AGENTS_CHAT, REPORTS_WRITE
from services.content_reports import AUDIT_ACTION
from services.content_reports import service, targets
from services.content_reports.reporter import KEY_ACTOR_TYPE

ALPHA, BETA, GONE = "product:alpha", "product:beta", "product:gone"
REPLY = "Try a short walk before lunch."


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


def _user(s, name):
    tag = uuid.uuid4().hex[:10]
    return s.execute(text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
                     {"e": f"{name}-{tag}@reports.example.test", "u": f"{name}-{tag}", "c": f"user_{tag}"}).scalar()


def _mint(new_session, ws, permissions=(REPORTS_WRITE, AGENTS_CHAT)):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        return ApiKeyService.create_api_key(s, uuid.UUID(ws), "product backend", key_type="server",
                                            permissions=list(permissions))
    finally:
        s.close()


def _service_user(new_session, key):
    from api.chat_server_key import service_user_for

    s = new_session()
    try:
        return service_user_for(s, key["id"])
    finally:
        s.close()


def _message(s, ws, owner, *, end_user=None, role="assistant", visibility="private"):
    chat = s.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility, end_user_id) VALUES "
                          "(gen_random_uuid(), :u, CAST(:w AS uuid), 'walk', :v, :e) RETURNING id"),
                     {"u": owner, "w": ws, "v": visibility, "e": end_user}).scalar()
    return str(s.execute(text("INSERT INTO messages (id, chat_id, workspace_id, role, parts) VALUES "
                              "(gen_random_uuid(), :c, CAST(:w AS uuid), :r, CAST(:p AS jsonb)) RETURNING id"),
                         {"c": chat, "w": ws, "r": role, "p": f'[{{"type": "text", "text": "{REPLY}"}}]'}).scalar())


@pytest.fixture(autouse=True)
def unlimited(monkeypatch):
    """The gate's per-key limits never refuse (``tests/test_afp_w1_document_keys.py``)."""

    class _Open:
        def check(self, *args, **kwargs):
            return (True, 1, 0, 60)

    monkeypatch.setattr("api.widgets.rate_limit.RateLimitStore", _Open)


def _consent(s, ws, end_user, *, erasing=False):
    s.execute(text("INSERT INTO end_users (id, workspace_id, end_user_id, erasing_at) VALUES (gen_random_uuid(), "
                   "CAST(:w AS uuid), :e, CASE WHEN :erasing THEN now() END)"),
              {"w": ws, "e": end_user, "erasing": erasing})


@pytest.fixture
def floor(new_session):
    """Key A (chatted), key B (chatted), key C (never chatted), an owner, and one message per case."""
    ws, other_ws = str(uuid.uuid4()), str(uuid.uuid4())
    s = new_session()
    for workspace_id in (ws, other_ws):
        s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'report-keys')"),
                  {"id": workspace_id})
    for end_user in (ALPHA, BETA):
        _consent(s, ws, end_user)
    _consent(s, ws, GONE, erasing=True)
    owner = _user(s, "owner")
    s.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                   "VALUES (CAST(:w AS uuid), :u, 'owner', TRUE)"), {"w": ws, "u": owner})
    s.commit()
    a, b, c = _mint(new_session, ws), _mint(new_session, ws), _mint(new_session, ws)
    a_user, b_user = _service_user(new_session, a), _service_user(new_session, b)
    msgs = {
        "alphas_reply": _message(s, ws, a_user, end_user=ALPHA),
        "alphas_own_words": _message(s, ws, a_user, end_user=ALPHA, role="user"),
        "betas_reply": _message(s, ws, a_user, end_user=BETA),
        "the_keys_own_reply": _message(s, ws, a_user),
        "another_keys_reply_for_alpha": _message(s, ws, b_user, end_user=ALPHA),
        "a_members_public_reply": _message(s, ws, owner, visibility="public"),
        "a_members_private_reply": _message(s, ws, owner),
        "alphas_reply_in_another_workspace": _message(s, other_ws, a_user, end_user=ALPHA),
        "an_erased_end_users_reply": _message(s, ws, a_user, end_user=GONE),
    }
    s.commit()
    s.close()
    yield SimpleNamespace(ws=ws, owner=owner, a=a, b=b, c=c, a_user=a_user, msgs=msgs)
    s = new_session.sweep()
    both = {"w": ws, "o": other_ws}
    for table in ("chats", "audit_logs", "workspace_members", "end_users", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id IN (CAST(:w AS uuid), CAST(:o AS uuid))"),  # noqa: S608
                  both)
    s.execute(text("DELETE FROM users WHERE id IN :ids").bindparams(bindparam("ids", expanding=True)),
              {"ids": [owner, a_user, b_user]})
    s.execute(text("DELETE FROM workspaces WHERE id IN (CAST(:w AS uuid), CAST(:o AS uuid))"), both)
    s.commit()
    s.close()


@pytest.fixture
def fakes(monkeypatch):
    """The owner's notices, the rate-limit checks, and any Deliverable read (none is expected)."""
    seen = SimpleNamespace(notices=[], limits=[], deliverables=[])

    class _Dispatcher:
        def __init__(self, db, workspace_id):
            self.workspace_id = str(workspace_id)

        async def dispatch(self, **kwargs):
            seen.notices.append(kwargs)
            return {"dispatched_to": ["in_app"]}

    async def _check(workspace_id, operation, **kwargs):
        seen.limits.append((operation, kwargs["subject_id"]))

    async def _deliverable(db, **kwargs):
        seen.deliverables.append(kwargs)
        return "A Deliverable"

    monkeypatch.setattr(service, "NotificationDispatcher", _Dispatcher)
    monkeypatch.setattr(service.rate_limiter, "check_rate_limit", _check)
    monkeypatch.setattr(targets, "deliverable_excerpt", _deliverable)
    return seen


def _key_ctx(key):
    request = Request({"type": "http", "method": "POST", "path": "/", "query_string": b"", "path_params": {},
                       "headers": [(b"authorization", f"Bearer {key['key']}".encode())]})
    gated = gate.require_workspace_permission_or_server_key("members:read", REPORTS_WRITE)
    return asyncio.run(gated(request, db=None))


def _report(new_session, key, target_id, *, end_user=ALPHA, target_type="message"):
    s = new_session()
    try:
        return asyncio.run(service.file_report(s, _key_ctx(key), target_type=target_type, target_id=target_id,
                                               reason="harmful", note=None, end_user_id=end_user))
    finally:
        s.close()


def _rows(new_session, ws):
    s = new_session()
    try:
        return s.execute(text("SELECT user_id, actor_type, resource_id, details FROM audit_logs WHERE "
                              "workspace_id = CAST(:w AS uuid) AND action = :a"), {"w": ws, "a": AUDIT_ACTION}).all()
    finally:
        s.close()


def test_the_route_takes_a_member_or_a_key_listing_reports_write():
    from fastapi.routing import APIRoute, iter_route_contexts

    from api.api_keys import VALID_PERMISSIONS, ApiKeyCreateRequest
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app

    route = next(r for r in iter_route_contexts(app.routes)
                 if isinstance(r.original_route, APIRoute) and r.path == "/api/v1/reports" and "POST" in r.methods)
    marked = [(getattr(d.call, PERMISSION_MARKER_ATTR, None), getattr(d.call, gate.KEY_SCOPE_MARKER_ATTR))
              for d in route.dependant.dependencies if getattr(d.call, gate.KEY_SCOPE_MARKER_ATTR, None)]
    assert marked == [("members:read", REPORTS_WRITE)]
    assert REPORTS_WRITE in VALID_PERMISSIONS
    assert ApiKeyCreateRequest(name="p", permissions=[REPORTS_WRITE]).permissions == [REPORTS_WRITE]


def test_a_key_without_reports_write_is_refused(new_session, floor):
    unscoped = _mint(new_session, floor.ws, permissions=(AGENTS_CHAT,))
    with pytest.raises(HTTPException) as refused:
        _key_ctx(unscoped)
    assert refused.value.status_code == 403


def test_a_key_reports_its_end_users_reply_as_its_service_user(new_session, floor, fakes):
    filed = _report(new_session, floor.a, floor.msgs["alphas_reply"])

    assert (filed.duplicate, filed.target_id, filed.excerpt) == (False, floor.msgs["alphas_reply"], REPLY)
    (row,) = _rows(new_session, floor.ws)
    assert (row.user_id, row.actor_type, row.resource_id) == (floor.a_user, KEY_ACTOR_TYPE, filed.target_id)
    assert row.details == {"reason": "harmful", "note": None, "excerpt": REPLY, "api_key_id": floor.a["id"]}
    assert fakes.limits == [(service.RATE_LIMIT_OPERATION, f"{floor.a_user}:end_user:{ALPHA}")]
    assert [n["user_id"] for n in fakes.notices] == [floor.owner]


def test_the_same_reply_again_answers_the_first_report(new_session, floor, fakes):
    first = _report(new_session, floor.a, floor.msgs["alphas_reply"])
    again = _report(new_session, floor.a, floor.msgs["alphas_reply"])

    assert (again.duplicate, again.report_id) == (True, first.report_id)
    assert len(_rows(new_session, floor.ws)) == 1 and len(fakes.notices) == 1


@pytest.mark.parametrize("case", [
    "betas_reply", "the_keys_own_reply", "another_keys_reply_for_alpha",
    "a_members_public_reply", "a_members_private_reply", "alphas_own_words", "alphas_reply_in_another_workspace",
])
def test_a_reply_outside_the_end_users_chats_with_this_key_is_not_found(new_session, floor, fakes, case):
    with pytest.raises(service.ReportTargetNotFound):
        _report(new_session, floor.a, floor.msgs[case])
    assert _rows(new_session, floor.ws) == [] and fakes.notices == []


def test_an_end_user_whose_erase_started_is_not_reported_for(new_session, floor, fakes):
    with pytest.raises(service.ReportTargetNotFound):
        _report(new_session, floor.a, floor.msgs["an_erased_end_users_reply"], end_user=GONE)
    assert fakes.limits == []
    assert _rows(new_session, floor.ws) == [] and fakes.notices == []


def test_a_key_naming_no_end_user_reaches_only_its_chats_for_no_one(new_session, floor, fakes):
    filed = _report(new_session, floor.a, floor.msgs["the_keys_own_reply"], end_user=None)
    assert filed.duplicate is False
    with pytest.raises(service.ReportTargetNotFound):
        _report(new_session, floor.a, floor.msgs["alphas_reply"], end_user=None)
    assert fakes.limits[0] == (service.RATE_LIMIT_OPERATION, str(floor.a_user))


def test_a_key_reports_no_deliverable(new_session, floor, fakes):
    with pytest.raises(service.ReportTargetNotFound):
        _report(new_session, floor.a, str(uuid.uuid4()), target_type="deliverable")
    assert fakes.deliverables == [] and _rows(new_session, floor.ws) == []


def test_a_key_that_never_chatted_finds_nothing_and_gets_no_service_user(new_session, floor, fakes):
    with pytest.raises(service.ReportTargetNotFound):
        _report(new_session, floor.c, floor.msgs["alphas_reply"])
    s = new_session()
    try:
        assert s.execute(text("SELECT service_user_id FROM sdk_api_keys WHERE id = CAST(:k AS uuid)"),
                         {"k": floor.c["id"]}).scalar() is None
    finally:
        s.close()
    assert fakes.limits == [] and _rows(new_session, floor.ws) == []
