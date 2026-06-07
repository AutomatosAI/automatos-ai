"""PRD-142 Wave 4 (W4-S1): authenticated HARNESS approve/reject HTTP API.

The design decision behind this wave: the inbound webhook does NO sender
authorization, so HARNESS approval moves to this authenticated Command Center
surface. These unit tests pin the two things ``api/harness.py`` actually does —
resolve the authenticated principal to the integer ``users.id`` the handler's
admin gate expects, and map the handler result to HTTP status codes — with the
underlying ``handle_harness_command`` (already covered by test_harness_commands)
faked out. Dummy POSTGRES_* and the apscheduler stub let the harness_service
import chain load without a real DB or the prod-only scheduler dependency.
"""
import asyncio
import os
import sys
import types
from uuid import UUID

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()

from fastapi import HTTPException  # noqa: E402

import api.harness as h  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402

_WS_ID = UUID("00000000-0000-0000-0000-000000000001")
_RX_ID = "rx-esc-1"


# --- fakes -----------------------------------------------------------------

class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result


class _FakeDB:
    """Minimal stand-in for the Session — only ``query(User).filter().first()``
    is exercised by the principal resolver."""

    def __init__(self, user=None):
        self._user = user

    def query(self, _model):
        return _FakeQuery(self._user)


class _ModelAwareDB:
    """Routes ``query()`` by model name so the REAL handler authz can run
    end-to-end: ``User`` -> the resolver row, ``WorkspaceMember`` -> the
    admin-gate row. A ``member`` of None means 'not an admin' -> refused."""

    def __init__(self, user=None, member=None):
        self._user = user
        self._member = member

    def query(self, model):
        if getattr(model, "__name__", "") == "WorkspaceMember":
            return _FakeQuery(self._member)
        return _FakeQuery(self._user)


def _ctx(*, clerk=None, email=None):
    return RequestContext(
        workspace_id=_WS_ID,
        user=UserContext(id=clerk or email, email=email, clerk_user_id=clerk),
    )


def _user(uid):
    return types.SimpleNamespace(id=uid)


def _run(coro):
    return asyncio.run(coro)


# --- principal resolution --------------------------------------------------

def test_resolve_user_id_via_clerk():
    uid = h._resolve_internal_user_id(_FakeDB(user=_user(42)), _ctx(clerk="clerk_admin"))
    assert uid == 42


def test_resolve_returns_none_when_no_match():
    # No matching users row → None → the handler's admin gate fails closed.
    uid = h._resolve_internal_user_id(_FakeDB(user=None), _ctx(clerk="ghost", email="x@y.z"))
    assert uid is None


def test_resolve_returns_none_without_user():
    assert h._resolve_internal_user_id(_FakeDB(user=_user(1)), RequestContext(workspace_id=_WS_ID, user=None)) is None


# --- approve / reject HTTP mapping -----------------------------------------

def test_approve_passes_resolved_user_id(monkeypatch):
    captured = {}

    async def fake_handle(db, ws, cmd, rx, identity):
        captured.update(ws=ws, cmd=cmd, rx=rx, identity=identity)
        return {"success": True, "message": "Applied heartbeat_tune for SCOUT"}

    monkeypatch.setattr(h, "handle_harness_command", fake_handle)
    result = _run(h.approve_prescription(_RX_ID, ctx=_ctx(clerk="clerk_admin"), db=_FakeDB(user=_user(7))))

    assert result["success"] is True
    assert captured["ws"] == _WS_ID
    assert captured["cmd"] == "/approve"
    assert captured["rx"] == _RX_ID
    assert captured["identity"] == {"user_id": 7}  # the integer users.id, not the clerk string


def test_reject_uses_reject_command(monkeypatch):
    captured = {}

    async def fake_handle(db, ws, cmd, rx, identity):
        captured["cmd"] = cmd
        return {"success": True, "rejected": True, "message": f"Rejected {rx}"}

    monkeypatch.setattr(h, "handle_harness_command", fake_handle)
    result = _run(h.reject_prescription(_RX_ID, ctx=_ctx(clerk="clerk_admin"), db=_FakeDB(user=_user(7))))

    assert result["rejected"] is True
    assert captured["cmd"] == "/reject"


def test_unauthorized_maps_to_403(monkeypatch):
    async def fake_handle(*a, **k):
        return {"success": False, "unauthorized": True, "message": "Only a workspace admin can approve"}

    monkeypatch.setattr(h, "handle_harness_command", fake_handle)
    try:
        _run(h.approve_prescription(_RX_ID, ctx=_ctx(clerk="not_admin"), db=_FakeDB(user=_user(9))))
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_disabled_flag_maps_to_409(monkeypatch):
    async def fake_handle(*a, **k):
        return {"success": False, "message": "HARNESS self-management is disabled"}

    monkeypatch.setattr(h, "handle_harness_command", fake_handle)
    try:
        _run(h.approve_prescription(_RX_ID, ctx=_ctx(clerk="clerk_admin"), db=_FakeDB(user=_user(7))))
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 409


def test_unresolved_principal_fails_closed(monkeypatch):
    # No users row → user_id None is passed through; the handler refuses it
    # (here faked as unauthorized) and the endpoint surfaces 403. Nothing mutates.
    captured = {}

    async def fake_handle(db, ws, cmd, rx, identity):
        captured["identity"] = identity
        return {"success": False, "unauthorized": True, "message": "unknown caller"}

    monkeypatch.setattr(h, "handle_harness_command", fake_handle)
    try:
        _run(h.approve_prescription(_RX_ID, ctx=_ctx(clerk="ghost"), db=_FakeDB(user=None)))
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403
    assert captured["identity"] == {"user_id": None}


def test_real_gate_refuses_non_admin(monkeypatch):
    """Integration through the REAL handler + REAL admin gate (not the faked
    mapping): a principal that resolves to a real users.id but is NOT an active
    owner/admin member is refused 403. This is the one assertion that catches a
    future refactor accidentally bypassing the authz gate.
    """
    from config import config

    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    # Resolves to users.id=7, but the WorkspaceMember admin query returns None.
    db = _ModelAwareDB(user=_user(7), member=None)
    try:
        _run(h.approve_prescription(_RX_ID, ctx=_ctx(clerk="clerk_member"), db=db))
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403
