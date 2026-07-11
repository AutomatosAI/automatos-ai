"""PRD-193 S2 (P2-12) — consume an authorising grant: the gate finally has a yes.

Immediately before the confirmation return, the executor consults for an
authorising ``tool_call`` grant on the same deterministic subject key:
GRANTED + unexpired (``is_authorising``) + params-hash equality ⇒ proceed to
the handler; anything else ⇒ the ask stands. Locked decisions applied:
destructive grants are SINGLE-USE (retired on use via
``revoke_grant(revoked_by="system:consumed")``); write-class grants stay
GRANTED for their TTL window per call-key.

Guarantees pinned here:
  1. ``test_granted_call_executes``            — an active, params-matching grant
     lets the gated action reach its handler; no ``requires_confirmation`` in
     the result; the result is audit-marked ``approved_via_grant_id``.
  2. ``test_params_drift_reasks``              — same tool, different params ⇒
     a new ask (the grant authorises *the* call, not the tool).
  3. ``test_expired_revoked_denied_reask``     — each non-authorising status ⇒ ask.
  4. ``test_destructive_grant_is_single_use``  — the second identical call after
     a consumed grant ⇒ new ask; the spent grant reads revoked-by
     ``system:consumed``.
  5. ``test_write_class_grant_reusable_within_ttl`` — write-class grants keep
     authorising inside the TTL window (no consumption).
  6. ``test_consult_error_fails_closed``       — a raising lookup ⇒ the ask,
     never execution.
  7. ``test_grant_execution_audited``          — the telemetry audit marker
     carries the grant id, distinct from the full-autonomy ``autonomous`` flag.

Pure: fake session (evaluates the service's equality filters), fake registry,
mocked rate limiter, muted notification seam. No DB / network / Redis.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys as _sys
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

from core.models.approval_grants import ApprovalGrant, GrantStatus
from core.services.approval_grants import deny_grant, grant_grant, revoke_grant
from modules.tools.discovery.action_registry import ActionDefinition
from modules.tools.discovery.platform_executor import PlatformActionExecutor
from modules.tools.execution import tool_grants

pytestmark = pytest.mark.asyncio


_DESTRUCTIVE_ACTION = "platform_delete_document"
_WRITE_ACTION = "platform_update_system_setting"
_MEMBER_CTX = {"user_id": "user_clerk_1", "workspace_role": "member"}


# ---------------------------------------------------------------------------
# Fake session — list-backed, evaluates the grant service's equality filters.
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            value = getattr(cond.right, "value", None)
            rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def order_by(self, *args):
        return _Query(list(reversed(self._rows)))

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self):
        self.rows = []

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def flush(self):
        pass

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


# ---------------------------------------------------------------------------
# Executor harness
# ---------------------------------------------------------------------------

def _action(name: str, **overrides) -> ActionDefinition:
    kwargs = dict(
        name=name,
        description="PRD-193 gate probe.",
        category="documents",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="destructive",
        requires_confirmation=True,
    )
    kwargs.update(overrides)
    return ActionDefinition(**kwargs)


class _FakeRegistry:
    def __init__(self, *defs: ActionDefinition):
        self._defs = {d.name: d for d in defs}

    def get(self, name: str):
        return self._defs.get(name)


_REGISTRY = _FakeRegistry(
    _action(_DESTRUCTIVE_ACTION),
    _action(_WRITE_ACTION, permission_level="write", category="workspace"),
)


def _executor(db: _FakeSession) -> PlatformActionExecutor:
    return PlatformActionExecutor(db, uuid.uuid4())


def _gate_patches():
    return (
        patch("modules.tools.discovery.get_action_registry", return_value=_REGISTRY),
        patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False),
        patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)),
    )


@pytest.fixture(autouse=True)
def _mute_notifications(monkeypatch):
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(tool_grants, "_dispatch_approval_pending", _noop)


def _seed_granted(
    db: _FakeSession,
    ex: PlatformActionExecutor,
    action: str,
    params: dict,
    *,
    permission_level: str = "destructive",
) -> ApprovalGrant:
    """Seed a GRANTED grant exactly as S1 issuance + a human grant would."""
    grant = tool_grants.issue_tool_grant(
        db, ex.workspace_id,
        action=action, params=params,
        permission_level=permission_level,
        description="probe", caller_context=dict(_MEMBER_CTX),
    )
    assert grant is not None, "seed failed — issuance broken"
    grant_grant(grant, granted_by="user:1")
    return grant


# ===========================================================================
# 1. The yes: an authorising grant opens the gate
# ===========================================================================

async def test_granted_call_executes():
    db = _FakeSession()
    ex = _executor(db)
    grant = _seed_granted(db, ex, _DESTRUCTIVE_ACTION, {"document_id": 7})

    sentinel = AsyncMock(return_value={"success": True, "_sentinel": "handler-ran"})
    ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

    p1, p2, p3 = _gate_patches()
    with p1, p2, p3:
        result = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))

    sentinel.assert_awaited_once()
    assert result.get("success") is True
    assert result.get("_sentinel") == "handler-ran"
    assert "requires_confirmation" not in result
    # S2 audit marking: WHICH grant authorised it — and it is not "autonomous".
    assert result.get("approved_via_grant_id") == grant.id
    assert result.get("autonomous") is None


# ===========================================================================
# 2. The grant authorises THE call, not the tool
# ===========================================================================

async def test_params_drift_reasks():
    db = _FakeSession()
    ex = _executor(db)
    _seed_granted(db, ex, _DESTRUCTIVE_ACTION, {"document_id": 7})

    sentinel = AsyncMock(return_value={"success": True})
    ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

    p1, p2, p3 = _gate_patches()
    with p1, p2, p3:
        result = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 8}, dict(_MEMBER_CTX))

    sentinel.assert_not_awaited()
    assert result["success"] is False
    assert result["requires_confirmation"] is True


async def test_params_hash_mismatch_reasks_even_on_same_subject():
    """Belt-and-braces: a stored grant whose details hash does not match the
    incoming call must not authorise it, even if the subject row was found."""
    db = _FakeSession()
    ex = _executor(db)
    grant = _seed_granted(db, ex, _DESTRUCTIVE_ACTION, {"document_id": 7})
    grant.details = {**(grant.details or {}), "params_hash": "tampered"}

    sentinel = AsyncMock(return_value={"success": True})
    ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

    p1, p2, p3 = _gate_patches()
    with p1, p2, p3:
        result = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))

    sentinel.assert_not_awaited()
    assert result["requires_confirmation"] is True


# ===========================================================================
# 3. Non-authorising statuses re-ask
# ===========================================================================

async def test_expired_revoked_denied_reask():
    for mutate in (
        lambda g: setattr(g, "expires_at", datetime.now(timezone.utc) - timedelta(seconds=5)),
        lambda g: revoke_grant(g, revoked_by="user:1"),
        lambda g: deny_grant(g, revoked_by="user:1"),
    ):
        db = _FakeSession()
        ex = _executor(db)
        grant = _seed_granted(db, ex, _DESTRUCTIVE_ACTION, {"document_id": 7})
        mutate(grant)

        sentinel = AsyncMock(return_value={"success": True})
        ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

        p1, p2, p3 = _gate_patches()
        with p1, p2, p3:
            result = await ex.execute(
                _DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX)
            )

        sentinel.assert_not_awaited()
        assert result["success"] is False
        assert result["requires_confirmation"] is True, (
            f"status={grant.status!r} must NOT authorise"
        )


async def test_pending_grant_does_not_authorise():
    """A PENDING grant blocks — only a live GRANTED one opens the gate."""
    db = _FakeSession()
    ex = _executor(db)
    tool_grants.issue_tool_grant(
        db, ex.workspace_id,
        action=_DESTRUCTIVE_ACTION, params={"document_id": 7},
        permission_level="destructive", caller_context=dict(_MEMBER_CTX),
    )

    sentinel = AsyncMock(return_value={"success": True})
    ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

    p1, p2, p3 = _gate_patches()
    with p1, p2, p3:
        result = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))

    sentinel.assert_not_awaited()
    assert result["requires_confirmation"] is True
    # And no second pending row was spammed (S1 idempotency, re-checked here).
    pendings = [
        r for r in db.rows
        if isinstance(r, ApprovalGrant) and r.status == GrantStatus.PENDING.value
    ]
    assert len(pendings) == 1


# ===========================================================================
# 4. Locked decision 1 — destructive ⇒ single-use, exact params
# ===========================================================================

async def test_destructive_grant_is_single_use():
    db = _FakeSession()
    ex = _executor(db)
    grant = _seed_granted(db, ex, _DESTRUCTIVE_ACTION, {"document_id": 7})

    sentinel = AsyncMock(return_value={"success": True})
    ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

    p1, p2, p3 = _gate_patches()
    with p1, p2, p3:
        first = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))
        second = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))

    sentinel.assert_awaited_once()  # the yes covered exactly one execution
    assert first.get("approved_via_grant_id") == grant.id
    # The spent grant is retired via the EXISTING lifecycle — no new status.
    assert grant.status == GrantStatus.REVOKED.value
    assert grant.revoked_by == tool_grants.GRANT_CONSUMED_BY
    # The identical retry is a fresh ask (new pending grant attached).
    assert second["requires_confirmation"] is True
    assert second.get("grant_id") not in (None, grant.id)


async def test_write_class_grant_reusable_within_ttl():
    """Locked decision 1: the 3 write-class actions get a TTL-window grant per
    call-key — repeated identical calls inside the window keep executing."""
    db = _FakeSession()
    ex = _executor(db)
    grant = _seed_granted(
        db, ex, _WRITE_ACTION, {"key": "rag_rerank_enabled", "value": "true"},
        permission_level="write",
    )

    sentinel = AsyncMock(return_value={"success": True})
    ex._handlers[_WRITE_ACTION] = sentinel

    p1, p2, p3 = _gate_patches()
    with p1, p2, p3:
        first = await ex.execute(
            _WRITE_ACTION, {"key": "rag_rerank_enabled", "value": "true"}, dict(_MEMBER_CTX)
        )
        second = await ex.execute(
            _WRITE_ACTION, {"key": "rag_rerank_enabled", "value": "true"}, dict(_MEMBER_CTX)
        )

    assert sentinel.await_count == 2
    assert grant.status == GrantStatus.GRANTED.value  # not consumed
    assert first.get("approved_via_grant_id") == grant.id
    assert second.get("approved_via_grant_id") == grant.id


# ===========================================================================
# 5. Fail closed: a raising consult lands on the ask, never execution
# ===========================================================================

async def test_consult_error_fails_closed(monkeypatch):
    db = _FakeSession()
    ex = _executor(db)
    _seed_granted(db, ex, _DESTRUCTIVE_ACTION, {"document_id": 7})

    import core.services.approval_grants as grant_service

    def _boom(*a, **k):
        raise RuntimeError("lookup exploded")

    monkeypatch.setattr(grant_service, "find_active_grant", _boom)

    sentinel = AsyncMock(return_value={"success": True})
    ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

    p1, p2, p3 = _gate_patches()
    with p1, p2, p3:
        result = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))

    sentinel.assert_not_awaited()
    assert result["success"] is False
    assert result["requires_confirmation"] is True, "errors land on the ask"


# ===========================================================================
# 6. Attribution is honest and queryable
# ===========================================================================

async def test_grant_execution_audited():
    """The telemetry audit marker carries the grant id, distinct from the
    full-autonomy dial's ``autonomous`` flag."""
    from modules.tools.execution.telemetry import _build_router_decision

    decision = _build_router_decision({}, autonomous=False, approved_via_grant_id=7)
    assert decision == {"approved_via_grant_id": 7}

    dial = _build_router_decision({}, autonomous=True)
    assert dial == {"autonomous": True}
    assert "approved_via_grant_id" not in dial
