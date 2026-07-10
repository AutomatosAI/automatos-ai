"""PRD-193 S1 (P2-12) — issue a durable grant at the ask.

The confirmation gate must stop returning a dead-end: when a
``requires_confirmation`` action fires, ``PlatformActionExecutor.execute``
issues (or reuses) a PENDING ``ApprovalGrant`` scoped ``tool_call`` and returns
the ask WITH the grant attached (``grant_id`` + risk class + AI-Act oversight
fields), so there is finally something a human can act on.

Guarantees pinned here:
  1. ``test_ask_issues_pending_tool_grant``   — ask ⇒ one pending grant scoped
     (workspace, tool_call, deterministic subject_id, tool_name, risk_tier),
     ask carries grant_id / risk_class / oversight fields.
  2. ``test_repeat_ask_reuses_pending_grant`` — same call twice ⇒ ONE pending row.
  3. ``test_deny_paths_issue_no_grant``       — su-gate and admin-deny returns
     create nothing (a deny is not an ask).
  4. ``test_grant_issue_failure_still_asks``  — grant creation raising ⇒ the ask
     is still returned (the ask is the floor: never an exception, never an
     execution).
  5. Subject-key determinism — server-injected ``_``-prefixed params do not
     perturb the call key (the ask and the retry must produce the same key).

Pure: fake session (list-backed, evaluates the service's equality filters),
fake registry, no DB / network / Redis. Notification dispatch is stubbed.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys as _sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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

from core.models.approval_grants import ApprovalGrant, GrantStatus, SUBJECT_TOOL_CALL
from modules.tools.discovery.action_registry import ActionDefinition
from modules.tools.discovery.platform_executor import PlatformActionExecutor
from modules.tools.execution import tool_grants

pytestmark = pytest.mark.asyncio


_DESTRUCTIVE_ACTION = "platform_delete_document"
_MEMBER_CTX = {"user_id": "user_clerk_1", "workspace_role": "member"}


# ---------------------------------------------------------------------------
# Fake session — list-backed, evaluates the grant service's equality filters
# (workspace_id / subject_type / subject_id / status) against stored rows.
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
# Executor harness (mirrors tests/test_prd143_su_executor_gate.py idiom)
# ---------------------------------------------------------------------------

def _action(name: str, **overrides) -> ActionDefinition:
    kwargs = dict(
        name=name,
        description="PRD-193 gate probe: delete a document permanently.",
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


def _executor(db=None) -> PlatformActionExecutor:
    ex = PlatformActionExecutor(db if db is not None else _FakeSession(), uuid.uuid4())
    return ex


def _gate_patches(registry: _FakeRegistry):
    return (
        patch("modules.tools.discovery.get_action_registry", return_value=registry),
        patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False),
    )


@pytest.fixture(autouse=True)
def _mute_notifications(monkeypatch):
    """Keep issuance tests pure — the S5 dispatch seam is tested separately."""
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(tool_grants, "_dispatch_approval_pending", _noop)


def _pending_grants(db: _FakeSession):
    return [
        r for r in db.rows
        if isinstance(r, ApprovalGrant) and r.status == GrantStatus.PENDING.value
    ]


# ===========================================================================
# 1. The ask issues a pending grant and returns it on the ask
# ===========================================================================

async def test_ask_issues_pending_tool_grant():
    db = _FakeSession()
    ex = _executor(db)
    registry = _FakeRegistry(_action(_DESTRUCTIVE_ACTION))
    p_registry, p_autonomy = _gate_patches(registry)

    params = {"document_id": 7, "_agent_id": 9}
    with p_registry, p_autonomy:
        result = await ex.execute(_DESTRUCTIVE_ACTION, params, dict(_MEMBER_CTX))

    # The ask still stands (fail-safe floor) …
    assert result["success"] is False
    assert result["requires_confirmation"] is True
    assert result["action"] == _DESTRUCTIVE_ACTION

    # … but it now carries the grant + risk + oversight fields for the card.
    grants = _pending_grants(db)
    assert len(grants) == 1, "the ask must stage exactly one pending grant"
    grant = grants[0]
    assert result.get("grant_id") == grant.id
    assert result.get("risk_class") == "destructive"
    assert result.get("risk_tier") == "human_in_the_loop"
    assert result.get("oversight_rationale")

    # Grant scoping: workspace + tool_call subject + deterministic key + risk.
    assert str(grant.workspace_id) == str(ex.workspace_id)
    assert grant.subject_type == SUBJECT_TOOL_CALL
    assert grant.subject_id == tool_grants.tool_call_subject_id(
        ex.workspace_id, _DESTRUCTIVE_ACTION, params
    )
    assert grant.tool_name == _DESTRUCTIVE_ACTION
    assert grant.risk_tier == "destructive"
    assert grant.agent_id == 9  # server-minted identity from _agent_id

    # The details snapshot carries what resume (S4) needs.
    details = grant.details or {}
    assert details.get("action") == _DESTRUCTIVE_ACTION
    assert details.get("params") == {"document_id": 7}  # model-provided only
    assert details.get("params_hash") == tool_grants.params_hash(params)


async def test_repeat_ask_reuses_pending_grant():
    db = _FakeSession()
    ex = _executor(db)
    registry = _FakeRegistry(_action(_DESTRUCTIVE_ACTION))
    p_registry, p_autonomy = _gate_patches(registry)

    with p_registry, p_autonomy:
        first = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))
        second = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))

    grants = _pending_grants(db)
    assert len(grants) == 1, "an identical retry must not spam a second grant"
    assert first["grant_id"] == second["grant_id"] == grants[0].id


async def test_different_params_issue_distinct_grants():
    """The grant authorises *the* call, not the tool — a different target is a
    different subject key and a different pending row."""
    db = _FakeSession()
    ex = _executor(db)
    registry = _FakeRegistry(_action(_DESTRUCTIVE_ACTION))
    p_registry, p_autonomy = _gate_patches(registry)

    with p_registry, p_autonomy:
        a = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))
        b = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 8}, dict(_MEMBER_CTX))

    assert len(_pending_grants(db)) == 2
    assert a["grant_id"] != b["grant_id"]


# ===========================================================================
# 2. Subject-key determinism (ask == retry, server plumbing stripped)
# ===========================================================================

async def test_subject_key_ignores_server_injected_params():
    ws = uuid.uuid4()
    bare = tool_grants.tool_call_subject_id(ws, _DESTRUCTIVE_ACTION, {"document_id": 7})
    injected = tool_grants.tool_call_subject_id(
        ws, _DESTRUCTIVE_ACTION,
        {"document_id": 7, "_agent_id": 9, "_agent_name": "Scribe"},
    )
    assert bare == injected
    # And the key is workspace- and params-sensitive.
    assert bare != tool_grants.tool_call_subject_id(ws, _DESTRUCTIVE_ACTION, {"document_id": 8})
    assert bare != tool_grants.tool_call_subject_id(uuid.uuid4(), _DESTRUCTIVE_ACTION, {"document_id": 7})


# ===========================================================================
# 3. Deny paths issue nothing — a deny is not an ask
# ===========================================================================

async def test_deny_paths_issue_no_grant():
    db = _FakeSession()
    ex = _executor(db)
    registry = _FakeRegistry(
        _action("platform_su_probe", super_admin_only=True, requires_confirmation=False,
                permission_level="read"),
        _action("platform_admin_probe", admin_only=True, requires_confirmation=False,
                permission_level="read"),
    )
    p_registry, p_autonomy = _gate_patches(registry)

    with p_registry, p_autonomy:
        su = await ex.execute("platform_su_probe", {}, dict(_MEMBER_CTX))
        admin = await ex.execute("platform_admin_probe", {}, dict(_MEMBER_CTX))

    assert su.get("permission_denied") is True
    assert admin.get("permission_denied") is True
    assert su.get("grant_id") is None and admin.get("grant_id") is None
    assert db.rows == [], "deny paths must not stage grants"


# ===========================================================================
# 4. Fail-safe floor: grant machinery raising still returns the ask
# ===========================================================================

async def test_grant_issue_failure_still_asks(monkeypatch):
    db = _FakeSession()
    ex = _executor(db)
    registry = _FakeRegistry(_action(_DESTRUCTIVE_ACTION))
    p_registry, p_autonomy = _gate_patches(registry)

    def _boom(*a, **k):
        raise RuntimeError("grant store down")

    import core.services.approval_grants as grant_service

    monkeypatch.setattr(grant_service, "create_grant", _boom)
    monkeypatch.setattr(grant_service, "find_pending_grant", _boom)

    sentinel = AsyncMock(return_value={"success": True})
    ex._handlers[_DESTRUCTIVE_ACTION] = sentinel

    with p_registry, p_autonomy:
        result = await ex.execute(_DESTRUCTIVE_ACTION, {"document_id": 7}, dict(_MEMBER_CTX))

    assert result["success"] is False
    assert result["requires_confirmation"] is True, "the ask is the floor"
    assert result.get("grant_id") is None
    sentinel.assert_not_awaited()


# ===========================================================================
# 5. The registry-miss fail-closed ask carries a grant too (best-effort)
# ===========================================================================

async def test_registry_miss_ask_still_returns_and_carries_grant():
    db = _FakeSession()
    ex = _executor(db)
    ex._handlers["platform_mystery"] = AsyncMock(return_value={"success": True})

    with patch(
        "modules.tools.discovery.get_action_registry",
        side_effect=RuntimeError("registry unavailable"),
    ):
        result = await ex.execute("platform_mystery", {"x": 1}, dict(_MEMBER_CTX))

    assert result["success"] is False
    assert result["requires_confirmation"] is True
    ex._handlers["platform_mystery"].assert_not_awaited()
    # Best-effort grant on the fail-closed ask (the same surface resolves it).
    assert len(_pending_grants(db)) == 1
    assert result.get("grant_id") == _pending_grants(db)[0].id
