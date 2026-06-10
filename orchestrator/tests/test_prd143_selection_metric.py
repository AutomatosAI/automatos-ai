"""PRD-143 S14 — selection-health metric: hit-rate / fallback-rate via the
EXISTING telemetry path, surfaced su-locked.

The narrowed-vs-not-narrowed distinction that lived only in the tool-trace
log (tool_router.py, PRD-138 US-009 block) is now recorded per selection on
the ToolSignalRecorder (in-memory counters + last-selection stash) and
persisted per dispatch through the PRD-139 universal telemetry hook into
``tool_execution_logs.router_decision->'selection'`` — no new infra, no new
table:

  * ``get_tools_for_agent`` records the selection outcome (narrowed with the
    allowed set | fallback with the narrow_reason) when it builds the
    dispatcher schema;
  * the ``platform_execute`` dispatch in UnifiedToolExecutor peeks the stash
    and attaches ``selection_outcome`` (incl. hit = chosen action in the
    narrowed enum) to caller_context;
  * ``write_telemetry`` persists it on the execution row (S8 audit idiom);
  * ``GET /api/analytics/selection-health`` on the ALREADY-LOCKED
    analytics_real router aggregates hit-rate / fallback-rate over a
    parameterized window — and 403s every non-super-admin (S6 inheritance).

Idioms: S3's _tool_surface fakes (real get_tools_for_agent, synthetic
registry), S8's MagicMock-db write_telemetry assertions, S6's router-mount
client with dependency overrides.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util as _ilu
import os
import sys as _sys
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Dummy POSTGRES_* satisfies the config chain at import (blessed pattern, see
# test_prd143_su_executor_gate.py) — the port points at nothing so any
# fail-soft connect refuses instantly. CI exports real vars (setdefault no-ops).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# Lean-venv shim: modules/tools/__init__ pulls modules.rag's ingestion chain
# (camelot at module top). Stub the missing leaf only when truly absent.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import modules.tools.discovery.signal_recorder as sr  # noqa: E402
import modules.tools.execution.unified_executor as ue  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.tools import tool_router as tr  # noqa: E402
from modules.tools.discovery.action_registry import (  # noqa: E402
    ActionDefinition,
    ActionRegistry,
)
from modules.tools.execution.telemetry import write_telemetry  # noqa: E402
from modules.tools.execution.unified_executor import UnifiedToolExecutor  # noqa: E402

_WS = uuid.uuid4()
_AGENT = 7

_OP_LIST = "platform_probe_list_agents"
_OP_CREATE = "platform_probe_create_agent"
_OP_OUTSIDE = "platform_probe_outside_topk"

_RANK_FAIL_REASON = "rank_actions returned empty or raised"


def _action(name: str, *, description: str) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=description,
        category="agents",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
    )


def _registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg._initialized = True  # bypass the live platform_actions registrar
    reg.register(_action(_OP_LIST, description="List the agents"))
    reg.register(_action(_OP_CREATE, description="Create an agent"))
    reg.register(_action(_OP_OUTSIDE, description="Probe outside the top-K"))
    return reg


class _RankIndex:
    """Async fake for ActionSemanticIndex (S3 idiom)."""

    def __init__(self, results: List[Tuple[str, float]]):
        self._results = list(results)

    async def rank_actions(self, query: str, top_k: int = 15, **kw: Any) -> List[Tuple[str, float]]:
        return list(self._results)[:top_k]


def _tool_surface(registry: ActionRegistry, index: _RankIndex):
    """Patch get_tools_for_agent's collaborators (S3 idiom) with semantic
    narrowing enabled and the fake ranker installed."""
    fake_tool_registry = MagicMock()
    fake_tool_registry.get_all_tools.return_value = []
    return (
        patch.object(tr, "registry_get_tool_registry", return_value=fake_tool_registry),
        patch.object(tr, "SessionLocal", return_value=MagicMock()),
        patch("modules.tools.discovery.get_action_registry", return_value=registry),
        patch.object(tr, "_semantic_routing_enabled", return_value=True),
        patch.object(tr, "_semantic_routing_top_k", return_value=10),
        patch(
            "modules.tools.discovery.action_semantic_index.get_action_semantic_index",
            return_value=index,
        ),
    )


@pytest.fixture()
def fresh_recorder(monkeypatch: pytest.MonkeyPatch) -> sr.ToolSignalRecorder:
    """Swap in a fresh process singleton so counters/stash start at zero."""
    recorder = sr.ToolSignalRecorder()
    monkeypatch.setattr(sr, "_instance", recorder)
    return recorder


def _build_surface(registry: ActionRegistry, index: _RankIndex) -> List[Dict[str, Any]]:
    patches = _tool_surface(registry, index)
    for p in patches:
        p.start()
    try:
        return tr.get_tools_for_agent(agent_id=_AGENT, workspace_id=_WS, query="list the agents")
    finally:
        for p in patches:
            p.stop()


async def _dispatch(registry: ActionRegistry, action: str) -> Dict[str, Any]:
    """Drive a real platform_execute dispatch through UnifiedToolExecutor with
    the platform executor stubbed; returns the caller_context that the
    universal telemetry hook received."""
    executor = UnifiedToolExecutor(MagicMock())
    telemetry = MagicMock()
    with patch("modules.tools.discovery.get_action_registry", return_value=registry), \
            patch.object(
                executor, "_execute_platform_action",
                new=AsyncMock(return_value={"success": True}),
            ), \
            patch.object(ue, "fire_telemetry", telemetry):
        result = await executor.execute_tool(
            "platform_execute",
            {"action": action, "params": {}},
            agent_id=_AGENT,
            workspace_id=_WS,
        )
    assert result.get("success") is True, f"dispatch failed: {result}"
    assert telemetry.call_count == 1
    return telemetry.call_args.kwargs.get("caller_context") or {}


# ===========================================================================
# 1. Selection outcome recorded on a narrowed dispatch
# ===========================================================================


def test_selection_outcome_recorded_on_narrowed_dispatch(fresh_recorder):
    """Building the surface with semantic narrowing ON records a 'narrowed'
    selection (counter + stash); the subsequent platform_execute dispatch
    carries selection_outcome with hit=True when the chosen action came from
    the narrowed enum — and hit=False when it escaped it (non-vacuous both
    ways) — and write_telemetry persists it on the execution row."""
    registry = _registry()
    _build_surface(registry, _RankIndex([(_OP_LIST, 0.9), (_OP_CREATE, 0.6)]))

    # Counter persisted instead of log-only (process-lifetime stats).
    stats = fresh_recorder.stats()
    assert stats["selection_narrowed"] == 1
    assert stats["selection_fallback"] == 0

    # Stash holds the narrowed outcome for this (workspace, agent) surface.
    sel = fresh_recorder.peek_selection(workspace_id=_WS, agent_id=_AGENT)
    assert sel is not None
    assert sel["narrowed"] is True
    assert _OP_LIST in sel["allowed"]

    # Dispatch from inside the narrowed enum → hit=True.
    ctx_hit = asyncio.run(_dispatch(registry, _OP_LIST))
    outcome = ctx_hit.get("selection_outcome")
    assert outcome is not None, "dispatch did not attach selection_outcome"
    assert outcome["narrowed"] is True
    assert outcome["hit"] is True
    assert outcome["action"] == _OP_LIST
    assert outcome["enum_size"] == 2

    # Dispatch that escaped the narrowed enum → hit=False (a real miss).
    ctx_miss = asyncio.run(_dispatch(registry, _OP_OUTSIDE))
    assert ctx_miss["selection_outcome"]["hit"] is False
    assert ctx_miss["selection_outcome"]["narrowed"] is True

    # The universal telemetry hook persists the outcome on the row (S8 idiom).
    db = MagicMock()
    asyncio.run(
        write_telemetry(
            db,
            tool_name="platform_execute",
            parameters={"action": _OP_LIST},
            agent_id=_AGENT,
            workspace_id=_WS,
            result={"success": True},
            execution_time_ms=3,
            caller_context=ctx_hit,
        )
    )
    db.add.assert_called_once()
    row = db.add.call_args[0][0]
    persisted = (row.router_decision or {}).get("selection")
    assert persisted is not None
    assert persisted["narrowed"] is True
    assert persisted["hit"] is True


# ===========================================================================
# 2. Fallback recorded when rank fails
# ===========================================================================


def test_fallback_recorded_when_rank_fails(fresh_recorder):
    """When rank_actions returns empty (or raises), the selection is recorded
    as a fallback with the narrow_reason; the dispatch carries
    narrowed=False / hit=None and telemetry persists it."""
    registry = _registry()
    _build_surface(registry, _RankIndex([]))  # rank → empty → None → fallback

    stats = fresh_recorder.stats()
    assert stats["selection_fallback"] == 1
    assert stats["selection_narrowed"] == 0

    sel = fresh_recorder.peek_selection(workspace_id=_WS, agent_id=_AGENT)
    assert sel is not None
    assert sel["narrowed"] is False
    assert sel["reason"] == _RANK_FAIL_REASON

    ctx = asyncio.run(_dispatch(registry, _OP_LIST))
    outcome = ctx.get("selection_outcome")
    assert outcome is not None, "fallback dispatch did not attach selection_outcome"
    assert outcome["narrowed"] is False
    assert outcome["hit"] is None
    assert outcome["reason"] == _RANK_FAIL_REASON

    db = MagicMock()
    asyncio.run(
        write_telemetry(
            db,
            tool_name="platform_execute",
            parameters={"action": _OP_LIST},
            agent_id=_AGENT,
            workspace_id=_WS,
            result={"success": True},
            execution_time_ms=3,
            caller_context=ctx,
        )
    )
    row = db.add.call_args[0][0]
    persisted = (row.router_decision or {}).get("selection")
    assert persisted is not None
    assert persisted["narrowed"] is False
    assert persisted["reason"] == _RANK_FAIL_REASON


# ===========================================================================
# 3. + 4. The aggregate endpoint — su-locked, windowed
# ===========================================================================

MEMBER = UserContext(id="u-member", role="member", system_role="user")
WS_ADMIN = UserContext(id="u-ws-admin", role="admin", system_role="user")
# hybrid.py:783 — API-key principals carry system_role='admin'.
API_KEY_ADMIN = UserContext(id="api_key", email=None, role="admin", system_role="admin")
SUPER_ADMIN = UserContext(id="u-gerard", role="admin", system_role="super_admin")

_PATH = "/api/analytics/selection-health"


def _client(user: UserContext, agg_row: Optional[tuple]) -> Tuple[TestClient, MagicMock]:
    """Mount the REAL analytics_real router (S6 idiom) with a fake db whose
    aggregate query returns ``agg_row``."""
    module = importlib.import_module("api.analytics_real")

    db = MagicMock()
    db.execute.return_value.fetchone.return_value = agg_row

    app = FastAPI()
    app.include_router(module.router)

    auth_type = "api_key" if user is API_KEY_ADMIN else "clerk"

    def _override_ctx():
        return RequestContext(workspace_id=_WS, user=user, auth_type=auth_type)

    def _override_db():
        yield db

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db
    return TestClient(app, raise_server_exceptions=False), db


def test_metric_endpoint_aggregates():
    """hit-rate = hits/narrowed, fallback-rate = fallback/selections, window
    parameterized (the SQL is bound to executed_at >= now - window)."""
    client, db = _client(SUPER_ADMIN, agg_row=(8, 2, 6))
    resp = client.get(_PATH, params={"window": "24h"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["window"] == "24h"
    assert body["narrowed"] == 8
    assert body["fallback"] == 2
    assert body["hits"] == 6
    assert body["selections"] == 10
    assert body["hit_rate"] == pytest.approx(0.75)
    assert body["fallback_rate"] == pytest.approx(0.2)

    sql = str(db.execute.call_args[0][0])
    assert "tool_execution_logs" in sql
    assert "router_decision" in sql
    assert "executed_at >= :window_start" in sql
    window_start = db.execute.call_args[0][1]["window_start"]
    expected = datetime.utcnow() - timedelta(hours=24)
    assert abs((window_start - expected).total_seconds()) < 300

    # The window parameter genuinely moves the cutoff.
    client7d, db7d = _client(SUPER_ADMIN, agg_row=(0, 0, 0))
    resp7d = client7d.get(_PATH, params={"window": "7d"})
    assert resp7d.status_code == 200, resp7d.text
    body7d = resp7d.json()
    assert body7d["hit_rate"] == 0.0  # zero-division safe
    assert body7d["fallback_rate"] == 0.0
    start7d = db7d.execute.call_args[0][1]["window_start"]
    expected7d = datetime.utcnow() - timedelta(days=7)
    assert abs((start7d - expected7d).total_seconds()) < 300


def test_metric_endpoint_su_locked():
    """The metric inherits the S6 router-wide lock: member, workspace admin
    and API-key admin all 403; the super admin passes."""
    for principal in (MEMBER, WS_ADMIN, API_KEY_ADMIN):
        client, _ = _client(principal, agg_row=(0, 0, 0))
        resp = client.get(_PATH)
        assert resp.status_code == 403, f"{principal.id}: {resp.status_code} {resp.text}"
        assert resp.json()["detail"] == "Super admin only"

    su_client, _ = _client(SUPER_ADMIN, agg_row=(0, 0, 0))
    resp = su_client.get(_PATH)
    assert resp.status_code not in (401, 403), resp.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
