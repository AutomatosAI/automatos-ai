"""PRD-185 S11: heartbeat write-side noise removal.

Two write-side fixes:
  1. The 30s Mem0 health probe used to write one ``heartbeat_results`` row per
     workspace EVERY tick (~2880/ws/day). It must now emit only on a health
     STATE CHANGE (baseline + transitions), killing the steady-state spam while
     keeping the memory-primitive tile fed.
  2. The daily summary used to double-write its digest into memory — once as a
     fabricated user/assistant L3 conversation that got injected into real
     prompts, once as an L2 heartbeat_log row. Both are gone.

Pure — the probe is driven with mocked Mem0 + a fake scheduler; the daily-summary
removal is asserted against the source. No DB / network.
"""
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from services.heartbeat_service import HeartbeatService


def _fake_scheduler_with_ws(ws_id: str):
    job = MagicMock()
    job.id = f"orch_hb_{ws_id}"
    sched = MagicMock()
    sched.get_jobs.return_value = [job]
    return sched


def _install_mem0(monkeypatch, healthy: bool):
    """Point the probe at a fake Mem0 client with the given health.

    Stubs ``modules.memory.unified_memory_service`` in ``sys.modules`` BEFORE the
    tick's local ``from ... import get_unified_memory_service`` runs, so the real
    (pgvector/asyncpg-heavy) memory chain never loads — the project's standard
    pattern (see test_heartbeat_primitive_findings.py)."""
    fake_client = MagicMock()
    fake_client.api_url = "http://mem0.local"
    fake_client.run_health_probe = AsyncMock(return_value=healthy)
    fake_unified = MagicMock()
    fake_unified._mem0 = fake_client
    fake_module = types.ModuleType("modules.memory.unified_memory_service")
    fake_module.get_unified_memory_service = lambda: fake_unified
    monkeypatch.setitem(
        sys.modules, "modules.memory.unified_memory_service", fake_module
    )
    return fake_client


@pytest.mark.asyncio
async def test_probe_emits_only_on_state_change(monkeypatch):
    svc = HeartbeatService()
    svc._scheduler = _fake_scheduler_with_ws("ws-1")
    fake_client = _install_mem0(monkeypatch, healthy=True)

    emits = []
    monkeypatch.setattr(
        "services.heartbeat_service.emit_primitive_finding",
        lambda ws, prim, status, detail="": emits.append((ws, prim, status)),
    )

    # Tick 1 — baseline green → one emit.
    await svc._mem0_health_probe_tick()
    assert emits == [("ws-1", "memory", "green")]

    # Tick 2 — still green → NO new row (the anti-spam guarantee).
    await svc._mem0_health_probe_tick()
    assert len(emits) == 1

    # Tick 3 — Mem0 goes down → transition → one more emit.
    fake_client.run_health_probe = AsyncMock(return_value=False)
    await svc._mem0_health_probe_tick()
    assert emits[-1] == ("ws-1", "memory", "down")
    assert len(emits) == 2

    # Tick 4 — still down → no new row.
    await svc._mem0_health_probe_tick()
    assert len(emits) == 2


@pytest.mark.asyncio
async def test_probe_still_steers_breakers_every_tick(monkeypatch):
    # The breaker trip/reset is the functional part and must run on EVERY tick,
    # independent of the (now transition-gated) primitive emit.
    svc = HeartbeatService()
    svc._scheduler = _fake_scheduler_with_ws("ws-1")
    fake_client = _install_mem0(monkeypatch, healthy=True)
    monkeypatch.setattr(
        "services.heartbeat_service.emit_primitive_finding",
        lambda *a, **k: True,
    )

    await svc._mem0_health_probe_tick()
    await svc._mem0_health_probe_tick()
    assert fake_client.run_health_probe.await_count == 2


def test_daily_summary_no_fabricated_conversation_or_double_write():
    src = (
        Path(_orchestrator_root) / "services" / "heartbeat_service.py"
    ).read_text()
    # The fabricated "User: … / Assistant: …" heartbeat turn is gone.
    assert "Daily heartbeat summary request" not in src, (
        "the fabricated heartbeat conversation must be removed"
    )
    # The daily summary must not write the digest into the memory plane at all.
    assert "store_conversation(" not in src, (
        "heartbeat must not write the daily digest to L3 memory"
    )
    assert "store_short_term(" not in src, (
        "heartbeat must not write the daily digest to L2 memory"
    )
