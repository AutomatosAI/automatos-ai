"""PRD-142 Wave 3 · W3-S7 — Memory failure-path tests.

BRAIN §3.x Memory contract: *degrades when Mem0 down (circuit breaker built)*.
GUARDRAILS §H DoD #2: *failure path tested — the primitive degrades or errors
visibly, never silently*.

These tests pin the failure-path behaviour for the chat → memory write path:

1. ``test_l2_transcript_still_writes_when_l3_raises`` — when Mem0 raises during
   ``store_two_tier`` (L3), the L2 transcript write must still happen. The
   turn is *not lost* even though L3 enrichment failed.

2. ``test_l2_transcript_still_writes_when_l3_unconfigured`` — when Mem0 is
   unconfigured (no api_url), L3 writes resolve to no-ops; L2 still persists
   the verbatim transcript.

3. ``test_mem0_failure_is_visible_via_return_value`` — when L3 fully fails and
   nothing was durable, ``store_conversation`` returns ``False`` so the caller
   can observe the degradation (the turn is *visible*, not silent).

4. ``test_per_workspace_breaker_isolation`` — a Mem0 outage in workspace A's
   breaker does not trip workspace B's breaker. ``Mem0Client._breakers`` is a
   per-workspace map; this test re-proves the isolation invariant from
   PRD-141 US-006.

Imports use the same package-stub isolation as
``test_memory_single_write_path.py`` so heavy modules (asyncpg / pdfplumber /
tiktoken) are not pulled in.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Isolation — stub consumers.chatbot package init + camelot
# ---------------------------------------------------------------------------

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

for _pkg in ("consumers", "consumers.chatbot"):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub

sys.modules.setdefault("camelot", types.ModuleType("camelot"))

# Stub core.llm.manager — store_conversation calls _get_store_max_chars which
# imports get_system_setting at call-time; the real module triggers DB / Redis
# IO chains we don't need (max_chars defaults to 6000 on import failure anyway,
# but the chain can HANG on first import). We only register the stubs when
# nothing is already there — once a sibling test (e.g. test_l3_distill_input)
# has loaded the real package, we keep using it. The stub for ``manager``
# carries placeholder ``create_llm_manager`` so the real ``core.llm.__init__``
# ``from core.llm.manager import create_llm_manager`` line still resolves
# even when our parent-package stub is what's in play.
for _pkg in ("core", "core.llm"):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        # Cross-test compatibility: ``test_l3_distill_input`` does
        # ``import core.llm as core_llm`` then ``monkeypatch.setattr(core_llm,
        # "create_llm_manager", ...)``. If our stub is what's in sys.modules
        # when that import runs, the attr must already exist for setattr to
        # succeed. Adding a placeholder is harmless when the real module wins.
        if _pkg == "core.llm":
            _stub.create_llm_manager = lambda *a, **k: MagicMock()
        sys.modules[_pkg] = _stub
if "core.llm.manager" not in sys.modules:
    _mgr_stub = types.ModuleType("core.llm.manager")
    _mgr_stub.get_system_setting = lambda *a, **k: "6000"
    _mgr_stub.create_llm_manager = lambda *a, **k: MagicMock()
    sys.modules["core.llm.manager"] = _mgr_stub

from consumers.chatbot.smart_memory import SmartMemoryManager  # noqa: E402


# ---------------------------------------------------------------------------
# Fake unified service that records calls and lets the test choose behaviour
# ---------------------------------------------------------------------------


class _FakeUnified:
    """Mimics the UnifiedMemoryService surface store_conversation touches."""

    def __init__(
        self,
        *,
        l3_raises: bool = False,
        l3_returns_error: bool = False,
        l3_unconfigured: bool = False,
        l2_raises: bool = False,
    ):
        self.l3_raises = l3_raises
        self.l3_returns_error = l3_returns_error
        self.l3_unconfigured = l3_unconfigured
        self.l2_raises = l2_raises
        self.two_tier_calls: list[dict] = []
        self.transcript_calls: list[dict] = []
        self.short_term_calls: list[dict] = []

    async def store_two_tier(self, **kwargs):
        self.two_tier_calls.append(kwargs)
        if self.l3_unconfigured:
            # Mirror store_two_tier behaviour when Mem0 is disabled — every
            # tier silently no-ops with an error dict. The shared client logs
            # at warning level once and returns ``None`` from ``_request``.
            return [
                ("global", {"success": False, "error": "store_global failed"}),
            ]
        if self.l3_raises:
            raise RuntimeError("mem0 add boom")
        if self.l3_returns_error:
            return [
                ("global", {"success": False, "error": "store_global failed"}),
            ]
        return [("global", {"id": "m1"})]

    async def store_transcript(self, **kwargs):
        self.transcript_calls.append(kwargs)
        if self.l2_raises:
            raise RuntimeError("l2 transcript boom")
        return "row-transcript"

    async def store_short_term(self, **kwargs):
        self.short_term_calls.append(kwargs)
        return "row-short-term"

    def namespace(self, workspace_id):
        return MagicMock(workspace=lambda: f"mem:{workspace_id}")


def _make_mgr(unified: _FakeUnified) -> SmartMemoryManager:
    mgr = SmartMemoryManager()
    # Bypass the lazy unified_service loader — inject the fake directly.
    mgr._unified_service = unified  # type: ignore[attr-defined]
    return mgr


def _patch_distill(mgr: SmartMemoryManager, result):
    """Inject a deterministic distill result so the test does not need an LLM.

    ``result``:
      * ``list[str]`` — durable facts → store_two_tier is called
      * ``[]``        — nothing durable → store_two_tier is skipped
      * ``None``      — distill failed → fall back to raw exchange
    """
    async def _stub(*args, **kwargs):
        return result
    mgr._distill_durable_facts = _stub  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 1. L2 still writes when L3 (Mem0) raises mid-call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_transcript_still_writes_when_l3_raises():
    """A Mem0 outage during ``store_two_tier`` must not lose the turn — the
    verbatim transcript still persists to L2.
    """
    unified = _FakeUnified(l3_raises=True)
    mgr = _make_mgr(unified)
    _patch_distill(mgr, ["I prefer tea over coffee."])

    # The orchestrator-level smart_memory wraps L3 in try/except for graceful
    # degradation; the raise inside store_two_tier surfaces back, but the L2
    # transcript write happens regardless via its own try/except.
    await mgr.store_conversation(
        workspace_id="ws-A",
        agent_id=42,
        user_message="remember I prefer tea over coffee",
        assistant_response="Logged. Tea > coffee.",
        chat_id="conv-l3boom",
    )

    # L2 transcript STILL ran — the turn is preserved.
    assert len(unified.transcript_calls) == 1, (
        "L2 transcript must persist even when L3 (Mem0) raises — turn loss "
        "is the failure mode we are explicitly preventing"
    )
    # L3 was attempted exactly once (no silent retry storm).
    assert len(unified.two_tier_calls) == 1


# ---------------------------------------------------------------------------
# 2. L2 still writes when Mem0 is unconfigured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_transcript_still_writes_when_l3_unconfigured():
    """When Mem0 has no API URL (disabled), L3 writes resolve to error dicts;
    the L2 transcript still persists so the chat history is not lost.
    """
    unified = _FakeUnified(l3_unconfigured=True)
    mgr = _make_mgr(unified)
    _patch_distill(mgr, ["alice prefers light mode"])

    ok = await mgr.store_conversation(
        workspace_id="ws-B",
        agent_id=11,
        user_message="I prefer light mode for the UI",
        assistant_response="Noted.",
        chat_id="conv-no-mem0",
    )

    # L2 transcript persisted.
    assert len(unified.transcript_calls) == 1
    # store_conversation returns False because L3 failed visibly — the turn is
    # NOT silently swallowed.
    assert ok is False


# ---------------------------------------------------------------------------
# 3. Failure is observable via return value (visible, not silent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mem0_failure_is_visible_via_return_value():
    """When L3 (Mem0) write errors and no durable facts could be stored,
    ``store_conversation`` returns ``False`` so the caller sees the failure.
    This is the §H "failure path tested — never silently swallowed" check.
    """
    unified = _FakeUnified(l3_returns_error=True)
    mgr = _make_mgr(unified)
    _patch_distill(mgr, ["durable fact here"])

    ok = await mgr.store_conversation(
        workspace_id="ws-C",
        agent_id=7,
        user_message="something durable for L3",
        assistant_response="Logged.",
        chat_id="conv-err-visible",
    )

    assert ok is False, (
        "L3 write failure must be visible via store_conversation's return; a "
        "silent True would hide the degradation from observers"
    )
    # And L2 still got the transcript — no turn loss.
    assert len(unified.transcript_calls) == 1


# ---------------------------------------------------------------------------
# 4. Per-workspace breaker isolation
# ---------------------------------------------------------------------------


def test_per_workspace_breaker_isolation():
    """A Mem0 outage in workspace A's breaker must not trip workspace B's.

    The shared Mem0Client carries a per-workspace breaker registry
    (``_breakers``). Calls scoped to ``workspace_id`` use that workspace's
    breaker. Calls without a workspace scope share the ``_global`` breaker.
    This re-proves the isolation invariant (PRD-141 US-006) — it underpins
    the W3-S7 claim that one workspace's Mem0 outage does not break every
    other workspace's chat turn.
    """
    # Stdlib-only stub for the modules.memory parents — avoid loading the full
    # memory chain (asyncpg / pdfplumber would be pulled).
    for _pkg in ("modules", "modules.memory", "modules.memory.integrations"):
        if _pkg not in sys.modules:
            _stub = types.ModuleType(_pkg)
            _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _stub

    # Stub config.config for Mem0Client init (it reads MEM0_API_URL etc.).
    if "config" not in sys.modules:
        cfg = types.ModuleType("config")
        cfg.config = MagicMock(
            MEM0_API_URL="",
            MEM0_API_KEY="",
            MEM0_TIMEOUT_SECONDS=3.0,
            MEM0_WRITE_TIMEOUT_SECONDS=5.0,
            MEM0_CIRCUIT_THRESHOLD=3,
            MEM0_CIRCUIT_COOLDOWN_SECONDS=60,
        )
        sys.modules["config"] = cfg

    from modules.memory.integrations.mem0_client import Mem0Client

    # Reset the breaker registry for a clean test.
    Mem0Client._breakers = {}

    breaker_a = Mem0Client._get_breaker("ws-iso-A")
    breaker_b = Mem0Client._get_breaker("ws-iso-B")
    assert breaker_a is not breaker_b, (
        "Each workspace must get its own breaker instance"
    )

    # Trip workspace A's breaker by recording failures up to the threshold.
    for _ in range(breaker_a.threshold):
        breaker_a.record_failure()

    assert breaker_a.is_open is True, "A's breaker should be open after threshold failures"
    assert breaker_b.is_open is False, (
        "B's breaker must remain closed — per-workspace isolation"
    )
    assert breaker_b.allow_request() is True, (
        "Workspace B should still be allowed to call Mem0 even while A is tripped"
    )
