"""PRD-142 Wave 3 · W3-S7 — write-once-per-layer (G12) for Memory.

The Memory primitive's BRAIN §3.x contract says: *exactly one write path per
layer*. Today the chat path dual-writes L2 — the same chat exchange is stored
both as ``content_type='exchange'`` via ``UnifiedMemoryService.store_exchange``
*and* as ``content_type='transcript'`` via
``SmartMemoryManager.store_conversation`` (which routes through
``UnifiedMemoryService.store_transcript``). The PRD-131d ``store_transcript``
path is the canonical one (it carries the verbatim multi-turn payload and
richer metadata); the older ``store_exchange`` path is the duplicate this
story collapses.

These tests pin write-once-per-layer for a single chat turn. They mock at the
seams above the database so the assertion is purely about *call sites fired*,
not Postgres rows committed (the integration check for that lives in
``test_memory_restart_and_isolation.py``).

ISOLATION: stubs ``consumers`` / ``consumers.chatbot`` parent packages so
importing ``consumers.chatbot.smart_orchestrator`` does NOT run
``consumers/chatbot/__init__.py`` (which transitively pulls
asyncpg/pdfplumber/tiktoken — not present in the unit-test env). Mirrors the
``test_tool_loop_characterization`` pattern.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Package-stub isolation — bypass the consumers.chatbot package __init__ that
# would otherwise pull the full tool/RAG/memory chain. We still load the real
# ``smart_orchestrator.py`` source via its normal import path.
# ---------------------------------------------------------------------------

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

for _pkg in ("consumers", "consumers.chatbot"):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub

# ``intent_classifier`` is the only sibling smart_orchestrator imports at
# module level; stub it so we don't trigger its dependency chain either.
_ic_stub_installed = "consumers.chatbot.intent_classifier" not in sys.modules
if _ic_stub_installed:
    _ic_stub = types.ModuleType("consumers.chatbot.intent_classifier")
    _ic_stub.Intent = MagicMock()
    _ic_stub.IntentResult = MagicMock()
    _ic_stub.get_intent_classifier = lambda: MagicMock()
    sys.modules["consumers.chatbot.intent_classifier"] = _ic_stub

# ``smart_memory`` is imported lazily by smart_orchestrator only for type
# hints / runtime calls *we mock on the fake self* — but we DO want to be able
# to test the real ``SmartMemoryManager.store_conversation`` shape below, so
# stub anything it would pull at module load time. We avoid importing it here
# at module load (the source is loaded on-demand inside one of the tests).
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from consumers.chatbot.smart_orchestrator import SmartChatOrchestrator  # noqa: E402

# De-pollute: our intent_classifier stub is intentionally partial (no
# SmartIntentClassifier). Once smart_orchestrator is imported we no longer need
# it, so drop it to let a later-collected sibling (e.g.
# test_w2s9_reasoning_entry) import the REAL module — its __path__ resolves via
# the consumers.chatbot package stub above. Leaving the stub cached caused the
# "(unknown location)" collection error in the full-suite CI run.
if _ic_stub_installed:
    sys.modules.pop("consumers.chatbot.intent_classifier", None)

# Unbound — invoked with an explicit fake ``self``.
_orchestrator_store_exchange = SmartChatOrchestrator.store_exchange


def _make_fake_orchestrator():
    """SimpleNamespace standing in for an initialised SmartChatOrchestrator.

    Mirrors ``test_smart_orchestrator_store_exchange._make_fake_self`` so the
    two test files stay byte-aligned on the contract surface that
    ``store_exchange`` touches.
    """
    return SimpleNamespace(
        workspace_id="ws-w3s7",
        agent_id=7,
        widget_mode=False,
        memory_manager=MagicMock(
            store_conversation=AsyncMock(return_value=True),
            store_daily_summary=AsyncMock(return_value=True),
        ),
        _unified_memory=MagicMock(
            store_exchange=AsyncMock(return_value=None),
            update_session=AsyncMock(return_value=None),
        ),
    )


async def _drain_background():
    """Await all fire-and-forget tasks scheduled on this test's event loop."""
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# G12 — one write per layer per logical event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_turn_writes_l1_exactly_once():
    """L1 (Redis session via ``update_session``) is written exactly once per
    turn (with a chat_id; the no-chat_id case is covered separately in
    ``test_smart_orchestrator_store_exchange``)."""
    fake = _make_fake_orchestrator()

    result = await _orchestrator_store_exchange(
        fake,
        "what hours are you open?",
        "9 to 5 Monday to Friday.",
        chat_id="conv-1",
    )
    assert result is True

    await _drain_background()

    assert fake._unified_memory.update_session.await_count == 1, (
        "L1 session must be written exactly once per chat turn; got "
        f"{fake._unified_memory.update_session.await_count}"
    )


@pytest.mark.asyncio
async def test_chat_turn_writes_l2_exactly_once():
    """L2 (Postgres ``memory_short_term``) is written exactly once per turn —
    via ``SmartMemoryManager.store_conversation`` (which routes through
    ``store_transcript``). The duplicate
    ``UnifiedMemoryService.store_exchange`` direct call MUST NOT be awaited
    (that path is what G12 collapses).
    """
    fake = _make_fake_orchestrator()

    result = await _orchestrator_store_exchange(
        fake,
        "please remember my prefs",
        "Noted — I'll remember.",
        chat_id="conv-2",
    )
    assert result is True

    await _drain_background()

    # Canonical L2 path: store_conversation → store_transcript.
    assert fake.memory_manager.store_conversation.await_count == 1, (
        "L2 transcript must fire exactly once per turn via store_conversation"
    )
    # Collapsed L2 path: the legacy store_exchange direct call must not run.
    assert fake._unified_memory.store_exchange.await_count == 0, (
        "Dual L2 write: UnifiedMemoryService.store_exchange should not be "
        "awaited from the chat path — that path is the duplicate G12 "
        "(write-once-per-layer) collapses."
    )


@pytest.mark.asyncio
async def test_chat_turn_writes_l3_exactly_once():
    """L3 (Mem0) is invoked exactly once per turn via ``store_conversation``.

    ``store_conversation`` routes to a SINGLE namespace per turn — "global" by
    default, or the agent namespace only on an explicit agent-scoped instruction
    (PRD-159 S5 removed the old 'both'-tier double-write default). What G12
    forbids is two SEPARATE call sites both pushing the same chat exchange to L3
    — the chat path must have exactly one.
    """
    fake = _make_fake_orchestrator()

    result = await _orchestrator_store_exchange(
        fake,
        "log this exchange please",
        "Logged.",
        chat_id="conv-3",
    )
    assert result is True

    await _drain_background()

    assert fake.memory_manager.store_conversation.await_count == 1, (
        "L3 (chat → Mem0 distilled facts) must fire exactly once per turn"
    )
