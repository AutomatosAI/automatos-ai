"""Regression tests for fire-and-forget memory writes in SmartChatOrchestrator.

PRD-141 widget latency fix: store_exchange() must never block the streaming
response on Mem0 fact extraction (a multi-second, server-side LLM call). Every
write — distilled facts, daily summary, L2 transcript, L1 session — is scheduled
fire-and-forget and the turn returns without waiting on the outcome.

PRD-142 W3-S7 / G12 (write-once-per-layer): the L2 transcript is the SINGLE L2
write per chat turn — fanned out via ``memory_manager.store_conversation``.
The older direct ``_unified_memory.store_exchange`` spawn (a duplicate L2 row
with content_type='exchange') was retired; the assertions below pin that the
collapsed path stays collapsed.

The orchestrator is exercised via its *unbound* store_exchange method bound to a
lightweight fake ``self``. This deliberately skips ``__init__`` (which lazily
imports ``.smart_memory`` / the unified memory service) so the test stays fully
isolated from sys.modules state mutated by sibling test modules.
"""
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

# Stub the consumers / consumers.chatbot package inits so importing
# smart_orchestrator does NOT pull the full chain (asyncpg / pdfplumber /
# tiktoken / camelot — not installed in the unit-test env). Mirrors the
# pattern in ``test_memory_single_write_path``.
_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

for _pkg in ("consumers", "consumers.chatbot"):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub

if "consumers.chatbot.intent_classifier" not in sys.modules:
    _ic_stub = types.ModuleType("consumers.chatbot.intent_classifier")
    _ic_stub.Intent = MagicMock()
    _ic_stub.IntentResult = MagicMock()
    _ic_stub.get_intent_classifier = lambda: MagicMock()
    sys.modules["consumers.chatbot.intent_classifier"] = _ic_stub

sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from consumers.chatbot.smart_orchestrator import SmartChatOrchestrator  # noqa: E402

# Unbound coroutine function — invoked with an explicit fake ``self`` below.
store_exchange = SmartChatOrchestrator.store_exchange


def _make_fake_self():
    """A stand-in for an initialised SmartChatOrchestrator.

    store_exchange only touches these attributes, so we provide them directly
    rather than running the real constructor.
    """
    return SimpleNamespace(
        workspace_id="ws-test",
        agent_id=1,
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


@pytest.mark.asyncio
async def test_store_exchange_does_not_block_on_slow_mem0_write():
    """store_exchange returns before a slow Mem0 write completes."""
    fake = _make_fake_self()

    gate = asyncio.Event()
    write_started = asyncio.Event()

    async def blocking_store_conversation(**kwargs):
        write_started.set()
        await gate.wait()  # would hang store_exchange if awaited inline
        return True

    fake.memory_manager.store_conversation = blocking_store_conversation

    # If the Mem0 write were awaited inline, `gate` is never set before
    # store_exchange awaits it, so wait_for would raise TimeoutError. Because the
    # write is backgrounded, store_exchange returns immediately.
    result = await asyncio.wait_for(
        store_exchange(fake, "what are your opening hours?", "We're open 9-5.", chat_id="c1"),
        timeout=2.0,
    )
    assert result is True
    # Yield once so the scheduled background tasks actually run up to their
    # first ``await`` (where ``blocking_store_conversation`` sets the event).
    await asyncio.sleep(0)
    assert write_started.is_set()  # the write was scheduled and started running

    gate.set()
    await _drain_background()


@pytest.mark.asyncio
async def test_store_exchange_schedules_all_memory_tiers():
    """Three fanout writes are scheduled per turn — distilled facts +
    transcript via ``store_conversation``, daily log via ``store_daily_summary``,
    L1 via ``update_session``. The retired ``_unified_memory.store_exchange``
    duplicate is NOT awaited (W3-S7 G12 collapse).
    """
    fake = _make_fake_self()

    result = await store_exchange(fake, "remember I prefer tea", "Noted!", chat_id="c1")
    assert result is True

    await _drain_background()

    fake.memory_manager.store_conversation.assert_awaited_once()
    fake.memory_manager.store_daily_summary.assert_awaited_once()
    fake._unified_memory.update_session.assert_awaited_once()
    # G12 — the dual L2 write was retired.
    fake._unified_memory.store_exchange.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_exchange_skips_l1_session_without_chat_id():
    """L1 session update only fires when a chat_id is present. The retired
    direct L2 ``_unified_memory.store_exchange`` spawn does not fire either —
    the L2 write goes through ``memory_manager.store_conversation``.
    """
    fake = _make_fake_self()

    result = await store_exchange(fake, "hello there friend", "Hi!", chat_id=None)
    assert result is True

    await _drain_background()

    fake._unified_memory.update_session.assert_not_awaited()
    fake._unified_memory.store_exchange.assert_not_awaited()
    # Transcript write still routes through store_conversation.
    fake.memory_manager.store_conversation.assert_awaited_once()
