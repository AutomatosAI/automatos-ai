"""Regression tests for fire-and-forget memory writes in SmartChatOrchestrator.

PRD-141 widget latency fix: store_exchange() must never block the streaming
response on Mem0 fact extraction (a multi-second, server-side LLM call). Every
write — distilled facts, daily summary, L2 transcript, L1 session — is scheduled
fire-and-forget and the turn returns without waiting on the outcome.

The orchestrator is exercised via its *unbound* store_exchange method bound to a
lightweight fake ``self``. This deliberately skips ``__init__`` (which lazily
imports ``.smart_memory`` / the unified memory service) so the test stays fully
isolated from sys.modules state mutated by sibling test modules.
"""
import asyncio
import sys
import types
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

# `camelot` is an optional PDF table-extraction dep pulled in transitively when
# `consumers` is imported. It isn't installed in the unit-test env; stub it so
# the import chain resolves.
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
    assert write_started.is_set()  # the write was scheduled and started running

    gate.set()
    await _drain_background()


@pytest.mark.asyncio
async def test_store_exchange_schedules_all_memory_tiers():
    """All four memory writes are scheduled (facts, daily log, L2, L1)."""
    fake = _make_fake_self()

    result = await store_exchange(fake, "remember I prefer tea", "Noted!", chat_id="c1")
    assert result is True

    await _drain_background()

    fake.memory_manager.store_conversation.assert_awaited_once()
    fake.memory_manager.store_daily_summary.assert_awaited_once()
    fake._unified_memory.store_exchange.assert_awaited_once()
    fake._unified_memory.update_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_exchange_skips_l1_session_without_chat_id():
    """L1 session update only fires when a chat_id is present."""
    fake = _make_fake_self()

    result = await store_exchange(fake, "hello there friend", "Hi!", chat_id=None)
    assert result is True

    await _drain_background()

    fake._unified_memory.update_session.assert_not_awaited()
    fake._unified_memory.store_exchange.assert_awaited_once()
