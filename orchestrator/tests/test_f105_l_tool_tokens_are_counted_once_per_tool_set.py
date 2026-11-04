"""F105 (4 Nov): a turn's tool list is counted once per tool set, and never on the event loop.

The loop watchdog caught the context guard encoding ``json.dumps(tools)`` with tiktoken on
every chat turn, on the event loop: 3.6 s inside ``check_and_compact`` (the chat's
``_prepare_messages``). The tool list is nearly the same every turn, so its count is kept
per tool set (a bounded LRU keyed by the hash of the serialized list, shared by every
thread), and ``check_and_compact`` counts the payload in a worker thread.

No tokenizer, model or database: the counter is patched to record what it encodes.
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from core import context_guard
from core.context_guard import ContextGuard, count_tool_tokens


def _tools(name: str = "") -> list:
    """A tool set no other test uses (the cache is process-wide)."""
    name = name or f"platform_probe_{uuid.uuid4().hex}"
    return [{"type": "function", "function": {"name": name, "description": "Reads the board.",
                                               "parameters": {"type": "object", "properties": {}}}}]


@pytest.fixture
def encoded(monkeypatch):
    """Every text the guard encodes; each counts as its length."""
    texts = []

    def _count(text):
        texts.append(text)
        return len(text)

    monkeypatch.setattr(context_guard, "count_tokens", _count)
    return texts


def test_a_repeated_tool_set_is_encoded_once(encoded):
    tools = _tools()

    first = count_tool_tokens(tools)
    again = count_tool_tokens([dict(tool) for tool in tools])  # an equal list, built afresh

    assert first == again == len(encoded[0])
    assert len(encoded) == 1


def test_another_tool_set_is_counted_on_its_own(encoded):
    first, second = _tools(), _tools()

    assert count_tool_tokens(first) == len(encoded[0])
    assert count_tool_tokens(second) == len(encoded[1])
    assert len(encoded) == 2


def test_no_tools_count_nothing(encoded):
    assert count_tool_tokens(None) == 0
    assert count_tool_tokens([]) == 0
    assert encoded == []


def test_the_cache_is_bounded_and_drops_the_least_recently_used(encoded, monkeypatch):
    monkeypatch.setattr(context_guard, "TOOL_TOKEN_CACHE_SIZE", 3)
    oldest, *rest = [_tools() for _ in range(4)]
    for tools in [oldest, *rest]:
        count_tool_tokens(tools)

    assert len(context_guard._tool_token_counts) <= 3
    count_tool_tokens(rest[-1])  # still kept: not encoded again
    assert len(encoded) == 4
    count_tool_tokens(oldest)  # dropped: encoded again
    assert len(encoded) == 5


def test_threads_counting_one_tool_set_agree(encoded):
    tools = _tools()

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(lambda _: count_tool_tokens(tools), range(64)))

    assert set(counts) == {len(encoded[0])}


def test_the_guard_counts_the_payload_off_the_event_loop(monkeypatch):
    counted_on = []

    def _tool_count(tools):
        counted_on.append(threading.get_ident())
        return 10

    def _message_count(messages):
        counted_on.append(threading.get_ident())
        return 20

    monkeypatch.setattr(context_guard, "count_tool_tokens", _tool_count)
    monkeypatch.setattr(context_guard, "count_message_tokens", _message_count)
    messages, tools = [{"role": "user", "content": "What needs me?"}], _tools()

    async def guard():
        result = await ContextGuard().check_and_compact(messages=messages, model_name="", llm_manager=None,
                                                        tools=tools)
        return threading.get_ident(), result

    loop_thread, result = asyncio.run(guard())

    assert result == (messages, False, tools)  # well under the window: passed through unchanged
    assert len(counted_on) == 2 and loop_thread not in counted_on
