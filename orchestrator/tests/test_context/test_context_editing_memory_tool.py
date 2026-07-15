"""PRD-201 S5 — context-editing + memory-tool on the headless Anthropic loop.

Pure/mocked — assert request SHAPE, never call a provider. Covers the
headless-gated context-editing / memory-tool request shaping, the memory-tool
declaration, and the /memories traversal guard.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_ORCH = Path(__file__).resolve().parent.parent.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.llm.request_scope import headless_run, is_headless_run
from modules.memory.memory_tool import (
    MEMORY_TOOL_TYPE,
    MemoryPathError,
    MemoryToolBackend,
    memory_tool_definition,
    resolve_memory_path,
)


# --- traversal guard ---


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "/memories/../secret",
        "../../etc/shadow",
        "/etc/passwd",
        "%2e%2e/x",
        "/memories/a/../../etc",
        "",
    ],
)
def test_memory_path_traversal_rejected(bad):
    with pytest.raises(MemoryPathError):
        resolve_memory_path(bad)


@pytest.mark.parametrize(
    "good,expected",
    [
        ("notes.md", "/memories/notes.md"),
        ("/memories/notes.md", "/memories/notes.md"),
        ("/memories/a/b.md", "/memories/a/b.md"),
        ("/memories", "/memories"),
        ("a/b/c.txt", "/memories/a/b/c.txt"),
    ],
)
def test_memory_path_confined_to_memories(good, expected):
    assert resolve_memory_path(good) == expected


def test_backend_rejects_traversal_without_touching_store():
    store = AsyncMock()
    backend = MemoryToolBackend(store, workspace_id="ws")
    out = asyncio.run(backend.handle({"command": "view", "path": "../../etc/passwd"}))
    assert out.startswith("Error:")
    store.get_text.assert_not_called()
    store.list_paths.assert_not_called()


def test_backend_create_then_view_roundtrips_through_store():
    store = AsyncMock()
    store.get_text.return_value = "remembered"
    backend = MemoryToolBackend(store, workspace_id="ws")
    asyncio.run(backend.handle({"command": "create", "path": "notes.md", "file_text": "remembered"}))
    store.put_text.assert_awaited_once()
    # The path handed to the store is always the guarded, normalised form.
    assert store.put_text.await_args.args[1] == "/memories/notes.md"
    view = asyncio.run(backend.handle({"command": "view", "path": "notes.md"}))
    assert view == "remembered"


# --- memory-tool declaration ---


def test_memory_tool_declared():
    assert memory_tool_definition() == {"type": "memory_20250818", "name": "memory"}
    assert MEMORY_TOOL_TYPE == "memory_20250818"


# --- request shaping: headless-gated ---


def test_request_scope_default_off():
    assert is_headless_run() is False
    with headless_run():
        assert is_headless_run() is True
    assert is_headless_run() is False


def _anthropic_provider():
    from core.llm.clients.anthropic_client import AnthropicProvider
    from core.llm.clients.base import LLMConfig, LLMProvider

    prov = AnthropicProvider.__new__(AnthropicProvider)
    prov.config = LLMConfig(
        provider=LLMProvider.ANTHROPIC, model="claude-opus-4-8", max_tokens=1024, temperature=0.7
    )
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="claude-opus-4-8",
        stop_reason="end_turn",
    )
    prov.client = MagicMock()
    prov.client.messages.create.return_value = resp
    prov.client.beta.messages.create.return_value = resp
    return prov


_MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
_TOOLS = [{"name": "search", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def test_context_editing_on_headless_anthropic_loop():
    prov = _anthropic_provider()

    async def _run():
        # Enter the headless scope inside the coroutine so the flag is set in the
        # same task context generate_response reads it from.
        with headless_run():
            return await prov.generate_response(_MSGS, tools=_TOOLS)

    asyncio.run(_run())
    # Headless → beta endpoint, context-editing edit, memory tool declared.
    prov.client.beta.messages.create.assert_called_once()
    prov.client.messages.create.assert_not_called()
    kwargs = prov.client.beta.messages.create.call_args.kwargs
    assert kwargs["betas"] == ["context-management-2025-06-27"]
    assert kwargs["context_management"] == {"edits": [{"type": "clear_tool_uses_20250919"}]}
    tool_types = [t.get("type") for t in kwargs["tools"]]
    assert "memory_20250818" in tool_types


def test_no_context_editing_off_headless():
    prov = _anthropic_provider()
    asyncio.run(prov.generate_response(_MSGS, tools=_TOOLS))  # no headless scope
    prov.client.messages.create.assert_called_once()
    prov.client.beta.messages.create.assert_not_called()
    kwargs = prov.client.messages.create.call_args.kwargs
    assert "context_management" not in kwargs
    tool_types = [t.get("type") for t in (kwargs.get("tools") or [])]
    assert "memory_20250818" not in tool_types
