"""PRD-201 S4 — prompt caching (the cost prize).

Pure/mocked — assert request SHAPE only, never call a provider. Covers the
cache-stable assembler ordering, the single-breakpoint client seam, and that
``cache_control`` is emitted on the Anthropic route only.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_ORCH = Path(__file__).resolve().parent.parent.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.llm.prompt_cache import build_cached_system, count_cache_breakpoints, read_cache_usage
from modules.context.budget import RenderedSection
from modules.context.service import ContextService


def _section(name, content, priority=5):
    return RenderedSection(name=name, priority=priority, content=content, token_estimate=len(content) // 4)


# --- assembler: cache-stable ordering ---


def test_assemble_orders_static_first_volatile_last():
    sections = [
        _section("datetime_context", "NOW=2026-07-15T10:00Z"),  # volatile
        _section("identity", "You are Auto."),                   # stable
        _section("memory", "user likes tabs"),                   # volatile
        _section("skills", "SKILL: search"),                     # stable
    ]
    system, prefix = ContextService._assemble_prompt(sections)
    # Stable blocks lead the prefix; volatile blocks are absent from it.
    assert prefix is not None
    assert "You are Auto." in prefix and "SKILL: search" in prefix
    assert "NOW=" not in prefix and "user likes tabs" not in prefix
    # The full system still contains everything (content unchanged, order only).
    assert system.startswith(prefix)
    assert "NOW=" in system and "user likes tabs" in system


def test_volatile_change_preserves_cached_prefix():
    stable = [_section("identity", "You are Auto."), _section("skills", "SKILL: search")]
    a = ContextService._assemble_prompt(stable + [_section("memory", "fact A")])
    b = ContextService._assemble_prompt(stable + [_section("memory", "a completely different fact B")])
    # A change in a volatile section leaves the cache-stable prefix bytes intact.
    assert a[1] == b[1]
    # And the cached block the client would emit is byte-identical across the two.
    sys_a, sys_b = build_cached_system(a[0], a[1]), build_cached_system(b[0], b[1])
    assert sys_a[0] == sys_b[0]


# --- client seam: exactly one breakpoint ---


def test_stable_prefix_single_breakpoint():
    system = "STATIC PREFIX\n\nvolatile tail"
    blocks = build_cached_system(system, "STATIC PREFIX")
    assert isinstance(blocks, list) and len(blocks) == 2
    assert count_cache_breakpoints(blocks) == 1
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[0]["text"] == "STATIC PREFIX"
    assert "cache_control" not in blocks[1]
    # Reassembled content is byte-identical to the input (content unchanged).
    assert blocks[0]["text"] + blocks[1]["text"] == system


def test_single_block_when_no_prefix():
    blocks = build_cached_system("whole system, no split")
    assert count_cache_breakpoints(blocks) == 1  # still exactly one


def test_empty_system_is_not_cached():
    assert build_cached_system("") == ""  # opt-in on content, never a bare marker


def test_read_cache_usage_extracts_prize_fields():
    usage = SimpleNamespace(cache_read_input_tokens=900, cache_creation_input_tokens=100, input_tokens=50)
    got = read_cache_usage(usage)
    assert got == {"cache_read_input_tokens": 900, "cache_creation_input_tokens": 100, "input_tokens": 50}
    assert read_cache_usage(None)["cache_read_input_tokens"] == 0


# --- cache_control only on the Anthropic route ---


def _anthropic_provider():
    from core.llm.clients.anthropic_client import AnthropicProvider
    from core.llm.clients.base import LLMConfig, LLMProvider

    prov = AnthropicProvider.__new__(AnthropicProvider)  # bypass _initialize_client
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


def test_cache_control_emitted_on_anthropic_route():
    prov = _anthropic_provider()
    messages = [
        {"role": "system", "content": "STATIC\n\nvolatile", "cache_prefix": "STATIC"},
        {"role": "user", "content": "hi"},
    ]
    asyncio.run(prov.generate_response(messages, tools=None))
    system = prov.client.messages.create.call_args.kwargs["system"]
    assert count_cache_breakpoints(system) == 1
    assert system[0]["text"] == "STATIC"  # split at the assembler's cache_prefix


def test_no_cache_control_on_non_anthropic_client_source():
    # Structural: cache_control emission lives only in the Anthropic client seam
    # (+ the pre-existing dead Bedrock passthrough). OpenAI/OpenRouter/etc. routes
    # never emit it.
    clients = _ORCH / "core" / "llm" / "clients"
    offenders = [
        p.name
        for p in clients.glob("*.py")
        if "cache_control" in p.read_text(encoding="utf-8")
        and p.name not in {"anthropic_client.py", "bedrock_client.py"}
    ]
    assert offenders == [], f"unexpected cache_control on non-Anthropic clients: {offenders}"
