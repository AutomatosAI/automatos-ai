"""Prompt caching on the OpenRouter route + the governor on reported cost (2026-09-16).

Ground truth that motivated this: one Auto turn on ``anthropic/claude-opus-4.6``
through OpenRouter made nine model calls, each re-sending the same ~28k-token
system context and ~11k of tool schemas at full price (``cache_read_tokens`` = 0
on every row), while the cost governor summed a static-map estimate that had no
entry for the model and let the turn through a $1.50 ceiling.

Pinned here (request SHAPE only — no provider is ever called):
1. an Anthropic model on a provider that passes breakpoints through gets its
   first system message as content parts with ONE ``cache_control`` marker on
   the assembler's stable prefix, plus request-level automatic caching for the
   conversation tail; the assembler's ``cache_prefix`` hint never leaves;
2. a non-Anthropic model on the same provider, and any model on a provider
   without the capability, are sent unchanged (hint stripped);
3. the forced-tool detection still reads a system prompt that became parts;
4. ``usage_dict`` reads OpenRouter's ``cache_write_tokens``;
5. ``call_cost_usd`` books the reported cost first, the estimate second;
6. the static price map is ordered longest-key-first, so ``gpt-4.1-mini`` is
   priced as itself, not as ``gpt-4.1``;
7. the chat lane stamps the assembler's prefix on its system message.
"""
from __future__ import annotations

import inspect
import os
from types import MethodType, SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.llm.clients.openai_compatible_client import OpenAICompatibleProvider, usage_dict  # noqa: E402
from core.llm.manager import MODEL_COST_MAP, estimate_cost_usd  # noqa: E402
from core.llm.prompt_cache import (  # noqa: E402
    EPHEMERAL,
    EPHEMERAL_1H,
    apply_openai_cache_control,
    count_cache_breakpoints,
    needs_explicit_breakpoints,
    strip_cache_hints,
    text_of,
)
from core.llm.turn_cost import call_cost_usd  # noqa: E402

STABLE = "IDENTITY + SKILLS + PLATFORM ACTIONS " * 40
VOLATILE = "\n\nToday is 2026-09-16 18:00 UTC."
SYSTEM = STABLE + VOLATILE


def _messages(hint=STABLE):
    system = {"role": "system", "content": SYSTEM}
    if hint is not None:
        system["cache_prefix"] = hint
    return [system, {"role": "user", "content": "create the agents"}]


# ── 1/2. the pure transform ───────────────────────────────────────

def test_anthropic_model_gets_one_marker_on_the_stable_prefix_and_automatic_tail():
    out, top = apply_openai_cache_control(_messages(), "anthropic/claude-opus-4.6")
    assert top == EPHEMERAL
    system = out[0]
    assert "cache_prefix" not in system
    parts = system["content"]
    assert isinstance(parts, list) and len(parts) == 2
    assert parts[0] == {"type": "text", "text": STABLE, "cache_control": EPHEMERAL}
    assert parts[1] == {"type": "text", "text": VOLATILE}  # the volatile tail carries no marker
    assert count_cache_breakpoints(parts) == 1
    assert out[1] == {"role": "user", "content": "create the agents"}


def test_without_a_hint_the_whole_system_is_one_cached_block():
    out, top = apply_openai_cache_control(_messages(hint=None), "anthropic/claude-sonnet-4.6")
    assert top == EPHEMERAL
    assert out[0]["content"] == [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}]


def test_one_hour_ttl_is_a_dial():
    out, top = apply_openai_cache_control(_messages(), "anthropic/claude-opus-4.6", ttl_1h=True)
    assert top == EPHEMERAL_1H
    assert out[0]["content"][0]["cache_control"] == EPHEMERAL_1H


def test_non_anthropic_models_are_only_stripped():
    for model in ("google/gemini-2.5-flash", "openai/gpt-5.5", "z-ai/glm-5"):
        out, top = apply_openai_cache_control(_messages(), model)
        assert top is None
        assert out[0] == {"role": "system", "content": SYSTEM}  # a string, hint gone


def test_only_the_first_system_message_is_converted_and_inputs_are_not_mutated():
    msgs = _messages() + [{"role": "system", "content": "You are in PLAN MODE."}]
    before = [dict(m) for m in msgs]
    out, _ = apply_openai_cache_control(msgs, "anthropic/claude-opus-4.6")
    assert msgs == before  # immutable: new dicts, the caller's list untouched
    assert isinstance(out[0]["content"], list)
    assert out[2] == {"role": "system", "content": "You are in PLAN MODE."}


def test_needs_explicit_breakpoints_and_strip():
    assert needs_explicit_breakpoints("anthropic/claude-opus-4.6")
    assert needs_explicit_breakpoints("claude-sonnet-4-6")
    assert not needs_explicit_breakpoints("google/gemini-2.5-flash")
    assert not needs_explicit_breakpoints(None)
    assert strip_cache_hints([{"role": "system", "content": "x", "cache_prefix": "x"}]) == [
        {"role": "system", "content": "x"}
    ]
    assert text_of("plain") == "plain"
    assert text_of([{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]) == "a b"


# ── 1/2/3. through the client's request assembly ─────────────────

def _client(model: str, *, prompt_cache_control: bool, reports_cost: bool = True):
    """A provider instance without an SDK client: the three helpers the request
    assembly uses are bound from the real class (the tests/test_prd240 idiom)."""
    stub = SimpleNamespace(
        spec=SimpleNamespace(prompt_cache_control=prompt_cache_control, reports_cost=reports_cost, web_search_tool=None),
        config=SimpleNamespace(model=model, temperature=0.2, max_tokens=1024, top_p=None,
                               frequency_penalty=None, presence_penalty=None, stop=None),
        _extra_body=None,
    )
    for name in ("_base_kwargs", "_sanitize_tools", "_web_search_server_tool", "_request_kwargs"):
        attr = inspect.getattr_static(OpenAICompatibleProvider, name)
        if isinstance(attr, staticmethod):
            setattr(stub, name, attr.__func__)
        elif isinstance(attr, classmethod):
            setattr(stub, name, MethodType(attr.__func__, OpenAICompatibleProvider))
        else:
            setattr(stub, name, MethodType(attr, stub))
    return stub


FN_TOOL = {"type": "function", "function": {"name": "platform_execute", "parameters": {"type": "object", "properties": {}}}}


def test_request_to_openrouter_for_anthropic_carries_the_marker_and_automatic_caching(monkeypatch):
    monkeypatch.setattr(config, "PROMPT_CACHE_TTL_1H", False)
    kwargs = _client("anthropic/claude-opus-4.6", prompt_cache_control=True)._request_kwargs(_messages(), [FN_TOOL])
    system = kwargs["messages"][0]
    assert "cache_prefix" not in system
    assert system["content"][0]["cache_control"] == EPHEMERAL
    assert kwargs["extra_body"] == {"usage": {"include": True}, "cache_control": EPHEMERAL}
    assert kwargs["tool_choice"] == "auto"


def test_request_to_openrouter_for_gemini_is_unchanged_but_stripped(monkeypatch):
    kwargs = _client("google/gemini-2.5-flash", prompt_cache_control=True)._request_kwargs(_messages(), [FN_TOOL])
    assert kwargs["messages"][0] == {"role": "system", "content": SYSTEM}
    assert kwargs["extra_body"] == {"usage": {"include": True}}


def test_request_on_a_provider_without_the_capability_is_stripped_only():
    kwargs = _client("anthropic/claude-opus-4.6", prompt_cache_control=False, reports_cost=False)._request_kwargs(_messages(), None)
    assert kwargs["messages"][0] == {"role": "system", "content": SYSTEM}
    assert "extra_body" not in kwargs


def test_forced_tool_turn_is_still_detected_inside_cached_parts():
    forced = [{"role": "system", "content": STABLE + "\nYou MUST call platform_execute now.", "cache_prefix": STABLE},
              {"role": "user", "content": "go"}]
    kwargs = _client("anthropic/claude-opus-4.6", prompt_cache_control=True)._request_kwargs(forced, [FN_TOOL])
    assert isinstance(kwargs["messages"][0]["content"], list)
    assert kwargs["tool_choice"] == "required"


# ── 4. usage parsing ──────────────────────────────────────────────

def test_usage_dict_reads_cache_writes_beside_reads():
    details = SimpleNamespace(cached_tokens=39000, cache_write_tokens=1500)
    usage = SimpleNamespace(prompt_tokens=41000, completion_tokens=150, total_tokens=41150, prompt_tokens_details=details,
                            model_extra={"cost": 0.0412})
    out = usage_dict(usage)
    assert out["cache_read_tokens"] == 39000
    assert out["cache_write_tokens"] == 1500
    assert out["cost"] == pytest.approx(0.0412)


# ── 5. the governor's per-call figure ─────────────────────────────

def test_call_cost_prefers_the_reported_cost_then_the_estimate():
    estimate = lambda i, o: (i + o) / 1000 * 0.003  # noqa: E731
    assert call_cost_usd({"prompt_tokens": 39577, "completion_tokens": 750, "cost": 0.43327}, estimate) == pytest.approx(0.43327)
    assert call_cost_usd({"prompt_tokens": 1000, "completion_tokens": 1000}, estimate) == pytest.approx(0.006)
    assert call_cost_usd({"prompt_tokens": 1000, "cost": 0}, estimate) == pytest.approx(0.003)  # a zero report is not a price
    assert call_cost_usd({}, estimate) == 0.0
    assert call_cost_usd(None, estimate) == 0.0


# ── 6. the static map prices the specific model ──────────────────

def test_price_map_is_ordered_longest_key_first_and_knows_the_models_in_use():
    keys = list(MODEL_COST_MAP)
    assert keys == sorted(keys, key=lambda k: -len(k))
    assert estimate_cost_usd("openai/gpt-4.1-mini", 1000, 0) == pytest.approx(0.0004)
    assert estimate_cost_usd("openai/gpt-4.1", 1000, 0) == pytest.approx(0.002)
    assert estimate_cost_usd("anthropic/claude-opus-4.6", 1000, 0) == pytest.approx(0.005)
    assert estimate_cost_usd("google/gemini-2.5-flash", 1000, 0) == pytest.approx(0.0003)


# ── 7. the chat lane hands the prefix to the client ───────────────

def test_chat_lane_stamps_the_assembler_prefix_on_its_system_message():
    from consumers.chatbot.integration import apply_orchestration_to_messages

    request = SimpleNamespace(system_prompt=SYSTEM, messages=[{"role": "user", "content": "hi"}], cacheable_prefix=STABLE)
    msgs = apply_orchestration_to_messages(request)
    assert msgs[0] == {"role": "system", "content": SYSTEM, "cache_prefix": STABLE}
    assert msgs[1] == {"role": "user", "content": "hi"}

    plain = SimpleNamespace(system_prompt=SYSTEM, messages=[], cacheable_prefix=None)
    assert apply_orchestration_to_messages(plain) == [{"role": "system", "content": SYSTEM}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
