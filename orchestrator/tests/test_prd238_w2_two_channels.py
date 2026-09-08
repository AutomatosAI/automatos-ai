"""PRD-238 W2 — thinking and the answer as two channels.

Pure unit tests: the reasoning helpers, the stream assembler (fake chunks),
the reasoning frame, the manager's opt-in streaming, the chat service's
streaming helper (fake manager) and the prompt converter dropping reasoning.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# S1 · reasoning helpers
# ---------------------------------------------------------------------------

def test_split_think_tags_lifts_blocks_and_unclosed_tails():
    from core.llm.reasoning import split_think_tags

    assert split_think_tags("plain answer") == ("plain answer", None)
    answer, reasoning = split_think_tags("<think>step one\nstep two</think>\n\nThe answer.")
    assert answer == "The answer." and reasoning == "step one\nstep two"
    answer, reasoning = split_think_tags("A<think>x</think>B<think>y</think>C")
    assert answer == "ABC" and reasoning == "x\n\ny"
    answer, reasoning = split_think_tags("Hello <think>I ran out of tok")
    assert answer == "Hello" and reasoning == "I ran out of tok"
    assert split_think_tags(None) == ("", None)


def test_reasoning_from_fields_and_coalesce():
    from core.llm.reasoning import coalesce_reasoning, reasoning_from_fields

    assert reasoning_from_fields({"reasoning_content": " because "}) == " because "
    assert reasoning_from_fields({"reasoning": "r", "reasoning_content": ""}) == "r"
    assert reasoning_from_fields({"content": "x"}) is None
    assert reasoning_from_fields(None) is None
    assert coalesce_reasoning(None, " a ", "", "b") == "a\n\nb"
    assert coalesce_reasoning(None, "") is None


# ---------------------------------------------------------------------------
# S2 · the stream assembler
# ---------------------------------------------------------------------------

def _chunk(*, content=None, reasoning_content=None, tool_calls=None, finish=None, usage=None, model="m"):
    delta = SimpleNamespace(model_dump=lambda: {
        "content": content, "reasoning_content": reasoning_content, "tool_calls": tool_calls,
    })
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(**usage) if usage else None,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
    )


def test_stream_assembler_separates_channels_and_reassembles_tool_calls():
    try:
        from core.llm.clients.openai_compatible_client import _StreamAssembler
    except Exception as e:
        pytest.skip(f"client not importable here: {e}")

    a = _StreamAssembler()
    live = []
    live += a.feed(_chunk(reasoning_content="think "))
    live += a.feed(_chunk(reasoning_content="hard"))
    live += a.feed(_chunk(content="Hel"))
    live += a.feed(_chunk(content="lo"))
    live += a.feed(_chunk(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "platform_get_", "arguments": '{"na'}}]))
    live += a.feed(_chunk(tool_calls=[{"index": 0, "function": {"name": "agent", "arguments": 'me":"bob"}'}}], finish="tool_calls"))
    live += a.feed(SimpleNamespace(model="m", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15), choices=[]))

    assert live == [("reasoning", "think "), ("reasoning", "hard"), ("text", "Hel"), ("text", "lo")]
    resp = a.response("nvidia", streamed=True)
    assert resp.content == "Hello" and resp.reasoning == "think hard"
    assert resp.tool_calls == [{"id": "c1", "type": "function", "function": {"name": "platform_get_agent", "arguments": '{"name":"bob"}'}}]
    assert resp.finish_reason == "tool_calls" and resp.streamed is True
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_stream_assembler_lifts_inline_think_tags():
    try:
        from core.llm.clients.openai_compatible_client import _StreamAssembler
    except Exception as e:
        pytest.skip(f"client not importable here: {e}")
    a = _StreamAssembler()
    for piece in ("<think>", "why", "</think>", "Answer."):
        a.feed(_chunk(content=piece))
    resp = a.response("deepseek", streamed=True)
    assert resp.content == "Answer." and resp.reasoning == "why"


def test_reasoning_frame_shape():
    from consumers.chatbot.streaming import get_streaming_handler

    frame = get_streaming_handler().format_aisdk_reasoning("hmm")
    assert frame.startswith("d:")
    assert json.loads(frame[2:]) == {"type": "reasoning", "data": {"delta": "hmm"}}


# ---------------------------------------------------------------------------
# S2 · the manager streams only when asked and the provider can
# ---------------------------------------------------------------------------

class _Provider:
    def __init__(self, can_stream):
        self.calls = []
        if can_stream:
            async def stream_response(messages, tools, on_delta=None):
                self.calls.append("stream")
                await on_delta("text", "hi")
                return SimpleNamespace(content="hi", usage=None, streamed=True, reasoning=None)
            self.stream_response = stream_response

    async def generate_response(self, messages, tools=None):
        self.calls.append("whole")
        return SimpleNamespace(content="hi", usage=None, streamed=False, reasoning=None)


def test_manager_routes_to_stream_response_only_with_on_delta(monkeypatch):
    try:
        from core.llm.manager import LLMManager
    except Exception as e:
        pytest.skip(f"manager not importable here: {e}")

    mgr = object.__new__(LLMManager)
    monkeypatch.setattr(mgr, "_ensure_provider_initialized", lambda: None, raising=False)
    monkeypatch.setattr(mgr, "_track_usage", lambda *a, **k: None, raising=False)

    async def scenario(provider, on_delta):
        mgr.provider = provider
        return await LLMManager.generate_response(mgr, [{"role": "user", "content": "x"}], None, on_delta=on_delta)

    got = []
    async def on_delta(kind, text):
        got.append((kind, text))

    streaming = _Provider(can_stream=True)
    asyncio.run(scenario(streaming, on_delta))
    assert streaming.calls == ["stream"] and got == [("text", "hi")]

    whole = _Provider(can_stream=False)
    asyncio.run(scenario(whole, on_delta))
    assert whole.calls == ["whole"]

    again = _Provider(can_stream=True)
    asyncio.run(scenario(again, None))
    assert again.calls == ["whole"]  # no on_delta → the whole-response path, as before


# ---------------------------------------------------------------------------
# S2 · the chat service's streaming helper yields frames live, then the response
# ---------------------------------------------------------------------------

def test_stream_llm_call_yields_frames_then_response():
    try:
        from consumers.chatbot.service import StreamingChatService
    except Exception as e:
        pytest.skip(f"chat service not importable here: {e}")
    from consumers.chatbot.streaming import get_streaming_handler

    svc = object.__new__(StreamingChatService)
    svc.streaming_handler = get_streaming_handler()

    class _Manager:
        async def generate_response(self, messages, tools=None, on_delta=None):
            await on_delta("reasoning", "hmm")
            await on_delta("text", "Hello")
            return SimpleNamespace(content="Hello", reasoning="hmm", streamed=True)

    async def scenario():
        items = []
        async for item in StreamingChatService._stream_llm_call(svc, _Manager(), [], None):
            items.append(item)
        return items

    items = asyncio.run(scenario())
    assert items[0] == svc.streaming_handler.format_aisdk_reasoning("hmm")
    assert items[1] == svc.streaming_handler.format_aisdk_text("Hello")
    assert items[2]["_response"].content == "Hello"


def test_stream_llm_call_surfaces_provider_errors():
    try:
        from consumers.chatbot.service import StreamingChatService
    except Exception as e:
        pytest.skip(f"chat service not importable here: {e}")
    from consumers.chatbot.streaming import get_streaming_handler

    svc = object.__new__(StreamingChatService)
    svc.streaming_handler = get_streaming_handler()

    class _Boom:
        async def generate_response(self, messages, tools=None, on_delta=None):
            raise RuntimeError("provider down")

    async def scenario():
        async for _ in StreamingChatService._stream_llm_call(svc, _Boom(), [], None):
            pass

    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# S1 · stored reasoning never reaches the model or the previews
# ---------------------------------------------------------------------------

def test_prompt_converter_drops_reasoning_parts():
    try:
        from consumers.chatbot.prompt_analyzer import PromptAnalyzer
    except Exception as e:
        pytest.skip(f"prompt_analyzer not importable here: {e}")

    pa = object.__new__(PromptAnalyzer)
    out = PromptAnalyzer.convert_to_llm_messages(pa, [
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "parts": [
            {"type": "reasoning", "reasoning": "SECRET DELIBERATION"},
            {"type": "text", "text": "Hello!"},
        ]},
    ], system_prompt="sys")
    joined = json.dumps(out)
    assert "SECRET DELIBERATION" not in joined
    assert "Hello!" in joined


def test_chat_previews_ignore_reasoning_parts():
    try:
        from api.chat import _parts_text
    except Exception as e:
        pytest.skip(f"api.chat not importable here: {e}")
    assert _parts_text([{"type": "reasoning", "reasoning": "hidden"}, {"type": "text", "text": "shown"}]) == "shown"
