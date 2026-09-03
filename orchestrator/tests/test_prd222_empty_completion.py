"""PRD-222 — an empty completion never streams as a blank turn (live-test 2026-09-02)."""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

from consumers.chatbot.empty_completion import (
    EMPTY_COMPLETION_FALLBACK,
    is_empty_completion,
    with_fallback_content,
)
from core.llm.clients.base import LLMResponse


def test_empty_when_neither_text_nor_tool_calls():
    assert is_empty_completion(LLMResponse(content=""))
    assert is_empty_completion(LLMResponse(content="   \n"))
    assert is_empty_completion(LLMResponse(content=None))
    assert is_empty_completion(None)


def test_not_empty_with_text_or_a_tool_call():
    assert not is_empty_completion(LLMResponse(content="hello"))
    assert not is_empty_completion(LLMResponse(content="", tool_calls=[{"id": "t1"}]))


def test_fallback_is_a_copy_carrying_the_sentence():
    r = LLMResponse(content="", finish_reason="stop", model="google/gemini-2.5-flash")
    out = with_fallback_content(r)
    assert out is not r and r.content == ""            # never mutated in place
    assert out.content == EMPTY_COMPLETION_FALLBACK
    assert out.model == "google/gemini-2.5-flash"      # everything else preserved
    assert not is_empty_completion(out)


def test_fallback_handles_none_and_plain_objects():
    assert with_fallback_content(None).content == EMPTY_COMPLETION_FALLBACK
    r = SimpleNamespace(content="", tool_calls=None)
    out = with_fallback_content(r)
    assert out.content == EMPTY_COMPLETION_FALLBACK and r.content == ""


def test_chat_service_retries_once_then_falls_back():
    src = (pathlib.Path(__file__).resolve().parents[1] / "consumers/chatbot/service.py").read_text()
    assert "if is_empty_completion(response):" in src
    assert "retrying once" in src
    assert "response = with_fallback_content(response)" in src
