"""Empty completions — the provider returned nothing and the turn streamed blank.

Live test 2026-09-02 (prod, Gemini 2.5 Flash via OpenRouter): mid-onboarding the
LLM call logged ``input_tokens=0 output_tokens=0 status=success`` with
``content_length: 0, finish_reason: stop`` and the chat service streamed it as a
successful turn — to the user, Auto went silent. Two of ~60 turns that day.
Policy: retry once, then an honest sentence. Never a blank.
"""
from __future__ import annotations

import dataclasses
from typing import Any

EMPTY_COMPLETION_FALLBACK = (
    "I didn't get a response through that time — could you send that again?"
)

# F185 (night 6): the owner was told "encountered an issue … Please try again" after two asks. F264 (night 8):
# a blank answer in the loop is the account of the turn (turn_account); this is the rare blank after it
# (StreamingChatService, nothing_said_fallback).
NOTHING_SAID = "My reply didn't come through. Anything I did this turn is on your board: check there before asking again."


def is_empty_completion(response: Any) -> bool:
    """True when the model produced neither text nor a tool call."""
    if response is None:
        return True
    if getattr(response, "tool_calls", None):
        return False
    return not (getattr(response, "content", None) or "").strip()


def with_fallback_content(response: Any) -> Any:
    """A COPY of ``response`` carrying the fallback sentence — never mutated in place."""
    if response is None:
        from core.llm.clients.base import LLMResponse

        return LLMResponse(content=EMPTY_COMPLETION_FALLBACK, finish_reason="empty_completion")
    if dataclasses.is_dataclass(response):
        return dataclasses.replace(response, content=EMPTY_COMPLETION_FALLBACK)
    clone = type(response).__new__(type(response))
    clone.__dict__.update(getattr(response, "__dict__", {}))
    clone.content = EMPTY_COMPLETION_FALLBACK
    return clone
