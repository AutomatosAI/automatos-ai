"""PRD-008-A.2 — chat-intent → callback-form auto-trigger.

Unit tests for the intent matcher. End-to-end behaviour (SSE event emission
+ message augmentation) is covered by the widget_chat integration suite once
the orchestrator boots.
"""
from __future__ import annotations

import pytest

from api.widgets.chat import (
    _DEFAULT_CALLBACK_INTENT_PHRASES,
    _matches_callback_intent,
)


class TestMatchesCallbackIntent:
    """The matcher MUST be case-insensitive substring match. False positives
    are worse than false negatives — keep phrases specific."""

    @pytest.mark.parametrize(
        "message",
        [
            "Can someone call me back?",
            "give me a call back please",
            "I'd like to SPEAK TO SOMEONE",
            "Talk to a human - it's urgent",
            "phone me when you get a chance",
            "give me a call when free",
            "can someone call later",
        ],
    )
    def test_matches_default_phrases(self, message: str) -> None:
        assert _matches_callback_intent(message, _DEFAULT_CALLBACK_INTENT_PHRASES)

    @pytest.mark.parametrize(
        "message",
        [
            "Tell me about this product",
            "Do you ship to the UK?",
            "What's the price?",
            "I'd like to call this product 'amazing'",  # contains "call" but not intent
            "",
            "   ",
        ],
    )
    def test_does_not_match_unrelated_messages(self, message: str) -> None:
        assert not _matches_callback_intent(message, _DEFAULT_CALLBACK_INTENT_PHRASES)

    def test_none_message_returns_false(self) -> None:
        assert _matches_callback_intent(None, _DEFAULT_CALLBACK_INTENT_PHRASES) is False

    def test_empty_phrases_returns_false(self) -> None:
        assert _matches_callback_intent("call me back", tuple()) is False

    def test_custom_phrases_override_defaults(self) -> None:
        custom = ("appelez-moi", "rappel telephonique")
        assert _matches_callback_intent("Pouvez-vous m'appelez-moi?", custom)
        # Default English phrase shouldn't match against custom-only set
        assert not _matches_callback_intent("call me back", custom)

    def test_empty_string_phrase_is_skipped(self) -> None:
        # An empty string would substring-match anything — explicitly filtered.
        phrases = ("", "call me back")
        assert _matches_callback_intent("call me back", phrases)
        assert not _matches_callback_intent("hello", phrases)
