"""#1444: the page context on a chat turn reaches the model, and is never remembered.

``inject_page_preamble`` adds the page context as an extra text part on the last user
message. ``extract_latest_user_text`` joined every text part, so the preamble went into
``latest_text``, which the turn stores (``_remember_the_turn`` → ``store_conversation``
→ distillation and the L2 transcript) and searches with. The part is now marked, and
the remembered text is the person's own words; the model still reads the preamble.
"""
from __future__ import annotations

import pytest

from consumers.chatbot.prompt_analyzer import PromptAnalyzer
from services.page_context import (
    inject_page_preamble,
    is_page_context_part,
    sanitize_page_context,
)

SAID = "Which of our open deals close this month?"
PAGES = [
    pytest.param({"page": "command-center", "tab": "board"}, "[Page context]", id="manifest page"),
    pytest.param({"page": "Some Legacy Screen"}, "[Context: the user is currently on", id="legacy label"),
]


def _turn(raw_page: dict) -> list:
    history = [
        {"role": "user", "parts": [{"type": "text", "text": "earlier question"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "earlier answer"}]},
        {"role": "user", "parts": [{"type": "text", "text": SAID}]},
    ]
    return inject_page_preamble(history, sanitize_page_context(raw_page))


@pytest.mark.parametrize(("raw_page", "preamble"), PAGES)
def test_the_remembered_text_is_the_persons_words_alone(raw_page, preamble):
    messages = _turn(raw_page)
    assert any(preamble in part["text"] for part in messages[-1]["parts"])  # the preamble is on the turn
    assert PromptAnalyzer().extract_latest_user_text(messages) == SAID


@pytest.mark.parametrize(("raw_page", "preamble"), PAGES)
def test_the_model_still_reads_the_page_context(raw_page, preamble):
    """Both the chat paths' conversion and the context build's conversation section keep it."""
    from modules.context.sections.conversation import _parts_to_text

    messages = _turn(raw_page)
    converted = PromptAnalyzer().convert_to_llm_messages(messages, system_prompt="")[-1]["content"]
    rendered = _parts_to_text(messages[-1]["parts"])
    for text in (converted, rendered):
        assert SAID in text and preamble in text


def test_only_the_preamble_part_is_marked_and_the_history_is_untouched():
    history = [{"role": "user", "parts": [{"type": "text", "text": SAID}]}]
    messages = inject_page_preamble(history, sanitize_page_context({"page": "missions"}))
    said, hint = messages[0]["parts"]
    assert not is_page_context_part(said) and is_page_context_part(hint)
    PromptAnalyzer().extract_latest_user_text(messages)
    assert len(messages[0]["parts"]) == 2 and history[0]["parts"] == [{"type": "text", "text": SAID}]


def test_a_turn_with_no_page_context_is_unchanged():
    messages = [{"role": "user", "parts": [{"type": "text", "text": SAID}]}]
    assert PromptAnalyzer().extract_latest_user_text(messages) == SAID
    assert PromptAnalyzer().extract_latest_user_text([{"role": "user", "content": SAID}]) == SAID
