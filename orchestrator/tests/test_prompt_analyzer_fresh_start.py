from collections.abc import Callable

import pytest

from consumers.chatbot.prompt_analyzer import PromptAnalyzer


@pytest.mark.parametrize("text", [None, "", "Please summarize this document."])
def test_non_fresh_start_requests(text: str | None) -> None:
    """Empty and ordinary messages do not trigger a fresh start."""
    assert PromptAnalyzer().is_fresh_start_request(text) is False


@pytest.mark.parametrize(
    "phrase",
    [
        "clear memory",
        "reset memory",
        "start fresh",
        "fresh run",
        "new session",
        "ignore previous",
        "ignore earlier",
        "forget previous",
        "forget earlier",
    ],
)
@pytest.mark.parametrize(
    "change_case",
    [str.lower, str.upper, str.title],
    ids=["lowercase", "uppercase", "mixed-case"],
)
def test_fresh_start_phrases_ignore_case(
    phrase: str, change_case: Callable[[str], str]
) -> None:
    """Reset phrases match within a message regardless of case."""
    text = f"Please {change_case(phrase)} for this task."

    assert PromptAnalyzer().is_fresh_start_request(text) is True


def test_extract_latest_user_text_returns_most_recent_message() -> None:
    """The newest user message takes precedence over earlier turns."""
    messages = [
        {"role": "user", "content": "Earlier request"},
        {"role": "user", "content": "Latest request"},
    ]

    assert PromptAnalyzer().extract_latest_user_text(messages) == "Latest request"


@pytest.mark.parametrize("role", ["assistant", "system"])
def test_extract_latest_user_text_skips_other_roles(role: str) -> None:
    """Later assistant and system turns do not replace the user's text."""
    messages = [
        {"role": "user", "content": "User request"},
        {"role": role, "content": "Not a user request"},
    ]

    assert PromptAnalyzer().extract_latest_user_text(messages) == "User request"


@pytest.mark.parametrize("empty_content", ["", None])
def test_extract_latest_user_text_skips_empty_user_messages(
    empty_content: str | None,
) -> None:
    """Empty user turns do not hide the latest non-empty user text."""
    messages = [
        {"role": "user", "content": "Earlier request"},
        {"role": "user", "content": "Latest request"},
        {"role": "user", "content": empty_content},
        {"role": "assistant", "content": "Assistant reply"},
        {"role": "system", "content": "System instructions"},
    ]

    assert PromptAnalyzer().extract_latest_user_text(messages) == "Latest request"


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": ""}],
        [
            {"role": "system", "content": "System instructions"},
            {"role": "assistant", "content": "Assistant reply"},
        ],
    ],
    ids=["no-messages", "empty-user-message", "no-user-messages"],
)
def test_extract_latest_user_text_when_no_user_text(messages: list[dict]) -> None:
    """Histories without user text produce an empty string."""
    assert PromptAnalyzer().extract_latest_user_text(messages) == ""


def test_extract_latest_user_text_from_text_parts() -> None:
    """Text parts are supported when selecting the latest user message."""
    messages = [
        {"role": "user", "content": "Earlier request"},
        {"role": "user", "parts": [{"type": "text", "text": "Latest request"}]},
    ]

    assert PromptAnalyzer().extract_latest_user_text(messages) == "Latest request"
