"""A re-prompt after the model's own reply is sent as the user's turn (F295, night 8).

F295 (FIXER's finding): a re-prompt sent as a system message right after the model's
own reply never reaches it as a turn. OpenRouter folds system messages into
Anthropic's system prompt, so the conversation the model sees ends on its own reply,
and Sonnet answers with 3 tokens; the empty retry then replaces a real answer.
FIXER moved the agent tool loop's nudges into the user's turn. Auto's chat has two
re-prompts of its own built the same way: F187's "that id doesn't exist" and F205's
"say it in the owner's words" (consumers/chatbot/service.py, ``_ground_ids`` and
``_owner_words``), each an assistant answer followed by a system message.

``reprompts_in_the_users_turn`` wraps ``LLMManager.generate_response``: when a
conversation ends on system messages that come straight after an assistant reply
with no tool calls, they are sent as one user turn, so the model answers them on
every route. The caller's list is never changed (the chat keeps reading its own),
and any other conversation goes as it came: system messages after tool results, or
before the reply, stay where they are. Stdlib only.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional

Messages = List[Dict[str, Any]]

SYSTEM, ASSISTANT, USER = "system", "assistant", "user"


def as_the_users_turn(messages: Optional[Messages]) -> Optional[Messages]:
    """``messages`` with a trailing run of system messages after an answer sent as one
    user turn; a new list when it changes, the same list otherwise."""
    if not messages:
        return messages
    start = len(messages)
    while start > 0 and _role(messages[start - 1]) == SYSTEM:
        start -= 1
    if start == len(messages) or start == 0 or not _an_answer(messages[start - 1]):
        return messages
    text = "\n\n".join(str(m.get("content") or "") for m in messages[start:]).strip()
    return [*messages[:start], {"role": USER, "content": text}] if text else messages


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, dict) else None


def _an_answer(message: Any) -> bool:
    """An assistant reply that called no tools: the model's own words."""
    return _role(message) == ASSISTANT and not message.get("tool_calls")


def reprompts_in_the_users_turn(generate: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap ``LLMManager.generate_response`` (see the module)."""
    @functools.wraps(generate)
    async def wrapped(self: Any, messages: Messages, *args: Any, **kwargs: Any) -> Any:
        return await generate(self, as_the_users_turn(messages), *args, **kwargs)
    return wrapped


__all__ = ["as_the_users_turn", "reprompts_in_the_users_turn"]
