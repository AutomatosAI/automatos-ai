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

F314 (night 9): sent as the user's turn, a re-prompt read as the owner speaking.
"Your previous reply says something was under way, but no tool call in this turn did
that … Never report an action as done without a tool result" came back as "You are
absolutely right to call me out on that" when the owner had said nothing (Quay chat
ddd6799f, the timer chat b8d9121f, 02159865, 046788d5, f5142b57, a471c235), with an
apology to nobody before the real answer. In f5142b57 the owner had said "Just the
draft here, please"; told to "make the call now", Auto made a social post nobody
asked for. Now every re-prompt the platform sends in the user's turn says it is the
platform's check, not the owner, and asks for a reply that stands on its own, with no
apology and nothing done that the owner did not ask for (``as_the_platforms_check``).
That covers the trailing system messages turned into a user turn here, and a nudge
the tool loop already writes in the user's turn (FIXER's ``nudges.Nudge``, a dict
subclass recognised by its class name so this module stays stdlib-only).
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional

Messages = List[Dict[str, Any]]

SYSTEM, ASSISTANT, USER = "system", "assistant", "user"
# F314: what frames a re-prompt sent in the user's turn.
REPROMPT_FRAME = ("(An automatic check by the platform, not a message from the owner: the owner has said "
                  "nothing since your last reply.)")
REPROMPT_RULES = ("Your next reply replaces your last one, so write it in full, as it should stand. Don't "
                  "apologise, don't thank anyone for a correction, and don't mention this check. Do only what "
                  "the owner asked for in their own messages: if they didn't ask for something, say plainly "
                  "that it wasn't done instead of doing it now.")
# The tool loop's own user-turn nudge (FIXER, modules/tools/execution/nudges.py).
LOOP_NUDGE_CLASS = "Nudge"


def as_the_platforms_check(text: str) -> str:
    """``text``, a re-prompt, framed as the platform's check rather than the owner's words (F314)."""
    if text.startswith(REPROMPT_FRAME):
        return text
    return f"{REPROMPT_FRAME}\n\n{text}\n\n{REPROMPT_RULES}"


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
    return [*messages[:start], {"role": USER, "content": as_the_platforms_check(text)}] if text else messages


def with_the_loops_nudge_framed(messages: Optional[Messages]) -> Optional[Messages]:
    """``messages`` with a closing tool-loop nudge (already the user's turn) framed as
    the platform's check (F314); a new list when it changes, the same list otherwise."""
    if not messages or type(messages[-1]).__name__ != LOOP_NUDGE_CLASS:
        return messages
    last = messages[-1]
    framed = as_the_platforms_check(str(last.get("content") or ""))
    return [*messages[:-1], {**last, "content": framed}]


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, dict) else None


def _an_answer(message: Any) -> bool:
    """An assistant reply that called no tools: the model's own words."""
    return _role(message) == ASSISTANT and not message.get("tool_calls")


def reprompts_in_the_users_turn(generate: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap ``LLMManager.generate_response`` (see the module)."""
    @functools.wraps(generate)
    async def wrapped(self: Any, messages: Messages, *args: Any, **kwargs: Any) -> Any:
        return await generate(self, with_the_loops_nudge_framed(as_the_users_turn(messages)), *args, **kwargs)
    return wrapped


__all__ = ["as_the_platforms_check", "as_the_users_turn", "reprompts_in_the_users_turn",
           "with_the_loops_nudge_framed"]
