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
before the reply, stay where they are.

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
subclass recognised by its class name).

F295 (night 9b): the frame wasn't enough. After a claim check Auto still answered "You
are absolutely right to call me out on that, Gerard. My apologies…" (6586c8bf,
8578eeaf). The chat's re-prompts now open with FIXER's own line (``nudges.PLATFORM_CHECK``,
through ``as_a_check``), the one wording every re-prompt and nudge shares, and the reply
to a re-prompt loses the apology it opens with: on the text that streams
(``reprompt_reply.ApologyGate``) and on the reply that is saved (``without_the_apology``).
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
    """``text``, a re-prompt, framed as the platform's check rather than the owner's words
    (F314), with FIXER's line (F295, night 9b). A text already framed is left as it is."""
    from modules.tools.execution.nudges import PLATFORM_CHECK, as_a_check

    if text.startswith((PLATFORM_CHECK, REPROMPT_FRAME)):
        return text
    return as_a_check(f"{text}\n\n{REPROMPT_RULES}")


def is_a_reprompt(messages: Optional[Messages]) -> bool:
    """Whether the conversation ends on the platform's check: the model's next reply answers it."""
    from modules.tools.execution.nudges import PLATFORM_CHECK

    last = messages[-1] if messages else None
    return _role(last) == USER and str(last.get("content") or "").startswith((PLATFORM_CHECK, REPROMPT_FRAME))


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
        sent = with_the_loops_nudge_framed(as_the_users_turn(messages))
        if not is_a_reprompt(sent):
            return await generate(self, sent, *args, **kwargs)
        return await _answered_without_an_apology(generate, self, sent, *args, **kwargs)
    return wrapped


async def _answered_without_an_apology(generate: Callable[..., Awaitable[Any]], manager: Any, sent: Messages,
                                       *args: Any, **kwargs: Any) -> Any:
    """The model's reply to a re-prompt, with the apology it opens with taken off what
    streams and what is saved (F295, night 9b)."""
    from core.llm.reprompt_reply import gated
    from modules.tools.execution.nudges import without_the_apology

    gate = gated(kwargs.get("on_delta"))
    if gate is not None:
        kwargs = {**kwargs, "on_delta": gate}
    response = await generate(manager, sent, *args, **kwargs)
    if gate is not None:
        await gate.close()
    return without_the_apology(response)


__all__ = ["as_the_platforms_check", "as_the_users_turn", "is_a_reprompt", "reprompts_in_the_users_turn",
           "with_the_loops_nudge_framed"]
