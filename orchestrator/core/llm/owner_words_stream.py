"""F264 (night 8): Auto's replies to the owner never carry the platform's names.

F205 (night 6) re-prompts a reply that names the platform's tools or parameters,
once, and only after a call failed. Night 8 still had them in about 60 replies
("You can find a list of social posts using `platform_list_social_posts`", "It
doesn't accept `send_back_reason`", "the `platform_update_task_status` call"), in
answers after calls that worked, in the words Auto says before its calls, and in
re-prompts that kept them. The refusals Auto reads name calls on purpose, so it can
make the right one, and it repeats them.

Now what Auto streams to the owner in a chat turn is read through ``in_owner_words``:
a name of the platform's (an action, or one of its parameters, with or without
"platform_", or an offered tool's) is said as plain words ("list social posts",
"assigned agent name"), unless the owner used it this turn; so is a call or a
parameter Auto made up, written as code ("`send_back_reason`"). The live text and the
saved answer read the same: a word is held back until it is whole, so a name split
across two deltas is still found. Code in a fenced block is left as written. Only
the owner-facing text changes: tool calls, their arguments and results are untouched.
"""
from __future__ import annotations

import dataclasses
import logging
import re
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .screen_watch import watched

logger = logging.getLogger(__name__)

PLATFORM_PREFIX = "platform_"
_SNAKE = re.compile(r"`?\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b`?")
# The end of a delta that may be the start of a name: held back until the next delta.
_OPEN_WORD = re.compile(r"[A-Za-z0-9_`]+$")
FENCE = "```"
Delta = Callable[[str, str], Awaitable[None]]


def schema_names(name: str, parameters: Any) -> Set[str]:
    """An action's names: its own, without "platform_", and its snake_case parameters."""
    names = {name, name[len(PLATFORM_PREFIX):] if name.startswith(PLATFORM_PREFIX) else name}
    properties = (parameters or {}).get("properties") if isinstance(parameters, dict) else None
    names.update(p for p in (properties or {}) if "_" in p)
    return {n for n in names if "_" in n}


@lru_cache(maxsize=1)
def platform_names() -> FrozenSet[str]:
    """Every registered platform action's names and snake_case parameters."""
    try:
        from modules.tools.discovery.action_registry import get_action_registry

        names: Set[str] = set()
        for action in get_action_registry().get_all():
            names |= schema_names(action.name, action.parameters)
        return frozenset(names)
    except Exception:  # noqa: BLE001 — without the registry, the offered tools still count
        logger.debug("[F205] action registry unreadable", exc_info=True)
        return frozenset()


def offered_names(tools: Optional[Iterable[Dict[str, Any]]]) -> Set[str]:
    """The platform's names plus the offered tools' own names and parameters."""
    names: Set[str] = set(platform_names())
    for tool in tools or ():
        function = (tool or {}).get("function") if isinstance(tool, dict) else None
        if function and function.get("name"):
            names |= schema_names(function["name"], function.get("parameters"))
    return names


def plain(name: str) -> str:
    """``platform_list_social_posts`` said as "list social posts"."""
    return name.removeprefix(PLATFORM_PREFIX).replace("_", " ")


def _swapped(text: str, names: Set[str], owners: Set[str]) -> str:
    def swap(found: "re.Match[str]") -> str:
        name, whole = found.group(1), found.group(0)
        quoted = whole.startswith("`") and whole.endswith("`") and len(whole) > len(name) + 1
        if name in owners or not (quoted or name in names):
            return whole
        return plain(name) if quoted else whole.replace(name, plain(name))
    return _SNAKE.sub(swap, text)


def plain_words(text: str, names: Set[str], owners: Set[str], in_code: bool = False) -> Tuple[str, bool]:
    """``text`` with each platform name the owner did not use said in plain words,
    outside fenced code; and whether it ends inside a fence (``in_code``: it starts in one)."""
    parts = (text or "").split(FENCE)
    out: List[str] = []
    for index, part in enumerate(parts):
        code = in_code if index % 2 == 0 else not in_code
        out.append(part if code else _swapped(part, names, owners))
    return FENCE.join(out), in_code if len(parts) % 2 == 1 else not in_code


def in_plain_words(text: str, names: Set[str], owners: Set[str]) -> str:
    """A whole reply in the owner's words (see ``plain_words``)."""
    return plain_words(text, names, owners)[0]


class OwnerWords:
    """One streamed reply's text, said in the owner's words as it arrives."""

    def __init__(self, names: Set[str], owner_text: str) -> None:
        self.names = names
        self.owners = {m.group(1) for m in _SNAKE.finditer((owner_text or "").lower())}
        self._held = ""
        self._in_code = False

    def feed(self, text: str) -> str:
        """What can be shown now: everything but a word that may not be whole yet."""
        pending = self._held + (text or "")
        open_word = _OPEN_WORD.search(pending)
        cut = open_word.start() if open_word else len(pending)
        self._held = pending[cut:]
        shown, self._in_code = plain_words(pending[:cut], self.names, self.owners, self._in_code)
        return shown

    def flush(self) -> str:
        """The last word, once the reply has ended."""
        held, self._held = self._held, ""
        shown, self._in_code = plain_words(held, self.names, self.owners, self._in_code)
        return shown


def _owner_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _autos_turn() -> bool:
    from .usage_context import LANE_CHAT, current_usage_scope

    return current_usage_scope().get("request_type") == LANE_CHAT


@watched  # FX-017: the turn's screen records the exact text each call streamed
async def in_owner_words(stream: Callable[..., Awaitable[Any]], messages: List[Dict[str, Any]],
                         tools: Optional[List[Dict[str, Any]]], on_delta: Delta) -> Any:
    """``stream`` the call, with Auto's text to the owner said in the owner's words
    (live and in the response); any other lane's stream is left as it comes."""
    if not _autos_turn():
        return await stream(messages, tools, on_delta=on_delta)
    words = OwnerWords(offered_names(tools), _owner_text(messages))

    async def said(kind: str, text: str) -> None:
        shown = words.feed(text) if kind == "text" else text
        if shown:
            await on_delta(kind, shown)

    response = await stream(messages, tools, on_delta=said)
    tail = words.flush()
    if tail:
        await on_delta("text", tail)
    content = getattr(response, "content", None)
    if not isinstance(content, str) or not dataclasses.is_dataclass(response):
        return response
    return dataclasses.replace(response, content=in_plain_words(content, words.names, words.owners))


__all__ = ["OwnerWords", "in_owner_words", "in_plain_words", "offered_names", "platform_names", "schema_names"]
