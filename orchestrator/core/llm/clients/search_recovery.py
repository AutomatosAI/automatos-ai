"""A model call that failed only because the provider's web search did (F264, night 7b).

PRD-240 puts the provider's own web search (OpenRouter's ``openrouter:web_search``
server tool) on every call while web access is on, and the model decides when to
search. On night 7b most of Auto's replies ended "Auto could not finish this reply:
the AI provider's web search failed on its side", replies that had finished. The
search had failed on a later call in the turn (F205's re-prompt that takes the
platform's tool names out of a reply, F187's re-prompt about an id), and the
provider's 502 failed the whole call, so the turn was reported as failed and the
re-prompt's fix never landed.

A call refused that way is sent once more without the provider's search: the model
answers without searching, rather than not at all. Any other failure, and a call
that carried no such tool, is the caller's as before. The retry is logged.

F264 (night 8): "Auto could not finish this reply: Server tool "openrouter:web_search"
failed: upstream returned an invalid response" (02:34) came after the retry above.
Auto's chat streams (PRD-238), and on a streamed call the provider's failure arrives
inside the stream, while it is read, not when the call is made, so the retry never
saw it. A streamed call that carries the search is now read through
``_RecoveringStream``: a failure of the search before anything was said sends the
call again without it, and one after the reply had started ends the reply where it
stopped (sending it again would say it twice), never failing the turn.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterator, List

import openai

logger = logging.getLogger(__name__)

# The provider's own words: 'Server tool "openrouter:web_search" failed: upstream returned an invalid response'.
SERVER_TOOL_FAILED = re.compile(r'Server tool \\?"[\w.-]+:[\w.-]+\\?" failed')
SERVER_TOOL_PREFIX = "openrouter:"


def _server_tool(tool: Any) -> bool:
    """A provider-executed tool (PRD-240), not one of the platform's functions."""
    return isinstance(tool, dict) and str(tool.get("type", "")).startswith(SERVER_TOOL_PREFIX)


def without_server_tools(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The call's arguments without the provider's own tools (a new dict)."""
    kept: List[Any] = [t for t in kwargs.get("tools") or [] if not _server_tool(t)]
    if kept:
        return {**kwargs, "tools": kept}
    return {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}


def _search_failed(exc: Exception) -> bool:
    return isinstance(exc, openai.APIError) and bool(SERVER_TOOL_FAILED.search(str(exc)))


def _says_something(chunk: Any) -> bool:
    """A streamed chunk that carries text, reasoning or a tool call's fragment."""
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        if delta is not None and any(getattr(delta, key, None) for key in
                                     ("content", "reasoning", "reasoning_content", "tool_calls")):
            return True
    return False


class _RecoveringStream:
    """A streamed call's chunks. When the provider's search fails inside the stream:
    before anything was said, the chunks of the call sent again without it; after,
    the end of the reply as it stood."""

    def __init__(self, stream: Any, again: Callable[[], Any]) -> None:
        self._stream = stream
        self._again = again

    def __iter__(self) -> Iterator[Any]:
        said = False
        try:
            for chunk in self._stream:
                said = said or _says_something(chunk)
                yield chunk
        except openai.APIError as exc:
            if not _search_failed(exc):
                raise
            if said:
                logger.warning("[F264] the provider's web search failed mid-reply (%s); ending the reply there", exc)
                return
            logger.warning("[F264] the provider's web search failed in the stream (%s); sending the call again "
                           "without it", exc)
            yield from self._again()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _Completions:
    """``chat.completions`` with the one retry; anything else is the SDK's own."""

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def create(self, **kwargs: Any) -> Any:
        searches = any(_server_tool(t) for t in kwargs.get("tools") or [])
        try:
            result = self._completions.create(**kwargs)
        except openai.APIError as exc:
            if not _search_failed(exc) or not searches:
                raise
            logger.warning("[F264] the provider's web search failed (%s); sending the call again without it", exc)
            return self._completions.create(**without_server_tools(kwargs))
        if kwargs.get("stream") and searches:
            return _RecoveringStream(result, lambda: self._completions.create(**without_server_tools(kwargs)))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _Chat:
    def __init__(self, chat: Any) -> None:
        self._chat = chat
        self.completions = _Completions(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class SearchRecoveringClient:
    """An OpenAI SDK client whose chat completions are sent once more without the
    provider's web search when only that search failed. Everything else passes
    through to the client."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _Chat(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = ["SERVER_TOOL_FAILED", "SearchRecoveringClient", "without_server_tools"]
