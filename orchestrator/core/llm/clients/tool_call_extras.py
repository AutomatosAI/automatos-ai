"""Each tool call's provider extras go back with the call (PRD-262 US-024).

Gemini, on its OpenAI-compatible endpoint, returns a tool call with a thought
signature in ``extra_content`` (``{"google": {"thought_signature": "…"}}``). Gemini 3
refuses the next request of the turn with a 400 when the assistant message that
repeats the call lacks it. The tool loops rebuild that message from the call's id,
type and function alone (``modules/tools/execution/tool_loop.py``, the heartbeat),
so the signature never reached the next request.

``ExtrasKeepingClient`` wraps the adapter's OpenAI SDK client. It keeps the extras
the provider sent with each call, by call id, and puts them back on an outgoing
assistant tool call with that id that carries none. A provider that sends no extras
gets its requests unchanged, and what one client kept never reaches another
provider: each adapter has its own client.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional

EXTRA_KEY = "extra_content"
# Calls remembered per client, oldest dropped first. A turn makes a few dozen at most.
MAX_KEPT_CALLS = 512

Message = Dict[str, Any]


def field(obj: Any, key: str) -> Any:
    """A field of a dict or an SDK object (which keeps an untyped field in ``model_extra``)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    value = getattr(obj, key, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(key)
    return value


class ToolCallExtras:
    """The extras each tool call arrived with, by call id. Thread-safe: the SDK calls
    run on worker threads (``run_blocking``)."""

    def __init__(self) -> None:
        self._by_id: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def keep(self, call_id: Optional[str], extras: Any) -> None:
        """Remember ``extras`` (a JSON object) for ``call_id``; nothing when either is missing."""
        if not isinstance(call_id, str) or not call_id or not isinstance(extras, dict) or not extras:
            return
        with self._lock:
            self._by_id[call_id] = extras
            self._by_id.move_to_end(call_id)
            while len(self._by_id) > MAX_KEPT_CALLS:
                self._by_id.popitem(last=False)

    def keep_calls(self, calls: Any) -> None:
        """Remember the extras of every call in a response message's ``tool_calls``."""
        for call in calls or []:
            self.keep(field(call, "id"), field(call, EXTRA_KEY))

    def kept(self, call_id: Any) -> Any:
        with self._lock:
            return self._by_id.get(call_id) if isinstance(call_id, str) else None

    def restored(self, messages: List[Message]) -> List[Message]:
        """``messages`` with the kept extras back on their assistant tool calls.

        A new list holding new dicts where something was added; the caller's
        messages are never changed. Unchanged when nothing is kept.
        """
        with self._lock:
            if not self._by_id:
                return messages
        return [self._restored_message(message) for message in messages]

    def _restored_message(self, message: Any) -> Any:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return message
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return message
        restored = [self._restored_call(call) for call in calls]
        if all(new is old for new, old in zip(restored, calls)):
            return message
        return {**message, "tool_calls": restored}

    def _restored_call(self, call: Any) -> Any:
        if not isinstance(call, dict) or call.get(EXTRA_KEY):
            return call
        extras = self.kept(call.get("id"))
        return {**call, EXTRA_KEY: extras} if extras else call


class _KeepingStream:
    """A streamed call's chunks, passed on unchanged as they arrive. The extras on a
    tool-call fragment are kept under its id, or under the id an earlier fragment
    with the same index brought."""

    def __init__(self, stream: Any, extras: ToolCallExtras) -> None:
        self._stream = stream
        self._extras = extras

    def __iter__(self) -> Iterator[Any]:
        ids: Dict[Any, str] = {}
        for chunk in self._stream:
            self._note(chunk, ids)
            yield chunk

    def _note(self, chunk: Any, ids: Dict[Any, str]) -> None:
        for choice in field(chunk, "choices") or []:
            for call in field(field(choice, "delta"), "tool_calls") or []:
                index = field(call, "index")
                call_id = field(call, "id") or ids.get(index)
                if call_id:
                    ids[index] = call_id
                self._extras.keep(call_id, field(call, EXTRA_KEY))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _Completions:
    """``chat.completions`` that keeps and restores the extras; anything else is the SDK's own."""

    def __init__(self, completions: Any, extras: ToolCallExtras) -> None:
        self._completions = completions
        self._extras = extras

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            kwargs = {**kwargs, "messages": self._extras.restored(messages)}
        result = self._completions.create(**kwargs)
        if kwargs.get("stream"):
            return _KeepingStream(result, self._extras)
        for choice in field(result, "choices") or []:
            self._extras.keep_calls(field(field(choice, "message"), "tool_calls"))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _Chat:
    def __init__(self, chat: Any, extras: ToolCallExtras) -> None:
        self._chat = chat
        self.completions = _Completions(chat.completions, extras)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class ExtrasKeepingClient:
    """An OpenAI SDK client whose chat completions send each tool call's extras back
    with the call. Everything else passes through to the client."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.extras = ToolCallExtras()
        self.chat = _Chat(client.chat, self.extras)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = ["EXTRA_KEY", "ExtrasKeepingClient", "ToolCallExtras"]
