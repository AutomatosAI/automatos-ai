"""HTTP client for the simulation — standard library only, every call recorded.

Scoping is ``X-Workspace-ID`` on every call. On the local edition no key is
sent: the request resolves to the anonymous operator (super admin) and the
header alone decides the workspace — see ``config`` for why the minted
per-workspace key cannot be used here. ``X-Api-Key`` is added only when the
operator configured the stack's static key.

The trace keeps the full request and the full response of every call (PRD-247
D10: capture full payloads) so a finding can point at the exact exchange rather
than a summary of it. Headers are never traced.
"""

from __future__ import annotations

import json
import socket
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

USER_AGENT = "automatos-sim/0.1"
RETRY_STATUSES = (502, 503, 504)
RETRY_DELAYS_S = (2.0, 5.0, 10.0)
TRACE_BODY_CAP = 200_000  # characters kept per body in the trace


class ApiError(Exception):
    """A request the platform refused (4xx/5xx) or one that never connected (status 0)."""

    def __init__(self, method: str, path: str, status: int, body: str):
        self.method, self.path, self.status, self.body = method, path, status, body
        super().__init__(f"{method} {path} -> {status}: {body[:300]}")


@dataclass(frozen=True)
class Response:
    status: int
    body: str
    ms: int
    headers: Mapping[str, str]
    first_byte_ms: int | None = None

    def json(self) -> Any:
        try:
            return json.loads(self.body) if self.body else None
        except json.JSONDecodeError:
            return None


@dataclass
class Trace:
    """Append-only record of every HTTP exchange in a run."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, entry: Mapping[str, Any]) -> None:
        self.calls.append(dict(entry))

    def __len__(self) -> int:
        return len(self.calls)

    def dump_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for call in self.calls:
                fh.write(json.dumps(call, default=str) + "\n")


def _cap(text: str) -> str:
    return text if len(text) <= TRACE_BODY_CAP else text[:TRACE_BODY_CAP] + f"...[+{len(text) - TRACE_BODY_CAP} chars]"


class Api:
    """Workspace-scoped client. ``get/post/patch/delete`` return parsed JSON or raise ApiError."""

    def __init__(self, base_url: str, api_key: str | None, workspace_id: str, trace: Trace, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.workspace_id = workspace_id
        self.trace = trace
        self.timeout_s = timeout_s

    # -- plumbing -------------------------------------------------------------
    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        base = {"X-Workspace-ID": self.workspace_id, "Accept": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            base["X-Api-Key"] = self.api_key
        return {**base, **(extra or {})}

    def _url(self, path: str, params: Mapping[str, Any] | None) -> str:
        url = self.base_url + path
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        return url + ("?" + urllib.parse.urlencode(clean) if clean else "")

    def _record(self, method: str, path: str, params: Mapping[str, Any] | None, body: Any,
                response: Response | None, error: str | None, label: str | None, attempt: int) -> None:
        self.trace.record({
            "ts": time.time(), "label": label, "method": method, "path": path,
            "params": dict(params or {}), "request": body, "attempt": attempt,
            "status": response.status if response else 0,
            "ms": response.ms if response else None,
            "first_byte_ms": response.first_byte_ms if response else None,
            "response": _cap(response.body) if response else None,
            "error": error,
        })

    def _once(self, method: str, url: str, data: bytes | None, headers: Mapping[str, str],
              timeout_s: float) -> Response:
        req = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                first = resp.read(1)
                first_byte_ms = int((time.monotonic() - started) * 1000)
                body = (first + resp.read()).decode("utf-8", "replace")
                return Response(resp.status, body, int((time.monotonic() - started) * 1000),
                                dict(resp.headers.items()), first_byte_ms)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            return Response(exc.code, body, int((time.monotonic() - started) * 1000), dict(exc.headers.items()))

    def request(self, method: str, path: str, *, json_body: Any = None, params: Mapping[str, Any] | None = None,
                timeout_s: float | None = None, accept: str | None = None, label: str | None = None) -> Response:
        """One logical call; 502/503/504 and connection errors retry with backoff. Never raises on 4xx."""
        url = self._url(path, params)
        data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        headers = self._headers({"Content-Type": "application/json"} if data else None)
        if accept:
            headers["Accept"] = accept
        timeout = timeout_s or self.timeout_s
        last_error = ""
        last_response: Response | None = None
        for attempt, delay in enumerate((0.0, *RETRY_DELAYS_S), start=1):
            if delay:
                time.sleep(delay)
            try:
                response = self._once(method, url, data, headers, timeout)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._record(method, path, params, json_body, None, last_error, label, attempt)
                continue
            self._record(method, path, params, json_body, response, None, label, attempt)
            if response.status in RETRY_STATUSES:
                last_response = response
                continue
            return response
        if last_response is not None:  # every attempt answered 5xx: hand the last one back, status intact
            return last_response
        raise ApiError(method, path, 0, last_error or "gave up after retries")

    # -- JSON helpers -----------------------------------------------------------
    def call(self, method: str, path: str, **kw: Any) -> Any:
        response = self.request(method, path, **kw)
        if response.status >= 400:
            raise ApiError(method, path, response.status, response.body)
        return response.json()

    def get(self, path: str, **kw: Any) -> Any:
        return self.call("GET", path, **kw)

    def post(self, path: str, json_body: Any = None, **kw: Any) -> Any:
        return self.call("POST", path, json_body=json_body, **kw)

    def put(self, path: str, json_body: Any = None, **kw: Any) -> Any:
        return self.call("PUT", path, json_body=json_body, **kw)

    def patch(self, path: str, json_body: Any = None, **kw: Any) -> Any:
        return self.call("PATCH", path, json_body=json_body, **kw)

    def delete(self, path: str, **kw: Any) -> Any:
        return self.call("DELETE", path, **kw)

    # -- chat ---------------------------------------------------------------------
    def stream_chat(self, text: str, *, chat_id: str | None = None, agent_id: int | None = None,
                    timeout_s: float = 240.0, label: str | None = None) -> tuple[Response, str]:
        """POST /api/chat; returns the stream and the chat id the conversation continues under."""
        body, used = chat_body(text, chat_id=chat_id, agent_id=agent_id)
        response = self.request("POST", "/api/chat", json_body=body, timeout_s=timeout_s,
                                accept="text/event-stream", label=label or "chat")
        return response, used


def chat_body(text: str, *, chat_id: str | None = None, agent_id: int | None = None) -> tuple[dict[str, Any], str]:
    """The UI's request shape. The chat id is CLIENT-supplied (``api/chat.py`` reads ``request.id``
    and never returns one), so a new conversation mints its own; pass it back to continue."""
    used = chat_id or str(uuid.uuid4())
    body: dict[str, Any] = {"id": used, "chatId": used,
                            "message": {"role": "user", "parts": [{"type": "text", "text": text}]}}
    if agent_id is not None:
        body["agentId"] = agent_id
    return body, used


def items_of(payload: Any, *keys: str) -> list[Any]:
    """The list inside a list response, whatever the route called it."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in (*keys, "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def iter_pages(api: Api, path: str, *, params: Mapping[str, Any], keys: tuple[str, ...], limit: int = 100,
               max_pages: int = 20) -> Iterator[Any]:
    """Walk an offset-paginated list route until a short page."""
    for page in range(max_pages):
        payload = api.get(path, params={**params, "limit": limit, "offset": page * limit})
        rows = items_of(payload, *keys)
        yield from rows
        if len(rows) < limit:
            return
