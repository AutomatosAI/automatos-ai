"""The backend contract, from the host's side (``api/cli_hosts.py`` on the server).

Plain ``urllib`` — no dependencies. One host token in one header; every error
carries the server's detail so the operator sees the real reason.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

HOST_TOKEN_HEADER = "X-CLI-Host-Token"
DEFAULT_TIMEOUT = 20.0


class BackendError(RuntimeError):
    def __init__(self, status: int, detail: Any, url: str):
        self.status = status
        self.detail = detail
        self.url = url
        super().__init__(f"{status} from {url}: {detail}")


class BackendClient:
    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ── plumbing ────────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                 *, auth: bool = True) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if auth and self.token:
            req.add_header(HOST_TOKEN_HEADER, self.token)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
            try:
                detail = json.loads(raw).get("detail", raw) if raw else exc.reason
            except ValueError:
                detail = raw or exc.reason
            raise BackendError(exc.code, detail, url) from None
        except urllib.error.URLError as exc:
            raise BackendError(0, f"cannot reach the backend: {exc.reason}", url) from None
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise BackendError(500, f"non-JSON response: {raw[:200]}", url) from None
        return parsed if isinstance(parsed, dict) else {"data": parsed}

    # ── endpoints ───────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health", auth=False)

    def pair(self, code: str, name: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/v1/cli-hosts/pair",
                             {"code": code, "name": name, "capabilities": capabilities}, auth=False)

    def heartbeat(self, host_id: str, capabilities: Dict[str, Any], running: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/cli-hosts/{host_id}/heartbeat",
                             {"capabilities": capabilities, "running": running})

    def claim(self, host_id: str, limit: int) -> List[Dict[str, Any]]:
        out = self._request("POST", f"/api/v1/cli-hosts/{host_id}/claim", {"limit": max(1, limit)})
        tasks = out.get("tasks") or []
        return [t for t in tasks if isinstance(t, dict)]

    def events(self, host_id: str, task_id: int, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/cli-hosts/{host_id}/tasks/{task_id}/events",
                             {"events": events})

    def result(self, host_id: str, task_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/cli-hosts/{host_id}/tasks/{task_id}/result", payload)
