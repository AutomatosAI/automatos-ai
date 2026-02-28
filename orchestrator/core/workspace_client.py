"""
Workspace Client — Async proxy to the workspace worker HTTP API
=================================================================

Reusable httpx client for orchestrator code that needs to talk to
the workspace worker (file I/O, command execution, grep, git).

Used by:
  - unified_executor.py (agent workspace_* tools)
  - workspace_files.py  (frontend file browser API)

All methods return the worker's JSON response dict on success,
or {"success": False, "error": "..."} on connection/timeout errors.
"""

import logging
from typing import Any, Dict, Optional

import httpx
from config import config

logger = logging.getLogger(__name__)

# Singleton client — reused across the orchestrator process
_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Get or create the shared httpx client."""
    global _client
    if _client is None or _client.is_closed:
        headers = {}
        if config.WORKER_INTERNAL_TOKEN:
            headers["X-Internal-Token"] = config.WORKER_INTERNAL_TOKEN
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=130.0, write=30.0, pool=10.0),
            headers=headers,
        )
    return _client


def _worker_url(workspace_id: str, path: str) -> str:
    """Build full worker URL for a workspace endpoint."""
    return f"{config.WORKER_INTERNAL_URL}/workspaces/{workspace_id}{path}"


def _connection_error(action: str, err: Exception) -> Dict[str, Any]:
    """Standard error response for connection/timeout failures."""
    logger.error("WorkspaceClient %s failed: %s", action, err)
    return {
        "success": False,
        "error": f"Workspace worker unreachable ({action}): {err}",
    }


class WorkspaceClient:
    """Proxy client for workspace worker operations.

    Stateless — all state lives in the singleton httpx client.
    Instantiate with a workspace_id to scope all calls.
    """

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id

    # ── File operations ────────────────────────────────────────────

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file from the workspace."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/files/content")
        try:
            resp = await client.get(url, params={"path": path})
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("read_file", err)

    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write or create a file in the workspace."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/files/write")
        try:
            resp = await client.post(url, json={"path": path, "content": content})
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data["success"] = True
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("write_file", err)

    async def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/files")
        try:
            resp = await client.get(url, params={"path": path})
            if resp.status_code == 404:
                return {"path": path, "entries": [], "truncated": False}
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("list_dir", err)

    # ── Search ─────────────────────────────────────────────────────

    async def grep(
        self,
        pattern: str,
        path: str = ".",
        include: str = "",
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """Search for a regex pattern across workspace files."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/files/grep")
        params: Dict[str, Any] = {"pattern": pattern, "path": path, "max_results": max_results}
        if include:
            params["include"] = include
        try:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("grep", err)

    # ── Command execution ──────────────────────────────────────────

    async def exec_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """Run a sandboxed shell command in the workspace."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/exec")
        body: Dict[str, Any] = {"command": command, "timeout": min(timeout, 300)}
        if cwd:
            body["cwd"] = cwd
        try:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("exec_command", err)

    # ── Git ────────────────────────────────────────────────────────

    async def git(
        self,
        operation: str,
        cwd: Optional[str] = None,
        args: str = "",
    ) -> Dict[str, Any]:
        """Execute a git operation (status, diff, add, commit, push, etc.)."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/git")
        body: Dict[str, Any] = {"operation": operation}
        if cwd:
            body["cwd"] = cwd
        if args:
            body["args"] = args
        try:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("git", err)


def _parse_error(resp: httpx.Response) -> str:
    """Extract error message from worker response."""
    try:
        data = resp.json()
        return data.get("error", "Worker error")
    except (ValueError, KeyError):
        return resp.text or "Worker error"
