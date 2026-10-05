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

import base64
import logging
from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx
from config import config
from core.local_projects_mount import worker_workspace_for

logger = logging.getLogger(__name__)

# A binary file goes to the worker's /files/write a piece at a time: base64 grows a piece by a
# third, so each request stays under the worker's 1 MB body limit (aiohttp's default).
BINARY_PIECE_BYTES = 600 * 1024
PART_SUFFIX = ".part"

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
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
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

    def _read_url(self, path: str, endpoint: str) -> str:
        """The worker URL a READ of ``path`` goes to. F333: on a local stack a
        ``projects/…`` path is served from the one projects mount, whichever
        workspace asks (core/local_projects_mount.py); writes stay in their own."""
        return _worker_url(worker_workspace_for(self.workspace_id, path), endpoint)

    # ── File operations ────────────────────────────────────────────

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file from the workspace."""
        client = _get_client()
        url = self._read_url(path, "/files/content")
        try:
            resp = await client.get(url, params={"path": path})
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data["success"] = True
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("read_file", err)

    async def write_file(self, path: str, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Write or create a file in the workspace (text, or one piece of ``write_binary``)."""
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

    async def write_binary(self, path: str, pieces: AsyncIterator[bytes]) -> Dict[str, Any]:
        """Write a binary file (a picture, a video) at ``path``, a piece at a time.

        Every piece goes to ``<path>.part``; the last one moves it to ``path``, so the
        workspace never shows a file half-written. ``pieces`` may come in any sizes: they
        are cut to ``BINARY_PIECE_BYTES``. The first failure stops the write and is returned.
        """
        part, pending, appended = path + PART_SUFFIX, b"", False
        async for piece in pieces:
            pending += piece
            while len(pending) > BINARY_PIECE_BYTES:
                result = await self._write_piece(part, pending[:BINARY_PIECE_BYTES], appended)
                if not result.get("success"):
                    return result
                pending, appended = pending[BINARY_PIECE_BYTES:], True
        return await self._write_piece(part, pending, appended, rename_to=path)

    async def _write_piece(self, part: str, data: bytes, append: bool, rename_to: Optional[str] = None) -> Dict[str, Any]:
        piece: Dict[str, Any] = {"base64": base64.b64encode(data).decode("ascii"), "append": append}
        if rename_to:
            piece["rename_to"] = rename_to
        return await self.write_file(part, piece)

    async def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        client = _get_client()
        url = self._read_url(path, "/files")
        try:
            resp = await client.get(url, params={"path": path})
            if resp.status_code == 404:
                return {"path": path, "entries": [], "truncated": False}
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("list_dir", err)

    async def download_file(self, path: str) -> Dict[str, Any]:
        """Download a raw binary file from the workspace. Returns bytes content."""
        client = _get_client()
        url = self._read_url(path, "/files/download")
        try:
            resp = await client.get(url, params={"path": path})
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return {
                "success": True,
                "content": resp.content,
                "filename": path.split("/")[-1],
                "size": len(resp.content),
                "content_type": resp.headers.get("content-type", "application/octet-stream"),
            }
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("download_file", err)

    def download_file_sync(self, path: str, max_bytes: int, timeout_s: float = 30.0) -> Optional[bytes]:
        """``download_file`` for a thread with no event loop (F353's thumbnail thread).

        Its own short-lived client: the shared async one belongs to the server's
        loop. None, logged, when the worker is unreachable, the file is missing or
        it is over ``max_bytes``.
        """
        headers = {"X-Internal-Token": config.WORKER_INTERNAL_TOKEN} if config.WORKER_INTERNAL_TOKEN else {}
        url = self._read_url(path, "/files/download")
        try:
            with httpx.Client(timeout=timeout_s, headers=headers) as client:
                resp = client.get(url, params={"path": path})
        except httpx.HTTPError as err:
            logger.warning("WorkspaceClient download_file_sync %s failed: %s", path, err)
            return None
        if resp.status_code != 200:
            logger.info("WorkspaceClient download_file_sync %s: %s", path, _parse_error(resp))
            return None
        return resp.content if len(resp.content) <= max_bytes else None

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
        body: Dict[str, Any] = {"command": command, "timeout": min(timeout, 600)}
        if cwd:
            body["cwd"] = cwd
        try:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("exec_command", err)

    # ── Headless rendering ────────────────────────────────────────

    async def html_to_png(
        self,
        url: str,
        viewport_w: int,
        viewport_h: int,
        output_path: str,
        wait_for: str = "[data-render-ready='true']",
        full_page: bool = False,
    ) -> Dict[str, Any]:
        """Render an HTML page to a PNG in the workspace via headless Chromium.

        Args mirror ``WorkspaceToolExecutor.html_to_png``. ``output_path`` is
        workspace-relative (e.g. ``deliverables/social/2026-04-29/foo.png``).
        Returns the worker's response dict; success path includes
        ``file_path``, ``file_size_bytes``, ``w``, ``h``, ``ms``.
        """
        client = _get_client()
        url_endpoint = _worker_url(self.workspace_id, "/html-to-png")
        body: Dict[str, Any] = {
            "url": url,
            "viewport": {"w": int(viewport_w), "h": int(viewport_h)},
            "output_path": output_path,
            "wait_for": wait_for,
            "full_page": bool(full_page),
        }
        try:
            resp = await client.post(url_endpoint, json=body)
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": _parse_error(resp),
                    "status_code": resp.status_code,
                }
            data = resp.json()
            data.setdefault("success", True)
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("html_to_png", err)

    # ── Canvas SDK session (PRD-170 S1) ───────────────────────────

    async def canvas_session_start(self) -> Dict[str, Any]:
        """Start (or resume) the workspace's headless SDK canvas session."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/canvas/session")
        try:
            resp = await client.post(url)
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data.setdefault("success", True)
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("canvas_session_start", err)

    async def canvas_session_status(self) -> Dict[str, Any]:
        """Get the workspace's canvas session status (live or volume state)."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/canvas/session")
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data.setdefault("success", True)
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("canvas_session_status", err)

    async def canvas_session_stop(self) -> Dict[str, Any]:
        """Stop the workspace's canvas session."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/canvas/session")
        try:
            resp = await client.delete(url)
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data.setdefault("success", True)
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("canvas_session_stop", err)

    async def canvas_session_decide(self, request_id: str, approved: bool) -> Dict[str, Any]:
        """Resolve a pending canvas approval (S4): approve applies, deny informs."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/canvas/session/decision")
        try:
            resp = await client.post(url, json={"request_id": request_id, "approved": approved})
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data.setdefault("success", True)
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("canvas_session_decide", err)

    async def canvas_session_auto_accept(self, enabled: bool) -> Dict[str, Any]:
        """Toggle session-scoped auto-accept for file edits (S4)."""
        client = _get_client()
        url = _worker_url(self.workspace_id, "/canvas/session/auto-accept")
        try:
            resp = await client.post(url, json={"enabled": enabled})
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data.setdefault("success", True)
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("canvas_session_auto_accept", err)

    async def canvas_session_send(self, prompt: str) -> Dict[str, Any]:
        """Send a user prompt to the workspace's live canvas session (PRD-203 C·S7).

        The turn ingress that makes the canvas usable: the worker calls
        ``client.query(prompt)`` and streams the resulting turns back over the
        canvas event stream.
        """
        client = _get_client()
        url = _worker_url(self.workspace_id, "/canvas/session/message")
        try:
            resp = await client.post(url, json={"prompt": prompt})
            if resp.status_code != 200:
                return {"success": False, "error": _parse_error(resp), "status_code": resp.status_code}
            data = resp.json()
            data.setdefault("success", True)
            return data
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            return _connection_error("canvas_session_send", err)

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
