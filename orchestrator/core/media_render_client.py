"""
Media Render Client — async proxy to the media-render service (PRD-251 S1.1c)
============================================================================

The orchestrator's side of D3: an httpx client for the ``media-render``
service (``services/media-render``), shaped like ``core/workspace_client.py``.
It sends ``X-Internal-Token`` (``SOCIALS_RENDER_TOKEN``, the same name the
service reads) and takes its timeouts from config: the read timeout,
``SOCIALS_RENDER_TIMEOUT_SECONDS`` (900 s), covers ``POST /render``, which
answers only once the job is staged, spoken, mixed and checked.

Used by:
  - modules/socials/render.py (a post's render lifecycle)
  - api/socials.py            (the renderer check before a render starts)

The service's API is described in ``services/media-render/README.md``. Every
failure raises :class:`MediaRenderError` with a ``code``: the renderer's own
(``check_failed`` with its findings, ``invalid_bundle``, ``busy`` with
``retry_after``, ``media_fetch_failed``), or ``not_configured``,
``unreachable`` and ``timeout`` from this side. Nothing here renders or
generates anything: the renderer assembles (D3).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

import httpx

from config import config

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
DOWNLOAD_CHUNK_BYTES = 1 << 16

# The render job's states (services/media-render/media_render/jobs.py).
JOB_DONE = "done"
JOB_FAILED = "failed"
JOB_REJECTED = "rejected"
JOB_TERMINAL = frozenset({JOB_DONE, JOB_FAILED, JOB_REJECTED})

# Codes raised from this side of the wire.
NOT_CONFIGURED = "not_configured"
UNREACHABLE = "unreachable"
TIMEOUT = "timeout"
BUSY = "busy"
NOT_FOUND = "not_found"

# Singleton client — reused across the orchestrator process
_client: Optional[httpx.AsyncClient] = None


class MediaRenderError(Exception):
    """media-render could not do what was asked; ``code`` says why."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: Optional[int] = None,
        findings: Sequence[Mapping[str, Any]] = (),
        report: Optional[Mapping[str, Any]] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.findings = tuple(findings)
        self.report = dict(report or {})
        self.retry_after = retry_after


class MediaRenderUnavailable(MediaRenderError):
    """No renderer is configured, or it cannot be reached."""


def _get_client() -> httpx.AsyncClient:
    """Get or create the shared httpx client."""
    global _client
    if _client is None or _client.is_closed:
        headers = {}
        if config.SOCIALS_RENDER_TOKEN:
            headers[TOKEN_HEADER] = config.SOCIALS_RENDER_TOKEN
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                float(config.SOCIALS_RENDER_TIMEOUT_SECONDS),
                connect=float(config.SOCIALS_RENDER_CONNECT_TIMEOUT_SECONDS),
            ),
            headers=headers,
        )
    return _client


def _base_url() -> str:
    url = (config.SOCIALS_RENDER_URL or "").strip().rstrip("/")
    if not url:
        raise MediaRenderUnavailable(NOT_CONFIGURED, "no media-render service is configured (SOCIALS_RENDER_URL)")
    return url


def _retry_after(resp: httpx.Response) -> Optional[float]:
    try:
        return float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return None


def _error_for(resp: httpx.Response, action: str) -> MediaRenderError:
    """The renderer's own error body (``{error, message, ...}``) as an exception."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = str(body.get("error") or f"http_{resp.status_code}")
    message = str(body.get("message") or f"media-render answered {resp.status_code} to {action}")
    findings = body.get("findings") if isinstance(body.get("findings"), list) else ()
    report = body.get("report") if isinstance(body.get("report"), dict) else None
    return MediaRenderError(
        code,
        message,
        status=resp.status_code,
        findings=findings,
        report=report,
        retry_after=_retry_after(resp) if resp.status_code == 503 else None,
    )


def _transport_error(action: str, err: Exception) -> MediaRenderError:
    """Standard error for connection and timeout failures."""
    logger.warning("MediaRenderClient %s failed: %s", action, err)
    if isinstance(err, httpx.TimeoutException) and not isinstance(err, httpx.ConnectTimeout):
        return MediaRenderError(TIMEOUT, f"media-render did not answer {action} in time")
    return MediaRenderUnavailable(UNREACHABLE, f"media-render is unreachable ({action}): {err}")


class MediaRenderClient:
    """Proxy client for the media-render service.

    Stateless — all state lives in the httpx client. Tests pass their own
    ``http`` (an ``httpx.AsyncClient`` on a mock transport).
    """

    def __init__(self, http: Optional[httpx.AsyncClient] = None) -> None:
        self._http = http

    def _client(self) -> httpx.AsyncClient:
        return self._http if self._http is not None else _get_client()

    async def _send(self, method: str, path: str, action: str, **kwargs: Any) -> httpx.Response:
        url = f"{_base_url()}{path}"
        try:
            return await self._client().request(method, url, **kwargs)
        except httpx.HTTPError as err:
            raise _transport_error(action, err) from err

    async def health(self) -> Dict[str, Any]:
        """``GET /health``: the image's versions and the render queue."""
        resp = await self._send("GET", "/health", "health")
        if resp.status_code != 200:
            raise _error_for(resp, "health")
        return resp.json()

    async def submit(self, bundle: Mapping[str, Any]) -> Dict[str, Any]:
        """``POST /render``: the job once it is staged, spoken, mixed and checked.

        A composition its check refuses raises ``check_failed`` with the
        findings (nothing rendered); a full renderer raises ``busy`` with
        ``retry_after``.
        """
        resp = await self._send("POST", "/render", "render", json=dict(bundle))
        if resp.status_code != 202:
            raise _error_for(resp, "render")
        return resp.json()

    async def job(self, job_id: str) -> Dict[str, Any]:
        """``GET /render/{id}``: status, check report, timings and outputs."""
        resp = await self._send("GET", f"/render/{quote(job_id, safe='')}", "job status")
        if resp.status_code != 200:
            raise _error_for(resp, "job status")
        return resp.json()

    async def download(self, job_id: str, name: str, target: Path) -> Tuple[int, str]:
        """Stream a finished job's output to ``target``; ``(bytes, sha256)``.

        The file is never held in memory, and its digest is taken on the way
        through, so the approval can bind to these exact bytes (D6).
        """
        url = f"{_base_url()}/render/{quote(job_id, safe='')}/output/{quote(name, safe='')}"
        digest = hashlib.sha256()
        size = 0
        try:
            async with self._client().stream("GET", url) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise _error_for(resp, "output download")
                with target.open("wb") as handle:
                    async for chunk in resp.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
                        digest.update(chunk)
                        size += len(chunk)
                        handle.write(chunk)
        except httpx.HTTPError as err:
            raise _transport_error("output download", err) from err
        return size, digest.hexdigest()
