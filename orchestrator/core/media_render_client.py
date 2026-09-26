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
  - modules/socials/render.py                   (a post's render lifecycle)
  - modules/documents/generation_service.py     (generate_document with a social format)
  - api/socials.py                              (the renderer check before a render starts)

The service's API is described in ``services/media-render/README.md``. Every
failure raises :class:`MediaRenderError` with a ``code``: the renderer's own
(``check_failed`` with its findings, ``invalid_bundle``, ``busy`` with
``retry_after``, ``media_fetch_failed``), or ``not_configured``,
``unreachable``, ``timeout``, ``not_found``, ``bad_response`` and
``no_output`` from this side. ``submit_when_free`` and ``wait_for`` are the one
submit-and-poll loop both renders use; ``render_to_file`` runs a render start
to finish for a caller that waits for its file. Nothing here renders or
generates anything: the renderer assembles (D3).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
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
BAD_RESPONSE = "bad_response"
NO_OUTPUT = "no_output"

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


def job_error(record: Mapping[str, Any]) -> MediaRenderError:
    """The error a job that ended ``failed`` or ``rejected`` carries: the renderer's code and message."""
    error = record.get("error") if isinstance(record.get("error"), dict) else {}
    report = record.get("report") if isinstance(record.get("report"), dict) else {}
    findings = report.get("findings") if isinstance(report.get("findings"), list) else ()
    code = str(error.get("code") or record.get("status") or "failed")
    message = str(error.get("message") or "the render failed")
    return MediaRenderError(code, message, findings=findings, report=report)


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

    async def submit_when_free(
        self, bundle: Mapping[str, Any], *, deadline: float, poll_seconds: float
    ) -> Dict[str, Any]:
        """:meth:`submit`, asked again after the renderer's ``Retry-After`` while
        it is busy and ``deadline`` (a ``time.monotonic()`` instant) allows."""
        while True:
            try:
                return await self.submit(bundle)
            except MediaRenderError as exc:
                wait = exc.retry_after or poll_seconds
                if exc.code != BUSY or time.monotonic() + wait >= deadline:
                    raise
                logger.info("MediaRenderClient: the renderer is busy; asking again in %ss", wait)
                await asyncio.sleep(wait)

    async def wait_for(
        self, record: Mapping[str, Any], *, deadline: float, poll_seconds: float
    ) -> Dict[str, Any]:
        """Poll the job ``record`` names until it ends (done, failed or rejected); its last record.

        A poll that fails is asked again until ``deadline``. Raises ``timeout``
        at the deadline, ``not_found`` when the renderer lost the job (it
        restarted) and ``bad_response`` for a record without a job id.
        """
        job_id = str(record.get("id") or "")
        if not job_id:
            raise MediaRenderError(BAD_RESPONSE, "media-render accepted the render but gave no job id")
        current = dict(record)
        while current.get("status") not in JOB_TERMINAL:
            if time.monotonic() >= deadline:
                raise MediaRenderError(TIMEOUT, f"render {job_id} did not finish in time")
            await asyncio.sleep(poll_seconds)
            try:
                current = await self.job(job_id)
            except MediaRenderError as exc:
                if exc.code == NOT_FOUND or exc.status == 404:
                    raise MediaRenderError(NOT_FOUND, "media-render lost the render (it restarted)", status=404) from exc
                logger.warning("MediaRenderClient: polling render %s failed: %s", job_id, exc)
        return current

    async def render_to_file(
        self, bundle: Mapping[str, Any], target: Path, *, max_wait_seconds: float, poll_seconds: float
    ) -> Dict[str, Any]:
        """Render ``bundle`` start to finish and stream its first output to
        ``target``: the finished job record.

        For a caller that waits for its file (``generate_document`` with a social
        format). A busy renderer is asked again, the job is polled, and a job
        that ends other than done raises the renderer's own code and message.
        ``max_wait_seconds`` bounds all of it, the download included.
        """
        async def run() -> Dict[str, Any]:
            deadline = time.monotonic() + max_wait_seconds
            accepted = await self.submit_when_free(bundle, deadline=deadline, poll_seconds=poll_seconds)
            finished = await self.wait_for(accepted, deadline=deadline, poll_seconds=poll_seconds)
            if finished.get("status") != JOB_DONE:
                raise job_error(finished)
            outputs = [o for o in (finished.get("outputs") or []) if isinstance(o, dict) and o.get("name")]
            if not outputs:
                raise MediaRenderError(NO_OUTPUT, "media-render finished the render but returned no file")
            size, _digest = await self.download(str(finished["id"]), str(outputs[0]["name"]), target)
            if size <= 0:
                raise MediaRenderError(NO_OUTPUT, "media-render returned an empty file")
            return finished

        try:
            return await asyncio.wait_for(run(), timeout=max_wait_seconds)
        except asyncio.TimeoutError:
            raise MediaRenderError(TIMEOUT, f"the render did not finish within {max_wait_seconds:g} s") from None

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
