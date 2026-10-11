"""Web handlers for PlatformActionExecutor (PRD-240 S2/S3).

``platform_web_fetch`` is native and keyless: an HTTP GET from the backend container,
allowed only through ``core.security.web_access`` (the switch, the operator's
denylist, and the private-range check that keeps agents off the compose
network and the host), TLS verified, redirects re-checked, bytes capped, HTML
reduced to readable text. ``platform_web_search`` hands the query to
``services.web_search`` — whichever engine the deployment has.

Both answer honestly and never raise into the agent loop: web access off, a
refused URL, a missing search engine and a failed fetch are all plain results.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from uuid import UUID

import httpx
from sqlalchemy.orm import Session

from config import config
from core.security.web_access import (
    resolve_outbound_async,
    build_pinned_request,
    WEB_ACCESS_OFF_REASON,
    OutboundTarget,
    web_access_enabled,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 20_000
MAX_CHARS_CEILING = 60_000
_TEXT_CONTENT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml")
_DROP_TAGS = ("script", "style", "noscript", "svg", "canvas", "template", "iframe")
_CHROME_TAGS = ("nav", "footer", "header", "aside", "form")
_USER_AGENT = "AutomatosAgent/1.0 (+https://github.com/AutomatosAI/automatos-ai)"
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 5
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_PAGE_ACCEPT = "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5"
_CSS_ACCEPT = "text/css,*/*;q=0.1"
# Bytes are counted as sent, before anything is kept: a compressed body can decode to far more than the cap in
# one chunk, so only an unencoded one is read (the rule documents' fetches keep, modules/documents/pinned_fetch).
_IDENTITY_ENCODINGS = ("", "identity")
# brand: true reads at most this many of the page's stylesheets, in parallel. They are the page's own choice of
# address (often a CDN, so not limited to the page's site), so the fan-out is bounded three ways: how many,
# how big each one may be, and how long the whole read may take; a sheet not in by then is named as not read.
BRAND_MAX_STYLESHEETS = 6
BRAND_STYLESHEET_MAX_BYTES = 768 * 1024
BRAND_DEADLINE_SECONDS = 30.0
NOT_A_PAGE_NOTE = "This address is not a web page, so it shows no brand: fetch the site's home page."


def _async_client(**kwargs: Any) -> httpx.AsyncClient:
    """The HTTP client factory — tests replace it with a MockTransport client."""
    return httpx.AsyncClient(**kwargs)


def _unavailable(reason: str, **extra: Any) -> Dict[str, Any]:
    return {"success": True, "data": {"available": False, "reason": reason, **extra}}


def _clamp_chars(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    return max(500, min(n, MAX_CHARS_CEILING))


def html_to_text(html: str) -> tuple[str, str]:
    """(title, readable text) — headings and paragraphs kept, page chrome dropped."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for tag in soup(_CHROME_TAGS):
        tag.decompose()
    lines: List[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td", "th", "blockquote"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append("#" * int(el.name[1]) + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        else:
            lines.append(text)
    if not lines:  # a page with no block elements — fall back to all visible text
        lines = [" ".join(soup.get_text(" ", strip=True).split())]
    return title, "\n".join(lines).strip()


def _pinned_request(client: httpx.AsyncClient, url: str, target: OutboundTarget) -> httpx.Request:
    """The GET for ``url`` sent to the address that was checked — see
    :func:`core.security.web_access.build_pinned_request`, the one place the
    resolve-and-pin request is built (the heartbeat webhook shares it)."""
    return build_pinned_request(client, "GET", url, target)


async def _resolve(url: str) -> OutboundTarget:
    # System DNS is blocking; keep it off the event loop, and bounded — a
    # resolver that never answers is a refusal (RESOLVE_TIMEOUT_SECONDS).
    return await resolve_outbound_async(url)


@dataclass(frozen=True)
class _Page:
    """One fetched response's text, as served, after every redirect hop was checked."""

    final_url: str
    status: int
    content_type: str
    body: str
    truncated: bool
    redirects: int

    @property
    def is_html(self) -> bool:
        return self.content_type in _HTML_TYPES or (not self.content_type and "<html" in self.body[:2000].lower())


class _FetchFailed(Exception):
    """A fetch that ends in an answer for the agent (refused, off, too many redirects…), not a crash."""

    def __init__(self, result: Dict[str, Any]) -> None:
        super().__init__(result.get("error") or result.get("data", {}).get("reason", "fetch failed"))
        self.result = result


async def _send_checked(client: httpx.AsyncClient, url: str, target: OutboundTarget) -> Tuple[httpx.Response, str, int]:
    """Send the GET, following each redirect by hand: every hop is resolved and checked BEFORE a
    request goes to it, and each request goes to the address that was checked. httpx's own
    ``follow_redirects`` would have fetched the whole chain, a hop into a private range included."""
    current, hops = url, 0
    while True:
        resp = await client.send(_pinned_request(client, current, target), stream=True)
        location = resp.headers.get("location")
        if resp.status_code not in _REDIRECT_STATUSES or not location:
            return resp, current, hops
        await resp.aclose()
        hops += 1
        if hops > MAX_REDIRECTS:
            raise _FetchFailed({"success": False, "error": f"Too many redirects (more than {MAX_REDIRECTS})", "url": url})
        current = urljoin(current, location)
        target = await _resolve(current)
        if not target.ok:
            raise _FetchFailed({"success": False, "error": f"Redirected to a refused address: {target.reason}", "url": url})


async def _read_capped(resp: httpx.Response, max_bytes: int) -> Tuple[bytes, bool]:
    chunks: List[bytes] = []
    received = 0
    # aiter_bytes yields the bytes as sent: _get_text refuses any content-encoding before reading.
    async for chunk in resp.aiter_bytes():
        received += len(chunk)
        if received > max_bytes:
            chunks.append(chunk[: max(0, max_bytes - (received - len(chunk)))])
            return b"".join(chunks), True
        chunks.append(chunk)
    return b"".join(chunks), False


async def _get_text(url: str, accept: str, max_bytes: int) -> _Page:
    """GET one public URL's text through the platform's outbound check, or raise :class:`_FetchFailed`."""
    target = await _resolve(url)
    if not target.ok:
        raise _FetchFailed(_unavailable(target.reason) if not web_access_enabled()
                           else {"success": False, "error": target.reason, "url": url})
    timeout = httpx.Timeout(float(config.WEB_FETCH_TIMEOUT_SECONDS))
    try:
        async with _async_client(follow_redirects=False, verify=True, timeout=timeout,
                                 headers={"User-Agent": _USER_AGENT, "Accept": accept,
                                          "Accept-Encoding": "identity"}) as client:
            resp, final_url, hops = await _send_checked(client, url, target)
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith(_TEXT_CONTENT_TYPES):
                await resp.aclose()
                raise _FetchFailed({
                    "success": False,
                    "error": f"Unsupported content type '{content_type}' — web_fetch reads text, HTML, JSON and XML pages.",
                    "url": final_url,
                })
            encoding_sent = (resp.headers.get("content-encoding") or "").strip().lower()
            if encoding_sent not in _IDENTITY_ENCODINGS:
                await resp.aclose()
                raise _FetchFailed({
                    "success": False,
                    "error": f"The server sent a compressed body ({encoding_sent}) after being asked for none; "
                             "web_fetch reads uncompressed pages only.",
                    "url": final_url,
                })
            raw, truncated = await _read_capped(resp, max_bytes)
            status, encoding = resp.status_code, resp.charset_encoding or "utf-8"
            await resp.aclose()
    except _FetchFailed:
        raise
    except httpx.TimeoutException:
        raise _FetchFailed({"success": False, "error": f"Timed out after {config.WEB_FETCH_TIMEOUT_SECONDS}s fetching {url}", "url": url}) from None
    except httpx.HTTPError as exc:
        raise _FetchFailed({"success": False, "error": f"Fetch failed: {exc}", "url": url}) from None
    except Exception as exc:  # noqa: BLE001 — never crash the agent loop on a bad page
        logger.warning("[web_fetch] unexpected error for %s: %s", url, exc, exc_info=True)
        raise _FetchFailed({"success": False, "error": f"Fetch failed: {type(exc).__name__}", "url": url}) from None
    return _Page(final_url, status, content_type, raw.decode(encoding, errors="replace"), truncated, hops)


async def _stylesheet(url: str) -> Tuple[str, Optional[str]]:
    """One linked stylesheet's text, through the same checks as the page; ``(url, None)`` when not read."""
    try:
        page = await _get_text(url, _CSS_ACCEPT, BRAND_STYLESHEET_MAX_BYTES)
    except _FetchFailed as failed:
        logger.info("[web_fetch] brand: stylesheet %s not read: %s", url, failed)
        return url, None
    return url, page.body


async def _site_brand(page: _Page) -> Dict[str, Any]:
    """The page's brand evidence (``site_brand``), with up to BRAND_MAX_STYLESHEETS of its stylesheets read."""
    from modules.tools.discovery.site_brand import site_brand, stylesheet_urls

    # Parsing is CPU work on the page's own bytes: it runs in a thread, never on the event loop.
    sheets = await asyncio.to_thread(stylesheet_urls, page.body, page.final_url)
    read = await _stylesheets_within_deadline(sheets[:BRAND_MAX_STYLESHEETS])
    css = {url: read[url] for url in sheets if read.get(url) is not None}
    skipped = [url for url in sheets if read.get(url) is None]
    return await asyncio.to_thread(site_brand, page.body, page.final_url, css, skipped)


async def _stylesheets_within_deadline(urls: List[str]) -> Dict[str, Optional[str]]:
    """Each stylesheet's text, read in parallel; one still loading at BRAND_DEADLINE_SECONDS is cancelled
    and counts as not read, so a slow host never holds the call open."""
    tasks = [asyncio.ensure_future(_stylesheet(url)) for url in urls]
    if not tasks:
        return {}
    done, pending = await asyncio.wait(tasks, timeout=BRAND_DEADLINE_SECONDS)
    for task in pending:
        task.cancel()
    return dict(task.result() for task in done)


def _wants_brand(params: Dict[str, Any]) -> bool:
    value = params.get("brand")
    return value is True or (isinstance(value, str) and value.strip().lower() in ("true", "1", "yes"))


async def web_fetch(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Read one public URL and return its text (US: S2); with ``brand``, also what it shows of its brand."""
    url = (params.get("url") or "").strip()
    if not url:
        return {"success": False, "error": "url is required"}
    max_chars = _clamp_chars(params.get("max_chars", DEFAULT_MAX_CHARS))
    try:
        page = await _get_text(url, _PAGE_ACCEPT, int(config.WEB_FETCH_MAX_BYTES))
    except _FetchFailed as failed:
        return failed.result

    title, text = await asyncio.to_thread(html_to_text, page.body) if page.is_html else ("", page.body)
    truncated = page.truncated or len(text) > max_chars
    text = text[:max_chars]
    data: Dict[str, Any] = {
        "url": url,
        "final_url": page.final_url,
        "status_code": page.status,
        "content_type": page.content_type or "text/plain",
        "title": title,
        "content": text,
        "chars": len(text),
        "truncated": truncated,
        "redirects": page.redirects,
    }
    if _wants_brand(params):
        data["brand"] = await _site_brand(page) if page.is_html else {"notes": [NOT_A_PAGE_NOTE]}
    return {"success": True, "data": data}


async def web_search(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Find pages about a topic through the deployment's search engine (US: S3)."""
    query = (params.get("query") or "").strip()
    if not query:
        return {"success": False, "error": "query is required"}
    if not web_access_enabled():
        return _unavailable(WEB_ACCESS_OFF_REASON)

    from services.web_search import NO_BACKEND_OPTIONS, resolve_backend, search

    try:
        backend = resolve_backend(db, workspace_id)
    except Exception as exc:  # noqa: BLE001 — a probe fault is a result, not a crash
        logger.warning("[web_search] backend resolution failed: %s", exc, exc_info=True)
        return {"success": False, "error": f"Could not determine a search engine: {exc}"}
    if backend is None:
        return _unavailable(
            "No web search engine is configured on this server.",
            options=NO_BACKEND_OPTIONS,
        )
    try:
        max_results = int(params.get("max_results") or config.WEB_SEARCH_MAX_RESULTS)
    except (TypeError, ValueError):
        max_results = int(config.WEB_SEARCH_MAX_RESULTS)
    max_results = max(1, min(max_results, 10))
    try:
        results = await search(query, max_results=max_results, backend=backend, db=db, workspace_id=workspace_id)
    except Exception as exc:  # noqa: BLE001 — a search engine failure is a result, not a crash
        logger.warning("[web_search] %s backend failed: %s", backend, exc, exc_info=True)
        return {"success": False, "error": f"Search via {backend} failed: {exc}", "backend": backend}
    unverified = sum(1 for r in results if r.get("unverified"))
    data: Dict[str, Any] = {"query": query, "backend": backend, "results": results, "count": len(results)}
    if unverified:
        data["note"] = (
            f"{unverified} of these links came from the model's memory, not from a search — "
            "platform_web_fetch them before relying on them."
        )
    return {"success": True, "data": data}
