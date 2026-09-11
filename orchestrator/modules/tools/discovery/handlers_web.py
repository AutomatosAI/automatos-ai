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
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin
from uuid import UUID

import httpx
from sqlalchemy.orm import Session

from config import config
from core.security.web_access import (
    WEB_ACCESS_OFF_REASON,
    OutboundTarget,
    resolve_outbound,
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
    """The request for ``url`` sent to the address that was checked.

    The URL's host becomes the pinned IP; the real hostname rides as ``Host``
    and as SNI (``sni_hostname``), so TLS is still verified against the name.
    A DNS answer that changes after the check therefore changes nothing.
    """
    pinned_url = httpx.URL(url).copy_with(host=target.ip)  # httpx brackets IPv6 itself
    return client.build_request(
        "GET",
        pinned_url,
        headers={"Host": target.host},
        extensions={"sni_hostname": target.host},
    )


async def _resolve(url: str) -> OutboundTarget:
    # System DNS is blocking; keep it off the event loop.
    return await asyncio.to_thread(resolve_outbound, url)


async def web_fetch(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Read one public URL and return its text (US: S2).

    Redirects are followed by hand: every hop is resolved and checked BEFORE a
    request is sent to it, and each request goes to the address that was
    checked. httpx's own ``follow_redirects`` would have fetched the whole
    chain — including a hop into a private range — before returning.
    """
    url = (params.get("url") or "").strip()
    if not url:
        return {"success": False, "error": "url is required"}
    max_chars = _clamp_chars(params.get("max_chars", DEFAULT_MAX_CHARS))

    target = await _resolve(url)
    if not target.ok:
        if not web_access_enabled():
            return _unavailable(target.reason)
        return {"success": False, "error": target.reason, "url": url}

    max_bytes = int(config.WEB_FETCH_MAX_BYTES)
    timeout = httpx.Timeout(float(config.WEB_FETCH_TIMEOUT_SECONDS))
    current = url
    hops = 0
    resp: Optional[httpx.Response] = None
    try:
        async with _async_client(
            follow_redirects=False,
            verify=True,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5"},
        ) as client:
            while True:
                resp = await client.send(_pinned_request(client, current, target), stream=True)
                location = resp.headers.get("location")
                if resp.status_code in _REDIRECT_STATUSES and location:
                    await resp.aclose()
                    hops += 1
                    if hops > MAX_REDIRECTS:
                        return {"success": False, "error": f"Too many redirects (more than {MAX_REDIRECTS})", "url": url}
                    next_url = urljoin(current, location)
                    target = await _resolve(next_url)
                    if not target.ok:
                        return {"success": False, "error": f"Redirected to a refused address: {target.reason}", "url": url}
                    current = next_url
                    continue
                break

            final_url = current
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith(_TEXT_CONTENT_TYPES):
                await resp.aclose()
                return {
                    "success": False,
                    "error": f"Unsupported content type '{content_type}' — web_fetch reads text, HTML, JSON and XML pages.",
                    "url": final_url,
                }
            chunks: List[bytes] = []
            received = 0
            truncated = False
            async for chunk in resp.aiter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    chunks.append(chunk[: max(0, max_bytes - (received - len(chunk)))])
                    truncated = True
                    break
                chunks.append(chunk)
            status = resp.status_code
            encoding = resp.charset_encoding or "utf-8"
            await resp.aclose()
    except httpx.TimeoutException:
        return {"success": False, "error": f"Timed out after {config.WEB_FETCH_TIMEOUT_SECONDS}s fetching {url}", "url": url}
    except httpx.HTTPError as exc:
        return {"success": False, "error": f"Fetch failed: {exc}", "url": url}
    except Exception as exc:  # noqa: BLE001 — never crash the agent loop on a bad page
        logger.warning("[web_fetch] unexpected error for %s: %s", url, exc, exc_info=True)
        return {"success": False, "error": f"Fetch failed: {type(exc).__name__}", "url": url}

    body = b"".join(chunks).decode(encoding, errors="replace")
    if content_type in ("text/html", "application/xhtml+xml") or (not content_type and "<html" in body[:2000].lower()):
        title, text = html_to_text(body)
    else:
        title, text = "", body
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return {
        "success": True,
        "data": {
            "url": url,
            "final_url": final_url,
            "status_code": status,
            "content_type": content_type or "text/plain",
            "title": title,
            "content": text,
            "chars": len(text),
            "truncated": truncated,
            "redirects": hops,
        },
    }


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
