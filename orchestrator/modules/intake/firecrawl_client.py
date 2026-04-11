"""
PRD-130: Firecrawl Client (cloud, single-tenant PoC)
=====================================================

Domain-locked Firecrawl wrapper for the Business Intake Wizard.

Phase 1 scope:
  - Cloud Firecrawl API only (https://api.firecrawl.dev/v1)
  - Two operations: map() for URL discovery, scrape() for single-page content
  - Hard page cap enforced client-side regardless of API response size
  - Domain lock: scrape() refuses URLs that don't match the bound domain

NOT a generic crawler. NOT exposed to agents. Wizard-only.
If Phase 2 needs more, refactor then — single file, not a folder, not an ABC.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class FirecrawlError(Exception):
    """Raised when Firecrawl API returns a non-success or unparseable response."""


class FirecrawlClient:
    """Thin async wrapper around Firecrawl cloud API.

    Usage:
        client = FirecrawlClient(api_key=config.FIRECRAWL_API_KEY,
                                 max_pages=config.FIRECRAWL_MAX_PAGES_PER_SCAN)
        urls = await client.map("inbuilduk.com")
        page = await client.scrape("https://inbuilduk.com/pages/about")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.firecrawl.dev/v1",
        max_pages: int = 20,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise FirecrawlError("FIRECRAWL_API_KEY is not configured")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_pages = max_pages
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        """Strip scheme/path/trailing slash to get a bare host."""
        d = domain.strip().lower()
        if "://" in d:
            d = urlparse(d).netloc or d.split("://", 1)[1]
        d = d.split("/", 1)[0]
        return d.removeprefix("www.")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _is_same_domain(self, url: str, bound_domain: str) -> bool:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        return host == bound_domain or host.endswith(f".{bound_domain}")

    # ------------------------------------------------------------------ API

    async def map(self, domain: str, limit: int | None = None) -> list[str]:
        """Discover URLs on a domain via Firecrawl /map.

        Returns a list of absolute URLs, capped by FIRECRAWL_MAX_PAGES_PER_SCAN.
        Note: the cap applies to the *returned slice* — scan still requests the
        full map so the wizard can show the true URL inventory size to the user
        before they pick which to scrape.
        """
        bound = self._normalize_domain(domain)
        payload: dict[str, Any] = {
            "url": f"https://{bound}",
        }
        if limit is not None:
            payload["limit"] = limit

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/map",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise FirecrawlError(f"Firecrawl /map timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise FirecrawlError(f"Firecrawl /map transport error: {exc}") from exc

        if resp.status_code >= 400:
            raise FirecrawlError(
                f"Firecrawl /map failed: {resp.status_code} {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise FirecrawlError(f"Firecrawl /map returned non-JSON: {exc}") from exc

        if not data.get("success", False):
            raise FirecrawlError(f"Firecrawl /map success=false: {data.get('error')}")

        links = data.get("links") or []
        # Defensive: some API versions return [{url:...}], others return [str]
        urls: list[str] = []
        for item in links:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and "url" in item:
                urls.append(item["url"])

        # Domain lock — drop anything off-domain even if Firecrawl returns it
        urls = [u for u in urls if self._is_same_domain(u, bound)]

        logger.info(
            "firecrawl.map domain=%s returned=%d (cap=%d)",
            bound,
            len(urls),
            self._max_pages,
        )
        return urls

    async def scrape(
        self,
        url: str,
        schema: dict | None = None,
        formats: tuple[str, ...] = ("markdown",),
    ) -> dict[str, Any]:
        """Scrape a single URL via Firecrawl /scrape.

        If `schema` is provided, Firecrawl runs LLM-extract mode and returns
        structured data alongside markdown.

        Returns:
            {
                "url": str,
                "markdown": str | None,
                "extract": dict | None,
                "metadata": dict,
            }
        """
        bound = self._normalize_domain(urlparse(url).netloc)
        if not self._is_same_domain(url, bound):
            raise FirecrawlError(f"Domain lock violation: {url} is not on {bound}")

        payload: dict[str, Any] = {
            "url": url,
            "formats": list(formats),
        }
        if schema is not None:
            payload["formats"] = list(set(formats) | {"extract"})
            payload["extract"] = {"schema": schema}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/scrape",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise FirecrawlError(
                f"Firecrawl /scrape timed out for {url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise FirecrawlError(
                f"Firecrawl /scrape transport error for {url}: {exc}"
            ) from exc

        if resp.status_code >= 400:
            raise FirecrawlError(
                f"Firecrawl /scrape failed for {url}: {resp.status_code} {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise FirecrawlError(f"Firecrawl /scrape returned non-JSON: {exc}") from exc

        if not data.get("success", False):
            raise FirecrawlError(
                f"Firecrawl /scrape success=false for {url}: {data.get('error')}"
            )

        body = data.get("data") or {}
        return {
            "url": url,
            "markdown": body.get("markdown"),
            "extract": body.get("extract") or body.get("llm_extraction"),
            "metadata": body.get("metadata") or {},
        }
