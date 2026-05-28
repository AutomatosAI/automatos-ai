"""
Mem0 Client Integration
=======================

Wrapper around standard mem0ai usage to connect to internal Railway instance.

Includes:
- Configurable timeout
- Exponential backoff retry (max 2 retries)
- Circuit breaker: after N consecutive failures, fail fast for a cooldown period
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

import httpx

logger = logging.getLogger(__name__)

# PRD-137 Fix #6: tighter defaults — Mem0 is enrichment, not critical path.
# Values are read from config.py at Mem0Client init time.
_MAX_RETRIES = 1                # One retry with backoff


class _CircuitBreaker:
    """Simple circuit breaker for Mem0 calls."""
    __slots__ = ("failures", "last_failure_time", "is_open", "threshold", "cooldown")

    def __init__(self, threshold: int = 3, cooldown_seconds: int = 300):
        self.failures = 0
        self.last_failure_time = 0.0
        self.is_open = False
        self.threshold = threshold
        self.cooldown = cooldown_seconds

    def record_success(self):
        self.failures = 0
        self.is_open = False

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.monotonic()
        if self.failures >= self.threshold:
            self.is_open = True
            logger.warning(
                "[Mem0] Circuit breaker OPEN after %d failures — "
                "skipping Mem0 calls for %ds",
                self.failures, self.cooldown,
            )

    def allow_request(self) -> bool:
        if not self.is_open:
            return True
        # Check cooldown
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= self.cooldown:
            logger.info("[Mem0] Circuit breaker half-open — allowing probe request")
            return True  # Half-open: allow one probe
        return False


def _make_breaker() -> _CircuitBreaker:
    try:
        from config import config
        return _CircuitBreaker(
            threshold=int(getattr(config, "MEM0_CIRCUIT_THRESHOLD", 3)),
            cooldown_seconds=int(getattr(config, "MEM0_CIRCUIT_COOLDOWN_SECONDS", 300)),
        )
    except Exception:
        return _CircuitBreaker()


class Mem0Client:
    """
    Client for interacting with Mem0 server.
    """

    # Per-workspace circuit breakers, keyed by workspace_id. A failure in one
    # workspace's Mem0 path must not fail-fast every other workspace, so each
    # gets its own breaker. Calls without a workspace scope share "_global".
    # Class-level so the single shared Mem0Client (and any others) agree on a
    # workspace's breaker state. The global health probe (US-006) operates over
    # this whole registry to trip/reset all breakers at once.
    _breakers: Dict[str, _CircuitBreaker] = {}

    @classmethod
    def _get_breaker(cls, workspace_id: Optional[str] = None) -> _CircuitBreaker:
        """Return the circuit breaker for a workspace, creating it on first use."""
        key = workspace_id or "_global"
        breaker = cls._breakers.get(key)
        if breaker is None:
            breaker = _make_breaker()
            cls._breakers[key] = breaker
        return breaker

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        from config import config
        self.api_url = (api_url or config.MEM0_API_URL or "").strip()
        self.api_key = api_key or config.MEM0_API_KEY
        self.timeout = float(getattr(config, "MEM0_TIMEOUT_SECONDS", 3.0))
        self.write_timeout = float(getattr(config, "MEM0_WRITE_TIMEOUT_SECONDS", 15.0))
        # Pooled AsyncClient, created lazily inside the running event loop so
        # connections are reused across calls instead of opened per request.
        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_url:
            logger.info(
                "[Mem0] Disabled — missing MEM0_API_URL. Memory features will be skipped silently.",
            )
            self.api_url = ""
            self.headers = {}
            return

        if not self.api_url.startswith("http"):
            self.api_url = f"https://{self.api_url}"

        self.api_url = self.api_url.rstrip("/")

        if "/api/v1" not in self.api_url:
            self.api_url = f"{self.api_url}/api/v1"
        self.headers = {"Authorization": f"Token {self.api_key}"} if self.api_key else {}

        logger.info(f"Initialized Mem0Client with URL: {self.api_url}")

    def _get_client(self) -> httpx.AsyncClient:
        """Return the pooled AsyncClient, creating it lazily for connection reuse.

        Created on first use (inside the running loop) and recreated if a prior
        ``aclose()`` closed it. ``follow_redirects=True`` preserves the implicit
        redirect-following behaviour the old ``requests`` client had.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(follow_redirects=True)
        return self._client

    async def aclose(self) -> None:
        """Close the pooled client and its connections (call on shutdown)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _request(
        self, method: str, url: str, *, workspace_id: Optional[str] = None, **kwargs
    ) -> Optional[httpx.Response]:
        """
        Make an async HTTP request with retry + per-workspace circuit breaker.

        Returns None early if api_url is not configured.
        Returns the response or None if all attempts fail.
        """
        if not self.api_url:
            logger.warning("[Mem0] No API URL configured — skipping request to %s", url)
            return None

        breaker = self._get_breaker(workspace_id)
        if not breaker.allow_request():
            logger.warning("[Mem0] Circuit breaker open — skipping request to %s", url)
            return None

        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", self.headers)
        client = self._get_client()

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.request(method, url, **kwargs)

                if resp.status_code < 400:
                    breaker.record_success()
                    return resp

                if resp.status_code == 404:
                    logger.debug(
                        "[Mem0] 404 on %s %s — treating as empty result",
                        method.upper(), url,
                    )
                    return resp

                if resp.status_code in (400, 401, 403):
                    logger.warning(
                        "[Mem0] Client/config error %d on %s %s — not retrying",
                        resp.status_code, method.upper(), url,
                    )
                    return resp

                # 429 / 5xx — transient, retry then breaker
                if attempt < _MAX_RETRIES:
                    wait = 1.5 ** attempt
                    logger.warning(
                        "[Mem0] Transient %d (attempt %d/%d) — retrying in %.1fs",
                        resp.status_code, attempt + 1, _MAX_RETRIES + 1, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.error(
                    "[Mem0] Transient %d after %d attempts on %s %s",
                    resp.status_code, _MAX_RETRIES + 1, method.upper(), url,
                )
                breaker.record_failure()
                return resp

            except httpx.TransportError as e:
                # TransportError covers TimeoutException + NetworkError/ConnectError
                # (the async analog of requests.Timeout + requests.ConnectionError).
                if attempt < _MAX_RETRIES:
                    wait = 1.5 ** attempt
                    logger.warning(
                        "[Mem0] Request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES + 1, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("[Mem0] Request failed after %d attempts: %s", _MAX_RETRIES + 1, e, exc_info=True)
                    breaker.record_failure()
            except Exception as e:
                logger.error("[Mem0] Unexpected request error: %s", e, exc_info=True)
                breaker.record_failure()
                return None

        return None

    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        metadata: Optional[Dict] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict:
        """
        Add memories from messages.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            user_id: Unique user identifier
            metadata: Optional metadata to attach
            workspace_id: Scopes the circuit breaker so one workspace's Mem0
                outage does not trip the breaker for every other workspace.

        Returns:
            Response dict from server
        """
        url = f"{self.api_url}/memories/"

        # OpenMemory API expects {"text": "...", "user_id": "..."}
        # Convert messages list to a single text string
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                text_parts.append(f"{role}: {content}" if role != "user" else content)
        text = "\n".join(text_parts)

        payload: Dict[str, Any] = {
            "text": text,
            "user_id": user_id,
        }
        if metadata:
            payload["metadata"] = metadata

        logger.debug("[Mem0] Adding memory for user_id=%s (text_len=%d)", user_id, len(text))

        resp = await self._request("POST", url, json=payload, timeout=self.write_timeout, workspace_id=workspace_id)
        if resp is None:
            return {"success": False, "error": "Mem0 unavailable (circuit breaker or timeout)"}

        if resp.status_code >= 400:
            body_preview = (resp.text or "")[:300]
            logger.error("[Mem0] Add failed: status=%s body=%s", resp.status_code, body_preview)
            return {"error": f"HTTP {resp.status_code}: {body_preview}"}

        # OpenMemory server returns 200 with null/empty body on success —
        # processing happens server-side (OpenAI extraction + pgvector storage).
        # A 200 means the memory was accepted and processed.
        try:
            result = resp.json()
        except Exception:
            logger.debug("[Mem0] Add response not JSON (status=%s) — treating as success", resp.status_code)
            result = None

        if result:
            logger.info("[Mem0] Memory stored for user_id=%s: %s", user_id, str(result)[:200])
            return result
        else:
            logger.info("[Mem0] Memory accepted (status=%s) for user_id=%s", resp.status_code, user_id)
            return {"success": True}

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        workspace_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            user_id: User identifier to scope search
            limit: Max results
            workspace_id: Scopes the circuit breaker (see ``add``).

        Returns:
            List of memory items
        """
        # OpenMemory API supports:
        #   GET  /api/v1/memories/?user_id=...&search_query=...  (text filter)
        #   POST /api/v1/memories/filter  (body-based, supports search_query)
        # Use GET with search_query param for compatibility with both endpoints.
        url = f"{self.api_url}/memories/"
        params = {
            "user_id": user_id,
            "search_query": query,
            "size": min(limit, 100),
        }

        logger.debug("[Mem0] Searching memories for user=%s query=%r", user_id, query)

        resp = await self._request("GET", url, params=params, workspace_id=workspace_id)
        if resp is None:
            return []

        if resp.status_code == 404:
            logger.debug("[Mem0] Search: no memories yet for user_id=%s", user_id)
            return []

        if resp.status_code >= 400:
            logger.error("[Mem0] Search failed: status=%s body=%s", resp.status_code, (resp.text or "")[:300])
            return []

        try:
            data = resp.json()
        except Exception:
            logger.error("[Mem0] Search failed to parse JSON for user_id=%s body=%s", user_id, (resp.text or "")[:200], exc_info=True)
            return []

        # Response may be a paginated dict with "items" key, a dict with
        # "results" key, or a plain list depending on Mem0 version.
        if isinstance(data, dict):
            items = data.get("results", data.get("items", []))
        elif isinstance(data, list):
            items = data
        else:
            logger.warning("[Mem0] Unexpected search response format: %s", type(data))
            return []

        logger.info("[Mem0] Search returned %d results for query=%r", len(items), query[:60])

        results = []
        for m in items:
            results.append({
                "id": m.get("id"),
                "memory": m.get("memory") or m.get("content"),
                "score": m.get("score"),
                "metadata": m.get("metadata") or m.get("metadata_"),
                "created_at": m.get("created_at"),
            })

        if results:
            sample = [f"{r.get('memory', '')[:40]} (score={r.get('score')})" for r in results[:3]]
            logger.info("[Mem0] Top results: %s", sample)

        # Results from search endpoint are pre-ranked by similarity score.
        # Re-sort as fallback in case scores are missing.
        results.sort(
            key=lambda x: (
                x.get("score") is not None,
                x.get("score") or 0,
                x.get("created_at") or "",
            ),
            reverse=True,
        )
        return results[:limit]

    async def get_all(
        self, user_id: str, limit: int = 100, workspace_id: Optional[str] = None
    ) -> List[Dict]:
        """Get all memories for a user."""
        url = f"{self.api_url}/memories/"
        params = {"user_id": user_id}

        resp = await self._request("GET", url, params=params, workspace_id=workspace_id)
        if resp is None:
            return []

        if resp.status_code >= 400:
            logger.debug("[Mem0] get_all status=%s for user_id=%s", resp.status_code, user_id)
            return []

        try:
            data = resp.json()
        except Exception:
            logger.warning("[Mem0] get_all failed to parse JSON for user_id=%s body=%s", user_id, (resp.text or "")[:200], exc_info=True)
            return []

        # Debug: log what Mem0 actually returns so we can diagnose empty browse
        logger.info("[Mem0] get_all user_id=%s type=%s keys=%s len=%s sample=%s",
                     user_id, type(data).__name__,
                     list(data.keys())[:5] if isinstance(data, dict) else "n/a",
                     len(data) if isinstance(data, (list, dict)) else "?",
                     str(data)[:200] if data else "empty")

        if isinstance(data, list):
            return data[:limit]
        return data.get("memories", data.get("results", data.get("items", [])))[:limit]

    async def delete(self, memory_id: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a specific memory."""
        url = f"{self.api_url}/memories/{memory_id}/"

        resp = await self._request("DELETE", url, workspace_id=workspace_id)
        if resp is None:
            return False

        return resp.status_code < 400
