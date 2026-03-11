"""
Mem0 Client Integration
=======================

Wrapper around standard mem0ai usage to connect to internal Railway instance.

Includes:
- Configurable timeout
- Exponential backoff retry (max 2 retries)
- Circuit breaker: after N consecutive failures, fail fast for a cooldown period
"""

import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Circuit breaker settings
_CB_FAILURE_THRESHOLD = 5      # Open circuit after 5 consecutive failures
_CB_COOLDOWN_SECONDS = 60      # Stay open for 60 seconds before retrying
_DEFAULT_TIMEOUT = 15           # Seconds — Mem0 calls OpenAI for fact extraction, needs headroom
_MAX_RETRIES = 1                # One retry with backoff


class _CircuitBreaker:
    """Simple circuit breaker for Mem0 calls."""
    __slots__ = ("failures", "last_failure_time", "is_open")

    def __init__(self):
        self.failures = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_success(self):
        self.failures = 0
        self.is_open = False

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.monotonic()
        if self.failures >= _CB_FAILURE_THRESHOLD:
            self.is_open = True
            logger.warning(
                "[Mem0] Circuit breaker OPEN after %d failures — "
                "skipping Mem0 calls for %ds",
                self.failures, _CB_COOLDOWN_SECONDS,
            )

    def allow_request(self) -> bool:
        if not self.is_open:
            return True
        # Check cooldown
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= _CB_COOLDOWN_SECONDS:
            logger.info("[Mem0] Circuit breaker half-open — allowing probe request")
            return True  # Half-open: allow one probe
        return False


# Shared circuit breaker instance
_breaker = _CircuitBreaker()


class Mem0Client:
    """
    Client for interacting with Mem0 server.
    """

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        from config import config
        self.api_url = (api_url or config.MEM0_API_URL or "").strip()
        self.api_key = api_key or config.MEM0_API_KEY
        self.timeout = _DEFAULT_TIMEOUT

        if not self.api_url:
            logger.warning("[Mem0] No API URL configured (MEM0_API_URL). Memory storage disabled.")
            self.api_url = ""
            self.headers = {}
            return

        # Ensure URL has correct format
        if not self.api_url.startswith("http"):
            self.api_url = f"https://{self.api_url}"

        self.api_url = self.api_url.rstrip("/")

        # OpenMemory API mounts routes under /api/v1 prefix
        if "/api/v1" not in self.api_url:
            self.api_url = f"{self.api_url}/api/v1"
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Token {self.api_key}"

        logger.info(f"Initialized Mem0Client with URL: {self.api_url}")

    def _request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Make an HTTP request with retry + circuit breaker.

        Returns None early if api_url is not configured.
        Returns the response or None if all attempts fail.
        """
        if not self.api_url:
            logger.debug("[Mem0] No API URL configured — skipping request")
            return None

        if not _breaker.allow_request():
            logger.debug("[Mem0] Circuit breaker open — skipping request")
            return None

        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", self.headers)
        last_exc = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.request(method, url, **kwargs)
                _breaker.record_success()
                return resp
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    wait = 1.5 ** attempt  # 1s, 1.5s
                    logger.warning(
                        "[Mem0] Request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES + 1, e, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("[Mem0] Request failed after %d attempts: %s", _MAX_RETRIES + 1, e)
                    _breaker.record_failure()
            except Exception as e:
                logger.error("[Mem0] Unexpected request error: %s", e)
                _breaker.record_failure()
                return None

        return None

    def add(self, messages: List[Dict[str, str]], user_id: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Add memories from messages.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            user_id: Unique user identifier
            metadata: Optional metadata to attach

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

        resp = self._request("POST", url, json=payload)
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
            result = None

        if result:
            logger.info("[Mem0] Memory stored for user_id=%s: %s", user_id, str(result)[:200])
            return result
        else:
            logger.info("[Mem0] Memory accepted (status=%s) for user_id=%s", resp.status_code, user_id)
            return {"success": True}

    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            user_id: User identifier to scope search
            limit: Max results

        Returns:
            List of memory items
        """
        # OpenMemory API uses /memories/filter for searching
        url = f"{self.api_url}/memories/filter"
        payload = {
            "user_id": user_id,
            "search_query": query,
            "size": limit,
        }

        logger.debug("[Mem0] Searching memories for user=%s query=%r", user_id, query)

        resp = self._request("POST", url, json=payload)
        if resp is None:
            return []

        if resp.status_code >= 400:
            logger.warning("[Mem0] Fetch returned %s", resp.status_code)
            return []

        try:
            data = resp.json()
        except Exception:
            return []

        # Search endpoint returns a list of results (with scores) or a
        # dict wrapper with "results" key depending on Mem0 version.
        if isinstance(data, dict):
            items = data.get("results", data.get("items", []))
        elif isinstance(data, list):
            items = data
        else:
            logger.warning("[Mem0] Unexpected search response format: %s", type(data))
            return []

        logger.info("[Mem0] Search returned %d results", len(items))

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

    def get_all(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get all memories for a user."""
        url = f"{self.api_url}/memories/"
        params = {"user_id": user_id}

        resp = self._request("GET", url, params=params)
        if resp is None:
            return []

        if resp.status_code >= 400:
            logger.debug("[Mem0] get_all status=%s for user_id=%s", resp.status_code, user_id)
            return []

        try:
            data = resp.json()
        except Exception:
            logger.warning("[Mem0] get_all failed to parse JSON for user_id=%s body=%s", user_id, (resp.text or "")[:200])
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

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        url = f"{self.api_url}/memories/{memory_id}/"

        resp = self._request("DELETE", url)
        if resp is None:
            return False

        return resp.status_code < 400
