"""PRD-221 S9 — Auto's Read: cached workspace digest generation.

Turns the S8 snapshot into a short plain-English read of the workspace,
cached per (workspace, state_hash) so the LLM fires at most once per real
state change. Any LLM/cache failure falls back to the deterministic template
(services.workspace_digest.render_fallback_digest) — this path never raises,
so the endpoint never 500s on a degraded model.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from config import config
from services.workspace_digest import build_digest_snapshot, render_fallback_digest

logger = logging.getLogger(__name__)

_DIGEST_NAMESPACE = "digest"
_MAX_WORDS = 150


def _digest_ttl() -> int:
    try:
        return int(getattr(config, "DIGEST_CACHE_TTL_S", 900))
    except (TypeError, ValueError):
        return 900


def _cache_key(workspace_id: Any, state_hash: str) -> str:
    """Workspace-isolated cache key (mirrors CacheService._key convention)."""
    return f"cache:{_DIGEST_NAMESPACE}:{workspace_id}:{state_hash}"


def _default_redis():
    try:
        from core.cache.service import get_cache_service
        return get_cache_service().redis
    except Exception:  # pragma: no cover - redis wiring
        logger.warning("[digest] cache unavailable — digest will not be cached")
        return None


def _read_cache(redis_client, key: str) -> Optional[Dict[str, Any]]:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


def _write_cache(redis_client, key: str, value: Dict[str, Any], ttl: int) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception:
        logger.debug("[digest] cache write skipped", exc_info=True)


def _build_messages(snapshot: Dict[str, Any]) -> list:
    payload = json.dumps({
        "counts": snapshot.get("counts", {}),
        "needs_attention": snapshot.get("needs_attention", []),
        "active": snapshot.get("active", []),
        "recent_completions": snapshot.get("recent_completions", []),
    }, separators=(",", ":"))
    system = (
        "You are Auto, summarising a workspace for its owner. Write a calm, "
        f"plain-English read in at most {_MAX_WORDS} words. Name any blocked or "
        "failed items explicitly and say why. No raw logs, no JSON, no bullet "
        "lists — two or three sentences a non-technical owner can act on."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Workspace state:\n{payload}"},
    ]


async def _generate_text(
    snapshot: Dict[str, Any],
    workspace_id: Any,
    llm_factory: Optional[Callable[..., Any]],
) -> Tuple[str, bool]:
    """Return (text, from_llm). Falls back to the deterministic template on any
    failure — never raises."""
    try:
        if llm_factory is not None:
            llm = llm_factory()
        else:
            from core.llm import create_llm_manager
            llm = create_llm_manager(
                service_name="digest",
                workspace_id=workspace_id,
                request_type="digest",
            )
        response = await llm.generate_response(_build_messages(snapshot))
        text = (getattr(response, "content", None) or "").strip()
        if text:
            return text, True
    except Exception:
        logger.warning("[digest] LLM generation failed — using fallback", exc_info=True)
    return render_fallback_digest(snapshot), False


async def generate_digest(
    db,
    workspace_id: Any,
    period: str = "1d",
    *,
    redis_client: Any = None,
    llm_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Cached Auto's Read for a workspace. Cache hit on an unchanged state_hash
    returns without invoking the LLM; a miss generates once and caches. LLM
    failures return the fallback text and are NOT cached (so they self-heal)."""
    snapshot = build_digest_snapshot(db, workspace_id, period)
    state_hash = snapshot["state_hash"]
    key = _cache_key(workspace_id, state_hash)

    client = redis_client if redis_client is not None else _default_redis()
    cached = _read_cache(client, key)
    if cached is not None:
        return cached

    text, from_llm = await _generate_text(snapshot, workspace_id, llm_factory)
    result = {
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_hash": state_hash,
        "needs_attention_count": snapshot.get("needs_attention_count", 0),
    }
    if from_llm:
        _write_cache(client, key, result, _digest_ttl())
    return result
