"""
Rerank Manager - Cohere Rerank API Integration
===============================================

Singleton manager for document reranking using Cohere's Rerank API.
Uses httpx (already in requirements) — no cohere SDK dependency.

Graceful degradation: if no API key configured, returns documents unchanged.
"""

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import httpx

from config import config

logger = logging.getLogger(__name__)

COHERE_RERANK_URL = config.COHERE_RERANK_URL
COHERE_DOC_CHAR_LIMIT = 4096
DEFAULT_RERANK_MODEL = "rerank-v3.5"


@dataclass
class RerankResult:
    """Single reranked document result."""
    index: int
    relevance_score: float


class RerankManager:
    """Singleton reranker using Cohere Rerank API."""

    def __init__(self):
        self._api_key: Optional[str] = None
        self._model: str = DEFAULT_RERANK_MODEL
        self._loaded = False
        self._client: Optional[httpx.AsyncClient] = None

    def _load_config(self):
        """Lazy-load API key and model from system_settings, fallback to env."""
        if self._loaded:
            return

        # Try system_settings first
        try:
            from core.database.database import SessionLocal
            from core.models.system_settings import SystemSetting

            db = SessionLocal()
            try:
                key_setting = db.query(SystemSetting).filter(
                    SystemSetting.key == "cohere_api_key"
                ).first()
                if key_setting and key_setting.value:
                    self._api_key = key_setting.value

                model_setting = db.query(SystemSetting).filter(
                    SystemSetting.key == "rag_rerank_model"
                ).first()
                if model_setting and model_setting.value:
                    self._model = model_setting.value
            finally:
                db.close()
        except Exception:
            pass

        # Env var fallback for API key
        if not self._api_key:
            self._api_key = config.COHERE_API_KEY

        self._loaded = True

        if self._api_key:
            logger.info(f"RerankManager configured: model={self._model}")
        else:
            logger.debug("RerankManager: no Cohere API key found, reranking disabled")

    def is_available(self) -> bool:
        """Check if reranking is available (API key configured)."""
        self._load_config()
        return bool(self._api_key)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank documents using Cohere Rerank API.

        Returns List[RerankResult] sorted by relevance descending.
        If unavailable, returns identity ordering (original indices).
        """
        self._load_config()

        if not self._api_key or not documents:
            return [
                RerankResult(index=i, relevance_score=0.0)
                for i in range(min(top_n, len(documents)))
            ]

        # Truncate docs to Cohere's per-doc limit
        truncated_docs = [doc[:COHERE_DOC_CHAR_LIMIT] for doc in documents]

        try:
            client = self._get_client()
            response = await client.post(
                COHERE_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query,
                    "documents": truncated_docs,
                    "top_n": min(top_n, len(truncated_docs)),
                },
            )
            response.raise_for_status()
            data = response.json()

            results = [
                RerankResult(
                    index=item["index"],
                    relevance_score=item["relevance_score"],
                )
                for item in data.get("results", [])
            ]
            results.sort(key=lambda r: r.relevance_score, reverse=True)

            logger.info(
                f"Reranked {len(documents)} candidates via Cohere {self._model}, "
                f"returning top {len(results)}"
            )
            return results

        except httpx.HTTPStatusError as e:
            logger.warning(f"Cohere rerank API error: {e.response.status_code} {e.response.text[:200]}")
            return [
                RerankResult(index=i, relevance_score=0.0)
                for i in range(min(top_n, len(documents)))
            ]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return [
                RerankResult(index=i, relevance_score=0.0)
                for i in range(min(top_n, len(documents)))
            ]

    def reload_config(self):
        """Force reload configuration (e.g. after settings change)."""
        self._loaded = False
        self._load_config()


# Thread-safe singleton
_rerank_lock = threading.Lock()
_rerank_manager: Optional[RerankManager] = None


def get_rerank_manager() -> RerankManager:
    """Get or create the singleton RerankManager."""
    global _rerank_manager

    if _rerank_manager is not None:
        return _rerank_manager

    with _rerank_lock:
        if _rerank_manager is None:
            _rerank_manager = RerankManager()
            logger.info("RerankManager singleton initialized")

    return _rerank_manager
