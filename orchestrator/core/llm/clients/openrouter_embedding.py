"""
OpenRouter Embedding Provider Implementation
=============================================

Routes embedding requests through OpenRouter's unified API,
supporting 20+ embedding models from multiple providers.
Uses OpenAI-compatible API at https://openrouter.ai/api/v1/embeddings

Features:
- Parallel batch processing with configurable concurrency
- Automatic text truncation based on model context length
- All top MTEB models accessible via single API key
"""

import logging
import asyncio
from typing import List, Optional

from config import config
from .base import BaseEmbeddingProvider, EmbeddingConfig

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

# Model ID → (native_dimension, max_context_tokens)
# These are the NATIVE output dimensions — no truncation, full quality
OPENROUTER_EMBEDDING_MODELS = {
    # Top-tier (best quality/price)
    "qwen/qwen3-embedding-8b":        (4096, 32000),   # Matryoshka: supports 32-4096
    "qwen/qwen3-embedding-4b":        (2560, 32768),   # Matryoshka: supports 32-2560
    "google/gemini-embedding-001":     (3072, 20000),   # Fixed 3072
    # OpenAI
    "openai/text-embedding-3-large":   (3072, 8192),    # Matryoshka: supports 256-3072
    "openai/text-embedding-3-small":   (1536, 8192),    # Matryoshka: supports 256-1536
    "openai/text-embedding-ada-002":   (1536, 8192),    # Fixed 1536
    # Mistral
    "mistralai/mistral-embed-2312":    (1024, 8192),    # Fixed 1024
    "mistralai/codestral-embed-2505":  (1024, 8192),    # Fixed 1024
    # BAAI (open source)
    "baai/bge-m3":                     (1024, 8192),    # Fixed 1024
    "baai/bge-large-en-v1.5":         (1024, 512),      # Fixed 1024
    "baai/bge-base-en-v1.5":          (768,  512),      # Fixed 768
    # Intfloat / E5
    "intfloat/e5-large-v2":           (1024, 512),      # Fixed 1024
    "intfloat/e5-base-v2":            (768,  512),      # Fixed 768
    "intfloat/multilingual-e5-large": (1024, 512),      # Fixed 1024
    # Sentence Transformers
    "sentence-transformers/all-mpnet-base-v2":       (768, 512),
    "sentence-transformers/all-minilm-l6-v2":        (384, 512),
    "sentence-transformers/all-minilm-l12-v2":       (384, 512),
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": (768, 512),
    "sentence-transformers/paraphrase-minilm-l6-v2": (384, 512),
    # GTE
    "thenlper/gte-large":             (1024, 512),
    "thenlper/gte-base":              (768,  512),
}


class OpenRouterEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenRouter embedding provider — routes through OpenRouter's unified API.

    Uses the OpenAI SDK pointed at https://openrouter.ai/api/v1
    Supports parallel batch embedding for fast document ingestion.
    """

    def _initialize_client(self):
        if AsyncOpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = self.config.api_key or config.OPENROUTER_API_KEY
        self._api_key = api_key or None
        self._base_url = config.OPENROUTER_BASE_URL
        # AsyncOpenAI binds its httpx pool to the event loop it is created on.
        # The tool-router runs embedding coroutines in a fresh THREAD loop
        # (sync->async bridge); a main-loop-bound client used there retries for
        # ~18s. So build the client per loop: the primary loop reuses one, a
        # foreign loop gets a fresh ephemeral client bound to it.
        self._primary_loop = None
        self.client = None  # created lazily per loop via _client_for_loop()
        self._extra_body = self._provider_routing_extra_body()

        if not api_key:
            logger.warning(
                "OpenRouter API key not configured. "
                "Embedding features will fail until key is added."
            )
        else:
            model_info = OPENROUTER_EMBEDDING_MODELS.get(self.config.model, (4096, 8192))
            logger.info(
                f"Initialized OpenRouter embedding client — "
                f"model: {self.config.model}, dim: {self.config.dimension}, "
                f"max_ctx: {model_info[1]} tokens"
            )

    @staticmethod
    def _provider_routing_extra_body():
        """OpenRouter provider-routing preferences for embedding requests.

        OpenRouter's default routing is price-sorted, so the slowest upstream
        for a model can win ties — measured 37-67s/call on qwen3-embedding-8b
        (2026-07-09) while all three hosts showed ~100% uptime. sort=latency
        makes OpenRouter pick the fastest measured provider instead. Config
        empty string disables (None → field omitted from the request).
        """
        try:
            sort = (getattr(config, "OPENROUTER_EMBEDDING_PROVIDER_SORT", "") or "").strip()
        except Exception:
            sort = ""
        if not sort:
            return None
        return {"provider": {"sort": sort}}

    def _client_for_loop(self):
        """An AsyncOpenAI client bound to the CURRENT running event loop.

        Reusing one client across loops binds its httpx pool to a stale loop and
        makes calls retry for ~18s (the tool-router thread-bridge stall). The
        primary loop reuses a cached client; any other loop (e.g. the router's
        per-call thread loop) gets a fresh ephemeral one, GC'd with that loop.
        """
        if self._api_key is None:
            return None
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._primary_loop is None:
            self._primary_loop = loop
        if loop is self._primary_loop:
            if self.client is None:
                self.client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
            return self.client
        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding via OpenRouter API"""
        client = self._client_for_loop()
        if client is None:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY or configure in Settings > General"
            )

        # Truncate based on model context length
        model_info = OPENROUTER_EMBEDDING_MODELS.get(self.config.model, (4096, 8192))
        max_chars = model_info[1] * 4  # ~4 chars per token estimate
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Text truncated to ~{model_info[1]} tokens for {self.config.model}")

        try:
            response = await client.embeddings.create(
                model=self.config.model,
                input=text,
                extra_body=self._extra_body,
            )
            embedding = response.data[0].embedding

            # Truncate to configured dimension if model outputs more
            if len(embedding) > self.config.dimension:
                embedding = embedding[:self.config.dimension]

            return embedding

        except Exception as e:
            # Use repr(e) — some HTTP errors (e.g. 402 credits exhausted) have empty str()
            status_code = getattr(e, "status_code", "N/A")
            logger.error(
                "OpenRouter embedding error (%s, status=%s): %r",
                self.config.model, status_code, e,
            )
            raise

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        max_concurrent: int = 5
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with parallel processing.

        Args:
            texts: List of texts to embed
            max_concurrent: Max concurrent API calls (default 5)

        Returns:
            List of embedding vectors in same order as input texts
        """
        client = self._client_for_loop()
        if client is None:
            raise ValueError("OpenRouter API key not configured.")

        if not texts:
            return []

        model_info = OPENROUTER_EMBEDDING_MODELS.get(self.config.model, (4096, 8192))
        max_chars = model_info[1] * 4

        # Truncate all texts
        processed_texts = [
            t[:max_chars] if len(t) > max_chars else t
            for t in texts
        ]

        # Try single batch call first (most efficient)
        # OpenRouter supports array input like OpenAI
        try:
            response = await client.embeddings.create(
                model=self.config.model,
                input=processed_texts,
                extra_body=self._extra_body,
            )

            embeddings = [None] * len(processed_texts)
            for item in response.data:
                emb = item.embedding
                if len(emb) > self.config.dimension:
                    emb = emb[:self.config.dimension]
                embeddings[item.index] = emb

            logger.info(
                f"Batch embedded {len(texts)} texts via OpenRouter "
                f"({self.config.model}) in single request"
            )
            return embeddings

        except Exception as e:
            logger.warning(
                f"Batch embedding failed ({e}), falling back to parallel individual calls"
            )

        # Fallback: parallel individual calls with semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(texts)

        async def embed_one(idx: int, text: str):
            async with semaphore:
                try:
                    resp = await client.embeddings.create(
                        model=self.config.model,
                        input=text,
                        extra_body=self._extra_body,
                    )
                    emb = resp.data[0].embedding
                    if len(emb) > self.config.dimension:
                        emb = emb[:self.config.dimension]
                    results[idx] = emb
                except Exception as exc:
                    logger.error(f"Failed to embed text {idx}: {exc}")
                    raise

        tasks = [
            embed_one(i, t) for i, t in enumerate(processed_texts)
        ]
        await asyncio.gather(*tasks)

        logger.info(
            f"Parallel embedded {len(texts)} texts via OpenRouter "
            f"({self.config.model}, concurrency={max_concurrent})"
        )
        return results

    def get_dimension(self) -> int:
        """Return configured embedding dimension"""
        return self.config.dimension
