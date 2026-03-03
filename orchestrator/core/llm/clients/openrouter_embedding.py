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

        if not api_key:
            logger.warning(
                "OpenRouter API key not configured. "
                "Embedding features will fail until key is added."
            )
            self.client = None
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.OPENROUTER_BASE_URL,
            )
            model_info = OPENROUTER_EMBEDDING_MODELS.get(self.config.model, (4096, 8192))
            logger.info(
                f"Initialized OpenRouter embedding client — "
                f"model: {self.config.model}, dim: {self.config.dimension}, "
                f"max_ctx: {model_info[1]} tokens"
            )

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding via OpenRouter API"""
        if self.client is None:
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
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=text,
            )
            embedding = response.data[0].embedding

            # Truncate to configured dimension if model outputs more
            if len(embedding) > self.config.dimension:
                embedding = embedding[:self.config.dimension]

            return embedding

        except Exception as e:
            logger.error(f"OpenRouter embedding error ({self.config.model}): {e}")
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
        if self.client is None:
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
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=processed_texts,
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
                    resp = await self.client.embeddings.create(
                        model=self.config.model,
                        input=text,
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
