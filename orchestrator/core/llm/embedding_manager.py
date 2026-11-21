"""
Embedding Manager
================

Centralized manager for all embedding operations.
Reads configuration from General Settings and provides unified interface.
"""

import inspect
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple

from core.loop_scoped_clients import adopt
from .embed_once import embed_once
from .clients.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingProvider,
    DeterministicEmbeddingProvider,
    UnusableEmbeddingProvider,
)
from .clients.bedrock_embedding import BedrockEmbeddingProvider, DEFAULT_BEDROCK_EMBEDDING_MODEL
from .clients.openai_embedding import OpenAIEmbeddingProvider
from .clients.huggingface_embedding import HuggingFaceLocalEmbeddingProvider
from .clients.openrouter_embedding import OpenRouterEmbeddingProvider

logger = logging.getLogger(__name__)

# Thread-safe singleton lock
_embedding_lock = threading.Lock()
_embedding_manager: Optional["EmbeddingManager"] = None
_init_logged = False

# The seeded size; a stored "2048" means "not chosen here": VECTOR_STORE_DIMENSIONS decides.
DEFAULT_DIMENSION = "2048"
DISABLED = EmbeddingProvider.DISABLED.value
BEDROCK = EmbeddingProvider.BEDROCK.value
# Providers the manager looks up no API key for: the local model, and Bedrock, whose
# provider takes the Bedrock chat client's credentials itself (PRD-278 US-501).
NO_KEY_PROVIDERS = frozenset({EmbeddingProvider.HUGGINGFACE_LOCAL.value, BEDROCK})
# Where the size the provider cuts to came from, as the start-up check names it (PRD-278 US-502).
DIMENSION_FROM_CONFIG = "VECTOR_STORE_DIMENSIONS"
DIMENSION_FROM_SETTING = "the embeddings 'dimensions' system setting (Settings → System Settings → Embeddings)"


@dataclass(frozen=True)
class EmbeddingChoice:
    """What the settings choose: the provider (None when none is set), its model, the size its
    vectors are cut to and which setting gave that size, and the local model cache."""

    provider: Optional[str]
    model: Optional[str]
    dimension: int
    dimension_setting: str
    cache_dir: str

    @property
    def is_set(self) -> bool:
        """A real provider is set (not unset, not ``disabled``)."""
        return bool(self.provider) and self.provider != DISABLED


def get_system_setting(key: str, default_value: Optional[str] = None) -> Optional[str]:
    """Get embedding system setting from database (category='embeddings')."""
    try:
        from .manager import get_system_setting as _get_setting
        return _get_setting("embeddings", key, default_value)
    except Exception:
        return default_value


def get_credential_field(provider_type: str, field: str = "api_key") -> Optional[str]:
    """Get credential from credential resolver using the same multi-strategy
    lookup that LLM manager uses (settings mapping, name variations, type-based, env fallback)."""
    try:
        from core.llm.manager import get_credential_data
        cred_data = get_credential_data(provider_type, service_name="embeddings")
        if cred_data and field in cred_data:
            return cred_data[field]
        return None
    except Exception as e:
        logger.debug(f"Failed to get credential for {provider_type}: {e}")
        return None


def _names_bedrock(value: Optional[str]) -> bool:
    return (value or "").strip().lower() == BEDROCK


def _provider_and_model() -> Tuple[Optional[str], Optional[str]]:
    """The provider and model the settings choose.

    The ``embeddings`` system settings (PRD-197 S2: the canonical short keys), else
    ``EMBEDDING_PROVIDER`` / ``EMBEDDING_MODEL``. PRD-278 US-501: ``EMBEDDING_PROVIDER=bedrock``
    is read first, with ``EMBEDDING_MODEL`` (Titan Text Embeddings V2 when unset): a seeded
    install's ``embeddings.provider`` row says ``openrouter`` and would otherwise hide it.
    """
    from config import config as _config

    if _names_bedrock(_config.EMBEDDING_PROVIDER):
        return BEDROCK, _config.EMBEDDING_MODEL or DEFAULT_BEDROCK_EMBEDDING_MODEL
    provider = get_system_setting("provider") or _config.EMBEDDING_PROVIDER
    model = get_system_setting("model") or _config.EMBEDDING_MODEL
    return provider, model


def configured_provider() -> Optional[str]:
    """The embedding provider the settings choose, read as the manager reads it (None: unset)."""
    return _provider_and_model()[0]


def read_embedding_choice() -> EmbeddingChoice:
    """What the settings choose, read as the manager loads its provider: a settings read, so
    call it off the event loop. Raises ``ValueError`` on a size that is not a number."""
    from config import config as _config

    provider, model = _provider_and_model()
    stored = get_system_setting("dimensions") or DEFAULT_DIMENSION
    if stored == DEFAULT_DIMENSION:
        dimension, source = int(_config.VECTOR_STORE_DIMENSIONS or DEFAULT_DIMENSION), DIMENSION_FROM_CONFIG
    else:
        dimension, source = int(stored), DIMENSION_FROM_SETTING
    cache_dir = get_system_setting("cache_dir") or "./model_cache"
    return EmbeddingChoice(provider=provider, model=model, dimension=dimension, dimension_setting=source,
                           cache_dir=cache_dir)


def _api_key_for(provider_type: str) -> Optional[str]:
    """The API key for a provider that needs one: the operator workspace's key first (the
    credential-store copy can drift: a dead openrouter_api row served for months, 2026-07-30),
    then the credential store, then the env key."""
    from config import config as _config
    from core.llm.workspace_keys import get_platform_workspace_key

    api_key = get_platform_workspace_key(provider_type) or get_credential_field(provider_type)
    if api_key:
        return api_key
    env_map = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "huggingface_api": "HUGGINGFACE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY"
    }
    env_var = env_map.get(provider_type)
    return getattr(_config, env_var, None) if env_var else None


class EmbeddingManager:
    """
    Centralized manager for all embedding operations.
    
    Reads configuration from General Settings and provides
    a unified interface for generating embeddings.
    """
    
    def __init__(self):
        self.provider: Optional[BaseEmbeddingProvider] = None
        self._provider_loaded = False
        adopt(self)  # #837: made inside a closes_its_clients call, it is closed on that call's loop

    async def aclose(self) -> None:
        """Close the provider's async HTTP client on the running loop (#837).

        Left open, the OpenAI SDK's client closes itself when it is garbage-collected,
        with a task on whatever loop runs then, which for a client made on a worker
        thread's loop is one it never ran on. The shared manager
        (``get_embedding_manager``) serves every loop and is never closed here.
        """
        if self is _embedding_manager:
            return
        close = getattr(getattr(self.provider, "client", None), "close", None)
        if close is not None and inspect.iscoroutinefunction(close):
            await close()
    
    def _load_provider(self):
        """Load the embedding provider the settings choose (``read_embedding_choice``).

        No provider set (unset or ``disabled``): the deterministic stand-in, which refuses to
        embed. A provider set but unusable (no key, not built, or its set-up failed): a provider
        that names it and raises on every call (PRD-278 US-503), never the stand-in.
        """
        try:
            choice = read_embedding_choice()
        except Exception:
            logger.exception("Could not read the embedding settings; no provider is used")
            self.provider = DeterministicEmbeddingProvider(dimension=int(DEFAULT_DIMENSION))
            return
        logger.debug("Loaded embedding settings: provider=%s, model=%s, dim=%s",
                     choice.provider, choice.model, choice.dimension)
        if not choice.is_set:
            logger.debug("Embedding provider disabled - using deterministic fallback")
            self.provider = DeterministicEmbeddingProvider(dimension=choice.dimension)
            return
        self.provider = self._set_provider(choice)

    def _set_provider(self, choice: EmbeddingChoice) -> BaseEmbeddingProvider:
        """The provider ``choice`` sets, or ``UnusableEmbeddingProvider`` saying why it cannot be used."""
        try:
            api_key = None
            if choice.provider not in NO_KEY_PROVIDERS:
                api_key = _api_key_for(choice.provider)
                if not api_key:
                    return UnusableEmbeddingProvider(choice.provider, "no API key was found for it", choice.dimension)
            provider = self._create_provider(EmbeddingConfig(
                provider=EmbeddingProvider(choice.provider), model=choice.model, dimension=choice.dimension,
                api_key=api_key, cache_dir=choice.cache_dir))
        except Exception as exc:
            logger.exception("The embedding provider '%s' could not be set up", choice.provider)
            return UnusableEmbeddingProvider(choice.provider, f"it could not be set up ({type(exc).__name__})",
                                             choice.dimension)
        logger.debug("Initialized %s embedding provider (model: %s, dimension: %s)",
                     choice.provider, choice.model, choice.dimension)
        return provider

    def _create_provider(self, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create embedding provider from config"""
        if config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingProvider(config)
        elif config.provider == EmbeddingProvider.OPENROUTER:
            return OpenRouterEmbeddingProvider(config)
        elif config.provider == EmbeddingProvider.HUGGINGFACE_LOCAL:
            return HuggingFaceLocalEmbeddingProvider(config)
        elif config.provider == EmbeddingProvider.BEDROCK:
            return BedrockEmbeddingProvider(config)
        elif config.provider == EmbeddingProvider.DISABLED:
            return DeterministicEmbeddingProvider(config.dimension)
        return UnusableEmbeddingProvider(config.provider.value, "no embedding client is built for it", config.dimension)
    
    def _ensure_provider(self):
        """Lazy-load provider on first use (when database is ready)"""
        if not self._provider_loaded:
            self._load_provider()
            self._provider_loaded = True
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using configured provider.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        self._ensure_provider()  # Lazy-load on first use
        
        if self.provider is None:
            raise ValueError("Embedding provider not initialized")
        
        # Inside a chat turn's scope (TURN_EMBED_ONCE) a repeat of the same text is answered
        # from the turn's memo; with no scope open this is the provider's call, as before.
        return await embed_once(self._model_key(), text, self.provider.generate_embedding)

    def _model_key(self) -> str:
        """What tells one embedding space from another: the provider's class, model and dimension."""
        provider_config = getattr(self.provider, "config", None)
        model = getattr(provider_config, "model", None)
        dimension = getattr(provider_config, "dimension", None)
        return f"{type(self.provider).__name__}:{model}:{dimension}"

    def generate_embedding_sync(self, text: str) -> List[float]:
        """
        Generate embedding for text (synchronous).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        self._ensure_provider()  # Lazy-load on first synchronous use

        if self.provider is None:
            raise ValueError("Embedding provider not initialized")
        
        return self.provider.generate_embedding_sync(text)
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        max_concurrent: int = 5
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with parallel processing.

        If the provider supports native batch (OpenRouter), uses a single API call.
        Otherwise falls back to parallel individual calls with semaphore.

        Args:
            texts: List of texts to embed
            max_concurrent: Max concurrent API calls

        Returns:
            List of embedding vectors in same order as input texts
        """
        import asyncio

        self._ensure_provider()

        if self.provider is None:
            raise ValueError("Embedding provider not initialized")

        # Use native batch if provider supports it (OpenRouter)
        if hasattr(self.provider, 'generate_embeddings_batch'):
            return await self.provider.generate_embeddings_batch(texts, max_concurrent)

        # Fallback: parallel individual calls
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(texts)

        async def embed_one(idx: int, text: str):
            async with semaphore:
                results[idx] = await self.provider.generate_embedding(text)

        tasks = [embed_one(i, t) for i, t in enumerate(texts)]
        await asyncio.gather(*tasks)

        logger.info(f"Batch embedded {len(texts)} texts (concurrency={max_concurrent})")
        return results

    def get_dimension(self) -> int:
        """Get embedding dimension from provider or system settings"""
        if self.provider is None:
            # Read from system settings (canonical short key — PRD-197 S2)
            dimension_str = get_system_setting("dimensions")
            if dimension_str:
                return int(dimension_str)
            return 2048  # Last resort fallback
        return self.provider.get_dimension()
    
    def get_provider_info(self) -> dict:
        """Get information about configured provider"""
        if self.provider is None:
            return {"provider": "none", "status": "not_initialized"}
        
        return {
            "provider": self.provider.config.provider.value,
            "model": self.provider.config.model,
            "dimension": self.get_dimension(),
            "status": "active"
        }


def create_embedding_manager() -> EmbeddingManager:
    """
    Create or get singleton embedding manager (thread-safe).

    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager, _init_logged

    # Fast path - already initialized
    if _embedding_manager is not None:
        return _embedding_manager

    # Thread-safe initialization
    with _embedding_lock:
        # Double-check after acquiring lock
        if _embedding_manager is None:
            _embedding_manager = EmbeddingManager()
            if not _init_logged:
                logger.info("EmbeddingManager singleton initialized")
                _init_logged = True

    return _embedding_manager


def get_embedding_manager() -> EmbeddingManager:
    """Alias for create_embedding_manager"""
    return create_embedding_manager()
