"""
Embedding Manager
================

Centralized manager for all embedding operations.
Reads configuration from General Settings and provides unified interface.
"""

import logging
import threading
from typing import Optional, List

from .clients.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingProvider,
    DeterministicEmbeddingProvider
)
from .clients.openai_embedding import OpenAIEmbeddingProvider
from .clients.huggingface_embedding import HuggingFaceLocalEmbeddingProvider
from .clients.openrouter_embedding import OpenRouterEmbeddingProvider

logger = logging.getLogger(__name__)

# Thread-safe singleton lock
_embedding_lock = threading.Lock()
_embedding_manager: Optional["EmbeddingManager"] = None
_init_logged = False


def get_system_setting(key: str, default_value: Optional[str] = None) -> Optional[str]:
    """Get system setting from database"""
    try:
        import sys
        # Check if database module is loaded (use correct module path)
        if 'core.database.database' not in sys.modules:
            logger.debug(f"Database module not loaded yet, returning default for {key}")
            return default_value
        
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(
                SystemSetting.key == key
            ).first()
            
            if setting and setting.value:
                return setting.value
            return default_value
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Failed to get system setting {key}: {e}")
        return default_value


def get_credential_field(provider_type: str, field: str = "api_key") -> Optional[str]:
    """Get credential from credential resolver"""
    try:
        from core.credentials.resolver import get_credential_resolver, CredentialNotFoundError
        resolver = get_credential_resolver()
        
        # Try standard naming patterns
        credential_names = [
            f"development_{provider_type}_api",
            f"development_{provider_type}",
            f"{provider_type}_api",
            provider_type
        ]
        
        for cred_name in credential_names:
            try:
                return resolver.get_credential_field(cred_name, field)
            except CredentialNotFoundError:
                continue
        
        return None
    except Exception as e:
        logger.debug(f"Failed to get credential for {provider_type}: {e}")
        return None


class EmbeddingManager:
    """
    Centralized manager for all embedding operations.
    
    Reads configuration from General Settings and provides
    a unified interface for generating embeddings.
    """
    
    def __init__(self):
        self.provider: Optional[BaseEmbeddingProvider] = None
        self._provider_loaded = False
    
    def _load_provider(self):
        """Load embedding provider from system settings"""
        try:
            #Get from General Settings
            provider_type = get_system_setting("embedding_provider")
            model = get_system_setting("embedding_model")
            cache_dir = get_system_setting("embedding_cache_dir") or "./model_cache"
            dimension_str = get_system_setting("vector_store_dimensions") or "2048"

            logger.debug(f"Loaded embedding settings: provider={provider_type}, model={model}, dim={dimension_str}")

            # Fallback to environment variables if DB settings failed
            import os
            if not provider_type:
                provider_type = os.getenv("EMBEDDING_PROVIDER")

            if not model:
                model = os.getenv("EMBEDDING_MODEL")

            if dimension_str == "2048": # If default was used
                dimension_str = os.getenv("VECTOR_STORE_DIMENSIONS", "2048")
                
            dimension = int(dimension_str)
            
            if not provider_type or provider_type == "disabled":
                logger.debug("Embedding provider disabled - using deterministic fallback")
                self.provider = DeterministicEmbeddingProvider(dimension=dimension)
                return
            
            # Get API key for provider (if needed)
            api_key = None
            if provider_type != "huggingface_local":
                api_key = get_credential_field(provider_type)
                
                # Fallback to environment variables
                if not api_key:
                    import os
                    env_map = {
                        "openai": "OPENAI_API_KEY",
                        "google": "GOOGLE_API_KEY",
                        "cohere": "COHERE_API_KEY",
                        "huggingface_api": "HUGGINGFACE_API_KEY",
                        "openrouter": "OPENROUTER_API_KEY"
                    }
                    env_var = env_map.get(provider_type)
                    if env_var:
                        api_key = os.getenv(env_var)
                
                if not api_key:
                    logger.warning(
                        f"No API key found for {provider_type} embedding provider. "
                        "Using deterministic fallback."
                    )
                    self.provider = DeterministicEmbeddingProvider(dimension=dimension)
                    return
            
            # Create config
            config = EmbeddingConfig(
                provider=EmbeddingProvider(provider_type),
                model=model,
                dimension=dimension,
                api_key=api_key,
                cache_dir=cache_dir
            )
            
            # Create provider
            self.provider = self._create_provider(config)
            logger.debug(
                f"Initialized {provider_type} embedding provider "
                f"(model: {model}, dimension: {dimension})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding provider: {e}. Using fallback.")
            dimension_str = get_system_setting("vector_store_dimensions") or "2048"
            self.provider = DeterministicEmbeddingProvider(dimension=int(dimension_str))
    
    def _create_provider(self, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create embedding provider from config"""
        if config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingProvider(config)
        elif config.provider == EmbeddingProvider.OPENROUTER:
            return OpenRouterEmbeddingProvider(config)
        elif config.provider == EmbeddingProvider.HUGGINGFACE_LOCAL:
            return HuggingFaceLocalEmbeddingProvider(config)
        elif config.provider == EmbeddingProvider.DISABLED:
            return DeterministicEmbeddingProvider(config.dimension)
        else:
            logger.warning(f"Unsupported embedding provider: {config.provider}. Using fallback.")
            return DeterministicEmbeddingProvider(config.dimension)
    
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
        
        return await self.provider.generate_embedding(text)
    
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
            # Read from system settings
            dimension_str = get_system_setting("vector_store_dimensions")
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
