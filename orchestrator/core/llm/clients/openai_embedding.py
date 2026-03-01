"""
OpenAI Embedding Provider Implementation
========================================

OpenAI embedding models provider (extends OpenAI client).
"""

import logging
from typing import List

from config import config
from .base import BaseEmbeddingProvider, EmbeddingConfig

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider implementation"""
    
    def _initialize_client(self):
        if AsyncOpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        api_key = self.config.api_key or config.OPENAI_API_KEY
        
        if not api_key:
            logger.warning(
                "OpenAI API key not configured. "
                "Embedding features will fail until key is added."
            )
            self.client = None
        else:
            client_kwargs = {"api_key": api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            
            self.client = AsyncOpenAI(**client_kwargs)
            logger.info(
                f"Initialized OpenAI embedding client with model: {self.config.model} "
                f"({self.config.dimension}d)"
            )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        if self.client is None:
            raise ValueError(
                "OpenAI API key not configured. Cannot generate embedding. "
                "Please configure embedding provider in Settings > General"
            )
        
        try:
            # Truncate text if too long
            max_length = 8000  # OpenAI limit
            if len(text) > max_length:
                text = text[:max_length]
                logger.warning(f"Text truncated to {max_length} characters for embedding")
            
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Return embedding dimension based on model"""
        # Map OpenAI models to their dimensions
        if "3-large" in self.config.model:
            return 3072
        elif "3-small" in self.config.model:
            return 1536
        elif "ada-002" in self.config.model:
            return 1536
        else:
            # Return configured dimension or default
            return self.config.dimension
