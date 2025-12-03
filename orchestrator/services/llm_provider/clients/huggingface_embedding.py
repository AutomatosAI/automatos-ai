"""
HuggingFace Local Embedding Provider Implementation
==================================================

Local HuggingFace models via sentence-transformers (FREE!).
"""

import logging
from typing import List

from .base import BaseEmbeddingProvider, EmbeddingConfig

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class HuggingFaceLocalEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace local embedding provider (sentence-transformers)"""
    
    def _initialize_client(self):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        try:
            logger.info(
                f"Loading HuggingFace model: {self.config.model} "
                f"(cache: {self.config.cache_dir})"
            )
            self.client = SentenceTransformer(
                self.config.model,
                cache_folder=self.config.cache_dir
            )
            logger.info(
                f"Initialized HuggingFace embedding model: {self.config.model} "
                f"({self.get_dimension()}d)"
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using local HuggingFace model"""
        if self.client is None:
            raise ValueError("HuggingFace model not initialized")
        
        try:
            # Run in executor since sentence-transformers is sync
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.client.encode, text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        if self.client is not None:
            # Get actual dimension from loaded model
            return self.client.get_sentence_embedding_dimension()
        
        # Map common models to dimensions (fallback when model not loaded)
        dimension_map = {
            # Short names
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "bge-large-en-v1.5": 1024,
            "bge-m3": 1024,
            "e5-large-v2": 1024,
            "gte-large-en-v1.5": 1024,
            "nomic-embed-text-v1.5": 768,
            # Full HuggingFace paths
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "intfloat/e5-large-v2": 1024,
            "Alibaba-NLP/gte-large-en-v1.5": 1024,
            "nomic-ai/nomic-embed-text-v1.5": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        return dimension_map.get(self.config.model, self.config.dimension)
