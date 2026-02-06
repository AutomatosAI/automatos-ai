"""
Sentence Transformer Embedding Generator Wrapper
=================================================

Wraps SentenceTransformer to match the EmbeddingGenerator interface
expected by DocumentProcessor.
"""

import asyncio
from typing import List
import numpy as np


class SentenceTransformerWrapper:
    """
    Wrapper for SentenceTransformer that provides the generate_embeddings() interface.
    """

    def __init__(self, model):
        """
        Args:
            model: SentenceTransformer model instance
        """
        self.model = model

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        # Run CPU-bound encoding in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, convert_to_numpy=True)
        )

        # Convert to list of lists
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()

        return embeddings
