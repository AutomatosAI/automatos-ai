"""
Search Module Test Configuration and Fixtures
==============================================

Pytest configuration and shared fixtures for Search module testing.

PRD-197 S1: fixtures for the deleted F079 zombie layer (SearchService /
SearchConfig / EnhancedVectorStore / VectorDocument) are gone with it. What
remains serves the live surface: the context optimizer and the math
foundations.
"""

import os
import pytest
import numpy as np
from typing import List

TEST_EMBEDDING_DIM = int(os.getenv("VECTOR_STORE_DIMENSIONS", "2048"))

from modules.search import ContextOptimizer, ContextItem

# ``test_db_url``, ``test_engine`` and the transactional ``db_session`` fixture
# come from the root orchestrator/conftest.py (PRD-142 W2-S4).


@pytest.fixture
def context_optimizer():
    """Provide ContextOptimizer instance"""
    return ContextOptimizer()


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing"""
    def _generate(count: int = 10, dimension: int = TEST_EMBEDDING_DIM) -> List[List[float]]:
        """Generate random normalized embeddings"""
        embeddings = []
        for _ in range(count):
            vec = np.random.randn(dimension)
            vec = vec / np.linalg.norm(vec)  # Normalize
            embeddings.append(vec.tolist())
        return embeddings
    return _generate


@pytest.fixture
def sample_context_items(sample_embeddings):
    """Provide sample ContextItems for optimization testing"""
    embeddings = sample_embeddings(count=20)

    items = []
    for i, embedding in enumerate(embeddings):
        item = ContextItem(
            content=f"Context item {i}: " + " ".join([f"word{j}" for j in range(10 + i)]),
            source=f"source-{i % 5}",
            context_type="general",
            relevance_score=0.9 - (i * 0.02),
            embedding=embedding,
            metadata={"index": i}
        )
        items.append(item)

    return items


@pytest.fixture
def diverse_context_items():
    """Provide context items with known diversity characteristics"""
    # Create items with varying content and sources for diversity testing
    items = [
        ContextItem(
            content="Python programming language basics",
            source="python-docs",
            context_type="documentation",
            relevance_score=0.95
        ),
        ContextItem(
            content="JavaScript async/await patterns",
            source="js-guide",
            context_type="documentation",
            relevance_score=0.90
        ),
        ContextItem(
            content="Database indexing strategies",
            source="db-manual",
            context_type="documentation",
            relevance_score=0.85
        ),
        ContextItem(
            content="Python list comprehensions",
            source="python-docs",
            context_type="documentation",
            relevance_score=0.80
        ),
        ContextItem(
            content="React component lifecycle",
            source="react-docs",
            context_type="documentation",
            relevance_score=0.75
        ),
    ]
    return items
