"""
Search Module Test Configuration and Fixtures
==============================================

Pytest configuration and shared fixtures for Search module testing.
"""

import os
import pytest
import numpy as np
from datetime import datetime
from typing import List

TEST_EMBEDDING_DIM = int(os.getenv("VECTOR_STORE_DIMENSIONS", "2048"))

from modules.search import (
    SearchService, SearchConfig,
    ContextOptimizer, ContextItem,
    EnhancedVectorStore, VectorDocument, SearchMode, RankingStrategy
)

# ``test_db_url``, ``test_engine`` and the transactional ``db_session`` fixture
# come from the root orchestrator/conftest.py (PRD-142 W2-S4).


@pytest.fixture
async def vector_store(test_db_url):
    """Provide EnhancedVectorStore instance"""
    store = EnhancedVectorStore(
        database_url=test_db_url,
        embedding_dimension=TEST_EMBEDDING_DIM,
        similarity_function="cosine",
        table_name="test_vector_documents"
    )
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def context_optimizer():
    """Provide ContextOptimizer instance"""
    return ContextOptimizer()


@pytest.fixture
def search_config(test_db_url):
    """Provide SearchConfig for testing"""
    return SearchConfig(
        database_url=test_db_url,
        embedding_dimension=TEST_EMBEDDING_DIM,
        default_max_results=10,
        mmr_lambda=0.7
    )


@pytest.fixture
async def search_service(search_config):
    """Provide SearchService instance"""
    service = SearchService(config=search_config)
    yield service
    await service.close()


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
def sample_documents(sample_embeddings):
    """Provide sample VectorDocuments for testing"""
    embeddings = sample_embeddings(count=10)
    
    documents = []
    for i, embedding in enumerate(embeddings):
        doc = VectorDocument(
            id=f"test-doc-{i}",
            content=f"This is test document {i} with some content about topic {i % 3}",
            embedding=embedding,
            metadata={"topic": i % 3, "category": "test"},
            timestamp=datetime.now(),
            source=f"source-{i % 3}",
            document_type="general",
            importance_score=0.5 + (i * 0.05)
        )
        documents.append(doc)
    
    return documents


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


@pytest.fixture
def cleanup_vector_store():
    """Clean up test vector store after tests"""
    yield
    # Cleanup logic here if needed


# ---------------------------------------------------------------------------
# test_math_foundations.py — broken against a non-existent core.math API (F056)
# ---------------------------------------------------------------------------
#
# These tests were authored against a math API that this codebase never
# shipped and had NEVER run before F056 (W12-S2) widened testpaths and first
# collected this orphaned tree. They call methods that do not exist on the real
# core.math classes, e.g.:
#
#   test calls                         real core.math API
#   InformationTheory.entropy(probs)   InformationTheory.calculate_entropy(text)
#   InformationTheory.kl_divergence()  (no such method)
#   InformationTheory.mutual_information()  calculate_mutual_information(t1, t2)
#   VectorOperations.euclidean_distance()   DistanceMetrics.euclidean_distance()
#   VectorOperations.dot_product()     (no such method)
#   VectorOperations.normalize()       VectorOperations.normalize_vector()
#   OptimizationAlgorithms.knapsack()  gradient_descent / simulated_annealing
#   DistanceMetrics.hamming_distance() (no such method)
#   DistanceMetrics.jaccard_similarity() (no such method)
#
# They fail with AttributeError, not because of any missing service — the real
# classes work (the four tests that hit real methods, cosine_similarity* and
# manhattan_distance, pass and KEEP RUNNING). This is a genuine test/impl drift
# that needs a rewrite-to-the-real-API decision (whose expected values? which
# algorithms are actually wanted?) — an authorship call, not a CI fix. Per the
# repo's "surface, don't paper over" rule it is skipped with this honest reason
# and flagged for a human decision, never silently deleted, xfailed, or rewritten
# to invented expectations.
_BROKEN_FICTIONAL_API_TESTS = frozenset({
    # TestInformationTheory
    "test_entropy_uniform_distribution",
    "test_entropy_certain_distribution",
    "test_entropy_skewed_distribution",
    "test_kl_divergence_identical",
    "test_kl_divergence_different",
    "test_mutual_information_independent",
    "test_mutual_information_identical",
    # TestVectorOperations
    "test_euclidean_distance_identical",
    "test_euclidean_distance_known",
    "test_dot_product",
    "test_normalize_vector",
    # TestOptimizationAlgorithms
    "test_knapsack_simple",
    "test_knapsack_all_fit",
    "test_knapsack_none_fit",
    "test_knapsack_with_max_items",
    # TestDistanceMetrics
    "test_hamming_distance",
    "test_jaccard_similarity",
})


def pytest_collection_modifyitems(config, items):
    """Skip the test_math_foundations.py tests written against a non-existent
    core.math API (see the note above). Leaves the real-API tests running."""
    skip_marker = pytest.mark.skip(
        reason="test written against a non-existent core.math API (AttributeError, "
        "not a missing service); orphaned tree first collected by F056 — needs a "
        "rewrite-to-real-API decision, see modules/search/tests/conftest.py"
    )
    for item in items:
        node = item.nodeid.replace("\\", "/")
        if "modules/search/tests/test_math_foundations.py" in node and item.name in _BROKEN_FICTIONAL_API_TESTS:
            item.add_marker(skip_marker)
