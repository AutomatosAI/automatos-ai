"""
Search Module Test Configuration and Fixtures
==============================================

Pytest configuration and shared fixtures for Search module testing.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from modules.search import (
    SearchService, SearchConfig,
    ContextOptimizer, ContextItem,
    EnhancedVectorStore, VectorDocument, SearchMode, RankingStrategy
)


# Test database URL
TEST_DB_URL = "postgresql://postgres:secure_password_123@127.0.0.1:5432/orchestrator_db"


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DB_URL)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine):
    """Provide a transactional database session for each test"""
    connection = test_engine.connect()
    transaction = connection.begin()
    
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
async def vector_store():
    """Provide EnhancedVectorStore instance"""
    store = EnhancedVectorStore(
        database_url=TEST_DB_URL,
        embedding_dimension=1024,
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
def search_config():
    """Provide SearchConfig for testing"""
    return SearchConfig(
        database_url=TEST_DB_URL,
        embedding_dimension=1024,
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
    def _generate(count: int = 10, dimension: int = 1024) -> List[List[float]]:
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
