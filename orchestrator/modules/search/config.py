"""
Search Module Configuration
===========================

Configuration for the search module.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SearchConfig:
    """Configuration for the Search module"""
    
    # Database configuration
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    
    # Embedding configuration
    embedding_dimension: int = 1024  # Default for text-embedding-3-small
    embedding_model: str = "text-embedding-3-small"
    
    # Vector store configuration
    similarity_function: str = "cosine"  # cosine, l2, inner_product
    vector_table_name: str = "document_chunks"
    
    # Retrieval configuration
    default_max_results: int = 10
    default_min_relevance: float = 0.5
    cache_ttl_minutes: int = 30
    
    # Optimization configuration
    knapsack_enabled: bool = True
    mmr_lambda: float = 0.7  # Balance between relevance and diversity
    
    @classmethod
    def from_env(cls) -> "SearchConfig":
        """Create config from environment variables"""
        return cls(
            database_url=os.getenv("DATABASE_URL", ""),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            similarity_function=os.getenv("SIMILARITY_FUNCTION", "cosine"),
            vector_table_name=os.getenv("VECTOR_TABLE_NAME", "document_chunks"),
            default_max_results=int(os.getenv("DEFAULT_MAX_RESULTS", "10")),
            default_min_relevance=float(os.getenv("DEFAULT_MIN_RELEVANCE", "0.5")),
            cache_ttl_minutes=int(os.getenv("CACHE_TTL_MINUTES", "30")),
        )

