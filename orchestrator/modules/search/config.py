"""
Search Module Configuration
===========================

Configuration for the search module.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


def _get_system_dimension() -> int:
    """Get embedding dimension from system settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(
                SystemSetting.key == "vector_store_dimensions"
            ).first()
            if setting and setting.value:
                return int(setting.value)
        finally:
            db.close()
    except Exception:
        pass
    return 4096  # Fallback if DB unavailable


@dataclass
class SearchConfig:
    """Configuration for the Search module"""
    
    # Database configuration
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    
    # Embedding configuration - reads from system settings
    embedding_dimension: int = field(default_factory=_get_system_dimension)
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
        """Create config from environment variables (prefers system settings)"""
        return cls(
            database_url=os.getenv("DATABASE_URL", ""),
            embedding_dimension=_get_system_dimension(),  # Always from system settings
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            similarity_function=os.getenv("SIMILARITY_FUNCTION", "cosine"),
            vector_table_name=os.getenv("VECTOR_TABLE_NAME", "document_chunks"),
            default_max_results=int(os.getenv("DEFAULT_MAX_RESULTS", "10")),
            default_min_relevance=float(os.getenv("DEFAULT_MIN_RELEVANCE", "0.5")),
            cache_ttl_minutes=int(os.getenv("CACHE_TTL_MINUTES", "30")),
        )

