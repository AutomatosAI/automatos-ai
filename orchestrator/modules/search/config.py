"""
Search Module Configuration
===========================

Configuration for the search module.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from config import config as app_config


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
    return 2048  # Fallback if DB unavailable


@dataclass
class SearchConfig:
    """Configuration for the Search module"""

    # Database configuration
    database_url: str = field(default_factory=lambda: app_config.DATABASE_URL or "")

    # Embedding configuration - reads from system settings (no hardcoded defaults)
    embedding_dimension: int = field(default_factory=_get_system_dimension)
    embedding_model: str = field(default_factory=lambda: app_config.EMBEDDING_MODEL or "")

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
        """Create config from app config (prefers system settings)"""
        return cls(
            database_url=app_config.DATABASE_URL or "",
            embedding_dimension=_get_system_dimension(),
            embedding_model=app_config.EMBEDDING_MODEL or "",
        )

