"""
RAG Module Configuration
========================

Central configuration for RAG module components.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


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


def _get_system_setting_int(key: str, default: int) -> int:
    """Get integer setting from system settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(
                SystemSetting.key == key
            ).first()
            if setting and setting.value:
                return int(setting.value)
        finally:
            db.close()
    except Exception:
        pass
    return default


def _get_system_setting_float(key: str, default: float) -> float:
    """Get float setting from system settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(
                SystemSetting.key == key
            ).first()
            if setting and setting.value:
                return float(setting.value)
        finally:
            db.close()
    except Exception:
        pass
    return default


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TOPIC_BASED = "topic_based"
    STRUCTURAL = "structural"
    ADAPTIVE = "adaptive"


@dataclass
class ChunkingConfig:
    """Configuration for chunking - reads from system settings"""
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE
    target_size: int = field(default_factory=lambda: _get_system_setting_int("chunk_size", 500))
    min_size: int = field(default_factory=lambda: _get_system_setting_int("min_chunk_size", 100))
    max_size: int = field(default_factory=lambda: _get_system_setting_int("max_chunk_size", 1500))
    overlap_ratio: float = 0.1
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings - reads dimension from system settings"""
    provider: str = "huggingface_local"
    model: str = "BAAI/bge-large-en-v1.5"
    dimension: int = field(default_factory=_get_system_dimension)
    batch_size: int = 32
    normalize: bool = True


@dataclass
class RetrievalConfig:
    """Configuration for retrieval - reads from system settings"""
    default_top_k: int = field(default_factory=lambda: _get_system_setting_int("rag_default_top_k", 8))
    max_tokens: int = field(default_factory=lambda: _get_system_setting_int("max_tokens", 2000))
    min_similarity: float = field(default_factory=lambda: _get_system_setting_float("min_similarity", 0.5))
    diversity_factor: float = field(default_factory=lambda: _get_system_setting_float("diversity_factor", 0.3))
    rerank: bool = False
    rerank_model: Optional[str] = None


@dataclass
class IngestionConfig:
    """Configuration for ingestion"""
    batch_size: int = 10
    max_concurrent: int = 3
    skip_patterns: List[str] = field(default_factory=lambda: [
        '__pycache__', 'node_modules', '.git', '.venv', 'venv'
    ])
    supported_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.py', '.js', '.ts', '.java', '.go', '.rs',
        '.yaml', '.yml', '.json', '.xml', '.html', '.css'
    ])


@dataclass
class RAGModuleConfig:
    """
    Master configuration for RAG module.
    
    Usage:
        from modules.rag.config import RAGModuleConfig
        
        config = RAGModuleConfig()
        config.chunking.target_size = 800
        config.retrieval.max_tokens = 3000
    """
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    
    # Feature flags
    enable_caching: bool = True
    enable_reranking: bool = False
    enable_query_expansion: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGModuleConfig":
        """Create config from dictionary"""
        config = cls()
        
        if 'chunking' in config_dict:
            for key, value in config_dict['chunking'].items():
                if hasattr(config.chunking, key):
                    setattr(config.chunking, key, value)
        
        if 'embedding' in config_dict:
            for key, value in config_dict['embedding'].items():
                if hasattr(config.embedding, key):
                    setattr(config.embedding, key, value)
        
        if 'retrieval' in config_dict:
            for key, value in config_dict['retrieval'].items():
                if hasattr(config.retrieval, key):
                    setattr(config.retrieval, key, value)
        
        if 'ingestion' in config_dict:
            for key, value in config_dict['ingestion'].items():
                if hasattr(config.ingestion, key):
                    setattr(config.ingestion, key, value)
        
        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'chunking': {
                'strategy': self.chunking.strategy.value,
                'target_size': self.chunking.target_size,
                'min_size': self.chunking.min_size,
                'max_size': self.chunking.max_size,
                'overlap_ratio': self.chunking.overlap_ratio,
            },
            'embedding': {
                'provider': self.embedding.provider,
                'model': self.embedding.model,
                'dimension': self.embedding.dimension,
            },
            'retrieval': {
                'default_top_k': self.retrieval.default_top_k,
                'max_tokens': self.retrieval.max_tokens,
                'min_similarity': self.retrieval.min_similarity,
                'diversity_factor': self.retrieval.diversity_factor,
            },
            'ingestion': {
                'batch_size': self.ingestion.batch_size,
                'max_concurrent': self.ingestion.max_concurrent,
            }
        }


# Default configuration instance
default_config = RAGModuleConfig()

