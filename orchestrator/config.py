"""
Centralized Configuration Management
=====================================

ONLY PLACE where os.getenv() is called for configuration.
All other files import from here.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from uuid import UUID

logger = logging.getLogger(__name__)

# =============================================================================
# SINGLE-TENANT MODE CONSTANTS
# =============================================================================

# Default tenant UUID for single-tenant deployments
DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000000")

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Config:
    """
    Central configuration - ONLY place where os.getenv() is called
    Simple property access, no complex logic
    """
    
    # =============================================================================
    # DATABASE - PostgreSQL (required)
    # =============================================================================
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")
    DATABASE_URL: str = os.getenv("DATABASE_URL")  # If set, overrides individual params
    
    # =============================================================================
    # REDIS - Caching/PubSub (optional)
    # =============================================================================
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: str = os.getenv("REDIS_PORT")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    
    @property
    def REDIS_URL(self) -> str:
        """Get Redis URL from env or construct from parts"""
        url = os.getenv("REDIS_URL")
        if url:
            return url
        
        if self.REDIS_HOST and self.REDIS_PORT:
            auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
            return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        
        return None
    
    # =============================================================================
    # API SECURITY
    # =============================================================================
    API_KEY: str = os.getenv("API_KEY")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
    
    # =============================================================================
    # CORS (Frontend origins)
    # =============================================================================
    # Allow multiple origins (comma-separated) for Railway deployment
    # Default includes localhost for local dev and Railway frontend domain
    # Set CORS_ALLOW_ORIGINS in Railway to include your frontend domain
    # For Railway: https://automotas-ai-frontend-production.up.railway.app
    # For custom domain: https://ui.automatos.app
    _cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000")
    CORS_ALLOW_ORIGINS: str = ",".join([origin.strip() for origin in _cors_origins.split(",") if origin.strip()])
    
    # =============================================================================
    # LLM KEYS (Optional - LLM Manager handles these)
    # =============================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")

    # LLM settings - loaded from database system_settings (NO hardcoded defaults)
    @property
    def LLM_PROVIDER(self) -> str:
        """Get LLM provider from system settings (database) or environment"""
        try:
            from core.llm.manager import get_system_setting
            # Get from database settings, fallback to env var only (NO hardcoded default)
            return get_system_setting("orchestrator_llm", "provider", os.getenv("LLM_PROVIDER"))
        except Exception:
            return os.getenv("LLM_PROVIDER")  # No hardcoded default - must be set in settings or env
    
    @property
    def LLM_MODEL(self) -> str:
        """Get LLM model from system settings (database) or environment"""
        try:
            from core.llm.manager import get_system_setting
            # Get from database settings, fallback to env var only (NO hardcoded default)
            return get_system_setting("orchestrator_llm", "model", os.getenv("LLM_MODEL"))
        except Exception:
            return os.getenv("LLM_MODEL")  # No hardcoded default - must be set in settings or env
    
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # =============================================================================
    # ENVIRONMENT
    # =============================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    # =============================================================================
    # DEPLOYMENT (for scripts)
    # =============================================================================
    DEPLOY_HOST: str = os.getenv("DEPLOY_HOST")
    DEPLOY_PORT: int = int(os.getenv("DEPLOY_PORT", "22"))
    DEPLOY_USER: str = os.getenv("DEPLOY_USER", "root")
    
    # =============================================================================
    # FRONTEND ENV (NextAuth)
    # =============================================================================
    NEXTAUTH_SECRET: str = os.getenv("NEXTAUTH_SECRET")
    NEXTAUTH_URL: str = os.getenv("NEXTAUTH_URL")
    NEXT_PUBLIC_API_URL: str = os.getenv("NEXT_PUBLIC_API_URL")
    
    # =============================================================================
    # ROUTING (Universal Orchestrator Router)
    # =============================================================================
    COMPOSIO_WEBHOOK_SECRET: str = os.getenv("COMPOSIO_WEBHOOK_SECRET")
    ROUTING_CACHE_TTL_HOURS: int = int(os.getenv("ROUTING_CACHE_TTL_HOURS", "24"))
    ROUTING_LLM_CONFIDENCE_THRESHOLD: float = float(os.getenv("ROUTING_LLM_CONFIDENCE_THRESHOLD", "0.5"))

    # GitHub repo used by automated recipes (e.g., Jira Bug Triage → PR)
    GITHUB_REPO_OWNER: str = os.getenv("GITHUB_REPO_OWNER", "")
    GITHUB_REPO_NAME: str = os.getenv("GITHUB_REPO_NAME", "")
    GITHUB_DEFAULT_BRANCH: str = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    ENABLE_BATCH_API: bool = os.getenv("ENABLE_BATCH_API", "false").lower() == "true"
    HEARTBEAT_ENABLED: bool = os.getenv("HEARTBEAT_ENABLED", "true").lower() == "true"
    CHANNELS_ENABLED: bool = os.getenv("CHANNELS_ENABLED", "true").lower() == "true"
    SEMANTIC_TOOL_ROUTING: bool = os.getenv("SEMANTIC_TOOL_ROUTING", "true").lower() == "true"

    # =============================================================================
    # AWS S3 VECTORS (PRD-42: Cloud Document Sync)
    # =============================================================================
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")

    # S3 Vectors Configuration
    S3_VECTORS_ENABLED: bool = os.getenv("S3_VECTORS_ENABLED", "false").lower() == "true"
    S3_VECTORS_BUCKET: str = os.getenv("S3_VECTORS_BUCKET")  # e.g., "automatos-ai" or "automatos-vectors-{workspace_id}"
    S3_VECTORS_INDEX_NAME: str = os.getenv("S3_VECTORS_INDEX_NAME", "documents-index")
    S3_VECTORS_DIMENSION: int = int(os.getenv("S3_VECTORS_DIMENSION", "2048"))
    S3_VECTORS_METRIC: str = os.getenv("S3_VECTORS_METRIC", "cosine")
    
    # =============================================================================
    # PRD-58: FutureAGI Integration (Prompt Scoring & Optimization)
    # =============================================================================
    FUTUREAGI_API_KEY: str = os.getenv("FUTUREAGI_API_KEY")
    FUTUREAGI_SECRET_KEY: str = os.getenv("FUTUREAGI_SECRET_KEY")
    FUTUREAGI_ENABLED: bool = os.getenv("FUTUREAGI_ENABLED", "false").lower() == "true"

    # =============================================================================
    # JIRA BUG REPORTS (Pilot Helper Widget)
    # =============================================================================
    JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "PILOT")
    JIRA_BUG_REPORTS_ENABLED: bool = os.getenv("JIRA_BUG_REPORTS_ENABLED", "true").lower() == "true"

    # =============================================================================
    # MARKETPLACE / S3 (Plugin Marketplace)
    # =============================================================================
    MARKETPLACE_S3_BUCKET: str = os.getenv("MARKETPLACE_S3_BUCKET", "automatos-marketplace")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    PLUGIN_MAX_UPLOAD_SIZE_MB: int = int(os.getenv("PLUGIN_MAX_UPLOAD_SIZE_MB", "10"))
    PLUGIN_LLM_SCAN_MODEL: str = os.getenv("PLUGIN_LLM_SCAN_MODEL", "claude-haiku-4-20250414")
    PLUGIN_CACHE_TTL_SECONDS: int = int(os.getenv("PLUGIN_CACHE_TTL_SECONDS", "3600"))

    # =============================================================================
    # RECIPE EXECUTION (Scratchpad, Logs, Memory)
    # =============================================================================
    RECIPE_SCRATCHPAD_TTL: int = int(os.getenv("RECIPE_SCRATCHPAD_TTL", "3600"))
    RECIPE_LOG_RETENTION_DAYS: int = int(os.getenv("RECIPE_LOG_RETENTION_DAYS", "30"))
    RECIPE_LOG_S3_BUCKET: str = os.getenv("RECIPE_LOG_S3_BUCKET", "automatos-ai")
    MEM0_API_URL: str = os.getenv("MEM0_API_URL", "http://automatos-mem0-server.railway.internal")
    MEM0_API_KEY: str = os.getenv("MEM0_API_KEY")

    # =============================================================================
    # RAG / KNOWLEDGE SERVICES API
    # =============================================================================
    KNOWLEDGE_API_BASE_URL: str = os.getenv("KNOWLEDGE_API_BASE_URL", "http://127.0.0.1:8000")
    
    # =============================================================================
    # RAG SETTINGS - Centralized similarity thresholds (loaded from system_settings)
    # =============================================================================
    @property
    def RAG_MIN_SIMILARITY(self) -> float:
        """Get min similarity threshold from system settings (default: 0.65)"""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "min_similarity", "0.65")
            return float(val) if val else 0.65
        except Exception:
            return float(os.getenv("RAG_MIN_SIMILARITY", "0.65"))
    
    @property
    def RAG_TOP_K(self) -> int:
        """Get default top_k results from system settings (default: 5)"""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "top_k", "5")
            return int(val) if val else 5
        except Exception:
            return int(os.getenv("RAG_TOP_K", "5"))
    
    @property
    def RAG_RERANK_ENABLED(self) -> bool:
        """Get rerank enabled from system settings (default: False)"""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "rerank_enabled", "false")
            return str(val).lower() == "true" if val else False
        except Exception:
            return os.getenv("RAG_RERANK_ENABLED", "false").lower() == "true"
    
    def validate(self) -> bool:
        """
        Validate required configuration
        Returns True if all required config is present
        """
        errors = []
        
        # Check database
        if not all([self.POSTGRES_DB, self.POSTGRES_USER, self.POSTGRES_HOST, self.POSTGRES_PORT]):
            errors.append("Database not configured: POSTGRES_DB, POSTGRES_USER, POSTGRES_HOST, POSTGRES_PORT required")
        
        # Check API key
        if self.REQUIRE_API_KEY and not self.API_KEY:
            errors.append("API_KEY required when REQUIRE_API_KEY=true")
        
        # Print errors if any
        if errors:
            logger.error("❌ Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True
    
    def print_config(self, show_secrets: bool = False) -> None:
        """
        Print current configuration (masks secrets by default)
        """
        print("=" * 60)
        print("AUTOMATOS AI CONFIGURATION")
        print("=" * 60)
        
        print(f"Environment: {self.ENVIRONMENT}")
        print(f"Database: {self.POSTGRES_DB}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}")
        print(f"Redis: {self.REDIS_HOST}:{self.REDIS_PORT if self.REDIS_HOST else 'Not configured'}")
        print(f"LLM Provider: {self.LLM_PROVIDER} ({self.LLM_MODEL})")
        
        if show_secrets:
            print(f"OpenAI Key: {self.OPENAI_API_KEY[:10]}..." if self.OPENAI_API_KEY else "Not set")
            print(f"Anthropic Key: {self.ANTHROPIC_API_KEY[:10]}..." if self.ANTHROPIC_API_KEY else "Not set")
            print(f"API Key: {self.API_KEY[:10]}..." if self.API_KEY else "Not set")
        else:
            print(f"OpenAI Key: {'✅ Set' if self.OPENAI_API_KEY else '❌ Not set'}")
            print(f"Anthropic Key: {'✅ Set' if self.ANTHROPIC_API_KEY else '❌ Not set'}")
            print(f"API Key: {'✅ Set' if self.API_KEY else '❌ Not set'}")
        
        print(f"API Key Required: {self.REQUIRE_API_KEY}")
        print("=" * 60)

# Singleton instance
config = Config()

# Backward compatibility alias
orchestrator_config = config

# Validate on import (non-blocking)
# if not config.validate():
#     logger.warning("⚠️  WARNING: Configuration validation failed")

# Export for easy import
__all__ = ['config', 'Config', 'orchestrator_config']
