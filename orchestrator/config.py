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
    SQL_DEBUG: bool = os.getenv("SQL_DEBUG", "false").lower() == "true"

    # PRD-70 FIX-05: Enforce SSL for database connections in production.
    # Skipped for local dev (localhost / docker-compose internal hostnames).
    @staticmethod
    def get_database_url() -> str:
        """Return DATABASE_URL with sslmode=require enforced for non-local hosts."""
        url = Config.DATABASE_URL
        if not url:
            return url
        _local_hosts = ("localhost", "127.0.0.1", "postgres", "db")
        if any(h in url for h in _local_hosts):
            return url
        if "sslmode" not in url:
            url += "?sslmode=require" if "?" not in url else "&sslmode=require"
        return url
    
    # =============================================================================
    # REDIS - Caching/PubSub (optional)
    # =============================================================================
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: str = os.getenv("REDIS_PORT")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    REDIS_DB: str = os.getenv("REDIS_DB", "0")
    
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
    # MEMORY — Unified Memory Service (PRD-79)
    # =============================================================================
    # L1 Session: TTL for active sessions (24 hours)
    MEMORY_SESSION_TTL_SECONDS: int = int(os.getenv("MEMORY_SESSION_TTL_SECONDS", "86400"))
    # L1 Session: TTL after end_session() called (1 hour consolidation window)
    MEMORY_SESSION_CONSOLIDATION_TTL_SECONDS: int = int(os.getenv("MEMORY_SESSION_CONSOLIDATION_TTL_SECONDS", "3600"))
    # L3 Cache: TTL for Mem0 search result caching in Redis
    MEMORY_CACHE_TTL_SECONDS: int = int(os.getenv("MEMORY_CACHE_TTL_SECONDS", "300"))
    # Context Router: total token budget for pre-LLM context injection
    CONTEXT_BUDGET_TOKENS: int = int(os.getenv("CONTEXT_BUDGET_TOKENS", "4000"))
    # Context Router: per-source sub-budgets (tokens)
    CONTEXT_BUDGET_SESSION: int = int(os.getenv("CONTEXT_BUDGET_SESSION", "500"))
    CONTEXT_BUDGET_LONG_TERM: int = int(os.getenv("CONTEXT_BUDGET_LONG_TERM", "800"))
    CONTEXT_BUDGET_TEMPORAL: int = int(os.getenv("CONTEXT_BUDGET_TEMPORAL", "600"))
    CONTEXT_BUDGET_DAILY: int = int(os.getenv("CONTEXT_BUDGET_DAILY", "400"))
    CONTEXT_BUDGET_AWARENESS: int = int(os.getenv("CONTEXT_BUDGET_AWARENESS", "200"))
    # Knowledge awareness: TTL for per-workspace capability map cached in Redis
    MEMORY_AWARENESS_CACHE_TTL_SECONDS: int = int(os.getenv("MEMORY_AWARENESS_CACHE_TTL_SECONDS", "600"))
    # L2 Decay: Ebbinghaus decay rate (higher = faster forgetting)
    MEMORY_DECAY_RATE: float = float(os.getenv("MEMORY_DECAY_RATE", "0.1"))
    # L2 Decay: threshold below which items are archived
    MEMORY_DECAY_ARCHIVE_THRESHOLD: float = float(os.getenv("MEMORY_DECAY_ARCHIVE_THRESHOLD", "0.3"))
    # L2 Decay: batch size per workspace (rows per transaction)
    MEMORY_DECAY_BATCH_SIZE: int = int(os.getenv("MEMORY_DECAY_BATCH_SIZE", "100"))
    # L2→L3 Promotion: minimum importance score for promotion candidates
    MEMORY_PROMOTION_MIN_IMPORTANCE: float = float(os.getenv("MEMORY_PROMOTION_MIN_IMPORTANCE", "0.7"))
    # L2→L3 Promotion: minimum access count for promotion candidates
    MEMORY_PROMOTION_MIN_ACCESS_COUNT: int = int(os.getenv("MEMORY_PROMOTION_MIN_ACCESS_COUNT", "3"))
    # L2→L3 Promotion: batch size per workspace
    MEMORY_PROMOTION_BATCH_SIZE: int = int(os.getenv("MEMORY_PROMOTION_BATCH_SIZE", "50"))
    # Background job intervals (PRD-79 US-023)
    MEMORY_CONSOLIDATION_INTERVAL_SECONDS: int = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL_SECONDS", "3600"))
    MEMORY_DECAY_INTERVAL_SECONDS: int = int(os.getenv("MEMORY_DECAY_INTERVAL_SECONDS", "3600"))
    MEMORY_PROMOTION_HOUR_UTC: int = int(os.getenv("MEMORY_PROMOTION_HOUR_UTC", "3"))
    MEMORY_JOBS_ENABLED: bool = os.getenv("MEMORY_JOBS_ENABLED", "true").lower() in ("true", "1", "yes")
    MEMORY_LAYERS_CACHE_TTL_SECONDS: int = int(os.getenv("MEMORY_LAYERS_CACHE_TTL_SECONDS", "60"))

    # =============================================================================
    # API SECURITY
    # =============================================================================
    API_KEY: str = os.getenv("API_KEY")
    ORCHESTRATOR_API_KEY: str = os.getenv("ORCHESTRATOR_API_KEY") or os.getenv("AUTOMATOS_API_KEY") or os.getenv("API_KEY")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
    REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "true").strip().lower() in ("true", "1", "yes")
    AUTH_DEBUG: bool = os.getenv("AUTH_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    
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
    # LLM KEYS (All providers)
    # =============================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    XAI_API_KEY: str = os.getenv("XAI_API_KEY")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY")

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
    # AUTH (Clerk, Workspaces)
    # =============================================================================
    CLERK_SECRET_KEY: str = os.getenv("CLERK_SECRET_KEY")
    CLERK_PUBLISHABLE_KEY: str = os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")
    CLERK_JWKS_URL: str = os.getenv("CLERK_JWKS_URL")
    CLERK_AUDIENCE: str = os.getenv("CLERK_AUDIENCE", "")
    DEFAULT_WORKSPACE_ID: str = os.getenv("DEFAULT_WORKSPACE_ID")
    WORKSPACE_ID: str = os.getenv("WORKSPACE_ID")
    CREDENTIAL_ENCRYPTION_KEY: str = os.getenv("CREDENTIAL_ENCRYPTION_KEY")

    # =============================================================================
    # URLS (Backend, Frontend, API)
    # =============================================================================
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")

    # =============================================================================
    # EXTERNAL SERVICE URLS
    # =============================================================================
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "https://automatos.app")
    COHERE_RERANK_URL: str = os.getenv("COHERE_RERANK_URL", "https://api.cohere.com/v2/rerank")
    RAILWAY_GQL_URL: str = os.getenv("RAILWAY_GQL_URL", "https://backboard.railway.app/graphql/v2")
    INTERNAL_API_HOSTNAME: str = os.getenv("INTERNAL_API_HOSTNAME", "automatos-ai.railway.internal")
    INTERNAL_FRONTEND_HOSTNAME: str = os.getenv("INTERNAL_FRONTEND_HOSTNAME", "automatos-ai-frontend.railway.internal")
    SKILLS_SH_API_URL: str = os.getenv("SKILLS_SH_API_URL", "https://api.skills.sh/v1")
    COMPOSIO_API_KEY: str = os.getenv("COMPOSIO_API_KEY") or os.getenv("COMPOSIO_KEY")
    COMPOSIO_API_BASE_URL: str = os.getenv("COMPOSIO_API_BASE_URL", "https://backend.composio.dev/api/v3")

    # =============================================================================
    # ROUTING (Universal Orchestrator Router)
    # =============================================================================
    COMPOSIO_WEBHOOK_SECRET: str = os.getenv("COMPOSIO_WEBHOOK_SECRET")
    ROUTING_CACHE_TTL_HOURS: int = int(os.getenv("ROUTING_CACHE_TTL_HOURS", "24"))
    ROUTING_LLM_CONFIDENCE_THRESHOLD: float = float(os.getenv("ROUTING_LLM_CONFIDENCE_THRESHOLD", "0.5"))

    # GitHub
    GITHUB_PAT: str = os.getenv("GITHUB", "")  # Personal Access Token for git push
    GITHUB_REPO_OWNER: str = os.getenv("GITHUB_REPO_OWNER", "")
    GITHUB_REPO_NAME: str = os.getenv("GITHUB_REPO_NAME", "")
    GITHUB_DEFAULT_BRANCH: str = os.getenv("GITHUB_DEFAULT_BRANCH", "main")
    GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    GITHUB_WEBHOOK_WORKSPACE_ID: str = os.getenv("GITHUB_WEBHOOK_WORKSPACE_ID") or os.getenv("DEFAULT_WORKSPACE_ID")
    GITHUB_PR_WORKFLOW_NAME: str = os.getenv("GITHUB_PR_WORKFLOW_NAME", "PR Code Review")

    # Webhooks / Widgets
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET")
    WIDGET_TOKEN_SECRET: str = os.getenv("WIDGET_TOKEN_SECRET", "")
    WIDGET_ORIGIN_ALLOWLIST: str = os.getenv("WIDGET_ORIGIN_ALLOWLIST", "")
    PLAYBOOKS_REQUIRE_TENANT: bool = os.getenv("PLAYBOOKS_REQUIRE_TENANT", "0").lower() in ("1", "true", "yes")

    # =============================================================================
    # TASK RUNNER (PRD-56: Infrastructure Scaling & Physical Workspaces)
    # =============================================================================
    TASK_RUNNER_BACKEND: str = os.getenv("TASK_RUNNER_BACKEND", "local")  # local, queued, kubernetes
    WORKSPACE_VOLUME_PATH: str = os.getenv("WORKSPACE_VOLUME_PATH", "/workspaces")
    WORKSPACE_DEFAULT_QUOTA_GB: int = int(os.getenv("WORKSPACE_DEFAULT_QUOTA_GB", "5"))
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "3"))

    # Bounded Concurrency (per-workspace execution limits)
    DEFAULT_MAX_CONCURRENT_TOTAL: int = int(os.getenv("DEFAULT_MAX_CONCURRENT_TOTAL", "3"))
    DEFAULT_MAX_CONCURRENT_RUNNING: int = int(os.getenv("DEFAULT_MAX_CONCURRENT_RUNNING", "3"))
    DEFAULT_MAX_CONCURRENT_PENDING: int = int(os.getenv("DEFAULT_MAX_CONCURRENT_PENDING", "10"))

    WORKER_INTERNAL_URL: str = os.getenv("WORKER_INTERNAL_URL", "http://localhost:8081")
    WORKER_INTERNAL_TOKEN: str = os.getenv("WORKER_INTERNAL_TOKEN", "")

    # Task Reconciliation (Symphony-inspired stall detection)
    TASK_STALL_TIMEOUT_SECONDS: int = int(os.getenv("TASK_STALL_TIMEOUT_SECONDS", "300"))  # 5 min
    TASK_PENDING_TIMEOUT_SECONDS: int = int(os.getenv("TASK_PENDING_TIMEOUT_SECONDS", "120"))  # 2 min
    TASK_MAX_RETRIES: int = int(os.getenv("TASK_MAX_RETRIES", "2"))
    TASK_MAX_RETRY_BACKOFF_MS: int = int(os.getenv("TASK_MAX_RETRY_BACKOFF_MS", "300000"))  # 5 min cap
    TASK_RECONCILE_INTERVAL_SECONDS: int = int(os.getenv("TASK_RECONCILE_INTERVAL_SECONDS", "60"))

    # =============================================================================
    # RAILWAY API (Log retrieval for agents)
    # =============================================================================
    RAILWAY_API_TOKEN: str = os.getenv("RAILWAY_API_TOKEN", "")
    RAILWAY_PROJECT_ID: str = os.getenv("RAILWAY_PROJECT_ID", "")
    RAILWAY_ENVIRONMENT_ID: str = os.getenv("RAILWAY_ENVIRONMENT_ID", "")

    # =============================================================================
    # MONITORING (PRD-73)
    # =============================================================================
    LOKI_URL: str = os.getenv("LOKI_URL", "http://loki.railway.internal:3100")
    PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://prometheus.railway.internal:9090")
    ALERTMANAGER_URL: str = os.getenv("ALERTMANAGER_URL", "http://alertmanager.railway.internal:9093")

    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    ENABLE_BATCH_API: bool = os.getenv("ENABLE_BATCH_API", "false").lower() == "true"
    HEARTBEAT_ENABLED: bool = os.getenv("HEARTBEAT_ENABLED", "true").lower() == "true"
    RECIPE_SCHEDULER_ENABLED: bool = os.getenv("RECIPE_SCHEDULER_ENABLED", "true").lower() == "true"
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
    S3_VECTORS_BUCKET: str = os.getenv("S3_VECTORS_BUCKET")
    S3_VECTORS_INDEX_NAME: str = os.getenv("S3_VECTORS_INDEX_NAME", "documents-index")
    S3_VECTORS_DIMENSION: int = int(os.getenv("S3_VECTORS_DIMENSION", "2048"))
    S3_VECTORS_METRIC: str = os.getenv("S3_VECTORS_METRIC", "cosine")

    # S3 Documents (general storage bucket)
    S3_DOCUMENTS_BUCKET: str = os.getenv("S3_DOCUMENTS_BUCKET", "automatos-ai")
    
    # =============================================================================
    # PRD-58: FutureAGI Integration (Prompt Scoring & Optimization)
    # =============================================================================
    FUTUREAGI_API_KEY: str = os.getenv("FUTUREAGI_API_KEY")
    FUTUREAGI_SECRET_KEY: str = os.getenv("FUTUREAGI_SECRET_KEY")
    FUTUREAGI_ENABLED: bool = os.getenv("FUTUREAGI_ENABLED", "false").lower() == "true"
    AGENT_OPT_WORKER_URL: str = os.getenv("AGENT_OPT_WORKER_URL", "http://agent-opt-worker.railway.internal:8080")

    # =============================================================================
    # JIRA BUG REPORTS (Pilot Helper Widget)
    # =============================================================================
    JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "PILOT")
    JIRA_BUG_REPORTS_ENABLED: bool = os.getenv("JIRA_BUG_REPORTS_ENABLED", "true").lower() == "true"

    # =============================================================================
    # MARKETPLACE / S3 (Plugin Marketplace)
    # =============================================================================
    MARKETPLACE_S3_BUCKET: str = os.getenv("MARKETPLACE_S3_BUCKET", "automatos-marketplace")
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION defined above (AWS S3 Vectors section)
    MARKETPLACE_LOCAL_DIR: str = os.getenv("MARKETPLACE_LOCAL_DIR")
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
    # EMBEDDINGS
    # =============================================================================
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    VECTOR_STORE_DIMENSIONS: int = int(os.getenv("VECTOR_STORE_DIMENSIONS", "2048"))

    # =============================================================================
    # PANDASAI (Data Analysis)
    # =============================================================================
    PANDASAI_API_KEY: str = os.getenv("PANDASAI_API_KEY")
    PANDASAI_MODEL: str = os.getenv("PANDASAI_MODEL")
    PANDASAI_OUTPUT_DIR: str = os.getenv("PANDASAI_OUTPUT_DIR", "/tmp/pandasai_charts")

    # =============================================================================
    # AGENT EXECUTION
    # =============================================================================
    AUTOMATOS_WORKSPACE: str = os.getenv("AUTOMATOS_WORKSPACE", "/tmp/automatos_workspace")
    AUTOMATOS_RESULTS_BASE: str = os.getenv("AUTOMATOS_RESULTS_BASE", "/var/lib/automatos/results")
    AUTOMATOS_DOCUMENTS_DIRS: str = os.getenv("AUTOMATOS_DOCUMENTS_DIRS") or os.getenv("AUTOMATOS_DOCUMENTS_DIR") or os.getenv("DOCUMENTS_DIR")
    IMAGE_STORE_LOCAL_DIR: str = os.getenv("IMAGE_STORE_LOCAL_DIR")
    GOTENBERG_URL: str = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")
    DOCUMENT_STORAGE_DIR: str = os.getenv("DOCUMENT_STORAGE_DIR", "documents")

    # =============================================================================
    # FEATURE FLAGS (additional)
    # =============================================================================
    ENABLE_LLM_QUALITY_ASSESSMENT: bool = os.getenv("ENABLE_LLM_QUALITY_ASSESSMENT", "false").lower() == "true"
    ENABLE_CONTEXT_OPTIMIZATION: bool = os.getenv("ENABLE_CONTEXT_OPTIMIZATION", "false").lower() == "true"
    INJECT_DAILY_LOGS: bool = os.getenv("INJECT_DAILY_LOGS", "true").lower() == "true"
    COMPLEXITY_CACHE_TTL_HOURS: int = int(os.getenv("COMPLEXITY_CACHE_TTL_HOURS", "24"))

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
    
    # =============================================================================
    # LLM ANALYTICS (PRD-54: Model Tiers & Cost Optimization)
    # =============================================================================
    # Comma-separated model IDs considered "premium" (expensive) for cost recommendations
    PREMIUM_MODELS: str = os.getenv(
        "PREMIUM_MODELS",
        "gpt-4o,gpt-4-turbo,claude-sonnet-4-5-20250929,claude-3-5-sonnet-20241022,claude-3-opus-20240229",
    )
    # Comma-separated model IDs suggested as cheaper alternatives
    BUDGET_MODELS: str = os.getenv(
        "BUDGET_MODELS",
        "gpt-4o-mini,claude-haiku-4-5",
    )
    # Estimated savings multiplier when switching from premium to budget model (0.0–1.0)
    PREMIUM_TO_BUDGET_SAVINGS_RATIO: float = float(os.getenv("PREMIUM_TO_BUDGET_SAVINGS_RATIO", "0.85"))

    # =============================================================================
    # VOICE SERVICE (PRD-74)
    # =============================================================================
    VOICE_SERVICE_URL: str = os.getenv("VOICE_SERVICE_URL", "http://voice-service.railway.internal:8300")
    VOICE_SERVICE_TIMEOUT: int = int(os.getenv("VOICE_SERVICE_TIMEOUT", "30"))
    VOICE_STT_MODEL: str = os.getenv("VOICE_STT_MODEL", "Systran/faster-whisper-large-v3")
    VOICE_TTS_MODEL: str = os.getenv("VOICE_TTS_MODEL", "kokoro")
    VOICE_TTS_DEFAULT_VOICE: str = os.getenv("VOICE_TTS_DEFAULT_VOICE", "af_heart")
    VOICE_ENABLED: bool = os.getenv("VOICE_ENABLED", "true").lower() == "true"
    VOICE_MAX_AUDIO_SIZE_MB: int = int(os.getenv("VOICE_MAX_AUDIO_SIZE_MB", "25"))
    VOICE_MAX_DURATION_SECONDS: int = int(os.getenv("VOICE_MAX_DURATION_SECONDS", "120"))
    # Auto agent voice defaults (PRD-74 Phase 2)
    AUTO_VOICE_ID: str = os.getenv("AUTO_VOICE_ID", "auto_default")
    AUTO_VOICE_PROVIDER: str = os.getenv("AUTO_VOICE_PROVIDER", "chatterbox")

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
