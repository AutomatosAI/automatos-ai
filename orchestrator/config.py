"""
Centralized Configuration Management
=====================================

ONLY PLACE where os.getenv() is called for configuration.
All other files import from here.
"""

import copy
import json
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
    # L3 Cache: TTL for durable-store search result caching in Redis
    MEMORY_CACHE_TTL_SECONDS: int = int(os.getenv("MEMORY_CACHE_TTL_SECONDS", "300"))
    # Context Router: per-source sub-budgets (tokens).
    # Fallback only — used when the model context window is unknown. When the
    # window is known, ContextRouter._compute_budgets derives budgets as a
    # proportion of the usable window instead (PRD-141 US-011).
    CONTEXT_BUDGET_SESSION: int = int(os.getenv("CONTEXT_BUDGET_SESSION", "500"))
    CONTEXT_BUDGET_LONG_TERM: int = int(os.getenv("CONTEXT_BUDGET_LONG_TERM", "800"))
    CONTEXT_BUDGET_TEMPORAL: int = int(os.getenv("CONTEXT_BUDGET_TEMPORAL", "600"))
    CONTEXT_BUDGET_DAILY: int = int(os.getenv("CONTEXT_BUDGET_DAILY", "400"))
    CONTEXT_BUDGET_AWARENESS: int = int(os.getenv("CONTEXT_BUDGET_AWARENESS", "200"))
    CONTEXT_BUDGET_TOOLS: int = int(os.getenv("CONTEXT_BUDGET_TOOLS", "1000"))
    CONTEXT_BUDGET_SYSTEM_PROMPT: int = int(os.getenv("CONTEXT_BUDGET_SYSTEM_PROMPT", "600"))
    # Knowledge awareness: TTL for per-workspace capability map cached in Redis
    MEMORY_AWARENESS_CACHE_TTL_SECONDS: int = int(os.getenv("MEMORY_AWARENESS_CACHE_TTL_SECONDS", "600"))
    # L2 Decay: Ebbinghaus decay rate per hour (higher = faster forgetting).
    # Week-scale (PRD-154 S3): 0.1/hr archived importance-0.8 memories in ~15h;
    # 0.004/hr keeps them above the 0.3 threshold for ~16 days.
    MEMORY_DECAY_RATE: float = float(os.getenv("MEMORY_DECAY_RATE", "0.004"))
    # L2 Decay: threshold below which items are archived
    MEMORY_DECAY_ARCHIVE_THRESHOLD: float = float(os.getenv("MEMORY_DECAY_ARCHIVE_THRESHOLD", "0.3"))
    # L2 Decay: batch size per workspace (rows per transaction)
    MEMORY_DECAY_BATCH_SIZE: int = int(os.getenv("MEMORY_DECAY_BATCH_SIZE", "100"))
    # L2→L3 Promotion (PRD-187 S4): fires on distilled IMPORTANCE with
    # type-aware thresholds — the old `AND access_count > N` conjunct was a
    # bootstrap deadlock (promotion needs access → access needs recall → recall
    # couldn't match) and produced zero promotions ever. Policy lives in
    # modules/memory/promotion_policy.py. Field→durable promotion keeps ITS
    # access gate (FIELD_PROMOTION_MIN_ACCESS_COUNT) — there access is real.
    MEMORY_PROMOTION_MIN_IMPORTANCE: float = float(os.getenv("MEMORY_PROMOTION_MIN_IMPORTANCE", "0.7"))
    # Types durable memory exists for — promoted from a lower importance bar.
    MEMORY_PROMOTION_HIGH_SIGNAL_TYPES: str = os.getenv(
        "MEMORY_PROMOTION_HIGH_SIGNAL_TYPES", "user_fact,preference,procedure"
    )
    MEMORY_PROMOTION_HIGH_SIGNAL_MIN_IMPORTANCE: float = float(
        os.getenv("MEMORY_PROMOTION_HIGH_SIGNAL_MIN_IMPORTANCE", "0.5")
    )
    # L2→L3 Promotion: batch size per workspace
    MEMORY_PROMOTION_BATCH_SIZE: int = int(os.getenv("MEMORY_PROMOTION_BATCH_SIZE", "50"))
    # Background job intervals (PRD-79 US-023)
    MEMORY_CONSOLIDATION_INTERVAL_SECONDS: int = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL_SECONDS", "3600"))
    MEMORY_DECAY_INTERVAL_SECONDS: int = int(os.getenv("MEMORY_DECAY_INTERVAL_SECONDS", "3600"))
    MEMORY_PROMOTION_HOUR_UTC: int = int(os.getenv("MEMORY_PROMOTION_HOUR_UTC", "3"))
    MEMORY_JOBS_ENABLED: bool = os.getenv("MEMORY_JOBS_ENABLED", "true").lower() in ("true", "1", "yes")
    MEMORY_LAYERS_CACHE_TTL_SECONDS: int = int(os.getenv("MEMORY_LAYERS_CACHE_TTL_SECONDS", "60"))
    # Graphify archival (PRD-131d Phase 4): monthly job that folds aged L2+L3
    # memories into the workspace business knowledge graph, then purges sources.
    MEMORY_ARCHIVAL_ENABLED: bool = os.getenv("MEMORY_ARCHIVAL_ENABLED", "true").lower() in ("true", "1", "yes")
    MEMORY_ARCHIVAL_CRON_DAY: int = int(os.getenv("MEMORY_ARCHIVAL_CRON_DAY", "1"))
    MEMORY_ARCHIVAL_CRON_HOUR: int = int(os.getenv("MEMORY_ARCHIVAL_CRON_HOUR", "3"))
    MEMORY_ARCHIVAL_L2_DECAY_THRESHOLD: float = float(os.getenv("MEMORY_ARCHIVAL_L2_DECAY_THRESHOLD", "0.2"))
    MEMORY_ARCHIVAL_L3_RETENTION_DAYS: int = int(os.getenv("MEMORY_ARCHIVAL_L3_RETENTION_DAYS", "180"))
    MEMORY_ARCHIVAL_BATCH_SIZE: int = int(os.getenv("MEMORY_ARCHIVAL_BATCH_SIZE", "500"))
    # PRD-197 S3: Qdrant memory snapshots (durable_memory + field_memory) — the
    # memory planes' DR arm; the document plane is S3 Vectors (PRD-186's DR).
    # Built to the §8-Q3 proposal: daily, 7-day retention, the platform object
    # store. Hour 4 UTC = after the 03:00 L2→L3 promotion, so the snapshot
    # includes the night's promotions. Empty bucket = S3_DOCUMENTS_BUCKET at
    # run time (that attr is defined later in this file).
    MEMORY_SNAPSHOT_ENABLED: bool = os.getenv("MEMORY_SNAPSHOT_ENABLED", "true").lower() in ("true", "1", "yes")
    MEMORY_SNAPSHOT_CRON_HOUR_UTC: int = int(os.getenv("MEMORY_SNAPSHOT_CRON_HOUR_UTC", "4"))
    MEMORY_SNAPSHOT_RETENTION_DAYS: int = int(os.getenv("MEMORY_SNAPSHOT_RETENTION_DAYS", "7"))
    MEMORY_SNAPSHOT_S3_BUCKET: str = os.getenv("MEMORY_SNAPSHOT_S3_BUCKET", "")
    MEMORY_SNAPSHOT_S3_PREFIX: str = os.getenv("MEMORY_SNAPSHOT_S3_PREFIX", "qdrant-snapshots")
    # PRD-197 S4: substrate telemetry retention — the per-seam retrieval
    # metric rows behind the Command Center substrate tile are pruned past
    # this window (the heartbeat_results 148k-row lesson: no unbounded
    # telemetry tables). Sweep rides the memory-jobs scheduler daily.
    SUBSTRATE_METRICS_RETENTION_DAYS: int = int(os.getenv("SUBSTRATE_METRICS_RETENTION_DAYS", "14"))
    SUBSTRATE_METRICS_PRUNE_INTERVAL_SECONDS: int = int(os.getenv("SUBSTRATE_METRICS_PRUNE_INTERVAL_SECONDS", "86400"))
    # PRD-206 S7: composite recall ranking (semantic × recency × importance ×
    # pin), applied ABOVE the relevance floor + type exclusions. Conservative
    # defaults; the S10 continuity eval slice is the referee.
    MEMORY_RANK_ENABLED: bool = os.getenv("MEMORY_RANK_ENABLED", "true").lower() in ("true", "1", "yes")
    MEMORY_RANK_HALF_LIFE_DAYS: float = float(os.getenv("MEMORY_RANK_HALF_LIFE_DAYS", "30"))
    MEMORY_RANK_PIN_BOOST: float = float(os.getenv("MEMORY_RANK_PIN_BOOST", "2.0"))
    # PRD-206 S2: thread-checkpoint sweep — recently-idle chats get an LLM
    # checkpoint (chats.summary + typed decision/open_loop memories). The
    # batch cap bounds LLM spend per sweep; min-messages skips trivia.
    THREAD_CHECKPOINT_ENABLED: bool = os.getenv("THREAD_CHECKPOINT_ENABLED", "true").lower() in ("true", "1", "yes")
    THREAD_CHECKPOINT_SWEEP_INTERVAL_SECONDS: int = int(os.getenv("THREAD_CHECKPOINT_SWEEP_INTERVAL_SECONDS", "900"))
    THREAD_CHECKPOINT_IDLE_MINUTES: int = int(os.getenv("THREAD_CHECKPOINT_IDLE_MINUTES", "30"))
    THREAD_CHECKPOINT_LOOKBACK_HOURS: int = int(os.getenv("THREAD_CHECKPOINT_LOOKBACK_HOURS", "48"))
    THREAD_CHECKPOINT_BATCH: int = int(os.getenv("THREAD_CHECKPOINT_BATCH", "10"))
    THREAD_CHECKPOINT_MIN_MESSAGES: int = int(os.getenv("THREAD_CHECKPOINT_MIN_MESSAGES", "4"))

    # =============================================================================
    # BOOT REAPER — orphaned in-flight runs (PRD-142 Wave 1 · WS-C · W1-S6)
    # =============================================================================
    # On restart, in-flight rows whose background executor died with the old
    # process are stranded forever (a board task stuck 'in_progress', a wizard
    # profile stuck 'scraping', a workflow execution stuck 'running'). The reaper
    # sweeps them once per deploy under the boot leader lock.
    BOOT_REAPER_ENABLED: bool = os.getenv("BOOT_REAPER_ENABLED", "true").lower() in ("true", "1", "yes")
    # A row counts as orphaned only after it has been in-flight this long. Must
    # exceed the slowest legitimate job (the wizard scrape runs ~10–20 min) so a
    # live run is never reaped out from under itself.
    BOOT_REAPER_STALE_MINUTES: int = int(os.getenv("BOOT_REAPER_STALE_MINUTES", "30"))

    # =============================================================================
    # API SECURITY
    # =============================================================================
    ORCHESTRATOR_API_KEY: str = os.getenv("ORCHESTRATOR_API_KEY") or os.getenv("AUTOMATOS_API_KEY") or os.getenv("API_KEY")

    # PRD-175 (F008) — the open-core edition flag. One core, two editions, one seam.
    #   saas  → Clerk is the identity boundary (the running product; the default).
    #   local → a single auto-authenticated local user in a single local workspace;
    #           no login, no external SaaS, no Clerk env (git clone && docker up).
    # An unknown value falls back to `saas` so a typo never silently un-guards auth.
    # This is the ONE flag the frontend mount-gate and the backend local-session
    # posture both read (mirrored to the client as NEXT_PUBLIC_AUTH_EDITION).
    _AUTH_EDITION_RAW = (os.getenv("AUTH_EDITION", "saas") or "saas").strip().lower()
    AUTH_EDITION: str = _AUTH_EDITION_RAW if _AUTH_EDITION_RAW in ("local", "saas") else "saas"

    # The `local` edition *implies* the no-login posture: REQUIRE_AUTH is forced
    # false so operators set ONE flag, not three that can silently contradict
    # (PRD §4.1/§4.3). In `saas`, REQUIRE_AUTH stays secure-by-default from env.
    REQUIRE_AUTH: bool = (
        False if AUTH_EDITION == "local"
        else os.getenv("REQUIRE_AUTH", "true").strip().lower() in ("true", "1", "yes")
    )
    AUTH_DEBUG: bool = os.getenv("AUTH_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

    # PRD-175 (F075) — the platform-staff email domain used by the Clerk
    # defence-in-depth admin gate (core/auth/clerk.py). Configuration, not a
    # baked-in literal, so a self-hosted/SaaS operator sets their own staff domain.
    PLATFORM_STAFF_EMAIL_DOMAIN: str = (
        os.getenv("PLATFORM_STAFF_EMAIL_DOMAIN", "automatos.app") or "automatos.app"
    ).strip().lstrip("@").lower()
    
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

    # LLM settings - loaded from database system_settings
    # Fallback defaults come from core.llm.defaults (single source of truth)
    @property
    def LLM_PROVIDER(self) -> str:
        """Get LLM provider from system settings (database) or environment."""
        from core.llm.defaults import DEFAULT_LLM_PROVIDER
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting(
                "orchestrator_llm", "provider",
                os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
            )
        except Exception:
            return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)

    @property
    def LLM_MODEL(self) -> str:
        """Get LLM model from system settings (database) or environment."""
        from core.llm.defaults import DEFAULT_LLM_MODEL
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting(
                "orchestrator_llm", "model",
                os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            )
        except Exception:
            return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    
    @property
    def BLOG_COVER_MODEL(self) -> str:
        """
        Image-gen model used by platform_generate_cover_image to produce blog
        covers. Resolves from system_settings (category=content_creation,
        key=blog_cover_model) → env BLOG_COVER_MODEL → DEFAULT_IMAGE_GEN_MODEL.
        Operators change this per-deployment without code changes.
        """
        from core.llm.defaults import DEFAULT_IMAGE_GEN_MODEL
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting(
                "content_creation", "blog_cover_model",
                os.getenv("BLOG_COVER_MODEL", DEFAULT_IMAGE_GEN_MODEL),
            )
        except Exception:
            return os.getenv("BLOG_COVER_MODEL", DEFAULT_IMAGE_GEN_MODEL)

    @property
    def PLANNER_MODEL(self) -> str:
        """Planner model — resolves to System LLM tier (PRD-136)."""
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting("system_llm", "model", os.getenv("PLANNER_MODEL"))
        except Exception:
            return os.getenv("PLANNER_MODEL")

    @property
    def PLANNER_MAX_TOKENS(self) -> int:
        """Planner max_tokens — canonical System LLM max_tokens (PRD-136)."""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("system_llm", "max_tokens", "8000")
            return int(val)
        except Exception:
            return 8000

    @property
    def GRAPHIFY_MODEL(self) -> str:
        """Knowledge-graph extraction model — System LLM tier (PRD-136)."""
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting("system_llm", "model", os.getenv("GRAPHIFY_MODEL"))
        except Exception:
            return os.getenv("GRAPHIFY_MODEL")

    @property
    def MEMORY_DISTILL_MODEL(self) -> str:
        """Cheap-tier model for L3 memory distillation (PRD-159 D11/Q16).

        The distiller runs ~1×/chat turn, so it is deliberately pinned to a cheap
        model rather than the conversation tier. Resolves system_settings
        (memory.distill_model) → env MEMORY_DISTILL_MODEL → DEFAULT_LLM_MODEL
        (already a fast/cheap flash tier)."""
        from core.llm.defaults import DEFAULT_LLM_MODEL
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting(
                "memory", "distill_model",
                os.getenv("MEMORY_DISTILL_MODEL", DEFAULT_LLM_MODEL),
            )
        except Exception:
            return os.getenv("MEMORY_DISTILL_MODEL", DEFAULT_LLM_MODEL)

    @property
    def MEMORY_RELEVANCE_FLOOR(self) -> float:
        """Server-side similarity floor for L3 recall (PRD-159 S3).

        Scored search results below this are never injected, so low-relevance
        junk can't leak into context. Resolves system_settings
        (memory.relevance_floor) → env MEMORY_RELEVANCE_FLOOR → 0.3."""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting(
                "memory", "relevance_floor",
                os.getenv("MEMORY_RELEVANCE_FLOOR", "0.3"),
            )
            return float(val)
        except Exception:
            try:
                return float(os.getenv("MEMORY_RELEVANCE_FLOOR", "0.3"))
            except (TypeError, ValueError):
                return 0.3

    @property
    def COORDINATOR_TASK_MAX_TOKENS(self) -> int:
        """Mission task max_tokens — canonical System LLM max_tokens (PRD-136)."""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("system_llm", "max_tokens", os.getenv("COORDINATOR_TASK_MAX_TOKENS", "8000"))
            return int(val)
        except Exception:
            return int(os.getenv("COORDINATOR_TASK_MAX_TOKENS", "8000"))

    # =============================================================================
    # TOOL-LOOP LIMITS (chatbot + recipe step execution)
    # =============================================================================
    # These are required settings — no env-var or hardcoded fallback. The seed
    # in core/seeds/seed_system_settings.py creates the DB rows; these
    # properties read them and raise loudly if a deployment skipped seeding.

    def _required_int_setting(self, category: str, key: str) -> int:
        from core.llm.manager import get_system_setting
        val = get_system_setting(category, key, None)
        if val is None or str(val).strip() == "":
            raise RuntimeError(
                f"Missing required system_setting {category}.{key}. "
                f"Run `python -m core.seeds.seed_system_settings` to seed defaults, "
                f"or set the value via the System Settings UI."
            )
        try:
            return int(val)
        except (ValueError, TypeError):
            raise RuntimeError(
                f"Invalid system_setting {category}.{key}={val!r}: must be an integer."
            )

    def _required_float_setting(self, category: str, key: str) -> float:
        from core.llm.manager import get_system_setting
        val = get_system_setting(category, key, None)
        if val is None or str(val).strip() == "":
            raise RuntimeError(
                f"Missing required system_setting {category}.{key}. "
                f"Run `python -m core.seeds.seed_system_settings` to seed defaults, "
                f"or set the value via the System Settings UI."
            )
        try:
            return float(val)
        except (ValueError, TypeError):
            raise RuntimeError(
                f"Invalid system_setting {category}.{key}={val!r}: must be a number."
            )

    @property
    def CHATBOT_MAX_TOOL_ITERATIONS(self) -> int:
        """Max tool-call turns per chatbot user message (Auto's mid-convo budget)."""
        return self._required_int_setting("chatbot", "max_tool_iterations")

    @property
    def CHATBOT_TURN_COST_CEILING_USD(self) -> float:
        """Estimated USD a single chat turn may spend on LLM calls before the
        tool loop is forced to synthesize (PRD-223 cost governor). 0 disables."""
        return self._required_float_setting("model_policy", "turn_cost_ceiling_usd")

    @property
    def CHATBOT_ACTION_RETRY_BUDGET(self) -> int:
        """Retries when a tool returns 'action not mapped'."""
        return self._required_int_setting("chatbot", "action_retry_budget")

    @property
    def CHATBOT_PARAM_RETRY_BUDGET(self) -> int:
        """Retries when a tool returns 'invalid parameters'."""
        return self._required_int_setting("chatbot", "param_retry_budget")

    @property
    def RECIPE_DEFAULT_MAX_ITERATIONS(self) -> int:
        """Default max tool-call turns per recipe step."""
        return self._required_int_setting("recipe", "default_max_iterations")

    @property
    def RECIPE_EMPTY_COMPLETION_RETRY_BUDGET(self) -> int:
        """Retries for empty LLM completions (no content + no tool_calls).
        OpenRouter intermittently returns 'empty-choices' responses with
        zero tokens — without retry, the tool loop treats empty as 'done'
        and the step finishes without emitting its final tool call."""
        return self._required_int_setting("recipe", "empty_completion_retry_budget")

    @property
    def RECIPE_EMPTY_COMPLETION_FALLBACK_MODEL(self) -> str:
        """Fallback model used when same-model retries still return empty.
        Empty string disables fallback. Stored as system_settings.recipe.empty_completion_fallback_model."""
        from core.llm.manager import get_system_setting
        val = get_system_setting("recipe", "empty_completion_fallback_model", "")
        return str(val).strip() if val is not None else ""

    @property
    def AGENT_HEARTBEAT_MAX_TOOL_ITERATIONS(self) -> int:
        """Max tool-call turns per heartbeat tick."""
        return self._required_int_setting("agent_heartbeat", "max_tool_iterations")

    @property
    def COORDINATOR_TASK_MAX_TOOL_ITERATIONS(self) -> int:
        """Max tool-call turns per orchestrated mission task."""
        return self._required_int_setting("coordinator", "task_max_tool_iterations")

    # =============================================================================
    # ENVIRONMENT
    # =============================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    # PRD-175 (F008) — edition helpers (read the one AUTH_EDITION flag).
    @property
    def IS_LOCAL_EDITION(self) -> bool:
        return self.AUTH_EDITION == "local"

    @property
    def IS_SAAS_EDITION(self) -> bool:
        return self.AUTH_EDITION == "saas"

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
    # Workspace whose stored user_api_keys rows act as the platform-level
    # provider keys for background workers (embeddings, system LLM) and the
    # pilot chat fallback. Empty = disabled (legacy credential-store-only).
    PLATFORM_KEY_WORKSPACE_ID: str = os.getenv("PLATFORM_KEY_WORKSPACE_ID", "")

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
    # OpenRouter routes embeddings by PRICE by default, so the slowest upstream
    # can win ties (qwen3-embedding-8b measured 37-67s/call, 2026-07-09).
    # "latency" re-sorts to the fastest measured provider. Empty string disables.
    OPENROUTER_EMBEDDING_PROVIDER_SORT: str = os.getenv("OPENROUTER_EMBEDDING_PROVIDER_SORT", "latency")
    COHERE_RERANK_URL: str = os.getenv("COHERE_RERANK_URL", "https://api.cohere.com/v2/rerank")
    RAILWAY_GQL_URL: str = os.getenv("RAILWAY_GQL_URL", "https://backboard.railway.app/graphql/v2")
    # PRD-176 F068: local-safe defaults. SaaS supplies the railway.internal host
    # via env; a fresh local clone must not dial Railway topology by default.
    INTERNAL_API_HOSTNAME: str = os.getenv("INTERNAL_API_HOSTNAME", "localhost")
    INTERNAL_FRONTEND_HOSTNAME: str = os.getenv("INTERNAL_FRONTEND_HOSTNAME", "localhost")
    COMPOSIO_API_KEY: str = os.getenv("COMPOSIO_API_KEY") or os.getenv("COMPOSIO_KEY")
    # v3.1 default: tool endpoints automatically serve the latest toolkit version
    # (no `toolkit_versions=latest` param required). Only tool endpoints differ
    # between v3 and v3.1 — auth_configs / connected_accounts / triggers / toolkits
    # behave identically. Override via env if pinning to v3 becomes necessary.
    COMPOSIO_API_BASE_URL: str = os.getenv("COMPOSIO_API_BASE_URL", "https://backend.composio.dev/api/v3.1")

    # =============================================================================
    # ROUTING (Universal Orchestrator Router)
    # =============================================================================
    COMPOSIO_WEBHOOK_SECRET: str = os.getenv("COMPOSIO_WEBHOOK_SECRET")
    ROUTING_CACHE_TTL_HOURS: int = int(os.getenv("ROUTING_CACHE_TTL_HOURS", "24"))
    ROUTING_LLM_CONFIDENCE_THRESHOLD: float = float(os.getenv("ROUTING_LLM_CONFIDENCE_THRESHOLD", "0.5"))

    # GitHub
    GITHUB_PAT: str = os.getenv("GITHUB", "")
    GITHUB_REPO_OWNER: str = os.getenv("GITHUB_REPO_OWNER", "")
    GITHUB_REPO_NAME: str = os.getenv("GITHUB_REPO_NAME", "")
    GITHUB_DEFAULT_BRANCH: str = os.getenv("GITHUB_DEFAULT_BRANCH", "main")
    GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    GITHUB_WEBHOOK_WORKSPACE_ID: str = os.getenv("GITHUB_WEBHOOK_WORKSPACE_ID") or os.getenv("DEFAULT_WORKSPACE_ID")
    GITHUB_PR_WORKFLOW_NAME: str = os.getenv("GITHUB_PR_WORKFLOW_NAME", "PR Code Review")
    # PRD-165 S4 (Q36): GitHub App installation auth. When all three are set,
    # codegraph mints installation tokens instead of using the PAT above.
    GITHUB_APP_ID: str = os.getenv("GITHUB_APP_ID", "")
    GITHUB_APP_PRIVATE_KEY: str = os.getenv("GITHUB_APP_PRIVATE_KEY", "")
    GITHUB_APP_INSTALLATION_ID: str = os.getenv("GITHUB_APP_INSTALLATION_ID", "")

    # Webhooks / Widgets
    WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET")
    # PRD-194 S2 (P2-13): replay/dedup guard for the three EXTERNAL webhook
    # lanes (Composio /webhook, workspace /ws/{key}, playbook /recipe/{id}).
    # Dedup marks live in Redis (SETNX + TTL via core/redis/client.py — no
    # new table). TTL covers provider retry windows with slack to spare.
    WEBHOOK_DEDUP_TTL_SECONDS: int = int(os.getenv("WEBHOOK_DEDUP_TTL_SECONDS", "3600"))
    # Replay skew: reject events whose provider timestamp is further than
    # this from now (mirrors Slack's documented v0 5-minute window).
    WEBHOOK_TIMESTAMP_SKEW_SECONDS: int = int(os.getenv("WEBHOOK_TIMESTAMP_SKEW_SECONDS", "300"))
    WIDGET_TOKEN_SECRET: str = os.getenv("WIDGET_TOKEN_SECRET", "")
    # PRD-194 S5 (P2-13): Redis-backed shared widget rate limiter (replaces
    # the per-process in-memory window). One window length; per-key limits by
    # key type; and a per-IP ceiling on the two money-spending endpoints
    # (/api/widgets/chat, /api/widgets/callback) that applies even when a
    # key is presented.
    WIDGET_RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("WIDGET_RATE_LIMIT_WINDOW_SECONDS", "60"))
    WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW: int = int(os.getenv("WIDGET_RATE_LIMIT_PUBLIC_PER_WINDOW", "30"))
    WIDGET_RATE_LIMIT_SERVER_PER_WINDOW: int = int(os.getenv("WIDGET_RATE_LIMIT_SERVER_PER_WINDOW", "1000"))
    WIDGET_CHAT_IP_LIMIT_PER_WINDOW: int = int(os.getenv("WIDGET_CHAT_IP_LIMIT_PER_WINDOW", "30"))
    WIDGET_CALLBACK_IP_LIMIT_PER_WINDOW: int = int(os.getenv("WIDGET_CALLBACK_IP_LIMIT_PER_WINDOW", "10"))
    SHOPIFY_INTERNAL_API_KEY: str = os.getenv("SHOPIFY_INTERNAL_API_KEY", "")
    # BudStacks vertical (api/verticals.py) — same fail-closed posture: unset ⇒
    # the budstacks provision surface answers 503, never open.
    BUDSTACKS_INTERNAL_API_KEY: str = os.getenv("BUDSTACKS_INTERNAL_API_KEY", "")
    # PRD-189 S3: per-workspace debounce window (seconds) for catalog-webhook
    # re-syncs. A merchant bulk edit emits a burst of products/update webhooks;
    # /events coalesces the burst into ONE full Bulk-Op re-sync once it has
    # been quiet for this long, instead of firing N concurrent re-syncs (each
    # an embedding-bearing full rebuild). Same config-not-inline-getenv
    # convention as PLAYBOOK_BREAKER_THRESHOLD.
    SHOPIFY_SYNC_DEBOUNCE_SECONDS: float = float(os.getenv("SHOPIFY_SYNC_DEBOUNCE_SECONDS", "30"))
    PLAYBOOKS_REQUIRE_TENANT: bool = os.getenv("PLAYBOOKS_REQUIRE_TENANT", "0").lower() in ("1", "true", "yes")

    # =============================================================================
    # TASK RUNNER (PRD-56: Infrastructure Scaling & Physical Workspaces)
    # =============================================================================
    TASK_RUNNER_BACKEND: str = os.getenv("TASK_RUNNER_BACKEND", "local")  # local, queued, kubernetes
    WORKSPACE_VOLUME_PATH: str = os.getenv("WORKSPACE_VOLUME_PATH", "/workspaces")

    # Bounded Concurrency (per-workspace execution limits)
    DEFAULT_MAX_CONCURRENT_TOTAL: int = int(os.getenv("DEFAULT_MAX_CONCURRENT_TOTAL", "3"))
    DEFAULT_MAX_CONCURRENT_RUNNING: int = int(os.getenv("DEFAULT_MAX_CONCURRENT_RUNNING", "3"))
    DEFAULT_MAX_CONCURRENT_PENDING: int = int(os.getenv("DEFAULT_MAX_CONCURRENT_PENDING", "10"))

    # =============================================================================
    # BOARD DISPATCH SPINE (PRD-161: claim/lease/requeue)
    # =============================================================================
    # One Postgres-native dispatch loop: assigned BoardTasks are claimed with
    # FOR UPDATE SKIP LOCKED (exactly-once), leased, and requeued on crash.
    BOARD_DISPATCH_ENABLED: bool = os.getenv("BOARD_DISPATCH_ENABLED", "true").lower() == "true"
    # Lease a claimed task holds before the sweeper presumes the worker dead.
    BOARD_DISPATCH_LEASE_SECONDS: int = int(os.getenv("BOARD_DISPATCH_LEASE_SECONDS", "600"))
    # Poll fallback cadence when no NOTIFY arrives (NOTIFY drives sub-second pickup).
    BOARD_DISPATCH_POLL_SECONDS: float = float(os.getenv("BOARD_DISPATCH_POLL_SECONDS", "5"))
    # Tasks claimed per loop tick (a tick claims a batch, runs each individually).
    BOARD_DISPATCH_CLAIM_BATCH: int = int(os.getenv("BOARD_DISPATCH_CLAIM_BATCH", "10"))
    # Q41: attempts before a task is terminal 'failed' (crash → requeue until here).
    BOARD_DISPATCH_MAX_ATTEMPTS: int = int(os.getenv("BOARD_DISPATCH_MAX_ATTEMPTS", "2"))
    # Per-agent concurrency slots: at most this many of an agent's tasks run at
    # once; the rest stay 'assigned' (the DB is the queue — double-texting is
    # queued, never dropped). The claim honours this via in_progress counts.
    BOARD_DISPATCH_AGENT_SLOTS: int = int(os.getenv("BOARD_DISPATCH_AGENT_SLOTS", "2"))
    # S5: done tasks older than this drop off the active board (retained in DB).
    BOARD_ARCHIVE_DONE_DAYS: int = int(os.getenv("BOARD_ARCHIVE_DONE_DAYS", "30"))
    # PRD-180 S1: board SSE is now LISTEN/NOTIFY-driven; this is only the
    # connection-liveness heartbeat cadence (a ':hb' comment), not a refresh tick.
    BOARD_SSE_HEARTBEAT_SECONDS: float = float(os.getenv("BOARD_SSE_HEARTBEAT_SECONDS", "20"))
    # PRD-228: an agent shown as "working" whose last task activity is older than
    # this is flagged STALLED by the fleet-status anomaly surface (default 30 min).
    FLEET_STALL_SECONDS: int = int(os.getenv("FLEET_STALL_SECONDS", "1800"))

    # =============================================================================
    # AUTO WATCHER (PRD-204: persistent supervision of launched work)
    # =============================================================================
    # The watcher tick rides the fcntl-locked UnifiedScheduler (single owner
    # across workers). Each tick claims due watches with FOR UPDATE SKIP
    # LOCKED, sweeps terminal states the S3 event hooks missed, detects
    # missed cron fires / benched schedules on scheduled-playbook watches,
    # and expires past-deadline watches.
    WATCHER_ENABLED: bool = os.getenv("WATCHER_ENABLED", "true").lower() == "true"
    # Sweep cadence. The S3 hooks are the fast path; the tick is the
    # fallback and the missed-run/trend brain, so 5 minutes is plenty.
    WATCHER_TICK_SECONDS: int = int(os.getenv("WATCHER_TICK_SECONDS", "300"))
    # PRD-224 US-005: auto-attach a run_and_report watch to every ASSIGN-lane
    # board ticket Auto files, so an assigned ticket reports its verdict back
    # into the originating thread. Default ON — an unsupervised assigned ticket
    # is the current failure mode, not a feature (Gerard, 2026-08-27).
    AUTO_TICKET_WATCH: bool = os.getenv("AUTO_TICKET_WATCH", "true").lower() in ("true", "1", "yes")

    WORKER_INTERNAL_URL: str = os.getenv("WORKER_INTERNAL_URL", "http://localhost:8081")
    WORKER_INTERNAL_TOKEN: str = os.getenv("WORKER_INTERNAL_TOKEN", "")

    # PRD-202 S2 (Q4): the small set of "core" skills that stay always-L2 — their
    # full body renders every turn because they are an agent's core operating
    # manual (Auto's platform-management), not an optional capability. Every
    # OTHER attached skill renders only its L1 metadata (~50-100 tokens) and
    # loads its body on trigger via the load_skill tool. Comma-separated names.
    SKILL_CORE_ALWAYS_ON = [
        s.strip()
        for s in os.getenv("SKILL_CORE_ALWAYS_ON", "platform-management").split(",")
        if s.strip()
    ]

    # PRD-202 S3: L3 skill-script execution caps (via the workspace worker only).
    # Wall-clock cap (seconds) and output-size cap (chars) — the worker is the
    # isolation boundary; only the script OUTPUT (capped) enters context.
    SKILL_SCRIPT_TIMEOUT_SECONDS: int = int(os.getenv("SKILL_SCRIPT_TIMEOUT_SECONDS", "60"))
    SKILL_SCRIPT_OUTPUT_MAX_CHARS: int = int(os.getenv("SKILL_SCRIPT_OUTPUT_MAX_CHARS", "20000"))

    # Task Reconciliation (Symphony-inspired stall detection)
    TASK_STALL_TIMEOUT_SECONDS: int = int(os.getenv("TASK_STALL_TIMEOUT_SECONDS", "300"))  # 5 min
    TASK_PENDING_TIMEOUT_SECONDS: int = int(os.getenv("TASK_PENDING_TIMEOUT_SECONDS", "120"))  # 2 min
    TASK_MAX_RETRIES: int = int(os.getenv("TASK_MAX_RETRIES", "2"))
    TASK_MAX_RETRY_BACKOFF_MS: int = int(os.getenv("TASK_MAX_RETRY_BACKOFF_MS", "300000"))  # 5 min cap
    TASK_RECONCILE_INTERVAL_SECONDS: int = int(os.getenv("TASK_RECONCILE_INTERVAL_SECONDS", "60"))

    # Playbook (Recipe) execution timeouts — defaults used when a recipe's
    # execution_config is empty; the MIN_* values floor whatever the recipe configures
    # so an under-spec'd recipe (e.g. 120s) cannot kill itself prematurely.
    PLAYBOOK_DEFAULT_STEP_TIMEOUT_SECONDS: int = int(os.getenv("PLAYBOOK_DEFAULT_STEP_TIMEOUT_SECONDS", "600"))   # 10 min
    PLAYBOOK_DEFAULT_TOTAL_TIMEOUT_SECONDS: int = int(os.getenv("PLAYBOOK_DEFAULT_TOTAL_TIMEOUT_SECONDS", "1800"))  # 30 min
    PLAYBOOK_MIN_STEP_TIMEOUT_SECONDS: int = int(os.getenv("PLAYBOOK_MIN_STEP_TIMEOUT_SECONDS", "300"))            # 5 min floor
    PLAYBOOK_MIN_TOTAL_TIMEOUT_SECONDS: int = int(os.getenv("PLAYBOOK_MIN_TOTAL_TIMEOUT_SECONDS", "900"))          # 15 min floor
    # PRD-185 S4: repeated-failure circuit breaker for cron playbooks. Once the
    # last N *terminal* runs of a playbook are all 'failed', the cron scheduler
    # stops re-firing it (the 2026-06 daily OpenRouter-402 spam re-fired forever).
    # A manual run that succeeds breaks the streak and auto-resets the breaker.
    # Set to 0 to disable the breaker entirely.
    PLAYBOOK_BREAKER_THRESHOLD: int = int(os.getenv("PLAYBOOK_BREAKER_THRESHOLD", "3"))

    # =============================================================================
    # RAILWAY API (Log retrieval for agents)
    # =============================================================================
    RAILWAY_API_TOKEN: str = os.getenv("RAILWAY_API_TOKEN", "")
    RAILWAY_PROJECT_ID: str = os.getenv("RAILWAY_PROJECT_ID", "")
    RAILWAY_ENVIRONMENT_ID: str = os.getenv("RAILWAY_ENVIRONMENT_ID", "")

    # =============================================================================
    # MONITORING (PRD-73)
    # =============================================================================
    # PRD-176 F068: local-safe defaults (SaaS sets the railway host via env).
    LOKI_URL: str = os.getenv("LOKI_URL", "http://localhost:3100")
    PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    GRAFANA_URL: str = os.getenv("GRAFANA_URL", "")
    GRAFANA_SERVICE_ACCOUNT_TOKEN: str = os.getenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", "")
    GRAFANA_LOKI_DATASOURCE_UID: str = os.getenv("GRAFANA_LOKI_DATASOURCE_UID", "loki")

    # =============================================================================
    # OBSERVABILITY — log relay, Prometheus metrics, Loki query API, alerts
    # (PRD-142 W3-S5 / G7 — centralized so monitoring modules stop reading env directly)
    # =============================================================================
    # Log relay client (core/monitoring/automatos_logging.py).
    # PRD-176 F068: local-safe default host + OFF by default. SaaS points this at
    # log-relay.railway.internal and sets LOG_RELAY_ENABLED=true via env; a fresh
    # local clone must not try to push logs to Railway topology.
    LOG_RELAY_URL: str = os.getenv(
        "LOG_RELAY_URL",
        "http://localhost:8080/push",
    )
    LOG_RELAY_ENABLED: bool = os.getenv("LOG_RELAY_ENABLED", "false").lower() == "true"
    LOG_RELAY_BATCH_SIZE: int = int(os.getenv("LOG_RELAY_BATCH_SIZE", "50"))
    LOG_RELAY_FLUSH_INTERVAL: float = float(os.getenv("LOG_RELAY_FLUSH_INTERVAL", "2.0"))
    # SERVICE_NAME has two distinct historical defaults — preserve both exactly.
    # The logging client used "unknown"; the Prometheus exporter used "automatos-backend".
    LOG_RELAY_SERVICE_NAME: str = os.getenv("SERVICE_NAME", "unknown")
    METRICS_SERVICE_NAME: str = os.getenv("SERVICE_NAME", "automatos-backend")
    # ENVIRONMENT also has two distinct historical defaults — preserve both.
    # The logging client also falls back to RAILWAY_ENVIRONMENT; metrics defaulted to "unknown".
    LOG_RELAY_ENVIRONMENT: str = os.getenv(
        "ENVIRONMENT",
        os.getenv("RAILWAY_ENVIRONMENT", "development"),
    )
    METRICS_ENVIRONMENT: str = os.getenv("ENVIRONMENT", "unknown")
    # Loki query proxy (core/monitoring/automatos_logs_api.py) — separate from the
    # PRD-73 LOKI_URL above because the existing module reads LOKI_QUERY_URL.
    # PRD-176 F068: local-safe default (SaaS sets the railway host via env).
    LOKI_QUERY_URL: str = os.getenv("LOKI_QUERY_URL", "http://localhost:3100")
    # Shared by automatos_logs_api + automatos_alerts for HMAC verification.
    ALERT_INGEST_TOKEN: str = os.getenv("ALERT_INGEST_TOKEN", "")
    # PRD-180 S5: default measurement window (seconds) for the tracked SLOs
    # (tool-call success rate, board dispatch p95). Overridable per-request.
    SLO_DEFAULT_WINDOW_SECONDS: int = int(os.getenv("SLO_DEFAULT_WINDOW_SECONDS", "86400"))

    # =============================================================================
    # CHANNELS — public-facing host used to build inbound webhook URLs
    # (PRD-142 W3-S5 / G7)
    # =============================================================================
    # rstrip('/') preserves api/channels.py's prior call-site behaviour.
    PUBLIC_API_HOST: str = os.getenv("PUBLIC_API_HOST", "api.automatos.app").rstrip("/")

    # =============================================================================
    # RATE LIMITING — per-operation override helper
    # (PRD-142 W3-S5 / G7; replaces core/security/rate_limiter.py::_env_limit)
    # =============================================================================
    @staticmethod
    def rate_limit_for(name: str, default_max: int, default_window: int) -> tuple[int, int]:
        """Read RATE_LIMIT_<NAME>_MAX and RATE_LIMIT_<NAME>_WINDOW_SECONDS env vars.

        Returns ``(max_requests, window_seconds)``, both clamped to ``>= 1`` so
        a malformed override cannot disable the bucket. Invalid integers (or any
        TypeError/ValueError) fall back to the supplied defaults — same shape as
        the previous in-module helper. No behaviour change.
        """
        try:
            max_req = int(os.getenv(f"RATE_LIMIT_{name.upper()}_MAX", str(default_max)))
            window = int(os.getenv(f"RATE_LIMIT_{name.upper()}_WINDOW_SECONDS", str(default_window)))
            return max(1, max_req), max(1, window)
        except (TypeError, ValueError):
            return default_max, default_window

    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    # PRD-155 S3: startup mount honesty. By default a required router that fails
    # to import aborts boot (RouterMountError names it) instead of being silently
    # dropped. Set true to downgrade that to a logged skip and boot degraded —
    # an operator escape hatch, not the norm. Default OFF.
    ALLOW_DEGRADED_BOOT: bool = os.getenv("ALLOW_DEGRADED_BOOT", "false").lower() == "true"
    HEARTBEAT_ENABLED: bool = os.getenv("HEARTBEAT_ENABLED", "true").lower() == "true"
    RECIPE_SCHEDULER_ENABLED: bool = os.getenv("RECIPE_SCHEDULER_ENABLED", "true").lower() == "true"
    COORDINATOR_ENABLED: bool = os.getenv("COORDINATOR_ENABLED", "true").lower() == "true"
    HARNESS_ENABLED: bool = os.getenv("HARNESS_ENABLED", "true").lower() == "true"
    # PRD-141 Phase 5: gates HARNESS self-management (auto-applying approved
    # config changes back onto the platform). HIGH RISK — default OFF. Nothing
    # in Phase 5 may take effect unless this is true.
    HARNESS_SELF_MANAGEMENT_ENABLED: bool = os.getenv("HARNESS_SELF_MANAGEMENT_ENABLED", "false").lower() == "true"
    # PRD-142 Wave 4 (§12.3): HARNESS risk thresholds — Railway-overridable env vars
    # (change in Railway + restart, no file edit). Auto-apply ceilings: a prescription
    # auto-applies only when its risk_score <= the workspace's ceiling; higher risk is
    # queued for human approval. Workspaces at autonomy=full get the higher ceiling,
    # standard workspaces the lower one.
    HARNESS_AUTO_APPLY_MAX_RISK_STANDARD: int = int(os.getenv("HARNESS_AUTO_APPLY_MAX_RISK_STANDARD", "2"))
    HARNESS_AUTO_APPLY_MAX_RISK_FULL: int = int(os.getenv("HARNESS_AUTO_APPLY_MAX_RISK_FULL", "3"))
    # Escalation threshold: prescriptions at or above this risk are flagged high
    # priority and escalated for human approval (when self-management is on).
    HARNESS_HIGH_PRIORITY_RISK: int = int(os.getenv("HARNESS_HIGH_PRIORITY_RISK", "4"))

    # PRD-174 Wave 4 / PRD-192 S1 (P2-11) — Unified Policy Plane staged mode
    # dial. ONE env, `AUTOMATOS_POLICY_PLANE = off | shadow | destructive | on`:
    #   off         ⇒ byte-for-byte today's per-router gates (no bus fire, no audit)
    #   shadow      ⇒ evaluate + audit every verdict; NEVER block
    #   destructive ⇒ enforce deny/ask only for the fail-closed risk classes
    #                 (destructive / external_side_effect / publish); shadow-log the rest
    #   on          ⇒ enforce all (PRD-174's original ON)
    # Legacy booleans map (true/1/yes ⇒ on, false/0/no ⇒ off); unknown values
    # fail safe to "off". Ships default OFF; stage flips are ops actions on the
    # deploy env (Railway), never code — each retreat is one env value.
    _POLICY_PLANE_RAW = os.getenv("AUTOMATOS_POLICY_PLANE", "off").strip().lower()
    POLICY_PLANE_MODE: str = {
        "true": "on", "1": "on", "yes": "on",
        "false": "off", "0": "off", "no": "off", "": "off",
    }.get(_POLICY_PLANE_RAW, _POLICY_PLANE_RAW)
    POLICY_PLANE_MODE = POLICY_PLANE_MODE if POLICY_PLANE_MODE in ("off", "shadow", "destructive", "on") else "off"
    # Derived boolean (mode ≠ off) so the existing registration sites — the
    # audit-handler attach (main.py), limiter arming (main.py F040), roles.py
    # F043, widgets/auth.py F042 — arm on ANY live stage, unchanged.
    POLICY_PLANE_ENABLED: bool = POLICY_PLANE_MODE != "off"

    # PRD-192 S3 (locked #2a): autonomy-enabled workspaces get a DEFAULT budget
    # ceiling — max_cost_usd 50 per month — applied in the budget reader when
    # the workspace has no explicit `plan_limits.budget` (code default, no
    # migration; explicit budgets always win). 0 disables the default.
    AUTONOMY_DEFAULT_BUDGET_USD: float = float(os.getenv("AUTOMATOS_AUTONOMY_DEFAULT_BUDGET_USD", "50"))

    # PRD-196 S5 — audit-log retention. EU-AI-Act Art.12 mandates >= 6 months
    # (a floor, so retention is a compliance requirement, not housekeeping) while
    # GDPR data-minimisation forbids forever. Platform-wide default 365 days; a
    # configured value below the 180-day Art.12 floor is CLAMPED UP at read
    # (services/audit_retention.effective_retention_days) — a config can never
    # dip under the legal floor. No per-workspace override in v1 (Gerard's call).
    AUDIT_RETENTION_DAYS: int = int(os.getenv("AUDIT_RETENTION_DAYS", "365"))
    # How often the retention sweep runs (default daily).
    AUDIT_RETENTION_SWEEP_INTERVAL_SECONDS: int = int(
        os.getenv("AUDIT_RETENTION_SWEEP_INTERVAL_SECONDS", "86400")
    )

    # PRD-185 S2 — per-lane telemetry canary. The type-poison outage S1 repaired
    # went unseen for ~2 months because nothing alarmed on "organic tool-execution
    # rows/day = 0". This canary counts production (telemetry_source='production')
    # ToolExecutionLog rows per lane (app_name) over a window and logs LOUD when a
    # lane — or the platform — has gone silent. Default ON (it only reads + logs).
    TELEMETRY_CANARY_ENABLED: bool = os.getenv("TELEMETRY_CANARY_ENABLED", "true").lower() == "true"
    # How often the scheduled check runs (default hourly). Its first run fires at
    # boot as the boot-probe.
    TELEMETRY_CANARY_INTERVAL_SECONDS: int = int(os.getenv("TELEMETRY_CANARY_INTERVAL_SECONDS", "3600"))
    # Look-back window for "have any organic rows landed?" (default 24h).
    TELEMETRY_CANARY_WINDOW_SECONDS: int = int(os.getenv("TELEMETRY_CANARY_WINDOW_SECONDS", "86400"))
    # Platform-wide organic-row count at/under which the canary alarms (default 0
    # → alarm only on a totally silent platform; raise to catch partial silence).
    TELEMETRY_CANARY_MIN_ROWS: int = int(os.getenv("TELEMETRY_CANARY_MIN_ROWS", "0"))

    # =============================================================================
    # PRD-181 W11 — Governance & Compliance staging
    # =============================================================================
    # EU-AI-Act Art.14 human-oversight tiers (S6 scaffold). The tier *mapping*
    # from risk class → oversight lives in modules/policy/ai_act.py (pure); this
    # constant is the canonical ordered vocabulary so config/UI reference the same
    # strings. Do NOT branch autonomy on these — the policy plane's risk routing
    # is authoritative; these describe the oversight posture for the approval card.
    EU_AI_ACT_OVERSIGHT_TIERS: tuple = ("monitor", "human_on_the_loop", "human_in_the_loop")
    # Default TTL (seconds) for a durable approval grant awaiting a human (S2).
    APPROVAL_GRANT_TTL_SECONDS: int = int(os.getenv("APPROVAL_GRANT_TTL_SECONDS", str(24 * 3600)))

    # =============================================================================
    # PRD-130 — Business Intake Wizard (PoC)
    # =============================================================================
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY")
    FIRECRAWL_BASE_URL: str = os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")
    FIRECRAWL_MAX_PAGES_PER_SCAN: int = int(os.getenv("FIRECRAWL_MAX_PAGES_PER_SCAN", "20"))

    WIZARD_ENABLED: bool = os.getenv("WIZARD_ENABLED", "true").lower() == "true"
    WIZARD_REQUIRE_DOMAIN_VERIFY: bool = os.getenv("WIZARD_REQUIRE_DOMAIN_VERIFY", "false").lower() == "true"

    # =============================================================================
    # COORDINATOR — PRD-82A Sequential Mission Coordinator
    # =============================================================================
    COORDINATOR_TICK_INTERVAL_SECONDS: int = int(os.getenv("COORDINATOR_TICK_INTERVAL_SECONDS", "5"))
    COORDINATOR_ASSIGNED_STALL_THRESHOLD_SECONDS: int = int(os.getenv("COORDINATOR_ASSIGNED_STALL_THRESHOLD_SECONDS", "60"))
    COORDINATOR_RUNNING_STALL_THRESHOLD_SECONDS: int = int(os.getenv("COORDINATOR_RUNNING_STALL_THRESHOLD_SECONDS", "300"))
    COORDINATOR_MAX_TASK_RETRIES: int = int(os.getenv("COORDINATOR_MAX_TASK_RETRIES", "3"))
    # PRD-164 S4: consecutive churn-without-progress joiner checks before a
    # looping mission is auto-replanned (or halted once replans are exhausted).
    COORDINATOR_STALL_LEDGER_LIMIT: int = int(os.getenv("COORDINATOR_STALL_LEDGER_LIMIT", "3"))
    COORDINATOR_MAX_VERIFICATION_RETRIES: int = int(os.getenv("COORDINATOR_MAX_VERIFICATION_RETRIES", "2"))
    # PRD-200 S1: how many times a FAIL verdict may requeue a COMPLETED task
    # with the verifier's feedback so the agent can revise (the judge "gates
    # once"). This is a SEPARATE budget from COORDINATOR_MAX_VERIFICATION_RETRIES
    # above (the LLM-judge's own malformed-response retry count) and from
    # COORDINATOR_MAX_TASK_RETRIES (agent-error retries). Capped at 1 by
    # decision: PARTIAL stays advisory, so the only re-judged verdict is FAIL,
    # bounding the token-burn the advisory retreat originally closed.
    COORDINATOR_MAX_VERIFICATION_REQUEUES: int = int(os.getenv("COORDINATOR_MAX_VERIFICATION_REQUEUES", "1"))
    COORDINATOR_VERIFICATION_PASS_THRESHOLD: float = float(os.getenv("COORDINATOR_VERIFICATION_PASS_THRESHOLD", "0.7"))
    COORDINATOR_VERIFICATION_FAIL_THRESHOLD: float = float(os.getenv("COORDINATOR_VERIFICATION_FAIL_THRESHOLD", "0.4"))
    COORDINATOR_VERIFICATION_CONFIDENCE_ESCALATION: float = float(os.getenv("COORDINATOR_VERIFICATION_CONFIDENCE_ESCALATION", "0.5"))
    # PRD-200 S3: awaiting-approval re-notify + optional expiry sweep. A parked
    # plan re-dispatches its mission_plan_ready notification every
    # RENOTIFY_SECONDS so it does not die after one notification (the 47%-parked
    # unblock). Expiry is OFF by default — under the always_ask posture,
    # terminating an unapproved plan is the operator's call (Q5); when enabled, a
    # plan older than MAX_AGE_SECONDS is cancelled.
    COORDINATOR_APPROVAL_RENOTIFY_SECONDS: int = int(os.getenv("COORDINATOR_APPROVAL_RENOTIFY_SECONDS", "86400"))
    COORDINATOR_APPROVAL_EXPIRY_ENABLED: bool = os.getenv("COORDINATOR_APPROVAL_EXPIRY_ENABLED", "false").lower() == "true"
    COORDINATOR_APPROVAL_MAX_AGE_SECONDS: int = int(os.getenv("COORDINATOR_APPROVAL_MAX_AGE_SECONDS", "604800"))
    # Cross-model verification: reads from system_settings → env fallback
    @property
    def COORDINATOR_VERIFIER_MODEL_MAPPING(self) -> str:
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting(
                "coordination", "verifier_model_mapping",
                os.getenv(
                    "COORDINATOR_VERIFIER_MODEL_MAPPING",
                    "anthropic=openai/gpt-4o-mini,openai=anthropic/claude-haiku-4-5,"
                    "google=openai/gpt-4o-mini,deepseek=openai/gpt-4o-mini,meta=openai/gpt-4o-mini",
                ),
            )
        except Exception:
            return os.getenv("COORDINATOR_VERIFIER_MODEL_MAPPING", "")

    @property
    def COORDINATOR_VERIFIER_FALLBACK_MODEL(self) -> str:
        try:
            from core.llm.manager import get_system_setting
            return get_system_setting(
                "coordination", "verifier_fallback_model",
                os.getenv("COORDINATOR_VERIFIER_FALLBACK_MODEL", "openai/gpt-4o-mini"),
            )
        except Exception:
            return os.getenv("COORDINATOR_VERIFIER_FALLBACK_MODEL", "openai/gpt-4o-mini")
    # History-based agent scoring (PRD-82B US-003)
    COORDINATOR_HISTORY_LOOKBACK_DAYS: int = int(os.getenv("COORDINATOR_HISTORY_LOOKBACK_DAYS", "30"))
    COORDINATOR_HISTORY_MIN_DATAPOINTS: int = int(os.getenv("COORDINATOR_HISTORY_MIN_DATAPOINTS", "3"))
    # PRD-164 S2 (Q21): upper bound on the per-dispatch semantic-signal
    # computation (task embedding + capability-card cosine + field query) so a
    # hung embedding/Qdrant backend can never stall the dispatch tick — on
    # timeout the matcher falls back to lexical-only scoring.
    AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS: float = float(os.getenv("AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS", "10"))
    # Cost estimation per 1K tokens (PRD-82B US-004). PRD-192 S3 (F059 finish):
    # DEMOTED to modules/policy/pricing.py's registry-miss last resort — pricing
    # is this constant's ONLY consumer (source-grep-guarded); every dollar
    # figure routes through the one pricing source.
    COORDINATOR_COST_PER_1K_TOKENS: float = float(os.getenv("COORDINATOR_COST_PER_1K_TOKENS", "0.003"))
    # Replanning limits (PRD-82B US-005)
    COORDINATOR_MAX_REPLANS: int = int(os.getenv("COORDINATOR_MAX_REPLANS", "2"))
    # PRD-227 US-002: mission-narration throttle. A mission narrates its lifecycle
    # back into the launching chat (approved → task done/failed → completed/failed/
    # cancelled). Run-level lines always send; task-level lines are SUPPRESSED for
    # runs with more than this many tasks, to keep large plans readable. Default 8
    # (Gerard, 2026-08-27). Narration itself is on for all missions.
    MISSION_NARRATION_TASK_CAP: int = int(os.getenv("MISSION_NARRATION_TASK_CAP", "8"))
    # COORDINATOR_TASK_MAX_TOKENS is now a @property above (reads from system_settings)
    # Maximum seconds a single task execution can take before being timed out
    COORDINATOR_TASK_EXECUTION_TIMEOUT: int = int(os.getenv("COORDINATOR_TASK_EXECUTION_TIMEOUT", "240"))
    # PRD-229: mid-run clarifications (ask_orchestrator). CLARIFICATION_BUDGET
    # caps how many questions Auto ANSWERS per run from retrievable context;
    # once spent, everything escalates (escalations are never budget-limited —
    # they are visible and cheap by design). Default 3 (Gerard, 2026-08-27).
    CLARIFICATION_BUDGET: int = int(os.getenv("CLARIFICATION_BUDGET", "3"))
    # PRD-229: hard time-box for ONE ask_orchestrator answer round (retrieval +
    # one composition call). It runs INSIDE the executing task's asyncio.wait_for
    # envelope (coordinator_service._run_agent_io), whose timeout is the power
    # mode's timeout_seconds (_POWER_MODE_DEFAULTS: light=120s, standard=240s,
    # max=600s). 30s sits well inside the SMALLEST (light, 120s) envelope; the
    # cumulative N-round bound is CLARIFICATION_MAX_ROUNDS_PER_TASK below (this
    # constant only bounds a SINGLE round). On time-box expiry the tool takes the
    # cannot_answer path.
    CLARIFICATION_ANSWER_TIMEOUT: int = int(os.getenv("CLARIFICATION_ANSWER_TIMEOUT", "30"))
    # PRD-229 (P229-RVW-8): CLARIFICATION_ANSWER_TIMEOUT bounds ONE round and
    # CLARIFICATION_BUDGET caps ANSWERED rounds per RUN — but neither bounds the
    # CUMULATIVE answer-round time a SINGLE task can spend (N rounds ×
    # CLARIFICATION_ANSWER_TIMEOUT could approach the task envelope, then the outer
    # asyncio.wait_for hard-cancels the WHOLE task — lost work + retry). This caps
    # the answer rounds one task may ENTER: worst-case clarification time =
    # CLARIFICATION_MAX_ROUNDS_PER_TASK × CLARIFICATION_ANSWER_TIMEOUT (2 × 30s =
    # 60s) — half the smallest (light, 120s) envelope, leaving ~60s for the task's
    # own LLM turns. The (cap+1)th ask_orchestrator call short-circuits to
    # escalation (park + human) WITHOUT entering a 30s round.
    CLARIFICATION_MAX_ROUNDS_PER_TASK: int = int(os.getenv("CLARIFICATION_MAX_ROUNDS_PER_TASK", "2"))
    # Note: synthesis-task model selection is now driven by power_mode +
    # the agent's own configured model — no synthesis-specific override.
    # System LLM (gemini-2.5-flash) is reserved for codegraph / memory / planner.
    # Cross-task consistency verification — feature flag, lives in `general`
    # (post PRD-136 collapse — no longer an LLM-tier setting).
    @property
    def COORDINATOR_CONSISTENCY_CHECK(self) -> bool:
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("general", "coordinator_consistency_check_enabled", "true")
            return str(val).lower() in ("true", "1", "yes")
        except Exception:
            return os.getenv("COORDINATOR_CONSISTENCY_CHECK", "true").lower() in ("true", "1", "yes")
    # Parallel execution & budget governance (PRD-82C)
    # Token budgets per complexity tier — used for task estimation and budget gate
    COMPLEXITY_TOKEN_BUDGET_SIMPLE: int = int(os.getenv("COMPLEXITY_TOKEN_BUDGET_SIMPLE", "15000"))
    COMPLEXITY_TOKEN_BUDGET_MODERATE: int = int(os.getenv("COMPLEXITY_TOKEN_BUDGET_MODERATE", "40000"))
    COMPLEXITY_TOKEN_BUDGET_COMPLEX: int = int(os.getenv("COMPLEXITY_TOKEN_BUDGET_COMPLEX", "80000"))
    COMPLEXITY_TOKEN_BUDGET_SYNTHESIS: int = int(os.getenv("COMPLEXITY_TOKEN_BUDGET_SYNTHESIS", "50000"))

    # Archival: move terminal runs to archive after N days (PRD-82B US-009)
    COORDINATOR_ARCHIVE_AFTER_DAYS: int = int(os.getenv("COORDINATOR_ARCHIVE_AFTER_DAYS", "30"))
    COORDINATOR_ARCHIVE_BATCH_SIZE: int = int(os.getenv("COORDINATOR_ARCHIVE_BATCH_SIZE", "50"))
    CHANNELS_ENABLED: bool = os.getenv("CHANNELS_ENABLED", "true").lower() == "true"
    # SEMANTIC_TOOL_ROUTING (default ON) and TOOL_ROUTING_GRAPH (default OFF, below)
    # gate DIFFERENT surfaces — PRD-232 US-002 split them after C2 found the
    # inversion. SEMANTIC_TOOL_ROUTING gates EMBEDDING narrowing everywhere:
    # the platform_execute dispatcher's action-enum narrowing (tool_router.py
    # _semantic_routing_enabled) and PlatformActionsSection's embedding catalog
    # (_build_filtered). It does NOT gate the learned tool-routing GRAPH reads —
    # those are TOOL_ROUTING_GRAPH's job. A graph-off turn still narrows
    # semantically; it just never queries GraphRouter.rank_chains.
    SEMANTIC_TOOL_ROUTING: bool = os.getenv("SEMANTIC_TOOL_ROUTING", "true").lower() == "true"
    SEMANTIC_TOOL_ROUTING_TOP_K: int = int(os.getenv("SEMANTIC_TOOL_ROUTING_TOP_K", "15"))
    # PRD-221 S4: ceiling on the narrowed dispatcher enum after the current
    # page's manifest actions are unioned in with the semantic top-K. Bounds the
    # prompt cost of page-prior exposure; role gates still apply before the cap.
    TOOL_ROUTING_ENUM_CAP: int = int(os.getenv("TOOL_ROUTING_ENUM_CAP", "40"))
    # PRD-221 S9: Auto's Read digest is cached per (workspace, state_hash) for
    # this many seconds, so the digest LLM fires at most once per real state
    # change rather than once per Command Centre pageview.
    DIGEST_CACHE_TTL_S: int = int(os.getenv("DIGEST_CACHE_TTL_S", "900"))
    # Max seconds a live query embedding may take before narrowing falls back
    # to the full action enum (the embed keeps running and caches for next turn).
    SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S: float = float(os.getenv("SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S", "2.5"))
    # Tool-surface deep review PR-B (docs/reviews/TOOL-SURFACE-DEEP-REVIEW-2026-07-23.md).
    # Relevance floor on rank_actions: drop candidates scoring below
    # max(FLOOR, best*FLOOR_RATIO) so a greeting stops surfacing 15
    # least-dissimilar actions. 0 = off (today's blind top-K, default).
    SEMANTIC_TOOL_ROUTING_FLOOR: float = float(os.getenv("SEMANTIC_TOOL_ROUTING_FLOOR", "0"))
    SEMANTIC_TOOL_ROUTING_FLOOR_RATIO: float = float(os.getenv("SEMANTIC_TOOL_ROUTING_FLOOR_RATIO", "0"))
    # What ships when narrowing CAN'T decide (no query / rank error / embed
    # timeout) while SEMANTIC_TOOL_ROUTING is on:
    #   open-full   — today's posture: full enum + full catalog (default)
    #   closed-pins — the pin set below + platform_find_tools discovery
    # (flag off entirely = operator chose the wide surface; always open-full).
    TOOL_FALLBACK_MODE: str = os.getenv("TOOL_FALLBACK_MODE", "open-full")
    TOOL_FALLBACK_PINS: str = os.getenv(
        "TOOL_FALLBACK_PINS",
        "platform_find_tools,platform_search_memory,platform_store_memory,platform_resume_context",
    )
    # PRD-228 (P228-RVW-4): actions the dispatcher_only surface (the heartbeat
    # orchestrator) MUST keep reachable regardless of the semantic top-K ranking
    # outcome, so the standing health loop can always read live floor state.
    # Unioned onto the NARROWED enum only (an open-full/None surface already
    # exposes every action). CSV of action names; each is role-gate-checked
    # before admission, so a gated/unknown name can never be forced in. Default
    # pins platform_fleet_status (PRD-228 US-003 fleet read-model tool) — without
    # a reserved slot it can rank out of the top-15 on any given tick and the
    # loop silently loses situational awareness (signal-tool-routing-drop class).
    HEARTBEAT_DISPATCHER_ALWAYS_INCLUDE: str = os.getenv(
        "HEARTBEAT_DISPATCHER_ALWAYS_INCLUDE",
        "platform_fleet_status",
    )
    # Shadow telemetry: log (never ship) what the PR-C relevance-gated surface
    # WOULD have been for each turn — the eval data for the flip.
    TOOL_SURFACE_SHADOW: bool = os.getenv("TOOL_SURFACE_SHADOW", "true").lower() == "true"
    # PR-C hybrid: max promoted actions that earn per-turn first-class schemas.
    # Shadow-only until the flip.
    TOOL_SURFACE_HYBRID_CAP: int = int(os.getenv("TOOL_SURFACE_HYBRID_CAP", "6"))
    PLATFORM_ACTIONS_MAX_TOKENS: int = int(os.getenv("PLATFORM_ACTIONS_MAX_TOKENS", "4000"))
    PLAYBOOK_CONTEXT_MAX_TOKENS: int = int(os.getenv("PLAYBOOK_CONTEXT_MAX_TOKENS", "2000"))
    MEMORY_SECTION_MAX_TOKENS: int = int(os.getenv("MEMORY_SECTION_MAX_TOKENS", "1500"))
    COMPOSIO_SECTION_MAX_TOKENS: int = int(os.getenv("COMPOSIO_SECTION_MAX_TOKENS", "1000"))
    # TOOL_ROUTING_GRAPH (default OFF) gates the learned tool-routing GRAPH reads
    # — GraphRouter.rank_chains on BOTH surfaces: the schema path
    # (SmartToolRouter.route, PRD-232 US-002) and the prompt catalog
    # (PlatformActionsSection._build_graph_filtered). OFF = zero GraphRouter
    # queries anywhere (the PRD-177 S4/S6 governance posture Gerard holds until
    # the uplift number clears the flip gate). ON = both surfaces consult the
    # graph. Distinct from SEMANTIC_TOOL_ROUTING (above), which gates embedding
    # narrowing and stays ON independently. The flip is a POST-MERGE HUMAN step
    # (PRD-232 US-013), never toggled by the build loop.
    TOOL_ROUTING_GRAPH: bool = os.getenv("TOOL_ROUTING_GRAPH", "false").lower() == "true"
    TOOL_ROUTING_GRAPH_MIN_CONFIDENCE: float = float(os.getenv("TOOL_ROUTING_GRAPH_MIN_CONFIDENCE", "0.6"))
    TOOL_ROUTING_GRAPH_AGENT_SAMPLE_FLOOR: int = int(os.getenv("TOOL_ROUTING_GRAPH_AGENT_SAMPLE_FLOOR", "50"))
    EDGE_BUILDER_HOUR_UTC: int = int(os.getenv("EDGE_BUILDER_HOUR_UTC", "3"))
    EDGE_BUILDER_WINDOW_DAYS: int = int(os.getenv("EDGE_BUILDER_WINDOW_DAYS", "30"))
    # PRD-177 S3 (F018): Composio action-metadata sync scheduler + fail-CLOSED
    # destructive gate. When the metadata table is empty (sync not yet run), a
    # destructive intent is DENIED rather than silently permitted; clearly
    # non-destructive intents still pass so a cold start is not bricked. The sync
    # job refreshes classifications daily on the same scheduler as the nightly
    # edge recompute.
    COMPOSIO_DESTRUCTIVE_FAIL_CLOSED: bool = os.getenv("COMPOSIO_DESTRUCTIVE_FAIL_CLOSED", "true").lower() == "true"
    COMPOSIO_SYNC_ENABLED: bool = os.getenv("COMPOSIO_SYNC_ENABLED", "true").lower() == "true"
    COMPOSIO_SYNC_HOUR_UTC: int = int(os.getenv("COMPOSIO_SYNC_HOUR_UTC", "4"))
    # PRD-141 US-019: batched incremental tool-execution signal recorder.
    # Opt-in (default off). Drains an in-process queue with ONE DB session per
    # flush — never a DB session or task per tool call.
    TOOL_SIGNAL_RECORDER_ENABLED: bool = os.getenv("TOOL_SIGNAL_RECORDER_ENABLED", "false").lower() == "true"
    TOOL_SIGNAL_FLUSH_BATCH_SIZE: int = int(os.getenv("TOOL_SIGNAL_FLUSH_BATCH_SIZE", "50"))
    TOOL_SIGNAL_FLUSH_INTERVAL_SECONDS: float = float(os.getenv("TOOL_SIGNAL_FLUSH_INTERVAL_SECONDS", "5.0"))
    TOOL_SIGNAL_QUEUE_MAXSIZE: int = int(os.getenv("TOOL_SIGNAL_QUEUE_MAXSIZE", "10000"))
    # PRD-143 S14: bounded (workspace, agent) -> last-selection-outcome stash
    # used to attach hit/fallback telemetry to platform_execute dispatches.
    TOOL_SELECTION_STASH_MAXSIZE: int = int(os.getenv("TOOL_SELECTION_STASH_MAXSIZE", "512"))

    # =============================================================================
    # OBSERVABILITY / TRACING (PRD-185 S9) — vendor-neutral trace seam
    # =============================================================================
    # Default OFF: zero overhead + zero data egress until explicitly enabled.
    # When ON, live traces/scores land at the tool-dispatch and retrieval
    # chokepoints via a vendor-neutral seam (core/observability/tracer.py) —
    # "was the tool call good / was retrieval grounded" as a queryable number.
    # Backend today = Langfuse Cloud; swappable behind the seam (data residency
    # is the axis that flips to self-host). `langfuse` is an OPTIONAL import: the
    # OFF path never imports it, and enabled-but-missing degrades to no-op.
    TRACING_ENABLED: bool = os.getenv("TRACING_ENABLED", "false").lower() in ("true", "1", "yes")
    TRACING_BACKEND: str = os.getenv("TRACING_BACKEND", "langfuse")
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # =============================================================================
    # AWS S3 VECTORS (PRD-42: Cloud Document Sync)
    # =============================================================================
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    # PRD-176 F089: S3 endpoint override for a local S3-compatible object store
    # (MinIO). Empty by default so prod/boto talks to real AWS S3; local compose
    # sets it to the MinIO endpoint so the knowledge flywheel persists outputs
    # instead of fail-softing to None on ephemeral disk.
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "")

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
    # PRD-176 F068: local-safe default (SaaS sets the railway worker host via env).
    AGENT_OPT_WORKER_URL: str = os.getenv("AGENT_OPT_WORKER_URL", "http://localhost:8080")

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
    RECIPE_LOG_S3_BUCKET: str = os.getenv("RECIPE_LOG_S3_BUCKET", "automatos-ai")

    # =============================================================================
    # QDRANT — PRD-108 Memory Field (Shared Semantic Context)
    # =============================================================================
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    # PRD-187 S1: in-process L3 durable memory — a second collection on the same
    # running Qdrant (field memory is the first). Wave-3 P2-16 consolidates both
    # under one client/config; keep the collection name a config knob for that.
    DURABLE_MEMORY_COLLECTION: str = os.getenv("DURABLE_MEMORY_COLLECTION", "durable_memory")
    # Heartbeat pings the durable store on this interval and feeds the memory
    # primitive tile (transition-only emits per PRD-185 S11).
    DURABLE_MEMORY_PROBE_INTERVAL_SECONDS: int = int(os.getenv("DURABLE_MEMORY_PROBE_INTERVAL_SECONDS", "30"))
    FIELD_EMBEDDING_DIM: int = int(os.getenv("FIELD_EMBEDDING_DIM", "2048"))
    FIELD_DECAY_RATE: float = float(os.getenv("FIELD_DECAY_RATE", "0.1"))
    FIELD_REINFORCE_BONUS: float = float(os.getenv("FIELD_REINFORCE_BONUS", "0.05"))
    FIELD_REINFORCE_CAP: float = float(os.getenv("FIELD_REINFORCE_CAP", "2.0"))
    FIELD_ARCHIVAL_THRESHOLD: float = float(os.getenv("FIELD_ARCHIVAL_THRESHOLD", "0.05"))
    FIELD_BOUNDARY_PERMEABILITY: float = float(os.getenv("FIELD_BOUNDARY_PERMEABILITY", "1.0"))
    # PRD-166 S2: adaptive half-life — each access divides the decay rate by
    # (1 + scale·access_count), so reused patterns persist longer.
    FIELD_HALF_LIFE_ACCESS_SCALE: float = float(os.getenv("FIELD_HALF_LIFE_ACCESS_SCALE", "0.5"))
    # PRD-166 S2/D11: query shape is config, not hardcoded caps.
    FIELD_QUERY_TOP_K: int = int(os.getenv("FIELD_QUERY_TOP_K", "10"))
    FIELD_QUERY_OVER_FETCH: int = int(os.getenv("FIELD_QUERY_OVER_FETCH", "3"))
    # Token budget for a field digest/query result block; over-budget → truncated=True.
    FIELD_QUERY_TOKEN_BUDGET: int = int(os.getenv("FIELD_QUERY_TOKEN_BUDGET", "1200"))
    # PRD-166 S1: compaction prunes points whose decayed strength falls below this
    # HARD floor (stricter than archival — archived stays queryable, pruned is deleted).
    FIELD_PRUNE_THRESHOLD: float = float(os.getenv("FIELD_PRUNE_THRESHOLD", "0.01"))
    FIELD_COMPACTION_MAX_SCAN: int = int(os.getenv("FIELD_COMPACTION_MAX_SCAN", "10000"))
    SHARED_CONTEXT_BACKEND: str = os.getenv("SHARED_CONTEXT_BACKEND", "vector_field")  # "vector_field" or "redis"
    # PRD-179 S2 (F049): how many completed missions the synthesis-flywheel ingest
    # sweep processes per coordinator tick. The sweep now orders newest-first and
    # excludes already-ingested / previously-failed runs SQL-side, so raising this
    # drains a backlog faster without ever re-touching a done run.
    FLYWHEEL_INGEST_BATCH: int = int(os.getenv("FLYWHEEL_INGEST_BATCH", "3"))

    # PRD-178 S4 — Field → durable (L3) promotion.
    # Strong, frequently-recalled field patterns are distilled into durable
    # memory BEFORE compaction hard-deletes them (else the field never becomes
    # durable). Thresholds are config, not hardcoded (D11).
    FIELD_PROMOTION_ENABLED: bool = os.getenv("FIELD_PROMOTION_ENABLED", "true").lower() in ("true", "1", "yes")
    # Minimum DECAYED strength for a pattern to be promotion-eligible.
    FIELD_PROMOTION_MIN_STRENGTH: float = float(os.getenv("FIELD_PROMOTION_MIN_STRENGTH", "0.5"))
    # Minimum access_count (reuse across tasks/missions) to promote.
    FIELD_PROMOTION_MIN_ACCESS_COUNT: int = int(os.getenv("FIELD_PROMOTION_MIN_ACCESS_COUNT", "3"))
    # Max points scanned per workspace per promotion run (bounds the scroll).
    FIELD_PROMOTION_MAX_SCAN: int = int(os.getenv("FIELD_PROMOTION_MAX_SCAN", "10000"))
    # Daily promotion job hour (UTC).
    FIELD_PROMOTION_HOUR_UTC: int = int(os.getenv("FIELD_PROMOTION_HOUR_UTC", "4"))
    # TAINT GATE (top-risk #4 — promotion is the memory-poisoning surface): a
    # pattern whose provenance names an untrusted external source is NEVER
    # promoted to durable memory. Comma-separated source tags treated as tainted.
    FIELD_PROMOTION_UNTRUSTED_SOURCES: str = os.getenv(
        "FIELD_PROMOTION_UNTRUSTED_SOURCES", "email,web,inbound,external,webhook,channel"
    )

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
    AUTOMATOS_DOCUMENTS_DIRS: str = os.getenv("AUTOMATOS_DOCUMENTS_DIRS") or os.getenv("AUTOMATOS_DOCUMENTS_DIR") or os.getenv("DOCUMENTS_DIR")
    IMAGE_STORE_LOCAL_DIR: str = os.getenv("IMAGE_STORE_LOCAL_DIR")
    GOTENBERG_URL: str = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")
    DOCUMENT_STORAGE_DIR: str = os.getenv("DOCUMENT_STORAGE_DIR", "documents")

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
        """Rerank on the retrieval hot path (default: ON — PRD-188 S1).

        Cohere stays a graceful-degrade seam: with no COHERE_API_KEY the
        reranker returns identity order, never an error, so ON is safe even
        before the key exists.
        """
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "rerank_enabled", "true")
            return str(val).lower() == "true" if val else True
        except Exception:
            return os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"

    @property
    def RAG_RERANK_MODEL(self) -> str:
        """Cohere rerank model id — the default lives here, not in
        rerank_manager (PRD-188 S1). rerank-v3.5 is the current verified
        Cohere id; upgrading is a setting/env flip, never a code change."""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "rerank_model", "rerank-v3.5")
            return str(val) if val else "rerank-v3.5"
        except Exception:
            return os.getenv("RAG_RERANK_MODEL", "rerank-v3.5")

    @property
    def RAG_HYBRID_ENABLED(self) -> bool:
        """Real dense+sparse hybrid retrieval (default: ON — PRD-188 S3).

        The BM25 leg reads the document_chunks.search_vector index the
        platform already maintains on every insert; OFF preserves the old
        dense-only behaviour exactly.
        """
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "hybrid_enabled", "true")
            return str(val).lower() == "true" if val else True
        except Exception:
            return os.getenv("RAG_HYBRID_ENABLED", "true").lower() == "true"

    @property
    def RAG_QUERY_ENHANCEMENT_ENABLED(self) -> bool:
        """LLM query enhancement — HyDE, decomposition, expansion — on the
        retrieval hot path (default: OFF).

        The 2026-07 live retrieval baseline measured enhancement at −26.9
        recall@5 points versus plain dense retrieval while adding ~4 LLM
        calls per query (evals/baseline/kg_retrieval_2026-07.json). OFF
        until an eval shows a variant that pays; the setting keeps it one
        flip away, and the eval lever grid still exercises it explicitly.
        """
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "query_enhancement_enabled", "false")
            return str(val).lower() == "true" if val else False
        except Exception:
            return os.getenv("RAG_QUERY_ENHANCEMENT_ENABLED", "false").lower() == "true"

    @property
    def RAG_CONTEXTUAL_ANNOTATIONS_ENABLED(self) -> bool:
        """Contextual chunk annotations at ingestion (default: OFF — PRD-188 S2).

        Stays OFF until the existing background reprocess has re-annotated the
        corpus, so live retrieval never sees a half-annotated store. Flipping
        it on only affects documents ingested/reprocessed after the flip.
        """
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "contextual_annotations_enabled", "false")
            return str(val).lower() == "true" if val else False
        except Exception:
            return os.getenv("RAG_CONTEXTUAL_ANNOTATIONS_ENABLED", "false").lower() == "true"

    @property
    def RAG_CONTEXTUAL_ANNOTATION_MODEL(self) -> str:
        """Haiku-class model for chunk situating contexts (PRD-188 S2) — cheap,
        prompt-cached over the parent document."""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("rag", "contextual_annotation_model", "claude-haiku-4-5")
            return str(val) if val else "claude-haiku-4-5"
        except Exception:
            return os.getenv("RAG_CONTEXTUAL_ANNOTATION_MODEL", "claude-haiku-4-5")

    # PRD-179 S4 (F070): rag_feedback feeds retrieval ranking. A document that
    # accrued negative feedback (thumbs_down, or a rating at/below the floor) has
    # its retrieval score multiplied by this factor on the live hot path — a doc
    # marked unhelpful de-ranks and can fall out of the top-K. 1.0 disables it.
    RAG_FEEDBACK_PENALTY_FACTOR: float = float(os.getenv("RAG_FEEDBACK_PENALTY_FACTOR", "0.5"))
    # A rating at/below this counts as negative (thumbs_down is always negative).
    RAG_FEEDBACK_NEGATIVE_RATING_MAX: int = int(os.getenv("RAG_FEEDBACK_NEGATIVE_RATING_MAX", "2"))
    # Only feedback from the last N days shapes ranking (stale opinions decay out).
    RAG_FEEDBACK_LOOKBACK_DAYS: int = int(os.getenv("RAG_FEEDBACK_LOOKBACK_DAYS", "90"))

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
    # TRIAL CREDIT (PRD-222 W1·S9: the $5 onboarding trial ledger)
    # =============================================================================
    # A one-time, platform-funded usage allowance that funds a new user's
    # onboarding "first mile" on the platform key, so the value moment (BOOM)
    # lands BEFORE the BYOK ask (D4). The ledger itself lives in
    # workspaces.onboarding.trial JSONB — no new table; enforcement + spend
    # accumulation live in services/trial_ledger.py. These are the only tunables.
    # Kill switch — false disables all trial grants; provisioning otherwise unchanged.
    TRIAL_ENABLED: bool = os.getenv("TRIAL_ENABLED", "true").lower() == "true"
    # Dollars granted per new Clerk user (once — checked across all their workspaces).
    TRIAL_CREDIT_USD: float = float(os.getenv("TRIAL_CREDIT_USD", "5.00"))
    # Platform-wide daily ceiling on aggregate trial spend; new grants PAUSE once
    # today's accumulated trial spend reaches this (US-005's daily counter feeds it).
    TRIAL_GLOBAL_DAILY_USD: float = float(os.getenv("TRIAL_GLOBAL_DAILY_USD", "25.00"))
    # Models a trial workspace may use on the platform key. Reuses the platform's
    # existing economical model comma-list (BUDGET_MODELS, PRD-54) — no new id invented.
    TRIAL_MODEL_ALLOWLIST: str = os.getenv("TRIAL_MODEL_ALLOWLIST", BUDGET_MODELS)

    # PRD-222 W1·S10 (D9) — DEV/OPS ONLY, TEMPORARY. Arms
    # POST /api/workspaces/current/onboarding/reset so the operator can re-run
    # onboarding in ONE workspace with a single alias account, instead of
    # provisioning and hard-deleting a workspace per attempt. Default OFF: when
    # false the endpoint 404s (unadvertised — never 403). This flag is
    # TEMPORARY: remove once onboarding QA moves to a seeded fixture flow (W2+).
    ONBOARDING_RESET_ENABLED: bool = os.getenv("ONBOARDING_RESET_ENABLED", "false").lower() == "true"

    # PRD-207 S4: Auto Live tuning constants. These are NUMERIC dials only —
    # the ON-switch (`voice.live_enabled`) and the Retell credentials live in
    # DB system_settings (category 'voice', masked; see
    # modules/voice/live_settings.py) so arming is a super-admin Settings-page
    # act: no env var, no redeploy. (The former RETELL_* env keys are gone —
    # PRD-143 killed .env-as-config; PRD-207 moved the seam to settings.)
    VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES: int = int(
        os.getenv("VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES", "100")
    )
    VOICE_LIVE_ACTIVE_CALL_RESERVE_MINUTES: int = int(
        os.getenv("VOICE_LIVE_ACTIVE_CALL_RESERVE_MINUTES", "10")
    )
    VOICE_LIVE_MAX_CALL_MINUTES: int = int(os.getenv("VOICE_LIVE_MAX_CALL_MINUTES", "30"))
    # Spoken turns ride a FAST PATH: only this many recent messages feed the
    # prompt (a call is a conversation, not an archive replay) — first-token
    # time is the product in voice.
    VOICE_LIVE_TURN_HISTORY_MESSAGES: int = int(
        os.getenv("VOICE_LIVE_TURN_HISTORY_MESSAGES", "12")
    )
    # If the brain has produced NO first frame this many seconds into a turn,
    # speak a short honest acknowledgment so the caller hears life instead of
    # dead air (0 disables the watchdog; empty text keeps the log but says
    # nothing). Retell severs a socket after ~5s without its ping_pong echoed,
    # so silence here used to read as a hang even when the turn was healthy.
    VOICE_LIVE_FIRST_FRAME_ACK_SECONDS: float = float(
        os.getenv("VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", "2.5")
    )
    VOICE_LIVE_FIRST_FRAME_ACK_TEXT: str = os.getenv(
        "VOICE_LIVE_FIRST_FRAME_ACK_TEXT", "One moment."
    )
    # PRD-207: Retell agent STT / turn-taking tuning — set at agent creation
    # AND re-applied on every one-click re-arm. Pinning the language stops
    # multilingual STT from hallucinating confident nonsense; background-speech
    # denoising kills TV / room bleed (the classic corruptor); accurate STT
    # trades a little latency for far fewer garbage transcripts; low
    # interruption sensitivity stops ambient noise from grabbing the turn
    # ("the mic is too sensitive"). All env-overridable — e.g. en-GB for a
    # British/Irish speaker whose accent transcribes better there.
    VOICE_LIVE_LANGUAGE: str = os.getenv("VOICE_LIVE_LANGUAGE", "en-GB")
    # ``noise-and-background-speech-cancellation`` (the aggressive mode) also
    # cancelled the SPEAKER at normal volume — measured live: a whole call
    # logged turns=0 because it treated his own voice as background and he had
    # to shout. ``noise-cancellation`` filters steady room noise and keeps the
    # person talking.
    VOICE_LIVE_DENOISING_MODE: str = os.getenv(
        "VOICE_LIVE_DENOISING_MODE", "noise-cancellation"
    )
    VOICE_LIVE_STT_MODE: str = os.getenv("VOICE_LIVE_STT_MODE", "accurate")
    # 0.2 made a normal-volume interruption unable to take the turn; 0.5 lets a
    # person barge in without ambient noise doing it for them.
    VOICE_LIVE_INTERRUPTION_SENSITIVITY: float = float(
        os.getenv("VOICE_LIVE_INTERRUPTION_SENSITIVITY", "0.5")
    )
    VOICE_LIVE_RESPONSIVENESS: float = float(os.getenv("VOICE_LIVE_RESPONSIVENESS", "0.9"))
    # Auto's voice. Retell ships no Irish voice; an ElevenLabs voice imported
    # into the Retell dashboard appears here as a normal voice id, so swapping
    # her accent is this one dial. Verified present in list-voices.
    VOICE_LIVE_VOICE_ID: str = os.getenv("VOICE_LIVE_VOICE_ID", "11labs-Willa")
    # <1 slower, >1 faster. Retell's default reads a touch brisk for a
    # colleague-across-the-desk register.
    VOICE_LIVE_VOICE_SPEED: float = float(os.getenv("VOICE_LIVE_VOICE_SPEED", "0.95"))
    # Prosody variation. Higher = more expressive, less predictable.
    VOICE_LIVE_VOICE_TEMPERATURE: float = float(
        os.getenv("VOICE_LIVE_VOICE_TEMPERATURE", "0.9")
    )
    # Let Retell speak "$5.20" / "2026-07-24" / "api.automatos.app" as words.
    # Belt to speechify()'s braces: the model is told not to emit them, this
    # rescues the ones that slip through.
    VOICE_LIVE_NORMALIZE_FOR_SPEECH: bool = os.getenv(
        "VOICE_LIVE_NORMALIZE_FOR_SPEECH", "true"
    ).strip().lower() == "true"
    # Retell nags ("are you still there?") when the caller goes quiet. One
    # reminder after 15s beats a conversation that keeps prodding you.
    VOICE_LIVE_REMINDER_TRIGGER_MS: int = int(
        os.getenv("VOICE_LIVE_REMINDER_TRIGGER_MS", "15000")
    )
    VOICE_LIVE_REMINDER_MAX_COUNT: int = int(
        os.getenv("VOICE_LIVE_REMINDER_MAX_COUNT", "1")
    )
    # Spoken replies are emitted as whole clauses rather than raw model tokens:
    # the sanitizer that strips markdown can only work on a complete unit
    # (``**`` straddles chunk boundaries). false = legacy token passthrough.
    VOICE_LIVE_SPEECH_UNITS: bool = os.getenv(
        "VOICE_LIVE_SPEECH_UNITS", "true"
    ).strip().lower() == "true"
    VOICE_LIVE_SPEECH_UNIT_MAX_CHARS: int = int(
        os.getenv("VOICE_LIVE_SPEECH_UNIT_MAX_CHARS", "180")
    )
    # Time-to-first-audio guard (PRD-203's property: the caller hears something
    # while the agent is still generating). The FIRST unit of a turn flushes at
    # a much smaller ceiling than the rest, so an opening clause starts playing
    # instead of waiting for the sentence to land. Later units use the full
    # ceiling, where prosody matters more than milliseconds.
    VOICE_LIVE_SPEECH_FIRST_UNIT_MAX_CHARS: int = int(
        os.getenv("VOICE_LIVE_SPEECH_FIRST_UNIT_MAX_CHARS", "60")
    )
    # Auto is told she is being HEARD on spoken turns (short sentences, no
    # markdown, no URLs read aloud). Same brain, different medium — false
    # restores the old behaviour of speaking her chat-formatted answer.
    VOICE_LIVE_SPOKEN_STYLE: bool = os.getenv(
        "VOICE_LIVE_SPOKEN_STYLE", "true"
    ).strip().lower() == "true"
    # Spoken turns skip the Composio EXECUTION surface (third-party app actions:
    # Gmail/Slack/etc.). Measured root cause of 20-90s voice latency: loading
    # all 23 Composio apps → 44 promoted action schemas + a 137-action
    # dispatcher enum inflated every turn's LLM calls to 24-36k input tokens and
    # drove a ~5-call agentic loop. Dropping Composio takes the agent from ~58
    # tools to ~14 core tools — memory, knowledge search, and reasoning are ALL
    # retained (this is NOT the force_text_only memory-lobotomy). Dial: set
    # false to restore full parity with typed chat at the latency cost.
    VOICE_LIVE_SKIP_COMPOSIO: bool = os.getenv(
        "VOICE_LIVE_SKIP_COMPOSIO", "true"
    ).strip().lower() == "true"
    # The public host Retell must reach for the custom-LLM socket + events
    # webhook (one-click arming builds the URLs from it). Railway injects
    # RAILWAY_PUBLIC_DOMAIN; override with PUBLIC_API_HOST where that's absent.
    PUBLIC_API_HOST: str = (
        os.getenv("PUBLIC_API_HOST", os.getenv("RAILWAY_PUBLIC_DOMAIN", "api.automatos.app"))
        or "api.automatos.app"
    ).strip().rstrip("/")

    def validate_security(self) -> None:
        """PRD-172: fail-closed validation of tenant-isolation secrets.

        Called from ``lifespan`` — outside any swallowing ``run_stage`` — so a
        misconfigured multi-tenant secret turns a silent cross-tenant leak into a
        loud boot failure rather than shipping fail-open.

        Raises ``RuntimeError`` (which aborts boot) when:

        - F005: S3 Vectors is enabled but ``S3_VECTORS_BUCKET`` is unset. (A
          shared bucket with no ``{workspace_id}`` placeholder is allowed —
          tenant isolation is enforced per-query by ``S3VectorsBackend.search()``
          (fail-closed on ``workspace_id``), not by the bucket layout.)

        Deliberately NOT checked here:

        - Widget CORS. Storefront origins are authorised from the per-key
          ``SdkApiKey.allowed_domains`` the merchant already maintains — there
          is no global widget allowlist to validate. See ``api/widgets/cors.py``.
        - ``SHOPIFY_INTERNAL_API_KEY`` (F004). The machine lanes it guards
          (Shopify app-install provisioning, GDPR verticals) fail CLOSED at the
          endpoint — ``_verify_internal_key`` returns 503 when the key is unset,
          with no fail-open branch. In the current deployment model there IS no
          Automatos-owned Shopify app: clients connect their own stores and all
          merchant traffic authenticates with per-workspace widget keys, so the
          key is intentionally absent and a boot-abort here would make every
          saas boot fail for a surface that is correctly dark (2026-08-01
          incident: exactly that, via #616).
        """
        errors: list[str] = []

        # F005 / PRD-186 S3 — vector-plane config integrity, extracted so CI
        # can pin the same rules the boot phase enforces (one assertion, no
        # duplicate strings).
        try:
            self.assert_vector_config_integrity()
        except RuntimeError as e:
            errors.append(str(e))

        if errors:
            raise RuntimeError(
                "Tenant-isolation security config invalid (PRD-172):\n  - "
                + "\n  - ".join(errors)
            )

        # PRD-175 (F008) — the edition boot guard runs in the same hard-fail phase
        # so a saas deploy that lost its Clerk env aborts boot rather than silently
        # downgrading to the anonymous local identity and serving tenant data.
        self.validate_auth_edition()

    def assert_vector_config_integrity(self) -> None:
        """PRD-186 S3: pure vector-plane config-integrity assertion.

        The shared-bucket rule set — raises ``RuntimeError`` when the committed
        config would ship the document plane dark or geometrically wrong:

        - S3 Vectors enabled with no ``S3_VECTORS_BUCKET`` set (the F005
          lesson: exactly this drift left the plane dark for weeks while a
          swallowing boot stage reported ``failed`` and served traffic anyway);
        - a non-positive ``S3_VECTORS_DIMENSION`` (writes/queries would carry
          meaningless geometry).

        A shared bucket with NO ``{workspace_id}`` placeholder is VALID — the
        live prod shape. Tenant isolation is enforced fail-closed at query
        time by ``S3VectorsBackend.search()`` (PRD-186 S1), not by bucket
        layout; the old hard placeholder requirement broke a working
        shared-bucket deployment (2026-07-02) and stays retired.

        Called from ``validate_security()`` in the hard-fail boot phase
        (``main._boot_phase_1_core`` — outside any swallowing ``run_stage``)
        and pinned directly by CI.
        """
        errors: list[str] = []

        if self.S3_VECTORS_ENABLED:
            bucket = (self.S3_VECTORS_BUCKET or "").strip()
            if not bucket:
                errors.append(
                    "S3_VECTORS_ENABLED=true but S3_VECTORS_BUCKET is unset."
                )
            try:
                dimension = int(self.S3_VECTORS_DIMENSION)
            except (TypeError, ValueError):
                dimension = 0
            if dimension <= 0:
                errors.append(
                    "S3_VECTORS_DIMENSION must be a positive integer "
                    f"(got {self.S3_VECTORS_DIMENSION!r})."
                )

        if errors:
            raise RuntimeError(
                "Vector config integrity invalid (PRD-186 S3):\n  - "
                + "\n  - ".join(errors)
            )

    def validate_auth_edition(self) -> None:
        """PRD-175 (F008): fail-closed boot guard for the open-core edition flag.

        Keeps the two editions from silently blending into the accidental
        "auth-not-configured" fallthrough that made local undeployable:

        - ``saas``  requires the Clerk env (``CLERK_JWKS_URL`` +
          ``CLERK_SECRET_KEY``). A saas boot with no Clerk is a misconfiguration,
          not a silent anonymous downgrade (the one real danger, review §5.3) —
          it must be a loud boot failure.
        - ``local`` requires a ``DEFAULT_WORKSPACE_ID`` — the single local
          workspace the auto-authenticated local session resolves to (W6 seeds it).
          It requires NO Clerk env.

        Raises ``RuntimeError`` (aborting boot) on a contradiction.
        """
        errors: list[str] = []

        if self.AUTH_EDITION == "saas":
            if not (self.CLERK_JWKS_URL or "").strip():
                errors.append("CLERK_JWKS_URL is unset")
            if not (self.CLERK_SECRET_KEY or "").strip():
                errors.append("CLERK_SECRET_KEY is unset")
            if errors:
                raise RuntimeError(
                    "AUTH_EDITION=saas requires Clerk to be configured, but "
                    + " and ".join(errors)
                    + ". A saas boot with no Clerk must fail fast, not fall through "
                    "to the anonymous local identity (which would serve tenant data "
                    "unauthenticated). Set the Clerk env, or set AUTH_EDITION=local "
                    "for a no-login local instance."
                )
        elif self.AUTH_EDITION == "local":
            if not (self.DEFAULT_WORKSPACE_ID or "").strip():
                raise RuntimeError(
                    "AUTH_EDITION=local requires DEFAULT_WORKSPACE_ID (the single "
                    "local workspace the auto-authenticated local session resolves "
                    "to). Set DEFAULT_WORKSPACE_ID to the seeded local workspace."
                )

    def validate(self) -> bool:
        """
        Validate required configuration
        Returns True if all required config is present
        """
        errors = []

        if not all([self.POSTGRES_DB, self.POSTGRES_USER, self.POSTGRES_HOST, self.POSTGRES_PORT]):
            errors.append("Database not configured: POSTGRES_DB, POSTGRES_USER, POSTGRES_HOST, POSTGRES_PORT required")

        if not self.ORCHESTRATOR_API_KEY:
            errors.append("ORCHESTRATOR_API_KEY (or AUTOMATOS_API_KEY / API_KEY) required")

        if errors:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        return True
    
    def print_config(self, show_secrets: bool = False) -> None:
        """Print current configuration (masks secrets by default)."""
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
        else:
            print(f"OpenAI Key: {'Set' if self.OPENAI_API_KEY else 'Not set'}")
            print(f"Anthropic Key: {'Set' if self.ANTHROPIC_API_KEY else 'Not set'}")

        print("=" * 60)

# Singleton instance
config = Config()

# ---------------------------------------------------------------------------
# PRD-82C: Complexity tier → token budget lookup
# Keys match ComplexityTier enum values (simple, moderate, complex) + "synthesis"
# ---------------------------------------------------------------------------
COMPLEXITY_TOKEN_BUDGET: dict[str, int] = {
    "simple": config.COMPLEXITY_TOKEN_BUDGET_SIMPLE,
    "moderate": config.COMPLEXITY_TOKEN_BUDGET_MODERATE,
    "complex": config.COMPLEXITY_TOKEN_BUDGET_COMPLEX,
    "synthesis": config.COMPLEXITY_TOKEN_BUDGET_SYNTHESIS,
}

# Backward compatibility alias
orchestrator_config = config

# ---------------------------------------------------------------------------
# PRD-222 W2·S1 (US-023): PLAN_TIERS — the v1 tier contract (approved strawman,
# 2026-08-28). DISPLAY PRICING ONLY — no billing or commerce is wired anywhere
# (PRD §12 Q5); ``display_price_usd`` sources an EARLY-ACCESS label, never a
# charge. Capability families (codegraph/nl2sql/team/voice) and the quota limits
# both derive from here: US-024 reads them for exposure profiles, and US-025's
# assignment helper (services/plan_tiers.py) writes the limits into
# ``workspaces.plan_limits`` under the keys live code already reads
# (seats→max_members, max_agents, budget). ``enterprise`` is a COMING-SOON
# display entry only — never assignable.
#
# Every number is env-overridable WITHOUT a redeploy via
# ``AUTOMATOS_PLAN_TIERS_JSON`` (a JSON object deep-merged onto these defaults)
# so tiers can be tuned live while testing. ``0`` means "unlimited" for
# max_agents / watcher_limit, and "no ceiling / custom" for budget_usd.
# ---------------------------------------------------------------------------
_PLAN_TIERS_DEFAULTS: dict[str, dict] = {
    "basic": {
        "display_name": "Basic",
        "display_price_usd": 19,
        "price_label": "early access",
        "assignable": True,
        "seats": 1,
        "max_agents": 5,
        "mission_concurrency": 1,
        "watcher_limit": 1,
        "marketplace_depth": 1,
        "budget_usd": 25,
        "families": {"codegraph": False, "nl2sql": False, "team": False, "voice": False},
    },
    "pro": {
        "display_name": "Pro",
        "display_price_usd": 49,
        "price_label": "early access",
        "assignable": True,
        "seats": 5,
        "max_agents": 20,
        "mission_concurrency": 3,
        "watcher_limit": 5,
        "marketplace_depth": 2,
        "budget_usd": 100,
        "families": {"codegraph": True, "nl2sql": True, "team": True, "voice": False},
    },
    "business": {
        "display_name": "Business",
        "display_price_usd": 99,
        "price_label": "early access",
        "assignable": True,
        "seats": 25,
        "max_agents": 0,
        "mission_concurrency": 10,
        "watcher_limit": 0,
        "marketplace_depth": 3,
        "budget_usd": 0,
        "families": {"codegraph": True, "nl2sql": True, "team": True, "voice": True},
    },
    "enterprise": {
        # Coming-soon placeholder — no ``families`` key on purpose. A tier
        # RESTRICTS only by declaring families, so a tier with none is treated as
        # UNRESTRICTED by both enforcement layers (nav + Auto's tool surface) —
        # see services/plan_tiers.exposure_for_plan / filter_tools_by_plan
        # (RVW-5). When enterprise is promoted assignable via
        # AUTOMATOS_PLAN_TIERS_JSON, add a ``families`` map to gate it; without
        # one it correctly exposes the FULL surface (top tier ≥ business).
        "display_name": "Enterprise",
        "assignable": False,
        "coming_soon": True,
    },
}


def load_plan_tiers(env_override=None) -> dict:
    """Resolve PLAN_TIERS from the defaults + an optional JSON env override.

    The override (``AUTOMATOS_PLAN_TIERS_JSON`` by default, or an explicit string
    for tests) is a JSON object of ``{tier: {field: value}}`` deep-merged onto
    the defaults — so one env var can reprice or re-gate any tier with no code
    change or redeploy. Malformed JSON is ignored (defaults stand) and logged.
    Returns a fresh deep copy so callers can never mutate the module constant.
    """
    tiers = copy.deepcopy(_PLAN_TIERS_DEFAULTS)
    raw = env_override if env_override is not None else os.getenv("AUTOMATOS_PLAN_TIERS_JSON", "")
    raw = (raw or "").strip()
    if not raw:
        return tiers
    try:
        override = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("[config] AUTOMATOS_PLAN_TIERS_JSON is not valid JSON — using tier defaults")
        return tiers
    if not isinstance(override, dict):
        return tiers
    for name, fields in override.items():
        if not isinstance(fields, dict):
            continue
        merged = dict(tiers.get(name, {}))
        for key, value in fields.items():
            if key == "families" and isinstance(value, dict):
                fam = dict(merged.get("families", {}))
                fam.update(value)
                merged["families"] = fam
            else:
                merged[key] = value
        tiers[name] = merged
    return tiers


PLAN_TIERS: dict[str, dict] = load_plan_tiers()

# ---------------------------------------------------------------------------
# PRD-222 W2·S1b (US-024): tool-name → capability-family map. A platform tool
# whose name matches an entry belongs to that family; a family DISABLED for the
# workspace's tier (PLAN_TIERS[plan]["families"]) is trimmed from Auto's per-turn
# tool surface at the single assembly seam
# (modules/tools/tool_router.get_tools_for_agent_async → services.plan_tiers.
# filter_tools_by_plan). Tools in NO family are CORE — always present. This map
# is config-side (the filter itself hardcodes no family names) so re-gating is a
# config/env change, and its keys MUST be the family names used in
# PLAN_TIERS[*]["families"]. Match semantics: exact name, OR prefix when the
# entry ends in '_'. ``voice`` gates nav/exposure only — Retell (PRD-207) is a
# separate lane with no platform_execute tools — so it maps to an empty list.
# Note: ``platform_get_activity_feed`` is deliberately NOT in nl2sql — the
# Command Center is available to every tier (only NL2SQL/analytics is gated).
TOOL_FAMILIES: dict[str, list] = {
    "codegraph": ["platform_codegraph_"],  # prefix — all 9 first-class CodeGraph tools
    "nl2sql": [
        "platform_query_data",              # the NL2SQL data query
        "platform_get_llm_usage", "platform_get_cost_breakdown", "platform_workspace_stats",
        "platform_get_success_rate", "platform_get_completion_time", "platform_get_error_rates",
        "platform_get_queue_depth", "platform_get_efficiency_score", "platform_get_cost_per_execution",
        "platform_get_peak_hours", "platform_get_bottlenecks", "platform_get_predictive_alerts",
        "platform_get_agent_ranking",
    ],
    "team": [
        "platform_list_members", "platform_invite_member",
        "platform_set_member_role", "platform_remove_member",
    ],
    "voice": [],
}

# Validate on import (non-blocking)
# if not config.validate():
#     logger.warning("⚠️  WARNING: Configuration validation failed")

# Export for easy import
__all__ = ['config', 'Config', 'orchestrator_config', 'PLAN_TIERS', 'load_plan_tiers', 'TOOL_FAMILIES']
