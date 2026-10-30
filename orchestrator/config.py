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

# The hosted product's API host: PUBLIC_API_HOST's default. A local install still on it
# has not named its own host, so it never advertises this one (PRD-266 K15).
HOSTED_PUBLIC_API_HOST = "api.automatos.app"

# An unset publish address is this machine only, as compose's own default (PRD-266 §4.3).
DEFAULT_PUBLISH_ADDRESS = "127.0.0.1"

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
    # For Railway: https://automatos-ai-frontend-production.up.railway.app
    # For custom domain: https://ui.automatos.app
    _cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000")
    CORS_ALLOW_ORIGINS: str = ",".join([origin.strip() for origin in _cors_origins.split(",") if origin.strip()])
    
    # =============================================================================
    # LLM KEYS (All providers)
    # =============================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    # The workspace an organization-level ANTHROPIC_API_KEY bills (wrkspc_…), sent as
    # anthropic-workspace-id. Unset for a key created inside a workspace (9 Oct 2026).
    ANTHROPIC_WORKSPACE_ID: str = os.getenv("ANTHROPIC_WORKSPACE_ID")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    # #873: resource URL; the client calls <endpoint>/openai/v1/ (no api-version, so AZURE_OPENAI_API_VERSION is retired)
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    XAI_API_KEY: str = os.getenv("XAI_API_KEY")
    # PRD-236: BYO-key providers served through the generic OpenAI-compatible
    # adapter. NVIDIA (build.nvidia.com) is a TRIAL endpoint — the key is the
    # user's own agreement with NVIDIA; the saas edition never holds one
    # (core/llm/providers.py marks it byok_only).
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
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

    # PRD-256 US-008 (Decision D5): the one place a failover model could be set.
    # Empty by default: a provider that rate-limits or refuses a call fails the
    # turn honestly (consumers/chatbot/turn_errors.py), never answering on another
    # model. Set, a refused call is asked once more on it (core/llm/failover.py).
    LLM_FAILOVER_MODEL: str = (os.getenv("LLM_FAILOVER_MODEL", "") or "").strip()
    
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

        Resolves system_settings (memory.distill_model) → env MEMORY_DISTILL_MODEL →
        the System LLM's own model, the tier the distiller runs on (P256-FIX-T1: the
        Gemini default 404'd on an Anthropic System LLM) → DEFAULT_LLM_MODEL."""
        from core.llm.defaults import DEFAULT_LLM_MODEL
        try:
            from core.llm.manager import get_system_setting
            return (get_system_setting("memory", "distill_model")
                    or os.getenv("MEMORY_DISTILL_MODEL")
                    or get_system_setting("system_llm", "model", DEFAULT_LLM_MODEL))
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
    def CHATBOT_RELEASE_DB_BETWEEN_MODEL_CALLS(self) -> bool:
        """F105-B: before each model call a chat turn ends its read-only
        transaction, so it holds no pool connection while the model thinks.
        Dial chatbot.release_db_between_model_calls; on unless set false."""
        from core.llm.manager import get_system_setting
        val = get_system_setting("chatbot", "release_db_between_model_calls", "true")
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    @property
    def CHATBOT_KNOWLEDGE_PREFETCH(self) -> bool:
        """F085-A: a question in a workspace with documents is searched before
        the model's first call. Dial chatbot.knowledge_prefetch; on unless set false."""
        from core.llm.manager import get_system_setting
        val = get_system_setting("chatbot", "knowledge_prefetch", "true")
        return str(val).strip().lower() in ("1", "true", "yes", "on")

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

    # =============================================================================
    # PRD-240 — agents can read and search the web (platform capability, not an app)
    # =============================================================================
    # WEB_ACCESS: on by default in the local edition (the operator's own machine,
    # their own keys), off in saas unless an operator sets it. When off, no
    # provider search tool is attached and web_fetch / web_search answer
    # {available:false, reason}. Private, loopback, link-local and metadata
    # ranges are ALWAYS refused (core/security/url_validator.py) — the denylist
    # adds to that, never replaces it.
    # Prompt caching on the OpenRouter route (2026-09-16): Anthropic models get an
    # explicit cache_control breakpoint on the stable system prefix plus automatic
    # caching of the conversation tail. Default TTL 5 minutes (a cache write costs
    # 1.25x input, a read 0.1x); PROMPT_CACHE_TTL_1H=on switches to the 1-hour TTL
    # (write 2x) for turns that usually arrive more than 5 minutes apart — it pays
    # for itself from the second read in an hour.
    PROMPT_CACHE_TTL_1H: bool = (os.getenv("PROMPT_CACHE_TTL_1H", "") or "").strip().lower() in ("on", "true", "1", "yes")
    _WEB_ACCESS_RAW = (os.getenv("WEB_ACCESS", "") or "").strip().lower()
    WEB_ACCESS: bool = (
        _WEB_ACCESS_RAW in ("on", "true", "1", "yes")
        if _WEB_ACCESS_RAW
        else AUTH_EDITION == "local"
    )
    # Hosts agents may never reach, suffix-matched: "example.com" also blocks
    # "www.example.com". Comma-separated, bare hostnames.
    WEB_ACCESS_DENY: tuple = tuple(
        h.strip().lower().lstrip(".")
        for h in (os.getenv("WEB_ACCESS_DENY", "") or "").split(",")
        if h.strip()
    )
    # Which engine answers web_search: auto = the first configured of
    # openrouter -> composio -> searxng. "off" disables search but not fetch.
    _WEB_SEARCH_PROVIDER_RAW = (os.getenv("WEB_SEARCH_PROVIDER", "auto") or "auto").strip().lower()
    WEB_SEARCH_PROVIDER: str = (
        _WEB_SEARCH_PROVIDER_RAW
        if _WEB_SEARCH_PROVIDER_RAW in ("auto", "openrouter", "composio", "searxng", "off")
        else "auto"
    )
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    # Searches the model may run inside ONE turn on a route with a server-side
    # search tool (OpenRouter). Each search is billed by the provider.
    WEB_SEARCH_MAX_USES_PER_TURN: int = int(os.getenv("WEB_SEARCH_MAX_USES_PER_TURN", "3"))
    # The cheap tool-capable model the web_search ACTION uses on OpenRouter to
    # run one search (the engine does the searching; the model only lists it).
    WEB_SEARCH_OPENROUTER_MODEL: str = os.getenv("WEB_SEARCH_OPENROUTER_MODEL", "openai/gpt-4o-mini")
    # Self-hosted SearXNG (docker compose --profile search): http://searxng:8080
    SEARXNG_URL: str = (os.getenv("SEARXNG_URL", "") or "").strip().rstrip("/")
    WEB_FETCH_MAX_BYTES: int = int(os.getenv("WEB_FETCH_MAX_BYTES", str(2 * 1024 * 1024)))
    WEB_FETCH_TIMEOUT_SECONDS: int = int(os.getenv("WEB_FETCH_TIMEOUT_SECONDS", "20"))

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
    # Where the auto-generated Fernet key is persisted when CREDENTIAL_ENCRYPTION_KEY
    # is unset. Default = the historical location (core/.credential_key, inside the
    # image's writable layer — lost on container recreate). The local edition
    # points this at a named volume (envs/api.defaults) so stored keys survive
    # `docker compose down` / rebuilds. PRD-209 live-test finding.
    CREDENTIAL_KEY_FILE: str = os.getenv("CREDENTIAL_KEY_FILE", "")

    # Local edition: the single operator's identity. The entrypoint seeds a users
    # row with this email (id 1 — the chat service's own fallback), and the
    # anonymous local session carries it so chat/attribution resolve a real user.
    # PRD-233 S6 adds the editable profile on top. Unused in saas.
    LOCAL_OPERATOR_EMAIL: str = os.getenv("LOCAL_OPERATOR_EMAIL", "local@automatos.local")
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
    # PRD-236 — base URLs of the other OpenAI-compatible providers (registry reads these).
    NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
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
    # PRD-266 US-016b: the Automatos moderation inbox for content reports. SaaS only
    # (a local install posts nothing even when it is set); an https URL; a credential,
    # never logged. PRD-266 open question 2 names the destination.
    MODERATION_WEBHOOK_URL: str = os.getenv("MODERATION_WEBHOOK_URL", "")
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
    # PRD-264 EUAI-10 (D16): the public detection route GET /api/public/provenance/{sha256}
    # takes no account, so each client IP gets this many checks a minute (at least one).
    PROVENANCE_RATE_PER_MINUTE: int = max(1, int(os.getenv("PROVENANCE_RATE_PER_MINUTE", "30")))
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
    # The backstop across EVERY requeue path, not just lease expiry. Night 1
    # re-dispatched three tickets 188 times between them and one 534 times,
    # because the paths that send a ticket back to `assigned` (an answered ask,
    # a resumed session) never consulted a ceiling at all. Single digits by
    # design: past this a ticket needs a person, not another attempt.
    BOARD_DISPATCH_HARD_ATTEMPT_CAP: int = int(os.getenv("BOARD_DISPATCH_HARD_ATTEMPT_CAP", "8"))
    # Per-agent concurrency slots: at most this many of an agent's tasks run at
    # once; the rest stay 'assigned' (the DB is the queue — double-texting is
    # queued, never dropped). The claim honours this via in_progress counts.
    BOARD_DISPATCH_AGENT_SLOTS: int = int(os.getenv("BOARD_DISPATCH_AGENT_SLOTS", "2"))

    # =============================================================================
    # SESSION MODE (PRD-234): tickets run as the user's OWN Claude Code sessions
    # =============================================================================
    # ``runtime: cli`` agents are claimed by a paired local CLI host, never by the
    # LLM factory. Local edition ONLY: validate_auth_edition() aborts a saas boot
    # that sets this (the SaaS path stays byte-identical). Default off everywhere.
    CLI_RUNTIME_ENABLED: bool = os.getenv("CLI_RUNTIME_ENABLED", "false").strip().lower() in ("true", "1", "yes", "on")
    # PRD-256 FX-016: the runtime a new agent gets when platform_create_agent names none:
    # 'api' (the platform's models) or 'cli' (a session, which needs CLI_RUNTIME_ENABLED).
    DEFAULT_AGENT_RUNTIME: str = os.getenv("DEFAULT_AGENT_RUNTIME", "api").strip().lower()
    # #848: the host shares no folder with this backend (a cluster): the claim asks it to
    # upload what its session left in the ticket's deliverables folder, and the backend writes
    # it into the workspace volume (services/session_uploads.py). Off = compose's shared folder.
    CLI_SESSION_FILE_UPLOAD: bool = os.getenv("CLI_SESSION_FILE_UPLOAD", "false").strip().lower() in ("true", "1", "yes", "on")
    # …and the most one run of a ticket may upload in all (each file: the document upload limit).
    CLI_SESSION_UPLOAD_MAX_TOTAL_MB: int = int(os.getenv("CLI_SESSION_UPLOAD_MAX_TOTAL_MB", "200"))
    # PRD-245 W1 (local edition): how many Automatos tool calls ONE ticket's
    # session may make through the loopback MCP bridge. A bound, not a budget —
    # the model is on the operator's own plan; this stops a looping session from
    # hammering the board. 0 = no cap.
    SESSION_TOOLS_MAX_CALLS_PER_TICKET: int = int(os.getenv("SESSION_TOOLS_MAX_CALLS_PER_TICKET", "200"))
    # F330 (night 9c): how many session tool calls run at once in this process. A
    # burst waits for a slot holding no pool connection; keep it well under the
    # pool (10 + 20), since each call can hold its own connection plus one more.
    SESSION_TOOL_CONCURRENCY: int = int(os.getenv("SESSION_TOOL_CONCURRENCY", "6"))
    # P256-FIX-N17-9 (F448): how long, from the call's arrival, a session's document or memory
    # read (services/session_read_deadline.DEADLINED_TOOLS) may take before it answers with a
    # worded failure, so the session's own client never gives up first ("The operation timed out.").
    SESSION_TOOL_READ_DEADLINE_S: float = float(os.getenv("SESSION_TOOL_READ_DEADLINE_S", "45"))
    # F161 (night 5): the most of one file an earlier mission step saved that a
    # session reads in one read_step_file call; a longer file comes back cut,
    # with a note saying so. Held under the bridge's own result cap.
    SESSION_STEP_FILE_MAX_CHARS: int = int(os.getenv("SESSION_STEP_FILE_MAX_CHARS", "30000"))
    # F413 (night 13): the most of a deliverable's text platform_get_deliverable hands the model to
    # quote in its reply; a longer text comes back cut there, with the link to the rest.
    DELIVERABLE_SHOWN_MAX_CHARS: int = int(os.getenv("DELIVERABLE_SHOWN_MAX_CHARS", "4000"))
    # F170 (night 5): how long a running mission's steps sit queued behind
    # another mission's Claude Code step before the mission says so.
    MISSION_WAIT_NOTE_AFTER_SECONDS: int = int(os.getenv("MISSION_WAIT_NOTE_AFTER_SECONDS", "300"))
    # PRD-234 S2 (local edition): the owner's projects folder on the HOST machine, as
    # the CLI host sees it. Only used to map a session's file paths onto the
    # worker's "projects/" view (the folder is bind-mounted read-only into the
    # workspace-worker, not into this container). Empty = no projects folder.
    LOCAL_PROJECTS_DIR: str = os.getenv("LOCAL_PROJECTS_DIR", "").strip()
    # PRD-239: how the projects folder is mounted into the worker (ro|rw) — shown in Settings → Session mode.
    LOCAL_PROJECTS_MOUNT: str = os.getenv("LOCAL_PROJECTS_MOUNT", "").strip().lower()
    # Local edition: the deliverables root on the HOST — the folder compose mounts as
    # the local workspace's root (/workspaces/<DEFAULT_WORKSPACE_ID>), so the host's
    # `<root>/sessions/68/x.py` is the worker's `sessions/68/x.py`. `make up` exports
    # it absolute; a relative value (plain `docker compose up`) is display-only.
    AUTOMATOS_WORKSPACE_DIR: str = os.getenv("AUTOMATOS_WORKSPACE_DIR", "").strip()
    # PRD-239 S3b: how long a playbook step waits for a session agent's Claude Code session
    # (a real session runs for minutes; API steps keep the recipe's own step timeout).
    CLI_LANE_STEP_TIMEOUT_SECONDS: int = int(os.getenv("CLI_LANE_STEP_TIMEOUT_SECONDS", "1800"))
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
    # PRD-238 S4: how long Auto may wait inside ONE turn for a ticket to finish
    # (platform_wait_for_task), and how often it re-checks while waiting. Beyond
    # the budget Auto reports "still running" and the watcher takes over.
    CHATBOT_WAIT_BUDGET_S: int = int(os.getenv("CHATBOT_WAIT_BUDGET_S", "90"))
    CHATBOT_WAIT_POLL_S: int = int(os.getenv("CHATBOT_WAIT_POLL_S", "5"))
    # PRD-239 S1: the cap on the skills text rendered into a session agent's
    # appended system prompt (persona + skills ride it, stable per agent).
    CLI_SESSION_SKILLS_MAX_CHARS: int = int(os.getenv("CLI_SESSION_SKILLS_MAX_CHARS", "24000"))
    # PRD-239 S3: how often a playbook step or mission task re-reads the ticket it
    # filed for a session agent while it waits for the session to end.
    CLI_LANE_POLL_SECONDS: int = int(os.getenv("CLI_LANE_POLL_SECONDS", "5"))
    # F114 (run 4): how long that wait keeps polling through a database outage
    # (Postgres crash-restarted mid-wait; ~10 s in recovery) before giving up.
    CLI_LANE_DB_OUTAGE_GRACE_SECONDS: int = int(os.getenv("CLI_LANE_DB_OUTAGE_GRACE_SECONDS", "180"))
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
    # PRD-251 US-120: how often a playbook step that waits on a render (a fixed
    # generate_document step) stamps the run's progress. Keep it well under
    # TASK_STALL_TIMEOUT_SECONDS, or the reconciler fails the run as stalled.
    PLAYBOOK_PROGRESS_STAMP_SECONDS: int = int(os.getenv("PLAYBOOK_PROGRESS_STAMP_SECONDS", "60"))

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
    # F132: how late a scheduled playbook may still start. APScheduler's default (1 s)
    # dropped a fire the loop reached a moment late, silently.
    PLAYBOOK_SCHEDULE_MISFIRE_GRACE_SECONDS: int = int(os.getenv("PLAYBOOK_SCHEDULE_MISFIRE_GRACE_SECONDS", "300"))

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
    # F078: whether an operator actually pointed us at them. The defaults above
    # only keep a fresh clone off Railway's topology — nothing answers there in
    # the compose stack — so the monitoring actions are offered only when set.
    LOKI_CONFIGURED: bool = bool(os.getenv("LOKI_URL"))
    PROMETHEUS_CONFIGURED: bool = bool(os.getenv("PROMETHEUS_URL"))
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
    # PRD-263 US-054: the installation's switch that allows HARNESS to raise approval
    # grants. HIGH RISK — default OFF. Only while it is on does a ``propose`` workspace
    # get an Approvals card per proposed change (services.harness_grants), applied only
    # on the owner's click; off, every workspace runs report_only and gets no card.
    HARNESS_SELF_MANAGEMENT_ENABLED: bool = os.getenv("HARNESS_SELF_MANAGEMENT_ENABLED", "false").lower() == "true"
    # PRD-263 US-012: when the HARNESS weekly sweep runs, as a five-field cron read in
    # UTC. Default Monday 10:00 UTC: a weekday daytime hour, outside the night window
    # where F408, F409 and F416 happened. It becomes a trigger only through
    # services.schedule_util.scheduler_cron_trigger (F240: weekday as day names).
    HARNESS_SWEEP_CRON: str = os.getenv("HARNESS_SWEEP_CRON", "0 10 * * 1")
    # PRD-263 US-014a/US-014b: how DIAGNOSE judges each agent x metric (design section
    # 5.1 ORG-4; services.harness_diagnose passes each to core.stats.harness_verdicts).
    # Below HARNESS_MIN_SAMPLES in the judged week or its reference a pair is
    # INSUFFICIENT_DATA. The reference is the HARNESS_REF_DAYS before the judged week,
    # in whole ISO weeks, less weeks already flagged and weeks after an applied change.
    # A rate flags only when it worsens by both HARNESS_ABS_DROP and HARNESS_REL_DROP
    # (fractions) with a one-sided p below HARNESS_ALPHA; cost or tokens per run only
    # when the median rises by both HARNESS_COST_REL_RISE and HARNESS_COST_ABS_RISE ($)
    # with a Mann-Whitney p below HARNESS_ALPHA. Benjamini-Hochberg at HARNESS_FDR_Q
    # runs across every test of a run. A first flag is WATCH; REGRESSION needs a flag
    # in the previous ISO week too, or p below HARNESS_STRONG_P. Below
    # HARNESS_MIN_COVERAGE (judged over active agents) convergence reads insufficient_data.
    HARNESS_MIN_SAMPLES: int = int(os.getenv("HARNESS_MIN_SAMPLES", "20"))
    HARNESS_REF_DAYS: int = int(os.getenv("HARNESS_REF_DAYS", "28"))
    HARNESS_ABS_DROP: float = float(os.getenv("HARNESS_ABS_DROP", "0.15"))
    HARNESS_REL_DROP: float = float(os.getenv("HARNESS_REL_DROP", "0.20"))
    HARNESS_ALPHA: float = float(os.getenv("HARNESS_ALPHA", "0.05"))
    HARNESS_FDR_Q: float = float(os.getenv("HARNESS_FDR_Q", "0.10"))
    HARNESS_COST_REL_RISE: float = float(os.getenv("HARNESS_COST_REL_RISE", "0.25"))
    HARNESS_COST_ABS_RISE: float = float(os.getenv("HARNESS_COST_ABS_RISE", "0.005"))
    HARNESS_STRONG_P: float = float(os.getenv("HARNESS_STRONG_P", "0.001"))
    HARNESS_MIN_COVERAGE: float = float(os.getenv("HARNESS_MIN_COVERAGE", "0.5"))
    # PRD-263 US-019 (FR-8): an agent's calls of one app in the judged week flag when there
    # are at least HARNESS_MIN_SAMPLES and the Wilson lower bound (95%) of their failure
    # rate is at least HARNESS_TOOL_FAIL_WILSON_MIN. A flag carried by sign-in and
    # connection failures is reported as "reconnect <app>", never as a bad tool.
    HARNESS_TOOL_FAIL_WILSON_MIN: float = float(os.getenv("HARNESS_TOOL_FAIL_WILSON_MIN", "0.5"))
    # PRD-263 US-050 (section 11-Q10 defaults): HARNESS suggests doubling an enabled heartbeat's
    # interval (clamped to 5-1440 minutes, as the heartbeat handler clamps it) only when its
    # judged week has at least HARNESS_HEARTBEAT_MIN_RUNS runs with an outcome (a run whose
    # objective_met is None is not counted) and the Wilson upper bound (95%) of the share
    # that met their objective is below HARNESS_HEARTBEAT_MAX_MET.
    HARNESS_HEARTBEAT_MIN_RUNS: int = int(os.getenv("HARNESS_HEARTBEAT_MIN_RUNS", "20"))
    HARNESS_HEARTBEAT_MAX_MET: float = float(os.getenv("HARNESS_HEARTBEAT_MAX_MET", "0.2"))
    # PRD-263 US-053 (section 11-Q7 default): a HARNESS proposal granted, or being applied,
    # longer than HARNESS_RECONCILE_GRACE_MINUTES ago with no outcome is closed as failed.
    # The daily sweep of HARNESS grants runs on HARNESS_GRANT_SWEEP_CRON, read in UTC
    # (default 09:00 every day; services.harness_grants).
    HARNESS_RECONCILE_GRACE_MINUTES: int = int(os.getenv("HARNESS_RECONCILE_GRACE_MINUTES", "60"))
    HARNESS_GRANT_SWEEP_CRON: str = os.getenv("HARNESS_GRANT_SWEEP_CRON", "0 9 * * *")
    # PRD-263 US-056: each change HARNESS applied is judged week by week against the
    # HARNESS_REF_DAYS before it (services.harness_post_change). A metric waits for a week
    # of at least HARNESS_MIN_SAMPLES for up to HARNESS_POST_CHANGE_MAX_WEEKS weeks after
    # the change; otherwise the report says it was not judged.
    HARNESS_POST_CHANGE_MAX_WEEKS: int = int(os.getenv("HARNESS_POST_CHANGE_MAX_WEEKS", "4"))

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
        """setting → env → the System LLM vendor's default (core/llm/vendor_fit.py)."""
        try:
            from core.llm.manager import get_system_setting
            from core.llm.vendor_fit import default_verifier_mapping
            return get_system_setting(
                "coordination", "verifier_model_mapping",
                os.getenv("COORDINATOR_VERIFIER_MODEL_MAPPING", default_verifier_mapping()),
            )
        except Exception:
            return os.getenv("COORDINATOR_VERIFIER_MODEL_MAPPING", "")

    @property
    def COORDINATOR_VERIFIER_FALLBACK_MODEL(self) -> str:
        """setting → env → the System LLM vendor's default (core/llm/vendor_fit.py)."""
        from core.llm.defaults import VERIFIER_FALLBACK_MODEL
        try:
            from core.llm.manager import get_system_setting
            from core.llm.vendor_fit import default_verifier_fallback
            return get_system_setting(
                "coordination", "verifier_fallback_model",
                os.getenv("COORDINATOR_VERIFIER_FALLBACK_MODEL", default_verifier_fallback()),
            )
        except Exception:
            return os.getenv("COORDINATOR_VERIFIER_FALLBACK_MODEL", VERIFIER_FALLBACK_MODEL)
    # History-based agent scoring (PRD-82B US-003)
    COORDINATOR_HISTORY_LOOKBACK_DAYS: int = int(os.getenv("COORDINATOR_HISTORY_LOOKBACK_DAYS", "30"))
    COORDINATOR_HISTORY_MIN_DATAPOINTS: int = int(os.getenv("COORDINATOR_HISTORY_MIN_DATAPOINTS", "3"))
    # PRD-164 S2 (Q21): upper bound on the per-dispatch semantic-signal
    # computation (task embedding + capability-card cosine + field query) so a
    # hung embedding/Qdrant backend can never stall the dispatch tick — on
    # timeout the matcher falls back to lexical-only scoring.
    AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS: float = float(os.getenv("AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS", "10"))
    # PRD-256 N15-6 (F427): an agent with a run this recent (a ticket in progress, or finished
    # into Review or Done) is said to be able to run, whatever its stored flag or a lapsed
    # CLI host heartbeat says (services/agent_last_run.py).
    AGENT_RECENT_RUN_WINDOW_SECONDS: int = int(os.getenv("AGENT_RECENT_RUN_WINDOW_SECONDS", "3600"))
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
    # A mission task worked by a Claude Code session is not an API turn: research
    # or a build routinely runs 10-30 minutes. The power-mode timeout still bounds
    # API turns; this is the ceiling for a session ticket that is demonstrably
    # alive (night 1, finding 21 — every mixed mission failed its session tasks at
    # four minutes and spawned duplicates beside the sessions still running).
    MISSION_CLI_TICKET_TIMEOUT_SECONDS: int = int(os.getenv("MISSION_CLI_TICKET_TIMEOUT_SECONDS", "3600"))
    # Graph extraction shares the ``system_llm`` tier, whose max_tokens is 8000.
    # Night 1 (2026-09-18): the model emitted ~2.4x its input and 311 of 497
    # calls were cut off MID-JSON at that ceiling — they parsed to nothing and
    # were billed in full ($9.06 of a $9.60 line). Extraction gets its own,
    # lower ceiling, and a prompt that fits inside it.
    GRAPH_EXTRACTION_MAX_OUTPUT_TOKENS: int = int(os.getenv("GRAPH_EXTRACTION_MAX_OUTPUT_TOKENS", "2000"))
    # #836: a thinking model's reasoning tokens count toward max_tokens. A call to
    # one reserves its visible-answer budget plus this many times it for thinking.
    # 4 keeps the whole answer at OpenRouter's "high" effort, which gives thinking
    # 80% of max_tokens (budget_tokens = max_tokens x 0.8).
    LLM_THINKING_ALLOWANCE_RATIO: float = float(os.getenv("LLM_THINKING_ALLOWANCE_RATIO", "4"))
    # #836: how often each worker re-reads the llm_output_budget rows and which
    # models think (core/llm/budget_snapshot.py). A new row is in force on every
    # worker within this many seconds; 0 reads once, at boot.
    LLM_OUTPUT_BUDGET_REFRESH_SECONDS: float = float(os.getenv("LLM_OUTPUT_BUDGET_REFRESH_SECONDS", "30"))
    # F206 (night 6): 20 of 25 calls ended at the cap. 25 nodes and 40 edges run
    # to about 68 lines, and the cap held about 36, so what was cut was mostly
    # edges. The contract now fits the cap: a full answer (node about 43
    # tokens, edge 61, hyperedge 80, cl100k) is about 1,670, about 83% of 2,000.
    GRAPH_EXTRACTION_MAX_NODES: int = int(os.getenv("GRAPH_EXTRACTION_MAX_NODES", "14"))
    GRAPH_EXTRACTION_MAX_EDGES: int = int(os.getenv("GRAPH_EXTRACTION_MAX_EDGES", "16"))
    GRAPH_EXTRACTION_MAX_HYPEREDGES: int = int(os.getenv("GRAPH_EXTRACTION_MAX_HYPEREDGES", "1"))
    # F051: a call that yields no parseable line is retried ONCE with the
    # ceiling lifted, rather than discarded after being paid for.
    GRAPH_EXTRACTION_RETRY_OUTPUT_TOKENS: int = int(os.getenv("GRAPH_EXTRACTION_RETRY_OUTPUT_TOKENS", "6000"))

    # The spend guard's own dials (fix-order §4). The first version borrowed
    # ``llm_cost_audit.daily_budget_alert_usd`` — a key whose name says "alert" —
    # and counted from date_trunc('day', NOW()), which is wrong three ways for
    # this workload: an overnight run spans two UTC days, so last night's spend
    # counted against tonight's ceiling ($10.89 of night 2's $20 before it had
    # spent anything); the window reset at 00:00Z = 01:00 local, in the MIDDLE of
    # every run it governs; and NOW() is the server's clock, not the owner's.
    #
    # A "spend day" therefore starts at SPEND_DAY_START_HOUR in SPEND_TIMEZONE —
    # midday by default, so a run from the evening to the small hours sits
    # wholly inside ONE window and last night sits wholly inside the previous
    # one. 0 turns the ceiling off, which is the historical behaviour.
    SPEND_CEILING_USD: float = float(os.getenv("SPEND_CEILING_USD", "0") or 0)
    SPEND_DAY_START_HOUR: int = int(os.getenv("SPEND_DAY_START_HOUR", "12"))
    SPEND_TIMEZONE: str = os.getenv("SPEND_TIMEZONE", "Europe/Lisbon")

    # F025 — the prompt-cache prefix. Tools are serialised BEFORE the system
    # prompt, so a tool block whose bytes move invalidates the whole cached
    # prefix. Night 1: call 1 of all 77 Auto turns read back only 2,432-3,456
    # tokens of a ~34k prefix while calls 2+ read ~33k, because the
    # platform_execute action enum is re-narrowed per query. With this on the
    # enum is the full eligible set (byte-stable between turns) and the
    # per-turn ranking is delivered as a late system line instead — the model
    # keeps the steer, the cache keeps the prefix.
    TOOL_ENUM_CACHE_STABLE: bool = os.getenv("TOOL_ENUM_CACHE_STABLE", "true").lower() == "true"
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
    # PRD-264 EUAI-03 (EU AI Act Art. 50(1)): a channel bot's reply carries "AI
    # assistant for {principal}." on first contact with a person and again once
    # this many hours have passed since the last notice. The last-notice time
    # lives in Redis with this TTL (channels/ai_notice.py). No setting turns it off.
    AI_DISCLOSURE_IDLE_HOURS: int = int(os.getenv("AI_DISCLOSURE_IDLE_HOURS", "24"))
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
    # F130: the playbook section carries the earlier steps' answers, so 2000 cut them
    # (the budget truncates from the end: the latest answer went first).
    PLAYBOOK_CONTEXT_MAX_TOKENS: int = int(os.getenv("PLAYBOOK_CONTEXT_MAX_TOKENS", "12000"))
    # F130: each earlier step's answer reaches a later step whole up to this many characters.
    PLAYBOOK_STEP_ANSWER_MAX_CHARS: int = int(os.getenv("PLAYBOOK_STEP_ANSWER_MAX_CHARS", "12000"))
    MEMORY_SECTION_MAX_TOKENS: int = int(os.getenv("MEMORY_SECTION_MAX_TOKENS", "1500"))
    # PRD-256 FX-015: the owner's standing rules ride every chat turn, newest first, up to this
    # many tokens; the block reads at most STANDING_RULES_SCAN_LIMIT candidate rows to find them
    # (a warning names a workspace that reaches it).
    STANDING_RULES_MAX_TOKENS: int = int(os.getenv("STANDING_RULES_MAX_TOKENS", "600"))
    STANDING_RULES_SCAN_LIMIT: int = int(os.getenv("STANDING_RULES_SCAN_LIMIT", "500"))
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
    # PRD-232 §6.5 (RVW-2, Gerard's ruling — the two-layer graph, amending PRD-177's
    # per-tenant lock): the learned graph reads as TWO layers — a tenant's own rows
    # (workspace_id == X) at full weight PLUS a text-free GLOBAL prior (workspace_id
    # IS NULL, aggregated across tenants) at REDUCED weight. This factor (0..1) scales
    # every global edge/affinity/failed_after row's contribution on a TENANT read, so a
    # tenant's own learned signal always dominates the borrowed cross-tenant prior — a
    # tenant edge of equal-or-higher confidence outranks the identical global edge. A
    # genuinely unscoped (workspace_id=None) system/eval read sees the global layer at
    # FULL weight (there is no tenant layer to prefer). Never hardcoded in the router.
    TOOL_ROUTING_GRAPH_GLOBAL_PRIOR_FACTOR: float = float(os.getenv("TOOL_ROUTING_GRAPH_GLOBAL_PRIOR_FACTOR", "0.5"))
    # PRD-232 US-010: cluster-aware reads — the graph expresses what it learned.
    # CLUSTER_MATCH_THRESHOLD is the cosine floor for assigning a live query to the
    # nearest ToolRoutingIntentCluster centroid (GraphRouter.rank_chains). Below it,
    # the query matches NO cluster and routing falls back to the embedding floor
    # (per-intent affinities do not apply) — a deliberate miss, never a wrong guess.
    TOOL_ROUTING_GRAPH_CLUSTER_MATCH_THRESHOLD: float = float(os.getenv("TOOL_ROUTING_GRAPH_CLUSTER_MATCH_THRESHOLD", "0.6"))
    # PRD-232 US-007: confidence stamped on the SEEDED cold-start affinity that the
    # synthetic-utterance seed writes alongside each seeded intent cluster. Set at the
    # graph min-confidence floor (matching metadata_graph_seed's meta_sibling floor) so
    # a seed PASSES the router filter but any organic succeeds_for_intent row of higher
    # Wilson confidence outranks it as real telemetry accrues — the seed is the floor,
    # learned usage always wins. Human-applied seed only; never read by a live turn.
    TOOL_ROUTING_SEED_CLUSTER_CONFIDENCE: float = float(os.getenv("TOOL_ROUTING_SEED_CLUSTER_CONFIDENCE", "0.6"))
    # PRD-232 US-014 (§6.2 LOCKED): promotion-as-prior. Only these pins ALWAYS attach
    # as first-class OpenAI schemas — the list starts from PRD-122's original ~13
    # promoted actions and adds platform_find_tools (the when-required discovery seam,
    # which MUST stay pinned). Every OTHER promoted=True action loses unconditional
    # attachment: its promoted flag becomes a ranking BOOST (below), and it attaches
    # first-class ONLY when it ranks into the query surface — otherwise it lives in the
    # platform_execute dispatcher enum like any action. Cuts the per-turn tool payload
    # from 47 unconditional schemas to ~14 pins + whatever the query ranked in. The
    # pin set lives HERE, never hardcoded in the router (CLAUDE.md §4). CSV, whitespace-
    # tolerant; role/tier gating (admin/super_admin_only) still applies to every pin.
    TOOL_ROUTING_PROMOTION_PINS: str = os.getenv(
        "TOOL_ROUTING_PROMOTION_PINS",
        "platform_find_tools,"
        "platform_list_agents,platform_get_agent,platform_create_agent,platform_update_agent,"
        "platform_browse_marketplace_agents,platform_browse_marketplace_skills,"
        "platform_browse_marketplace_plugins,platform_install_skill,platform_install_plugin,"
        "platform_get_system_health,platform_get_activity_feed,"
        "platform_search_memory,platform_store_memory,"
        # PRD-256 US-006: Auto's writes (and the two reads they lean on) as their own tools, strict schemas,
        # outside the dispatcher's enum — the used action sat outside the ranked top-15 on 53% of calls.
        "platform_create_task,platform_update_task,platform_update_task_status,platform_assign_task,"
        "platform_create_mission,platform_execute_playbook,platform_schedule_playbook,"
        "platform_create_social_post,platform_get_task,platform_query_data,"
        # PRD-256 FX-009: the owner's two mission decisions, first-class (night 12 guessed their ids).
        "platform_approve_mission,platform_cancel_mission",
    )
    # The additive ranking boost a promoted action gets in the shared cosine pass, so a
    # promoted action outranks an equal-cosine unpromoted one and is more likely to rank
    # into the surface (and thus attach first-class). Small — a PRIOR, not an override:
    # a strongly-relevant unpromoted action still outranks a barely-relevant promoted one.
    TOOL_ROUTING_PROMOTION_BOOST: float = float(os.getenv("TOOL_ROUTING_PROMOTION_BOOST", "0.05"))
    # When a cluster matches, per-intent affinity rows (intent_cluster_id == match)
    # apply at full weight; cluster-blind rows (intent_cluster_id IS NULL) still
    # apply as a WEAK global prior, discounted by this factor (0..1). Keeps a
    # global succeeds/fails signal informative without letting it override the
    # per-intent evidence the cluster match unlocked.
    TOOL_ROUTING_GRAPH_GLOBAL_AFFINITY_DISCOUNT: float = float(os.getenv("TOOL_ROUTING_GRAPH_GLOBAL_AFFINITY_DISCOUNT", "0.5"))
    # failed_after edges (a succeeded→b errored, within 2 steps) are read as an
    # EXPANSION PENALTY: a chain [a, b] whose transition reliably fails is de-ranked
    # by penalty = failed_after.confidence (Wilson lower bound of the failure rate)
    # * this weight. Scales how hard a learned failure suppresses a chain. This is
    # what turns the previously write-only failed_after rows into a live signal
    # (PRD-232 US-010c — no write-only tables).
    TOOL_ROUTING_GRAPH_FAILED_AFTER_PENALTY: float = float(os.getenv("TOOL_ROUTING_GRAPH_FAILED_AFTER_PENALTY", "1.0"))
    EDGE_BUILDER_HOUR_UTC: int = int(os.getenv("EDGE_BUILDER_HOUR_UTC", "3"))
    EDGE_BUILDER_WINDOW_DAYS: int = int(os.getenv("EDGE_BUILDER_WINDOW_DAYS", "30"))
    # F416 (night 14): the nightly run's memory is bounded. It clusters at most the
    # CLUSTER_MAX_QUERIES most recent queries (the embedding matrix's row cap), embeds
    # them EMBED_BATCH_SIZE at a time, reads the logs LOG_PAGE_SIZE rows a page, and
    # the k-means' working block stays under KMEANS_BLOCK_MB. It runs in a child
    # process whose address space is capped at WORKER_MEMORY_MB (0 = no cap), so
    # running out of memory ends the run, never the backend.
    EDGE_BUILDER_CLUSTER_MAX_QUERIES: int = int(os.getenv("EDGE_BUILDER_CLUSTER_MAX_QUERIES", "5000"))
    EDGE_BUILDER_EMBED_BATCH_SIZE: int = int(os.getenv("EDGE_BUILDER_EMBED_BATCH_SIZE", "256"))
    EDGE_BUILDER_LOG_PAGE_SIZE: int = int(os.getenv("EDGE_BUILDER_LOG_PAGE_SIZE", "1000"))
    EDGE_BUILDER_KMEANS_BLOCK_MB: int = int(os.getenv("EDGE_BUILDER_KMEANS_BLOCK_MB", "64"))
    EDGE_BUILDER_WORKER_MEMORY_MB: int = int(os.getenv("EDGE_BUILDER_WORKER_MEMORY_MB", "4096"))
    # PRD-232 US-011: the learning loop's gap + shown-vs-used signals (nightly).
    # GAP_RESOLUTION_HOURS: a tool_gap answered by a successful action within this
    # window in the SAME conversation becomes a succeeds_for_intent for that
    # action (the ground truth is what eventually served the intent).
    TOOL_ROUTING_GAP_RESOLUTION_HOURS: float = float(os.getenv("TOOL_ROUTING_GAP_RESOLUTION_HOURS", "24"))
    # SHOWN_DECAY_FACTOR: geometric erosion applied per shown-not-used excess to an
    # action's succeeds_for_intent weight (surfaced far more than used = dial down).
    TOOL_ROUTING_SHOWN_DECAY_FACTOR: float = float(os.getenv("TOOL_ROUTING_SHOWN_DECAY_FACTOR", "0.9"))
    # AFFINITY_WEIGHT_FLOOR: the shown-not-used decay never drops an affinity's
    # weight below this — a seeded/earned affinity is dialed down, never deleted.
    TOOL_ROUTING_AFFINITY_WEIGHT_FLOOR: float = float(os.getenv("TOOL_ROUTING_AFFINITY_WEIGHT_FLOOR", "0.5"))
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
    # PRD-232 US-009 (decision LOCKED §6.1): default ON. The write side of the
    # learning loop ships from day one — signals flow so harness stats are
    # non-zero within a day of deploy; rollback is this one env var. Drains an
    # in-process queue with ONE DB session per flush (never a DB session or task
    # per tool call), and flushes the pending batch on clean shutdown (stop()).
    # Bounded-loss window is honest: a HARD crash drops the in-process queue, but
    # the nightly edge_builder RECOMPUTES authoritative edges from the durable
    # tool_execution_logs, so only intra-day freshness is ever at risk.
    TOOL_SIGNAL_RECORDER_ENABLED: bool = os.getenv("TOOL_SIGNAL_RECORDER_ENABLED", "true").lower() == "true"
    TOOL_SIGNAL_FLUSH_BATCH_SIZE: int = int(os.getenv("TOOL_SIGNAL_FLUSH_BATCH_SIZE", "50"))
    TOOL_SIGNAL_FLUSH_INTERVAL_SECONDS: float = float(os.getenv("TOOL_SIGNAL_FLUSH_INTERVAL_SECONDS", "5.0"))
    TOOL_SIGNAL_QUEUE_MAXSIZE: int = int(os.getenv("TOOL_SIGNAL_QUEUE_MAXSIZE", "10000"))
    # F408 (night 13): a flush waits this long for a row lock, then drops its batch
    # with one warning (the nightly edge_builder recomputes what a dropped batch held).
    TOOL_SIGNAL_FLUSH_LOCK_TIMEOUT_MS: int = int(os.getenv("TOOL_SIGNAL_FLUSH_LOCK_TIMEOUT_MS", "5000"))
    # PRD-143 S14: bounded (workspace, agent) -> last-selection-outcome stash
    # used to attach hit/fallback telemetry to platform_execute dispatches.
    TOOL_SELECTION_STASH_MAXSIZE: int = int(os.getenv("TOOL_SELECTION_STASH_MAXSIZE", "512"))

    # =============================================================================
    # OBSERVABILITY / TRACING — the PRD-185 S9 seam (core/observability/tracer.py)
    # emits through OpenTelemetry (PRD-256 O3), so it follows OTEL_ENABLED below.
    # =============================================================================
    # PRD-256 (#847): OpenTelemetry traces over OTLP/HTTP, beside the PRD-73 metrics, logs
    # and correlation IDs (core/observability/otel.py). Default OFF: nothing from
    # `opentelemetry` is imported. The collector at the endpoint chooses the backend.
    OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "false").strip().lower() in ("true", "1", "yes", "on")
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "automatos-api")
    # The collector's OTLP/HTTP base address; /v1/traces is appended.
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    # `key=value,key2=value2` (URL-encoded values), e.g. a vendor's Authorization header.
    OTEL_EXPORTER_OTLP_HEADERS: str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    # The share of new traces kept (0.0–1.0); a request whose caller sampled it follows the caller.
    # Kept as text and parsed when tracing starts, so a bad value can never stop a boot (otel.py).
    OTEL_TRACES_SAMPLER_RATIO: str = os.getenv("OTEL_TRACES_SAMPLER_RATIO", "1.0")

    # =============================================================================
    # OBJECT STORAGE (S3 / MinIO) + AWS S3 VECTORS (PRD-42: Cloud Document Sync)
    # =============================================================================
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    # PRD-176 F089: S3 endpoint override for a local S3-compatible object store
    # (MinIO). Empty by default so prod/boto talks to real AWS S3; local compose
    # sets it to the MinIO endpoint so the knowledge flywheel persists outputs
    # instead of fail-softing to None on ephemeral disk.
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "")
    # PRD-233 S4 (absorbs PRD-151): every S3 client is built by core/storage/s3.py
    # from the three knobs above/below. S3_PUBLIC_ENDPOINT_URL is the host a
    # BROWSER can reach for presigned links (locally http://localhost:9000 while
    # the backend talks to http://minio:9000); empty = same as the endpoint, so
    # SaaS presigns are unchanged. S3_USE_PATH_STYLE forces path-style bucket
    # addressing (MinIO needs it, AWS uses virtual-host style); unset = derived
    # from S3_ENDPOINT_URL presence, so SaaS keeps boto's default.
    S3_PUBLIC_ENDPOINT_URL: str = os.getenv("S3_PUBLIC_ENDPOINT_URL", "")
    S3_USE_PATH_STYLE: bool = (
        os.getenv("S3_USE_PATH_STYLE") or ("true" if S3_ENDPOINT_URL else "false")
    ).lower() in ("true", "1", "yes")

    # S3 Vectors Configuration
    S3_VECTORS_ENABLED: bool = os.getenv("S3_VECTORS_ENABLED", "false").lower() == "true"
    S3_VECTORS_BUCKET: str = os.getenv("S3_VECTORS_BUCKET")
    S3_VECTORS_INDEX_NAME: str = os.getenv("S3_VECTORS_INDEX_NAME", "documents-index")
    S3_VECTORS_DIMENSION: int = int(os.getenv("S3_VECTORS_DIMENSION", "2048"))
    S3_VECTORS_METRIC: str = os.getenv("S3_VECTORS_METRIC", "cosine")

    # S3 Documents (general storage bucket)
    S3_DOCUMENTS_BUCKET: str = os.getenv("S3_DOCUMENTS_BUCKET", "automatos-ai")
    # PRD-251 S0.4c: GET /api/generated-images/{id} streams the S3 body in chunks
    # of this many bytes (never the whole object in memory).
    GENERATED_IMAGE_STREAM_CHUNK_BYTES: int = int(
        os.getenv("GENERATED_IMAGE_STREAM_CHUNK_BYTES", str(64 * 1024))
    )

    # =============================================================================
    # PRD-58: FutureAGI Integration (Prompt Scoring & Optimization)
    # =============================================================================
    FUTUREAGI_API_KEY: str = os.getenv("FUTUREAGI_API_KEY")
    # PRD-176 F068: local-safe default (SaaS sets the railway worker host via env).
    AGENT_OPT_WORKER_URL: str = os.getenv("AGENT_OPT_WORKER_URL", "http://localhost:8080")
    # PRD-262 US-006: sent as X-Internal-Token on every worker call. It must match the
    # worker's own AGENT_OPT_WORKER_TOKEN, which the worker refuses to start without.
    AGENT_OPT_WORKER_TOKEN: str = os.getenv("AGENT_OPT_WORKER_TOKEN", "").strip()

    # =============================================================================
    # JIRA BUG REPORTS (Pilot Helper Widget)
    # =============================================================================
    JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "PILOT")
    JIRA_BUG_REPORTS_ENABLED: bool = os.getenv("JIRA_BUG_REPORTS_ENABLED", "true").lower() == "true"

    # =============================================================================
    # MARKETPLACE / S3 (Plugin Marketplace)
    # =============================================================================
    MARKETPLACE_S3_BUCKET: str = os.getenv("MARKETPLACE_S3_BUCKET", "automatos-marketplace")
    # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_ENDPOINT_URL defined
    # above (object storage section). Local runs use MinIO through the same path —
    # the MARKETPLACE_LOCAL_DIR filesystem fallback was deleted (PRD-233 S4).
    PLUGIN_MAX_UPLOAD_SIZE_MB: int = int(os.getenv("PLUGIN_MAX_UPLOAD_SIZE_MB", "10"))
    PLUGIN_LLM_SCAN_MODEL: str = os.getenv("PLUGIN_LLM_SCAN_MODEL", "claude-haiku-4-5")
    PLUGIN_CACHE_TTL_SECONDS: int = int(os.getenv("PLUGIN_CACHE_TTL_SECONDS", "3600"))

    # =============================================================================
    # PRD-261 INSTALLABLE THEMES (Gerard D1, 18 Oct 2026; PRD Q25 size limits)
    # =============================================================================
    # Private code uploads platform-wide: on | off | admins | auto. `auto` is the
    # DEFAULT of each workspace's own setting (local edition on; SaaS on only in a
    # single-member workspace) that a workspace admin may switch on; `off` is a
    # ceiling. An unknown value is read as off (services/themes/records.py).
    THEMES_PRIVATE_UPLOAD: str = (os.getenv("THEMES_PRIVATE_UPLOAD", "auto") or "auto").strip().lower()
    # The theme Marketplace (W-C1 routes read it): on | off.
    THEMES_MARKETPLACE: str = (os.getenv("THEMES_MARKETPLACE", "on") or "on").strip().lower()
    # The .autotheme as uploaded; main.py sizes the import route's body limit from it.
    THEME_MAX_PACKAGE_MB: int = int(os.getenv("THEME_MAX_PACKAGE_MB", "10"))
    # JS plus CSS, the entry module, and each mascot clip (D1-a: 3 MB, 300 KB, 1 MB).
    THEME_MAX_CODE_KB: int = int(os.getenv("THEME_MAX_CODE_KB", "3072"))
    THEME_MAX_ENTRY_KB: int = int(os.getenv("THEME_MAX_ENTRY_KB", "300"))
    THEME_MAX_CLIP_KB: int = int(os.getenv("THEME_MAX_CLIP_KB", "1024"))
    # How long a browser keeps a theme file (immutable: the URL carries the digest).
    THEME_FILE_CACHE_SECONDS: int = int(os.getenv("THEME_FILE_CACHE_SECONDS", "86400"))
    # Step-up on sensitive routes (themes now; role changes with W-A2): a first factor this recent (D1-e: 10).
    STEP_UP_MAX_AGE_MINUTES: int = int(os.getenv("STEP_UP_MAX_AGE_MINUTES", "10"))
    # PRD-261 W-C1 Marketplace review (D3, 19 Oct: a person reviews every version, auto-publish
    # is off for themes). Q9: one pending submission per author at a time; on by default, so one
    # author can't queue many paid review stages at once.
    THEMES_ONE_OPEN_SUBMISSION: bool = (
        os.getenv("THEMES_ONE_OPEN_SUBMISSION", "true").strip().lower() in ("true", "1", "yes", "on")
    )
    # The automated stage's LLM check (services/themes/llm_stage.py), on PLUGIN_LLM_SCAN_MODEL:
    # tokens per chunk, chunks per package (more is "LLM stage did not run"), seconds for the whole
    # stage (parts unread by then are "did not run", so one upload holds its request for a bounded
    # time), seconds per call.
    THEME_LLM_CHUNK_TOKENS: int = int(os.getenv("THEME_LLM_CHUNK_TOKENS", "8000"))
    THEME_LLM_MAX_CHUNKS: int = int(os.getenv("THEME_LLM_MAX_CHUNKS", "48"))
    THEME_LLM_STAGE_SECONDS: int = int(os.getenv("THEME_LLM_STAGE_SECONDS", "300"))
    THEME_LLM_TIMEOUT_SECONDS: int = int(os.getenv("THEME_LLM_TIMEOUT_SECONDS", "60"))
    # What theme reviews may spend (services/themes/review_limits.py): review stages running at once
    # in one process (more is 503, try again); Marketplace submissions per person and private
    # uploads per workspace in a rolling 24 hours (more is 429); and the LLM tokens all theme
    # reviews may use in a rolling 24 hours (past it the LLM check doesn't run: review_required).
    THEME_REVIEW_MAX_CONCURRENT: int = int(os.getenv("THEME_REVIEW_MAX_CONCURRENT", "2"))
    THEME_SUBMISSIONS_PER_DAY: int = int(os.getenv("THEME_SUBMISSIONS_PER_DAY", "10"))
    THEME_UPLOADS_PER_DAY: int = int(os.getenv("THEME_UPLOADS_PER_DAY", "20"))
    THEME_REVIEW_DAILY_TOKENS: int = int(os.getenv("THEME_REVIEW_DAILY_TOKENS", "2000000"))
    # PRD-261 W-C3 theme revocations (QB-25: the local edition pulls the hosted feed by default).
    # The feed the local edition pulls (https, or http to loopback only; empty = no pull) and how
    # often (services/themes/revocation_pull.py, services/theme_revocation_scheduler.py). The hosted
    # edition serves the feed (api/themes_revocations_feed.py) with rows lifted in the last
    # THEME_REVOCATIONS_LIFTED_DAYS days, and neither side reads a feed over THEME_REVOCATIONS_FEED_MAX_KB.
    # THEME_REVOCATIONS_URL is a deploy-time trust boundary: only the environment sets it, and https
    # to any host passes revocation_pull.checked_url (no DNS or private-range check). Never let a
    # lower-trust actor set it (an admin setting, a per-workspace override) without a host allowlist
    # or a check of the resolved address.
    THEME_REVOCATIONS_URL: str = (
        os.getenv("THEME_REVOCATIONS_URL", "https://api.automatos.app/api/themes/revocations") or ""
    ).strip()
    THEME_REVOCATIONS_PULL_MINUTES: int = int(os.getenv("THEME_REVOCATIONS_PULL_MINUTES", "60"))
    THEME_REVOCATIONS_LIFTED_DAYS: int = int(os.getenv("THEME_REVOCATIONS_LIFTED_DAYS", "365"))
    THEME_REVOCATIONS_FEED_MAX_KB: int = int(os.getenv("THEME_REVOCATIONS_FEED_MAX_KB", "1024"))
    # No signing key here, ever: the kit repo's sign workflow holds it (D3-c, services/themes/signing.py).

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
    # F105: every embedding call has a bound. The OpenAI SDK default is a 600 s
    # read timeout with 2 retries — one stalled call held a search for minutes.
    # A single text (a query) is small, so it gets the shorter read bound.
    EMBEDDING_CONNECT_TIMEOUT_S: float = float(os.getenv("EMBEDDING_CONNECT_TIMEOUT_S", "5"))
    EMBEDDING_QUERY_TIMEOUT_S: float = float(os.getenv("EMBEDDING_QUERY_TIMEOUT_S", "15"))
    EMBEDDING_BATCH_TIMEOUT_S: float = float(os.getenv("EMBEDDING_BATCH_TIMEOUT_S", "60"))
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "1"))
    # F105: best-effort writes (usage rows, telemetry, heartbeat findings, the
    # NL2SQL audit) run on this many threads of their own, never on the loop.
    BEST_EFFORT_WRITE_THREADS: int = int(os.getenv("BEST_EFFORT_WRITE_THREADS", "4"))
    # ...and wait at most this long for a free pool connection, then drop the row.
    BEST_EFFORT_POOL_WAIT_S: float = float(os.getenv("BEST_EFFORT_POOL_WAIT_S", "2"))
    # F458: GET /health reads a full pool from its counters, never waiting on it, and
    # answers degraded (200). A pool seen full at every probe for this long is stuck:
    # /health answers 503, so the liveness watchdog (docker-entrypoint.sh, F416)
    # restarts the backend. A dead or hung backend still restarts within ~2 min.
    HEALTH_POOL_STUCK_SECONDS: float = float(os.getenv("HEALTH_POOL_STUCK_SECONDS", "300"))
    # F458: when a checkout leaves the pool full, one WARNING lists who holds every
    # connection and where each holder is now (core/database/pool_watch.py), at most
    # once per this many seconds. 0 = off.
    DB_POOL_FULL_LOG_SECONDS: float = float(os.getenv("DB_POOL_FULL_LOG_SECONDS", "60"))
    # F140: a playbook step's whole answer at most this long, ending by asking for
    # what it needs, stops the run and asks the owner. Night 4's 12 asks were at
    # most 402 characters; a longer answer is the step's work, even if it ends
    # on a question.
    PLAYBOOK_OWNER_ASK_MAX_CHARS: int = int(os.getenv("PLAYBOOK_OWNER_ASK_MAX_CHARS", "600"))
    # F183: an answer that says what it lacks, asks for it and puts its work off
    # until it has it ("Once I have …, I will …") is an ask up to this long,
    # wherever its questions sit. Night 6's #1097 was 662 characters.
    OWNER_ASK_DEFERRED_MAX_CHARS: int = int(os.getenv("OWNER_ASK_DEFERRED_MAX_CHARS", "1200"))
    # F105: document-vector searches run on this many threads of their own, never
    # on the loop; the local search is a full scan (F107) that holds a pool
    # connection for its whole length, so this also caps how many run at once.
    DOCUMENT_SEARCH_THREADS: int = int(os.getenv("DOCUMENT_SEARCH_THREADS", "4"))
    # F105: the Composio lookups before a model call (tool search, action hints)
    # run on this many threads of their own, never on the loop; each holds a pool
    # connection for its whole length, so this also caps how many run at once.
    COMPOSIO_LOOKUP_THREADS: int = int(os.getenv("COMPOSIO_LOOKUP_THREADS", "4"))
    # ...and an app's action list or a step search, once fetched, is answered
    # from memory for this long (core.composio.lookup_cache).
    COMPOSIO_LOOKUP_CACHE_TTL_SECONDS: float = float(os.getenv("COMPOSIO_LOOKUP_CACHE_TTL_SECONDS", "600"))
    # ...and a turn waits at most this long for one; then it goes on without
    # Composio tools. The SDK's lookup calls give up after as long, un-retried,
    # so a hung call frees its thread too.
    COMPOSIO_LOOKUP_TIMEOUT_SECONDS: float = float(os.getenv("COMPOSIO_LOOKUP_TIMEOUT_SECONDS", "20"))
    # F105: when the event loop stands still this long, the loop watchdog logs
    # one WARNING with the loop thread's stack (core/loop_watchdog.py). 0 = off.
    LOOP_STALL_LOG_SECONDS: float = float(os.getenv("LOOP_STALL_LOG_SECONDS", "2"))
    # F141: an OpenRouter catalog sync marks the models it no longer lists inactive
    # unless it fetched fewer than this share of the models listed before: a
    # partial answer would otherwise retire the rest of the catalog.
    OPENROUTER_DELIST_MIN_FETCH_RATIO: float = float(os.getenv("OPENROUTER_DELIST_MIN_FETCH_RATIO", "0.9"))
    # F085-A: how many passages the automatic search brings, and the relevance
    # (the retrieval funnel's final score) a passage needs to reach the prompt.
    KNOWLEDGE_PREFETCH_PASSAGES: int = int(os.getenv("KNOWLEDGE_PREFETCH_PASSAGES", "5"))
    KNOWLEDGE_PREFETCH_MIN_SCORE: float = float(os.getenv("KNOWLEDGE_PREFETCH_MIN_SCORE", "0.3"))
    # F103: a memory write (durable or mission field) that times out is tried
    # once more after this pause.
    MEMORY_WRITE_RETRY_PAUSE_S: float = float(os.getenv("MEMORY_WRITE_RETRY_PAUSE_S", "0.5"))

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
    DOCUMENT_STORAGE_DIR: str = os.getenv("DOCUMENT_STORAGE_DIR", "documents")
    # F353: draw a document Deliverable's first page for its card, after it is made.
    DOCUMENT_THUMBNAILS_ENABLED: bool = os.getenv("DOCUMENT_THUMBNAILS_ENABLED", "true").lower() == "true"

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

    # Cohere bills reranking per SEARCH UNIT (one query over up to 100 documents),
    # USD per 1,000 units — the analytics tracker prices each rerank call with it.
    # Cohere's published list price for rerank-v3.5 (2026): $2.00 / 1k searches.
    COHERE_RERANK_USD_PER_1K_SEARCHES: float = float(os.getenv("COHERE_RERANK_USD_PER_1K_SEARCHES", "2.0"))

    # PRD-248 — the decision seam (TypeSafe Jev direct or through OpenRouter).
    # Secrets and endpoints are env-only; the operating dials (route, model,
    # timeout, per-hook modes, confidence floor) live in system_settings
    # category ``decision_engine`` so a PoC flips from Settings → System with
    # no restart. A missing key leaves the engine idle — no route errors for it.
    TYPESAFE_API_KEY: str = os.getenv("TYPESAFE_API_KEY")
    TYPESAFE_API_URL: str = os.getenv("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone")
    # OpenRouter serves Jev on its alpha decisions path (2026-09); it may move.
    OPENROUTER_DECISIONS_URL: str = os.getenv("OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")
    # TypeSafe list price: $0.042 per million input tokens, output unmetered.
    DECISION_ENGINE_USD_PER_MTOK_IN: float = float(os.getenv("DECISION_ENGINE_USD_PER_MTOK_IN", "0.042"))
    # Shadow comparisons, one JSON line each — read by scripts/eval/decision_shadow.
    DECISION_SHADOW_LOG_PATH: str = os.getenv("DECISION_SHADOW_LOG_PATH", "logs/decision_shadow.jsonl")
    # P256-FIX-N14-8: an experiment, off by default. On, Auto's offered tool shortlist is
    # Jev's kept top few ahead of the platform's embedding top few (deduplicated) when Jev
    # answers within the timeout; late or silent, the platform's list is offered unchanged.
    TOOL_SHORTLIST_JEV_UNION: bool = os.getenv("TOOL_SHORTLIST_JEV_UNION", "false").lower() in ("true", "1", "yes")
    TOOL_SHORTLIST_JEV_TIMEOUT_S: float = float(os.getenv("TOOL_SHORTLIST_JEV_TIMEOUT_S", "1.0"))
    TOOL_SHORTLIST_JEV_TOP: int = int(os.getenv("TOOL_SHORTLIST_JEV_TOP", "5"))
    TOOL_SHORTLIST_PLATFORM_TOP: int = int(os.getenv("TOOL_SHORTLIST_PLATFORM_TOP", "10"))

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

    # F086: a document whose stored chunks hold less than this share of its
    # extracted text is shown to the owner as partial ("partial — 61% kept").
    RAG_KEPT_WARN_PCT: int = int(os.getenv("RAG_KEPT_WARN_PCT", "98"))
    # F311 (night 9): a Markdown or text document with headings is chunked by its sections, each
    # with its heading; a section under the minimum joins the next, one over the maximum
    # is split at its paragraphs, each piece under its headings. Characters.
    RAG_SECTION_CHUNKING_ENABLED: bool = os.getenv("RAG_SECTION_CHUNKING_ENABLED", "true").lower() == "true"
    RAG_SECTION_MIN_CHARS: int = int(os.getenv("RAG_SECTION_MIN_CHARS", "200"))
    RAG_SECTION_MAX_CHARS: int = int(os.getenv("RAG_SECTION_MAX_CHARS", "1500"))

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

    # F311 (night 9): agents' reports crowded the owner's documents out of the search.
    # When a search finds an agent's report, the owner's own documents are searched on
    # their own too (the newest RAG_OWNER_LEG_MAX_DOCUMENTS of them), and up to
    # RAG_OWNER_PASSAGES_RESERVED of the passages handed over are the owner's, placed first.
    RAG_OWNER_LEG_ENABLED: bool = os.getenv("RAG_OWNER_LEG_ENABLED", "true").lower() == "true"
    RAG_OWNER_LEG_MAX_DOCUMENTS: int = int(os.getenv("RAG_OWNER_LEG_MAX_DOCUMENTS", "200"))
    RAG_OWNER_PASSAGES_RESERVED: int = int(os.getenv("RAG_OWNER_PASSAGES_RESERVED", "3"))
    # F311: a passage from an owner's document this short (in tokens) is handed over as
    # the whole document, so the section beside the one that matched comes with it. 0 = off.
    RAG_WHOLE_DOCUMENT_MAX_TOKENS: int = int(os.getenv("RAG_WHOLE_DOCUMENT_MAX_TOKENS", "800"))

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
    # CREDITS (PRD-265 W1): the ledger behind hosted Studio's plans and trial
    # =============================================================================
    # 1 credit = 1 US cent of AI usage at Automatos prices. Every movement is a
    # credit_ledger row (services/credits/). The local edition never meters.
    # Kill switch — false disables all trial grants; provisioning otherwise unchanged.
    TRIAL_ENABLED: bool = os.getenv("TRIAL_ENABLED", "true").lower() == "true"
    # Credits granted once per person (checked across all their workspaces).
    TRIAL_CREDITS: int = int(os.getenv("TRIAL_CREDITS", "500"))
    # Platform-wide daily ceiling on trial credits spent; new grants PAUSE once
    # today's trial spend reaches it. Grants already made keep working.
    TRIAL_GLOBAL_DAILY_CREDITS: int = int(os.getenv("TRIAL_GLOBAL_DAILY_CREDITS", "2500"))
    # Paid plans are on sale. False (the beta): checkout is open to platform admins
    # only (test mode, live card tests), and plan prices carry the "beta" label
    # (facts sheet §4). Credits are granted only by a verified Stripe payment,
    # open or not. US-013 flips it.
    BILLING_OPEN: bool = os.getenv("BILLING_OPEN", "false").lower() == "true"
    # Metered credits (Gerard, 20 Oct 2026: tokens × model price, like OpenRouter —
    # replaces the fixed-per-ask bands). Every platform-funded call costs
    # ceil(its llm_usage cost in cents × this markup) credits. Decision D-1 sets it.
    CREDITS_MARKUP: float = float(os.getenv("CREDITS_MARKUP", "1.2"))
    # How often the meter turns the calls since its last run into ledger debits.
    CREDITS_METER_INTERVAL_SECONDS: int = int(os.getenv("CREDITS_METER_INTERVAL_SECONDS", "60"))
    # The share of a workspace's allowance (the trial, or the plan's monthly credits)
    # left when the "running low" alert fires; the second alert fires at zero.
    CREDITS_ALERT_LOW_SHARE: float = float(os.getenv("CREDITS_ALERT_LOW_SHARE", "0.20"))
    # The people who never pay (Gerard, 23 Oct 2026: "Super Admin doesn't have to pay"):
    # a workspace whose OWNER's email is in this comma-separated list runs on an
    # allowance, like Enterprise: its calls on the platform's AI keys are recorded
    # (for the margin numbers) but never charged and never paused. An email, not the
    # super-admin role, because that role lives only in Clerk's sign-in token, and the
    # meter and the scheduled work run with nobody signed in. Empty = nobody is free.
    BILLING_FREE_OWNER_EMAILS: frozenset = frozenset(
        email.strip().lower() for email in os.getenv("BILLING_FREE_OWNER_EMAILS", "").split(",") if email.strip()
    )
    # A month's credits stay spendable this long past their period end when the
    # renewal hasn't been paid yet (Stripe retries a failed card for days); then
    # the hourly sweep expires them (services/credits/expiry.py).
    BILLING_MONTHLY_EXPIRY_GRACE_HOURS: int = int(os.getenv("BILLING_MONTHLY_EXPIRY_GRACE_HOURS", "24"))
    # A top-up (PRD-265 §3): credits per pack, the display price, and how long they last.
    TOPUP_CREDITS: int = int(os.getenv("TOPUP_CREDITS", "1000"))
    TOPUP_PRICE_USD: int = int(os.getenv("TOPUP_PRICE_USD", "10"))
    TOPUP_VALID_DAYS: int = int(os.getenv("TOPUP_VALID_DAYS", "365"))

    # =============================================================================
    # STRIPE (PRD-265 W2): in-page Checkout, Stripe Tax, card and plan management, webhooks
    # =============================================================================
    # Gerard, 20 + 23 Oct 2026: Stripe takes the money, Clerk only signs people in
    # (Clerk Billing has no 3-D Secure, no VAT and no one-off purchases), and the
    # customer never leaves Studio: the card form is Stripe's fields inside our own
    # Billing tab (Checkout elements), and card, invoices, cancel and plan changes are
    # ours too — no hosted page, no portal. Keys come from the environment only.
    # Without STRIPE_SECRET_KEY billing is "not configured": its routes answer 503.
    STRIPE_SECRET_KEY: str = os.getenv("STRIPE_SECRET_KEY", "")
    # The publishable key (pk_…) Stripe.js needs in the browser; safe to expose.
    STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    # Every request pins this API version (Stripe-Version header), and
    # scripts/stripe_setup.py creates the webhook endpoint at it, so event shapes
    # never change under us when Stripe ships a new version. It must stay on the
    # release train of the frontend's @stripe/stripe-js (10.x = endive).
    STRIPE_API_VERSION: str = os.getenv("STRIPE_API_VERSION", "2026-09-30.endive")
    # The Stripe Price behind each plan and the top-up (scripts/stripe_setup.py
    # prints them). Business is sold only through the admin invite link.
    STRIPE_PRICE_BASIC: str = os.getenv("STRIPE_PRICE_BASIC", "")
    STRIPE_PRICE_PRO: str = os.getenv("STRIPE_PRICE_PRO", "")
    STRIPE_PRICE_BUSINESS: str = os.getenv("STRIPE_PRICE_BUSINESS", "")
    STRIPE_PRICE_TOPUP: str = os.getenv("STRIPE_PRICE_TOPUP", "")
    # How long a platform admin's Business offer stays open on the workspace's Billing tab.
    BUSINESS_OFFER_DAYS: int = int(os.getenv("BUSINESS_OFFER_DAYS", "14"))
    # The Stripe account (acct_…) payments go to, recorded on each billing account
    # so a later move to a company's account can tell old customers from new.
    STRIPE_ACCOUNT_ID: str = os.getenv("STRIPE_ACCOUNT_ID", "")
    # How old a webhook's signed timestamp may be before it is refused as a replay.
    STRIPE_WEBHOOK_TOLERANCE_SECONDS: int = int(os.getenv("STRIPE_WEBHOOK_TOLERANCE_SECONDS", "300"))
    STRIPE_TIMEOUT_SECONDS: float = float(os.getenv("STRIPE_TIMEOUT_SECONDS", "20"))
    # Counsel's EU withdrawal-consent wording for buying credits (decision D-4).
    # When set, the Billing tab's checkout shows it beside a box the buyer must tick,
    # and the acceptance is stamped on the Checkout Session; empty = no box.
    BILLING_WITHDRAWAL_CONSENT_TEXT: str = os.getenv("BILLING_WITHDRAWAL_CONSENT_TEXT", "")

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
    # PRD-265 D-6 (Gerard, 30 Oct 2026): an Auto Live call is paid from credits like chat, at what
    # Retell charges us x CREDITS_MARKUP (modules/voice/live_billing.py). The cost comes on the
    # call_ended event; when an event carries none, the call's minutes are charged at this rate:
    # Retell's list price for our voice at its dearest (voice infra $0.055 + an ElevenLabs voice
    # up to $0.10, retellai.com/pricing, 30 Oct 2026), as the meter does for an unpriced model.
    VOICE_LIVE_ESTIMATE_USD_PER_MINUTE: float = float(os.getenv("VOICE_LIVE_ESTIMATE_USD_PER_MINUTE", "0.155"))
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
    # PRD-264 EUAI-04 (EU AI Act Art. 50(1)): Auto Live speaks first, and its
    # opening line says it is an AI. A greeting that is empty or lacks the AI
    # term for VOICE_LIVE_LANGUAGE ("AI", "IA", "KI") is never spoken; the
    # default is (modules/policy/ai_disclosure.py). No setting silences it.
    VOICE_AI_GREETING: str = os.getenv(
        "VOICE_AI_GREETING", "Hi, this is Auto, your AI assistant."
    )
    # The spoken AI notice is once per call, the greeting above (policy, 23 Oct 2026).
    # Above 0: every this many minutes of call time and after a reconnect, the next
    # response opens with "Just so you know, I'm an AI assistant." (¶37). 0 or less: never.
    VOICE_AI_REMINDER_MINUTES: float = float(os.getenv("VOICE_AI_REMINDER_MINUTES", "0"))
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
    # Spoken turns (POST /api/chat with spokenMode, and Auto Live) answer on this model;
    # typed turns keep the agent's own (consumers/chatbot/spoken_model.py). 28 Oct 2026: a
    # spoken quick ask took 26-32 s on claude-opus-5-5; the 11 Oct benchmark scored
    # claude-haiku-5-5 the same at about 3% of the cost. The id is the provider's own (a
    # direct Anthropic id, never OpenRouter's dotted twin). Empty model or provider = off:
    # spoken turns use the agent's model too. A workspace that cannot reach it (no key for
    # the provider, a policy block) answers on the agent's model.
    SPOKEN_MODE_MODEL: str = os.getenv("SPOKEN_MODE_MODEL", "claude-haiku-5-5").strip()
    SPOKEN_MODE_PROVIDER: str = os.getenv("SPOKEN_MODE_PROVIDER", "anthropic").strip().lower()
    # Auto's server voice (services/voice-tts): POST /api/voice/speech streams a reply
    # as MP3 for the phone and the web quick-ask (modules/voice/speech.py). Empty URL =
    # off: the route answers 503 tts_off and discovery does not list `tts`, so clients
    # keep the device voice. The token is sent as `Authorization: Bearer` and must match
    # the service's own VOICE_TTS_TOKEN. TIMEOUT bounds one whole answer, the stream
    # included; CONNECT is short so a refusal is fast. RATE is sentences a workspace may
    # have spoken a minute (0 = no limit).
    VOICE_TTS_URL: str = os.getenv("VOICE_TTS_URL", "").strip().rstrip("/")
    VOICE_TTS_TOKEN: str = os.getenv("VOICE_TTS_TOKEN", "").strip()
    VOICE_TTS_TIMEOUT_SECONDS: float = float(os.getenv("VOICE_TTS_TIMEOUT_SECONDS", "15"))
    VOICE_TTS_CONNECT_TIMEOUT_SECONDS: float = float(os.getenv("VOICE_TTS_CONNECT_TIMEOUT_SECONDS", "2"))
    VOICE_TTS_DEFAULT_VOICE: str = os.getenv("VOICE_TTS_DEFAULT_VOICE", "af_heart").strip()
    VOICE_TTS_RATE_PER_MINUTE: int = int(os.getenv("VOICE_TTS_RATE_PER_MINUTE", "60"))
    # The ONE public host (PRD-258 US-003; it was defined twice): Retell's
    # custom-LLM socket + events webhook (arming builds the agent's URLs from
    # it) and the channels' inbound webhook URLs (PRD-142 W3-S5 / G7). Railway
    # injects RAILWAY_PUBLIC_DOMAIN; PUBLIC_API_HOST overrides it (the local
    # tunnel's domain, set in the compose override). No trailing slash.
    PUBLIC_API_HOST: str = (
        os.getenv("PUBLIC_API_HOST", os.getenv("RAILWAY_PUBLIC_DOMAIN", HOSTED_PUBLIC_API_HOST))
        or HOSTED_PUBLIC_API_HOST
    ).strip().rstrip("/")

    # =============================================================================
    # PRD-266 Automatos Go: pairing a phone (or a browser) with a local install
    # =============================================================================
    # LOCAL_REMOTE_ACCESS (local edition, §4.3): 'off' (the default) keeps the install on this
    # machine; 'paired' opens POST /api/v1/mobile/pair and closes the anonymous lane; 'open'
    # keeps it on a published install, warned at every boot. Boot refuses an unknown value and
    # 'off' with a publish address beyond loopback (services/mobile/remote_access.py).
    LOCAL_REMOTE_ACCESS: str = (os.getenv("LOCAL_REMOTE_ACCESS", "off") or "off").strip().lower()
    # The addresses compose publishes the API and the web app on (docker-compose.yml passes
    # both in), so the API knows whether it is reachable beyond this machine.
    API_PUBLISH_ADDRESS: str = (os.getenv("API_PUBLISH_ADDRESS") or DEFAULT_PUBLISH_ADDRESS).strip()
    WEB_PUBLISH_ADDRESS: str = (os.getenv("WEB_PUBLISH_ADDRESS") or DEFAULT_PUBLISH_ADDRESS).strip()
    # A paired device is asked to rotate its token this many days after it was issued
    # (``rotate_after`` in the pairing answer), at least one.
    MOBILE_TOKEN_ROTATE_DAYS: int = max(1, int(os.getenv("MOBILE_TOKEN_ROTATE_DAYS", "30")))
    # After a rotation the previous token keeps working this long (§4.2: ten minutes).
    MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS: int = max(
        0, int(os.getenv("MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS", "600"))
    )
    # Push through Expo's push service (services/push/expo.py, AGENTS.md exception 2). Expo's
    # optional access token: empty sends without one. A secret: never logged, never in
    # discovery, never a key in Settings.
    EXPO_PUSH_ACCESS_TOKEN: str = (os.getenv("EXPO_PUSH_ACCESS_TOKEN") or "").strip()
    # Each request to Expo times out after this many seconds, at least one.
    EXPO_PUSH_TIMEOUT_SECONDS: int = max(1, int(os.getenv("EXPO_PUSH_TIMEOUT_SECONDS", "10")))
    # The receipts job (services/push/receipts.py) reads Expo's receipts this often on the
    # leader's scheduler, at least a minute apart.
    PUSH_RECEIPTS_INTERVAL_SECONDS: int = max(60, int(os.getenv("PUSH_RECEIPTS_INTERVAL_SECONDS", "900")))
    # The daily digest (services/mobile/digest.py, US-013) is checked this often on the leader's
    # scheduler, at least a minute apart; a person is due for 90 minutes after their digest time.
    DIGEST_JOB_INTERVAL_SECONDS: int = max(60, int(os.getenv("DIGEST_JOB_INTERVAL_SECONDS", "900")))
    # The digest time of a phone that names none, 'HH:MM' in the phone's timezone; a value that
    # is not HH:MM reads as 08:00 (services.mobile.digest.default_digest_time).
    DIGEST_DEFAULT_TIME: str = (os.getenv("DIGEST_DEFAULT_TIME") or "08:00").strip()
    # "Send me today's brief now" (POST /api/v1/mobile/digest/now): one per person at most this
    # many seconds apart, at least one.
    DIGEST_NOW_MIN_INTERVAL_SECONDS: int = max(1, int(os.getenv("DIGEST_NOW_MIN_INTERVAL_SECONDS", "300")))

    # =============================================================================
    # PRD-268 Proactive Auto: check-ins that suggest
    # =============================================================================
    # K7: the platform mode is off | shadow | live, the ``proactive.mode`` system setting
    # (services/proactive/switches.py), read strictly. No row takes this default; a value
    # that is none of the three, or a read that cannot complete, is off.
    PROACTIVE_MODE_DEFAULT: str = (os.getenv("PROACTIVE_MODE_DEFAULT") or "off").strip().lower()
    # §6.9: the caps every workspace's check-ins are held to unless its owners or admins set
    # their own (settings.proactive.caps, services/proactive/limits.py). 0 allows nothing; a
    # value the caps refuse (a negative one, a minimum interval past the maximum) refuses
    # every check-in until it is fixed.
    PROACTIVE_MAX_MODEL_RUNS_PER_DAY: int = int(os.getenv("PROACTIVE_MAX_MODEL_RUNS_PER_DAY", "48"))
    PROACTIVE_MAX_SUGGESTIONS_PER_RUN: int = int(os.getenv("PROACTIVE_MAX_SUGGESTIONS_PER_RUN", "3"))
    PROACTIVE_MAX_NEW_PER_PERSON_PER_DAY: int = int(os.getenv("PROACTIVE_MAX_NEW_PER_PERSON_PER_DAY", "10"))
    # Past this the person's oldest open suggestions expire; nothing is refused.
    PROACTIVE_MAX_OPEN_PER_PERSON: int = int(os.getenv("PROACTIVE_MAX_OPEN_PER_PERSON", "20"))
    PROACTIVE_MAX_OPEN_PER_AGENT: int = int(os.getenv("PROACTIVE_MAX_OPEN_PER_AGENT", "3"))
    # A check-in's interval: one sooner than the minimum after the last is refused.
    PROACTIVE_MIN_INTERVAL_MINUTES: int = int(os.getenv("PROACTIVE_MIN_INTERVAL_MINUTES", "30"))
    PROACTIVE_MAX_INTERVAL_MINUTES: int = int(os.getenv("PROACTIVE_MAX_INTERVAL_MINUTES", "1440"))
    # §6.11.3: each readiness probe (services/readiness/) answers within this many seconds, or
    # its item reads not_checked. At least a tenth of a second.
    READINESS_PROBE_TIMEOUT_SECONDS: float = max(0.1, float(os.getenv("READINESS_PROBE_TIMEOUT_SECONDS", "3")))

    # =============================================================================
    # PRD-251 Socials
    # =============================================================================
    # Socials is gated two ways (D1): the platform master switch is the
    # ``socials.enabled`` system setting, a super-admin toggle in Settings →
    # System Settings, and each workspace has ``settings['socials'].enabled``,
    # the Socials card in its Settings (modules/socials/settings.py).
    # SOCIALS_ENABLED_DEFAULT is both switches' DEFAULT, on (owner, 2026-10-03; an
    # install that wants Socials off until asked, such as an enterprise one, sets
    # it false): the prd251_socials migration seeds the master row with it, it
    # applies wherever no row exists, and a workspace that never set its switch
    # takes it. Every plan gets Socials, so there is no plan exposure key.
    SOCIALS_ENABLED_DEFAULT: bool = os.getenv(
        "SOCIALS_ENABLED_DEFAULT", "true"
    ).strip().lower() == "true"
    # D9: the lifetime of the presigned media URL a channel fetches at publish time.
    SOCIALS_MEDIA_URL_TTL_SECONDS: int = int(os.getenv("SOCIALS_MEDIA_URL_TTL_SECONDS", "86400"))
    # D10: a post whose slot passed while the backend was down still publishes
    # within this grace; later than that it goes to ``missed``, never posted late.
    SOCIALS_MISFIRE_GRACE_SECONDS: int = int(os.getenv("SOCIALS_MISFIRE_GRACE_SECONDS", "1800"))
    # D8: transient-error retries per channel target.
    SOCIALS_MAX_TARGET_ATTEMPTS: int = int(os.getenv("SOCIALS_MAX_TARGET_ATTEMPTS", "3"))
    # D8 (US-301): the publisher. A transient failure tries again after this many
    # seconds times the attempt; a status step (an upload processing, a publish
    # finishing) is called every POLL seconds for at most MAX_WAIT; one post's whole
    # publish lives at most RUN_MAX, under BOOT_REAPER_STALE_MINUTES.
    SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS: int = int(os.getenv("SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS", "5"))
    SOCIALS_PUBLISH_POLL_SECONDS: int = int(os.getenv("SOCIALS_PUBLISH_POLL_SECONDS", "5"))
    SOCIALS_PUBLISH_MAX_WAIT_SECONDS: int = int(os.getenv("SOCIALS_PUBLISH_MAX_WAIT_SECONDS", "600"))
    SOCIALS_PUBLISH_RUN_MAX_SECONDS: int = int(os.getenv("SOCIALS_PUBLISH_RUN_MAX_SECONDS", "1500"))
    # D3: the media-render service (Wave 1). Empty = no renderer configured.
    SOCIALS_RENDER_URL: str = os.getenv("SOCIALS_RENDER_URL", "").strip()
    # S1.1c: the client (core/media_render_client.py). The token is sent as
    # X-Internal-Token and must match the service's own SOCIALS_RENDER_TOKEN
    # (empty = none sent). The read timeout covers the longest single call:
    # POST /render answers once the job is staged, spoken, mixed and checked.
    SOCIALS_RENDER_TOKEN: str = os.getenv("SOCIALS_RENDER_TOKEN", "").strip()
    SOCIALS_RENDER_TIMEOUT_SECONDS: int = int(os.getenv("SOCIALS_RENDER_TIMEOUT_SECONDS", "900"))
    SOCIALS_RENDER_CONNECT_TIMEOUT_SECONDS: int = int(os.getenv("SOCIALS_RENDER_CONNECT_TIMEOUT_SECONDS", "10"))
    # A render in flight is polled this often, and waited for this long (its
    # queue time and the render) before the post goes to failed. Keep the wait
    # under BOOT_REAPER_STALE_MINUTES, so a restart only ever reaps renders no
    # live process still owns.
    SOCIALS_RENDER_POLL_SECONDS: int = int(os.getenv("SOCIALS_RENDER_POLL_SECONDS", "5"))
    SOCIALS_RENDER_MAX_WAIT_SECONDS: int = int(os.getenv("SOCIALS_RENDER_MAX_WAIT_SECONDS", "1500"))
    # P251W1-RVW-3: the longest composition media-render renders (its own
    # MEDIA_RENDER_MAX_DURATION_SECONDS). A render holds its composition's
    # declared duration against the monthly quota while it runs, at most this;
    # one whose duration cannot be read holds this much.
    SOCIALS_RENDER_MAX_DURATION_SECONDS: int = int(os.getenv("SOCIALS_RENDER_MAX_DURATION_SECONDS", "180"))
    # PRD-264 EUAI-08a (EU AI Act Art. 50(4)): a labelled video shows "AI-generated" for
    # its first this many seconds, and again for as long every interval after
    # (modules/socials/deepfake_label.py; media-render burns it in). An interval no longer
    # than the label shows it the whole video.
    SOCIALS_AI_LABEL_SECONDS: float = float(os.getenv("SOCIALS_AI_LABEL_SECONDS", "3"))
    SOCIALS_AI_LABEL_INTERVAL_SECONDS: float = float(os.getenv("SOCIALS_AI_LABEL_INTERVAL_SECONDS", "30"))
    # How long a presigned link to a render's input in our storage (a voice line
    # spoken by a voice toolkit, S1.5) lives: media-render fetches it after the
    # render's queue wait, so keep it above SOCIALS_RENDER_MAX_WAIT_SECONDS.
    SOCIALS_RENDER_MEDIA_URL_TTL_SECONDS: int = int(os.getenv("SOCIALS_RENDER_MEDIA_URL_TTL_SECONDS", "3600"))
    # S1.5 (D11): a voice toolkit's line (Fish Audio, ElevenLabs through the
    # workspace's Composio connection) is copied into our storage the moment it
    # returns: at most this many bytes, fetched within this many seconds.
    SOCIALS_VOICE_LINE_MAX_BYTES: int = int(os.getenv("SOCIALS_VOICE_LINE_MAX_BYTES", "16777216"))
    SOCIALS_MEDIA_FETCH_TIMEOUT_SECONDS: int = int(os.getenv("SOCIALS_MEDIA_FETCH_TIMEOUT_SECONDS", "60"))
    # PRD-251B US-B109: a file uploaded as a post's visual (POST /api/socials/posts/{id}/media),
    # at most this many bytes: a PNG, JPEG or WebP image, or an MP4 video.
    SOCIALS_UPLOAD_IMAGE_MAX_BYTES: int = int(os.getenv("SOCIALS_UPLOAD_IMAGE_MAX_BYTES", "10485760"))
    SOCIALS_UPLOAD_VIDEO_MAX_BYTES: int = int(os.getenv("SOCIALS_UPLOAD_VIDEO_MAX_BYTES", "209715200"))
    # PRD-251B US-B205: the leader's plan tick makes each plan's due posts this often, at
    # most this many slots a tick (the rest wait for the next).
    SOCIALS_PLAN_TICK_SECONDS: int = int(os.getenv("SOCIALS_PLAN_TICK_SECONDS", "300"))
    SOCIALS_PLAN_MAX_SLOTS_PER_TICK: int = int(os.getenv("SOCIALS_PLAN_MAX_SLOTS_PER_TICK", "12"))
    # PRD-251C US-C103 (C5): the workspace's Socials history, as research reads it: this many
    # days back and at most this many posts, newest first, unless the caller asks for others.
    SOCIALS_HISTORY_DAYS: int = int(os.getenv("SOCIALS_HISTORY_DAYS", "90"))
    SOCIALS_HISTORY_LIMIT: int = int(os.getenv("SOCIALS_HISTORY_LIMIT", "50"))
    # PRD-251C US-C104 (C5): two titles whose content words overlap this much (shared over all,
    # 0 to 1) are one topic: research may not add the second (modules/socials/repeats.py).
    SOCIALS_REPEAT_OVERLAP: float = float(os.getenv("SOCIALS_REPEAT_OVERLAP", "0.75"))
    # PRD-251C US-C105: the composer is given how this many of the workspace's last posts began.
    SOCIALS_COMPOSE_RECENT_OPENINGS: int = int(os.getenv("SOCIALS_COMPOSE_RECENT_OPENINGS", "10"))
    # PRD-251C US-C402 (C7): the leader reads published posts' numbers this often, at most this
    # many reads a tick, each reading (1 and 7 days after a post went out) for this many days.
    SOCIALS_RESULTS_TICK_SECONDS: int = int(os.getenv("SOCIALS_RESULTS_TICK_SECONDS", "3600"))
    SOCIALS_RESULTS_MAX_READS_PER_TICK: int = int(os.getenv("SOCIALS_RESULTS_MAX_READS_PER_TICK", "50"))
    SOCIALS_RESULTS_READ_WINDOW_DAYS: int = int(os.getenv("SOCIALS_RESULTS_READ_WINDOW_DAYS", "2"))
    # PRD-251C US-C406 (C8): the workspace keeps this many voice examples, newest first.
    SOCIALS_VOICE_EXAMPLES: int = int(os.getenv("SOCIALS_VOICE_EXAMPLES", "10"))
    # PRD-251B US-B303: the vision read of the brand kit's style references (empty model: the
    # workspace's own), and how long it may take.
    BRAND_STYLE_READ_MODEL: str = os.getenv("BRAND_STYLE_READ_MODEL", "")
    BRAND_STYLE_READ_TIMEOUT_SECONDS: int = int(os.getenv("BRAND_STYLE_READ_TIMEOUT_SECONDS", "90"))
    # PRD-264 ART5-1 (EU AI Act Art. 5): the model read that checks every image, video and voice
    # request before it goes to a provider (core/generation_guard.py; empty model: the
    # workspace's own), and how long it may take. A read that fails refuses the request.
    GENERATION_GUARD_MODEL: str = os.getenv("GENERATION_GUARD_MODEL", "")
    GENERATION_GUARD_TIMEOUT_SECONDS: int = int(os.getenv("GENERATION_GUARD_TIMEOUT_SECONDS", "30"))
    # Off until after Web Summit (Gerard, 23 Oct 2026): no model read, so no cost and no refused
    # demo when the read fails. Only the CSAM keyword block and the providers' own refusals run.
    GENERATION_GUARD_READ_ENABLED: bool = os.getenv("GENERATION_GUARD_READ_ENABLED", "false").lower() in ("true", "1", "yes")
    # The voices a voice toolkit lists in the Socials voice picker, at most.
    SOCIALS_VOICE_LIST_LIMIT: int = int(os.getenv("SOCIALS_VOICE_LIST_LIMIT", "30"))
    # S1.8 (D12): footage and stills from the workspace's own Composio generation
    # toolkit (modules/socials/recipes/footage.py). The toolkits a render tries,
    # in this order, for a slot its post asks footage for; each toolkit's model
    # per kind (the owner sets these after the small-spend quality test, PRD-251
    # open question 7); the clip length asked for; the words every prompt ends
    # with (every word on screen is template text, never generated); how often a
    # job is polled and for how long (inside the render's own
    # SOCIALS_RENDER_MAX_WAIT_SECONDS, leaving the render its time); the largest
    # file kept.
    SOCIALS_FOOTAGE_TOOLKITS: str = os.getenv("SOCIALS_FOOTAGE_TOOLKITS", "fal_ai,kieai,higgsfield_mcp")
    SOCIALS_FOOTAGE_FAL_VIDEO_MODEL: str = os.getenv(
        "SOCIALS_FOOTAGE_FAL_VIDEO_MODEL", "fal-ai/kling-video/v2.1/standard/text-to-video"
    ).strip()
    SOCIALS_FOOTAGE_FAL_IMAGE_MODEL: str = os.getenv("SOCIALS_FOOTAGE_FAL_IMAGE_MODEL", "fal-ai/flux-pro/v1.1-ultra").strip()
    SOCIALS_FOOTAGE_KIEAI_VIDEO_MODEL: str = os.getenv("SOCIALS_FOOTAGE_KIEAI_VIDEO_MODEL", "veo3_fast").strip()
    SOCIALS_FOOTAGE_KIEAI_IMAGE_MODEL: str = os.getenv("SOCIALS_FOOTAGE_KIEAI_IMAGE_MODEL", "flux-kontext-pro").strip()
    SOCIALS_FOOTAGE_HIGGSFIELD_VIDEO_MODEL: str = os.getenv("SOCIALS_FOOTAGE_HIGGSFIELD_VIDEO_MODEL", "kling3_0").strip()
    SOCIALS_FOOTAGE_HIGGSFIELD_IMAGE_MODEL: str = os.getenv("SOCIALS_FOOTAGE_HIGGSFIELD_IMAGE_MODEL", "gpt_image_2").strip()
    SOCIALS_FOOTAGE_CLIP_SECONDS: int = int(os.getenv("SOCIALS_FOOTAGE_CLIP_SECONDS", "5"))
    SOCIALS_FOOTAGE_PROMPT_GUARD: str = os.getenv("SOCIALS_FOOTAGE_PROMPT_GUARD", "no readable text, no logos").strip()
    SOCIALS_FOOTAGE_POLL_SECONDS: int = int(os.getenv("SOCIALS_FOOTAGE_POLL_SECONDS", "10"))
    SOCIALS_FOOTAGE_MAX_WAIT_SECONDS: int = int(os.getenv("SOCIALS_FOOTAGE_MAX_WAIT_SECONDS", "900"))
    SOCIALS_FOOTAGE_MAX_BYTES: int = int(os.getenv("SOCIALS_FOOTAGE_MAX_BYTES", "134217728"))
    # D13: money. A toolkit that prices a call (fal: its estimate action) is
    # priced before any submit; one that prices nothing (Kie.ai, Higgsfield) is
    # budgeted at these ceilings per shot, so the caps still hold (the reference
    # client's rule: a 5 s 720p Cinema Studio shot was $2.31, an unpriced image
    # budgeted at $0.30). Every submit fits the post's cap and the workspace's
    # monthly media cap: the workspace setting socials.media_monthly_cap_usd, whose
    # default this is. A credit-billed toolkit books its balance difference at its
    # credit's price: Kie.ai $0.005 a credit; Higgsfield $0.0625, its smallest
    # top-up pack (the dearest credit, so a cap binds early).
    SOCIALS_FOOTAGE_CEILING_VIDEO_USD: float = float(os.getenv("SOCIALS_FOOTAGE_CEILING_VIDEO_USD", "2.50"))
    SOCIALS_FOOTAGE_CEILING_IMAGE_USD: float = float(os.getenv("SOCIALS_FOOTAGE_CEILING_IMAGE_USD", "0.30"))
    SOCIALS_MEDIA_POST_CAP_USD: float = float(os.getenv("SOCIALS_MEDIA_POST_CAP_USD", "10"))
    SOCIALS_MEDIA_MONTHLY_CAP_USD: float = float(os.getenv("SOCIALS_MEDIA_MONTHLY_CAP_USD", "30"))
    SOCIALS_KIEAI_USD_PER_CREDIT: float = float(os.getenv("SOCIALS_KIEAI_USD_PER_CREDIT", "0.005"))
    SOCIALS_HIGGSFIELD_USD_PER_CREDIT: float = float(os.getenv("SOCIALS_HIGGSFIELD_USD_PER_CREDIT", "0.0625"))
    # A voice toolkit's script (S1.5) is priced before its first line is spoken,
    # at these prices per unit, and must fit the same two caps (P251W1-RVW-2).
    # Fish Audio bills API credit per UTF-8 byte: $15 per million bytes
    # (docs.fish.audio, 2026-09-28); its render still books its balance
    # difference. ElevenLabs bills the customer's own plan per character: $0.08
    # per 1,000, its API rate for its dearest models (elevenlabs.io/pricing/api,
    # 2026-09-28), so a cap binds early; its render books its characters at this.
    SOCIALS_VOICE_FISH_AUDIO_USD_PER_BYTE: float = float(os.getenv("SOCIALS_VOICE_FISH_AUDIO_USD_PER_BYTE", "0.000015"))
    SOCIALS_VOICE_ELEVENLABS_USD_PER_CHARACTER: float = float(
        os.getenv("SOCIALS_VOICE_ELEVENLABS_USD_PER_CHARACTER", "0.00008")
    )
    # D9: the local edition's public bucket for channels that fetch media by URL
    # (Instagram, TikTok publish-from-URL, the YouTube thumbnail). Empty = those
    # channels show "needs public storage".
    SOCIALS_PUBLIC_MEDIA_BUCKET: str = os.getenv("SOCIALS_PUBLIC_MEDIA_BUCKET", "").strip()
    # S0.6 (D16): how long the Composio deny list is cached per process
    # (core/composio/deny_list.py). A warm cache answers every Composio call from
    # memory, so no call waits on system_settings; an edit applies within this.
    COMPOSIO_DENY_LIST_CACHE_TTL_SECONDS: int = int(os.getenv("COMPOSIO_DENY_LIST_CACHE_TTL_SECONDS", "30"))
    # S3.5 (D14b): how long the Socials post gate's list of posting actions (the
    # socials.post_actions system setting) is cached per process
    # (core/composio/post_gate.py). An edit applies within this, with no restart.
    SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS: int = int(os.getenv("SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS", "30"))
    # PRD-251 S2.2a (US-207): how long the composer waits for the model to turn a
    # brief into a draft proposal (POST /api/socials/compose answers 504 after it).
    SOCIALS_COMPOSE_TIMEOUT_SECONDS: int = int(os.getenv("SOCIALS_COMPOSE_TIMEOUT_SECONDS", "60"))

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
        - PRD-266 §4.3: a ``local`` install published beyond loopback needs
          ``LOCAL_REMOTE_ACCESS`` paired or open (``services.mobile.remote_access``).

        Raises ``RuntimeError`` (aborting boot) on a contradiction.
        """
        errors: list[str] = []

        # PRD-234 S1a: session mode is a local-edition feature. A saas boot with
        # CLI_RUNTIME_ENABLED set is a misconfiguration that would expose the CLI
        # host routes to tenants — abort, never a silent partial enable. Runs
        # outside run_stage (main.py) like the rest of this guard, so it is real.
        if self.CLI_RUNTIME_ENABLED and self.AUTH_EDITION != "local":
            raise RuntimeError(
                "CLI_RUNTIME_ENABLED=true requires AUTH_EDITION=local. Session mode "
                "runs the user's own CLI sessions on their machine and has no meaning "
                "in the hosted edition; unset CLI_RUNTIME_ENABLED or set AUTH_EDITION=local."
            )

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

        # PRD-266 §4.3: imported here, at boot, so config.py never imports the app.
        from services.mobile import remote_access
        remote_access.enforce_boot_decision(self.AUTH_EDITION, self.LOCAL_REMOTE_ACCESS, {
            "API_PUBLISH_ADDRESS": self.API_PUBLISH_ADDRESS,
            "WEB_PUBLISH_ADDRESS": self.WEB_PUBLISH_ADDRESS,
        })

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
# PRD-222 W2·S1 (US-023), re-priced by PRD-265 US-001: PLAN_TIERS — the tier
# contract the public facts sheet sells (§4): Own keys $9, Pro $29, Business $99
# (by invite), Enterprise coming soon. Capability families (codegraph/nl2sql/
# team/voice) and the quota limits derive from here: exposure profiles read them,
# and the assignment helper (services/plan_tiers.py) writes the limits into
# ``workspaces.plan_limits`` under the keys live code already reads
# (seats→max_members, max_agents, budget). ``enterprise`` is a COMING-SOON
# display entry only — never assignable.
#
# PRD-265 fields:
#   * ``platform_ai`` — whether the plan's monthly and top-up credits may pay for
#     calls on the platform's own AI keys. Own keys (DB key ``basic``) is False:
#     its calls run on the customer's keys only. Trial credits fund platform AI on
#     any plan (services/credits/routing.py).
#   * ``monthly_credits`` — the credits a paid period grants (granted by billing
#     from W2 on; nothing grants them while ``BILLING_OPEN`` is false).
#   * ``self_serve`` — bought through checkout by the owner. Business is False:
#     by invite, assigned by a platform admin only.
#   * ``metered`` — False only for enterprise: an explicit platform allowance,
#     contract-billed, so its platform calls are recorded but never gated.
# The DB key ``basic`` stays (workspaces.plan defaults to it); only the display
# name changed. Own keys is the same service as Pro (facts sheet §4): same seats,
# agents, render minutes and families — only the credits differ.
#
# Every number is env-overridable WITHOUT a redeploy via
# ``AUTOMATOS_PLAN_TIERS_JSON`` (a JSON object deep-merged onto these defaults).
# ``0`` means "unlimited" for max_agents / watcher_limit, and "no ceiling /
# custom" for budget_usd.
#
# PRD-251 S1.1c (owner, 2026-09-23): every plan gets Socials, and plans differ
# only in hosting: ``render_minutes_month`` is the monthly render quota
# (core/media_render_quota.py). ``0`` or no key means no quota, as for
# max_agents: enterprise has none until the owner sets one, and the local
# edition never has one.
# ---------------------------------------------------------------------------
_PRO_LIMITS: dict = {
    "seats": 5,
    "max_agents": 20,
    "mission_concurrency": 3,
    "watcher_limit": 5,
    "marketplace_depth": 2,
    "budget_usd": 100,
    "render_minutes_month": 60,
    # PRD-265 D-6 (30 Oct 2026): voice on every plan; an Auto Live call is paid by the minute
    # from credits, so it is the customer's choice (it was Business-only, PRD-266 D34).
    "families": {"codegraph": True, "nl2sql": True, "team": True, "voice": True},
}

_PLAN_TIERS_DEFAULTS: dict[str, dict] = {
    "basic": {
        **copy.deepcopy(_PRO_LIMITS),
        "display_name": "Own keys",
        "display_price_usd": 9,
        "assignable": True,
        "self_serve": True,
        "platform_ai": False,
        "metered": True,
        "monthly_credits": 0,
    },
    "pro": {
        **copy.deepcopy(_PRO_LIMITS),
        "display_name": "Pro",
        "display_price_usd": 29,
        "assignable": True,
        "self_serve": True,
        "platform_ai": True,
        "metered": True,
        "monthly_credits": 2000,
    },
    "business": {
        "display_name": "Business",
        "display_price_usd": 99,
        "assignable": True,
        "self_serve": False,
        "platform_ai": True,
        "metered": True,
        "monthly_credits": 8000,
        "seats": 25,
        "max_agents": 0,
        "mission_concurrency": 10,
        "watcher_limit": 0,
        "marketplace_depth": 3,
        "budget_usd": 0,
        "render_minutes_month": 240,
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
        "platform_ai": True,
        "metered": False,
        "monthly_credits": 0,
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
