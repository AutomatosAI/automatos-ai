"""
Seed System Settings
===================

Seeds default system settings.

PRD-26: System Settings Configuration — comprehensive settings seed.
PRD-136: Collapsed 12 LLM silos to 3 tiers (Auto / System / Embeddings).

CRITICAL: This script NEVER overwrites existing user values.
- If setting exists: only updates default_value, description, validation, and
  required/sensitive flags. The user's chosen `value` is preserved.
- If setting doesn't exist: creates with default_value as the initial value.

Tooltip copy lives on each setting's `description` field — the frontend reads
it directly. Keep descriptions plain-English, no jargon.
"""

import logging
from sqlalchemy.orm import Session

from config import config
from core.models.system_settings import SystemSetting, SettingCategory

logger = logging.getLogger(__name__)


# =============================================================================
# CANONICAL LLM-TIER SCHEMA (PRD-136)
# =============================================================================
#
# Every LLM tier exposes the same shape. We generate the per-tier setting list
# from this template so adding a field once propagates everywhere.

def _llm_tier_settings(category: str, defaults: dict) -> list[dict]:
    """Build the canonical per-tier setting blocks.

    `category` is one of: orchestrator_llm, system_llm.
    `defaults` provides per-tier default values for every key.
    """
    return [
        {
            "category": category,
            "key": "provider",
            "default_value": defaults["provider"],
            "value_type": "string",
            "description": (
                "Which LLM provider serves this tier. OpenRouter is recommended — "
                "one credential covers 100+ models across vendors."
            ),
            "is_required": True,
            "validation_rules": {
                "options": ["openrouter", "openai", "anthropic", "google", "azure", "huggingface"]
            },
        },
        {
            "category": category,
            "key": "model",
            "default_value": defaults["model"],
            "value_type": "string",
            "description": (
                "Which model handles requests for this tier. Use the OpenRouter "
                "format `provider/model` (e.g. `google/gemini-2.5-flash`). Auto = "
                "premium model for the brain. System = cheap-fast model for the "
                "dozens of internal calls per request."
            ),
            "is_required": True,
        },
        {
            "category": category,
            "key": "temperature",
            "default_value": defaults["temperature"],
            "value_type": "number",
            "description": (
                "How creative the LLM is. 0 = deterministic and factual. 1 = "
                "playful and varied. 0.7 is the sweet spot for chat; lower for "
                "internal tools."
            ),
            "is_required": False,
            "validation_rules": {"min": 0.0, "max": 2.0, "step": 0.1},
        },
        {
            "category": category,
            "key": "max_tokens",
            "default_value": defaults["max_tokens"],
            "value_type": "number",
            "description": (
                "The longest reply this LLM is allowed to write. Higher = more "
                "detail but slower and more expensive. 8000 is comfortable for "
                "most tasks. The LLM stops earlier if it finishes naturally — "
                "this is a ceiling, not a target."
            ),
            "is_required": False,
            "validation_rules": {"min": 256, "max": 32000},
        },
        {
            "category": category,
            "key": "top_p",
            "default_value": defaults["top_p"],
            "value_type": "number",
            "description": (
                "Nucleus sampling — narrows the LLM's word choices to the top "
                "fraction of the probability mass. 1.0 = no narrowing. Most "
                "users leave this at 1.0."
            ),
            "is_required": False,
            "validation_rules": {"min": 0.0, "max": 1.0, "step": 0.05},
        },
        {
            "category": category,
            "key": "frequency_penalty",
            "default_value": defaults["frequency_penalty"],
            "value_type": "number",
            "description": (
                "Discourages the LLM from repeating the same words. Positive "
                "values reduce repetition; 0 is neutral. Most users leave at 0."
            ),
            "is_required": False,
            "validation_rules": {"min": -2.0, "max": 2.0, "step": 0.1},
        },
        {
            "category": category,
            "key": "presence_penalty",
            "default_value": defaults["presence_penalty"],
            "value_type": "number",
            "description": (
                "Encourages the LLM to bring up new topics. Positive values push "
                "for novelty; 0 is neutral. Most users leave at 0."
            ),
            "is_required": False,
            "validation_rules": {"min": -2.0, "max": 2.0, "step": 0.1},
        },
        {
            "category": category,
            "key": "timeout_seconds",
            "default_value": defaults["timeout_seconds"],
            "value_type": "number",
            "description": (
                "How long to wait for the LLM to respond before giving up. "
                "Premium models can be slow on long generations — 120s is safe "
                "for Auto, 60s is plenty for the System tier."
            ),
            "is_required": False,
            "validation_rules": {"min": 10, "max": 600},
        },
        {
            "category": category,
            "key": "max_retries",
            "default_value": defaults["max_retries"],
            "value_type": "number",
            "description": (
                "How many times to retry a failed LLM call before surfacing the "
                "error. 3 covers transient network blips without inflating cost."
            ),
            "is_required": False,
            "validation_rules": {"min": 0, "max": 5},
        },
    ]


def seed_system_settings(db: Session):
    """
    Seed default system settings for all configuration categories.

    Categories:
    - general:           Environment, deployment, NextAuth, plus a small set of
                         non-LLM feature flags (memory truncation, consistency check).
    - orchestrator_llm:  Auto — the brain. Premium, user-facing, low-volume.
    - system_llm:        System — cheap-fast model for all internal calls.
    - embeddings:        Vectorization model + chunking + reranking.
    - system_logging, api_rate_limiting, backend_api_keys, llm_cost_audit:
                         Operational settings — untouched by PRD-136.

    CRITICAL: Never overwrites existing user values.
    """

    settings_to_create: list[dict] = []

    # =========================================================================
    # GENERAL SETTINGS (environment, deployment, frontend, feature flags)
    # =========================================================================
    settings_to_create.extend([
        {
            "category": SettingCategory.GENERAL.value,
            "key": "environment",
            "default_value": "development",
            "value_type": "string",
            "description": "Application environment (development, staging, production).",
            "is_required": True,
            "validation_rules": {"options": ["development", "staging", "production"]},
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "log_level",
            "default_value": "INFO",
            "value_type": "string",
            "description": "Python logging level.",
            "is_required": False,
            "validation_rules": {"options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
        },

        # PRD-136: non-LLM feature flags lifted out of retired LLM categories
        {
            "category": SettingCategory.GENERAL.value,
            "key": "coordinator_consistency_check_enabled",
            "default_value": "true",
            "value_type": "boolean",
            "description": (
                "Run cross-task consistency verification after each mission task. "
                "Catches contradictions between sibling tasks. Adds a small LLM "
                "call per task."
            ),
            "is_required": False,
            "validation_rules": {"options": ["true", "false"]},
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "memory_store_max_chars",
            "default_value": "6000",
            "value_type": "number",
            "description": (
                "Max characters per message stored in long-term memory. Higher = "
                "richer recall but more storage and cost per save."
            ),
            "is_required": False,
            "validation_rules": {"min": 500, "max": 20000, "step": 500},
        },

        # Deployment Configuration
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_host",
            "default_value": "",
            "value_type": "string",
            "description": "Deployment host (SSH).",
            "is_required": False,
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_port",
            "default_value": "22",
            "value_type": "number",
            "description": "Deployment SSH port.",
            "is_required": False,
            "validation_rules": {"min": 1, "max": 65535},
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_user",
            "default_value": "root",
            "value_type": "string",
            "description": "Deployment SSH user.",
            "is_required": False,
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_key_path",
            "default_value": "",
            "value_type": "string",
            "description": "Path to deployment SSH key.",
            "is_required": False,
            "is_sensitive": True,
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_enabled",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Enable automated deployment.",
            "is_required": False,
            "validation_rules": {"options": ["true", "false"]},
        },

        # Frontend Configuration
        {
            "category": SettingCategory.GENERAL.value,
            "key": "nextauth_secret",
            "default_value": "",
            "value_type": "string",
            "description": "NextAuth secret key.",
            "is_required": False,
            "is_sensitive": True,
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "nextauth_url",
            "default_value": "",
            "value_type": "string",
            "description": "NextAuth callback URL.",
            "is_required": False,
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "next_public_api_url",
            "default_value": config.NEXT_PUBLIC_API_URL or "",
            "value_type": "string",
            "description": "Public API URL for frontend.",
            "is_required": True,
        },
    ])

    # =========================================================================
    # ORCHESTRATOR LLM (Auto — the brain)
    # =========================================================================
    settings_to_create.extend(_llm_tier_settings(
        SettingCategory.ORCHESTRATOR_LLM.value,
        defaults={
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",  # overridden by user to gpt-5.5/opus-4.7 in production
            "temperature": "0.7",
            "max_tokens": "8000",
            "top_p": "1.0",
            "frequency_penalty": "0.0",
            "presence_penalty": "0.0",
            "timeout_seconds": "120",
            "max_retries": "3",
        },
    ))

    # Auto-only credential overrides (per-provider). Hidden behind "Advanced".
    for provider in ("openai", "anthropic", "google", "azure", "huggingface"):
        settings_to_create.append({
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": f"credential_name_{provider}",
            "default_value": "",
            "value_type": "string",
            "description": (
                f"Explicit credential name for {provider.title()} (e.g. "
                f"`development_{provider}`). Leave empty to auto-resolve."
            ),
            "is_required": False,
        })

    # =========================================================================
    # SYSTEM LLM (everything internal — cheap-fast)
    # =========================================================================
    settings_to_create.extend(_llm_tier_settings(
        SettingCategory.SYSTEM_LLM.value,
        defaults={
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
            "temperature": "0.3",  # lower than Auto — internal tasks favor consistency
            "max_tokens": "8000",
            "top_p": "1.0",
            "frequency_penalty": "0.0",
            "presence_penalty": "0.0",
            "timeout_seconds": "60",
            "max_retries": "3",
        },
    ))

    # =========================================================================
    # EMBEDDINGS (vectorization — different model family)
    # =========================================================================
    settings_to_create.extend([
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "provider",
            "default_value": "openrouter",
            "value_type": "string",
            "description": (
                "Which provider serves embeddings. OpenRouter routes to "
                "Qwen/Cohere/OpenAI; pick `disabled` to turn off RAG entirely."
            ),
            "is_required": True,
            "validation_rules": {
                "options": ["openrouter", "openai", "google", "cohere", "huggingface_local", "huggingface_api", "disabled"]
            },
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "model",
            "default_value": "qwen/qwen3-embedding-8b",
            "value_type": "string",
            "description": (
                "Which embedding model converts text to vectors. The default "
                "(Qwen3-8B) is multilingual and supports Matryoshka truncation."
            ),
            "is_required": True,
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "dimensions",
            "default_value": "2048",
            "value_type": "number",
            "description": (
                "Vector size. Qwen3-8B supports Matryoshka truncation to 2048 "
                "(default) or smaller. Must match the vector store's index dim."
            ),
            "is_required": False,
            "validation_rules": {"min": 128, "max": 4096},
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "max_seq_length",
            "default_value": "256",
            "value_type": "number",
            "description": (
                "Max tokens per chunk fed to the embedding model. 256 is the "
                "standard chunk size — increase only if your docs have long "
                "self-contained sections."
            ),
            "is_required": False,
            "validation_rules": {"min": 64, "max": 8192},
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "chunk_size",
            "default_value": "512",
            "value_type": "number",
            "description": "Document chunk size (characters) for embedding ingestion.",
            "is_required": False,
            "validation_rules": {"min": 64, "max": 4096},
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "chunk_overlap",
            "default_value": "50",
            "value_type": "number",
            "description": (
                "Overlap (characters) between adjacent chunks. Helps preserve "
                "context across chunk boundaries."
            ),
            "is_required": False,
            "validation_rules": {"min": 0, "max": 256},
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "vector_store_type",
            "default_value": "faiss",
            "value_type": "string",
            "description": "Vector store backend.",
            "is_required": False,
            "validation_rules": {"options": ["faiss", "chroma", "pinecone", "weaviate"]},
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "cache_dir",
            "default_value": "./model_cache",
            "value_type": "string",
            "description": "Directory for cached embedding model weights.",
            "is_required": False,
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "rerank_enabled",
            "default_value": "false",
            "value_type": "boolean",
            "description": (
                "Re-rank vector search hits with Cohere for 15-30% precision "
                "lift. Requires a Cohere API key in Backend API Keys."
            ),
            "is_required": False,
            "validation_rules": {"options": ["true", "false"]},
        },
        {
            "category": SettingCategory.EMBEDDINGS.value,
            "key": "rerank_model",
            "default_value": "rerank-v3.5",
            "value_type": "string",
            "description": "Cohere rerank model.",
            "is_required": False,
            "validation_rules": {
                "options": ["rerank-v3.5", "rerank-english-v3.0", "rerank-multilingual-v3.0"]
            },
        },
    ])

    # =========================================================================
    # SYSTEM LOGGING (operational — untouched by PRD-136)
    # =========================================================================
    settings_to_create.extend([
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_level", "default_value": "INFO", "value_type": "string", "description": "System logging level.", "is_required": False, "validation_rules": {"options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_format", "default_value": "json", "value_type": "string", "description": "Log format.", "is_required": False, "validation_rules": {"options": ["json", "text", "structured"]}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_file_path", "default_value": "/var/log/automatos/app.log", "value_type": "string", "description": "Path to log file.", "is_required": False},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_max_size", "default_value": "100", "value_type": "number", "description": "Maximum log file size (MB).", "is_required": False, "validation_rules": {"min": 1, "max": 1000}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_backup_count", "default_value": "7", "value_type": "number", "description": "Number of backup log files to keep.", "is_required": False, "validation_rules": {"min": 1, "max": 50}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_rotation_interval", "default_value": "24", "value_type": "number", "description": "Log rotation interval (hours).", "is_required": False, "validation_rules": {"min": 1, "max": 168}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_retention_days", "default_value": "30", "value_type": "number", "description": "Log retention period (days).", "is_required": False, "validation_rules": {"min": 1, "max": 365}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_compress", "default_value": "false", "value_type": "boolean", "description": "Enable log file compression.", "is_required": False, "validation_rules": {"options": ["true", "false"]}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_buffer_size", "default_value": "64", "value_type": "number", "description": "Log buffer size (KB).", "is_required": False, "validation_rules": {"min": 1, "max": 1024}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_flush_interval", "default_value": "5", "value_type": "number", "description": "Log flush interval (seconds).", "is_required": False, "validation_rules": {"min": 1, "max": 60}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_async", "default_value": "false", "value_type": "boolean", "description": "Enable async logging.", "is_required": False, "validation_rules": {"options": ["true", "false"]}},
        {"category": SettingCategory.SYSTEM_LOGGING.value, "key": "log_queue_size", "default_value": "1000", "value_type": "number", "description": "Log queue size for async logging.", "is_required": False, "validation_rules": {"min": 100, "max": 10000}},
    ])

    # =========================================================================
    # API RATE LIMITING (operational — untouched by PRD-136)
    # =========================================================================
    settings_to_create.extend([
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "enabled", "default_value": "true", "value_type": "boolean", "description": "Enable API rate limiting.", "is_required": False, "validation_rules": {"options": ["true", "false"]}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "requests_per_window", "default_value": "100", "value_type": "number", "description": "Maximum requests per time window.", "is_required": False, "validation_rules": {"min": 1, "max": 10000}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "window_seconds", "default_value": "60", "value_type": "number", "description": "Rate limiting time window (seconds).", "is_required": False, "validation_rules": {"min": 1, "max": 3600}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "burst_limit", "default_value": "200", "value_type": "number", "description": "Maximum burst requests.", "is_required": False, "validation_rules": {"min": 1, "max": 1000}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "recovery_time", "default_value": "300", "value_type": "number", "description": "Time to recover from rate limit (seconds).", "is_required": False, "validation_rules": {"min": 1, "max": 3600}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "skip_successful_requests", "default_value": "false", "value_type": "boolean", "description": "Don't count successful requests in rate limit.", "is_required": False, "validation_rules": {"options": ["true", "false"]}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "skip_failed_requests", "default_value": "false", "value_type": "boolean", "description": "Don't count failed requests in rate limit.", "is_required": False, "validation_rules": {"options": ["true", "false"]}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "storage_backend", "default_value": "redis", "value_type": "string", "description": "Storage backend for rate limiting.", "is_required": False, "validation_rules": {"options": ["redis", "memory"]}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "cleanup_interval", "default_value": "300", "value_type": "number", "description": "Cleanup interval for rate limit keys (seconds).", "is_required": False, "validation_rules": {"min": 60, "max": 3600}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "max_keys", "default_value": "100000", "value_type": "number", "description": "Maximum number of rate limit keys to store.", "is_required": False, "validation_rules": {"min": 1000, "max": 1000000}},
        {"category": SettingCategory.API_RATE_LIMITING.value, "key": "key_expiry", "default_value": "3600", "value_type": "number", "description": "Rate limit key expiry time (seconds).", "is_required": False, "validation_rules": {"min": 60, "max": 86400}},
    ])

    # =========================================================================
    # BACKEND API KEYS (operational — untouched by PRD-136)
    # =========================================================================
    settings_to_create.extend([
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "cohere_api_key", "default_value": "", "value_type": "string", "description": "Cohere API key for reranking. Get one at dashboard.cohere.com.", "is_required": False, "is_sensitive": True},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_key", "default_value": "", "value_type": "string", "description": "Backend API key for authentication.", "is_required": False, "is_sensitive": True},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_port", "default_value": "8000", "value_type": "number", "description": "Backend API port.", "is_required": False, "validation_rules": {"min": 1, "max": 65535}},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_url", "default_value": "", "value_type": "string", "description": "Backend API URL.", "is_required": False},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_key_length", "default_value": "32", "value_type": "number", "description": "API key length for generated keys.", "is_required": False, "validation_rules": {"min": 16, "max": 128}},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_key_expiry", "default_value": "90", "value_type": "number", "description": "API key expiry period (days).", "is_required": False, "validation_rules": {"min": 1, "max": 365}},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "max_api_keys_per_user", "default_value": "5", "value_type": "number", "description": "Maximum API keys per user.", "is_required": False, "validation_rules": {"min": 1, "max": 10}},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_key_prefix", "default_value": "ak_", "value_type": "string", "description": "Prefix for generated API keys.", "is_required": False},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_timeout", "default_value": "30", "value_type": "number", "description": "API request timeout (seconds).", "is_required": False, "validation_rules": {"min": 5, "max": 300}},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_retry_attempts", "default_value": "3", "value_type": "number", "description": "API retry attempts on failure.", "is_required": False, "validation_rules": {"min": 0, "max": 5}},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_rate_limit_per_key", "default_value": "1000", "value_type": "number", "description": "Rate limit per API key (requests per hour).", "is_required": False, "validation_rules": {"min": 1, "max": 10000}},
        {"category": SettingCategory.BACKEND_API_KEYS.value, "key": "api_monitoring_enabled", "default_value": "false", "value_type": "boolean", "description": "Enable API monitoring.", "is_required": False, "validation_rules": {"options": ["true", "false"]}},
    ])

    # =========================================================================
    # LLM COST AUDIT (cost tracking — keeps service-level granularity)
    # =========================================================================
    settings_to_create.extend([
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "enabled",
            "default_value": "true",
            "value_type": "boolean",
            "description": "Track every LLM call's tokens, model, and estimated cost.",
            "is_required": False,
        },
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "mission_budget_alert_usd",
            "default_value": "2.00",
            "value_type": "number",
            "description": "Log a WARNING when a single mission's LLM spend exceeds this (USD).",
            "is_required": False,
            "validation_rules": {"min": 0.10, "max": 100.00, "step": 0.50},
        },
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "daily_budget_alert_usd",
            "default_value": "20.00",
            "value_type": "number",
            "description": "Log a CRITICAL alert when daily LLM spend exceeds this (USD).",
            "is_required": False,
            "validation_rules": {"min": 1.00, "max": 500.00, "step": 1.00},
        },
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "log_every_call",
            "default_value": "true",
            "value_type": "boolean",
            "description": "Emit a structured log line for every LLM call (model, tokens, cost).",
            "is_required": False,
        },
    ])

    # ── One-time cleanup: fix stale orchestrator_llm values ──────────
    # The env vars LLM_PROVIDER=openai / LLM_MODEL=openai/gpt-5.4 were
    # captured into system_settings before they were deleted from Railway.
    # Reset to the correct defaults so config.LLM_PROVIDER returns the right value.
    from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
    _STALE_FIXES = {
        ("orchestrator_llm", "provider"): DEFAULT_LLM_PROVIDER,
        ("orchestrator_llm", "model"): DEFAULT_LLM_MODEL,
    }
    for (cat, key), correct_value in _STALE_FIXES.items():
        stale = db.query(SystemSetting).filter(
            SystemSetting.category == cat,
            SystemSetting.key == key,
        ).first()
        if stale and stale.value and stale.value != correct_value:
            old_val = stale.value
            stale.value = correct_value
            stale.default_value = correct_value
            logger.info(
                "Fixed stale system_setting %s.%s: '%s' → '%s'",
                cat, key, old_val, correct_value,
            )

    created_count = 0
    updated_count = 0
    preserved_count = 0

    for setting_data in settings_to_create:
        # Check if setting already exists
        existing = db.query(SystemSetting).filter(
            SystemSetting.category == setting_data["category"],
            SystemSetting.key == setting_data["key"]
        ).first()

        if existing:
            # CRITICAL: Never overwrite existing value - preserve user's setting
            metadata_changed = False

            if existing.default_value != setting_data.get("default_value"):
                existing.default_value = setting_data.get("default_value")
                metadata_changed = True

            if existing.description != setting_data.get("description"):
                existing.description = setting_data.get("description")
                metadata_changed = True

            existing_vrules = existing.validation_rules or {}
            new_vrules = setting_data.get("validation_rules") or {}
            if existing_vrules != new_vrules:
                existing.validation_rules = new_vrules
                metadata_changed = True

            if existing.is_required != setting_data.get("is_required", False):
                existing.is_required = setting_data.get("is_required", False)
                metadata_changed = True

            if existing.is_sensitive != setting_data.get("is_sensitive", False):
                existing.is_sensitive = setting_data.get("is_sensitive", False)
                metadata_changed = True

            if metadata_changed:
                updated_count += 1
                logger.debug(f"Updated metadata for {setting_data['category']}.{setting_data['key']}")
            else:
                preserved_count += 1
                logger.debug(f"Preserved existing setting {setting_data['category']}.{setting_data['key']}")

            # NEVER set existing.value = default_value - preserve user's choice!

        else:
            setting = SystemSetting(
                category=setting_data["category"],
                key=setting_data["key"],
                value=setting_data.get("default_value"),  # Initial value = default
                default_value=setting_data.get("default_value"),
                value_type=setting_data.get("value_type", "string"),
                description=setting_data.get("description"),
                is_sensitive=setting_data.get("is_sensitive", False),
                is_required=setting_data.get("is_required", False),
                validation_rules=setting_data.get("validation_rules"),
                created_by="system"
            )
            db.add(setting)
            created_count += 1
            logger.debug(f"Created setting {setting_data['category']}.{setting_data['key']}")

    try:
        db.commit()
        logger.info(
            f"✅ Seeded system settings: {created_count} created, {updated_count} metadata updated, "
            f"{preserved_count} preserved (user values not overwritten)"
        )
        return created_count, updated_count
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to seed system settings: {e}")
        raise
