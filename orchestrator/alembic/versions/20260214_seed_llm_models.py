"""PRD-54: Seed LLM model catalog with Tier 1 & Tier 2 models

Revision ID: 20260214_seed_llm_models
Revises: 20260214_llm_marketplace
Create Date: 2026-02-14

Populates llm_models with accurate pricing, capabilities, and marketplace metadata
for OpenAI, Anthropic, Google (Tier 1) and OpenRouter (Tier 2) models.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
import json
from datetime import datetime

revision = '20260214_seed_llm_models'
down_revision = '20260214_llm_marketplace'
branch_labels = None
depends_on = None


# Lightweight table reference for bulk inserts
llm_models = table(
    'llm_models',
    column('provider', sa.String),
    column('model_id', sa.String),
    column('display_name', sa.String),
    column('model_family', sa.String),
    column('capabilities', sa.JSON),
    column('context_window', sa.Integer),
    column('max_output_tokens', sa.Integer),
    column('supports_functions', sa.Boolean),
    column('supports_vision', sa.Boolean),
    column('supports_streaming', sa.Boolean),
    column('input_cost_per_1k_tokens', sa.Float),
    column('output_cost_per_1k_tokens', sa.Float),
    column('description', sa.Text),
    column('status', sa.String),
    column('recommended_for', sa.JSON),
    column('default_temperature', sa.Float),
    column('tier', sa.String),
    column('tags', sa.JSON),
    column('category', sa.String),
    column('is_featured', sa.Boolean),
    column('is_default', sa.Boolean),
    column('requires_plan', sa.String),
    column('external_id', sa.String),
)


MODELS = [
    # ═══════════════════════════════════════════════════════════════════
    # TIER 1 — DIRECT PROVIDERS
    # ═══════════════════════════════════════════════════════════════════

    # ── OpenAI ─────────────────────────────────────────────────────────
    {
        "provider": "openai", "model_id": "gpt-4o", "display_name": "GPT-4o",
        "model_family": "gpt-4o", "context_window": 128000, "max_output_tokens": 16384,
        "input_cost_per_1k_tokens": 0.0025, "output_cost_per_1k_tokens": 0.01,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "creativity": "good"},
        "recommended_for": ["code_review", "analysis", "complex_reasoning", "general_tasks"],
        "description": "OpenAI's flagship multimodal model. Excellent across all tasks.",
        "tier": "direct", "category": "premium", "is_featured": True, "is_default": False,
        "requires_plan": "pro", "tags": ["multimodal", "flagship", "popular"],
    },
    {
        "provider": "openai", "model_id": "gpt-4o-mini", "display_name": "GPT-4o Mini",
        "model_family": "gpt-4o", "context_window": 128000, "max_output_tokens": 16384,
        "input_cost_per_1k_tokens": 0.00015, "output_cost_per_1k_tokens": 0.0006,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "good", "coding": "good", "analysis": "good", "creativity": "good"},
        "recommended_for": ["routing", "simple_tasks", "content_generation", "general_tasks"],
        "description": "Affordable and fast. Great for routing, simple tasks, and high-volume use.",
        "tier": "direct", "category": "fast", "is_featured": False, "is_default": True,
        "requires_plan": None, "tags": ["affordable", "fast", "default"],
    },
    {
        "provider": "openai", "model_id": "gpt-4-turbo", "display_name": "GPT-4 Turbo",
        "model_family": "gpt-4", "context_window": 128000, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.01, "output_cost_per_1k_tokens": 0.03,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "creativity": "good"},
        "recommended_for": ["complex_reasoning", "code_review", "analysis"],
        "description": "Previous-gen flagship with 128K context. Still excellent for complex work.",
        "tier": "direct", "category": "premium", "is_featured": False, "is_default": False,
        "requires_plan": "enterprise", "tags": ["legacy", "reliable"],
    },

    # ── Anthropic ──────────────────────────────────────────────────────
    {
        "provider": "anthropic", "model_id": "claude-sonnet-4-5-20250929",
        "display_name": "Claude Sonnet 4.5", "model_family": "claude-4",
        "context_window": 200000, "max_output_tokens": 8192,
        "input_cost_per_1k_tokens": 0.003, "output_cost_per_1k_tokens": 0.015,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "creativity": "excellent"},
        "recommended_for": ["code_review", "analysis", "creative_writing", "complex_reasoning"],
        "description": "Anthropic's latest Sonnet — exceptional balance of intelligence and speed.",
        "tier": "direct", "category": "premium", "is_featured": True, "is_default": False,
        "requires_plan": "pro", "tags": ["latest", "balanced", "popular"],
    },
    {
        "provider": "anthropic", "model_id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5", "model_family": "claude-4",
        "context_window": 200000, "max_output_tokens": 8192,
        "input_cost_per_1k_tokens": 0.0008, "output_cost_per_1k_tokens": 0.004,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "good", "coding": "good", "analysis": "good", "creativity": "good"},
        "recommended_for": ["simple_tasks", "content_generation", "routing", "general_tasks"],
        "description": "Fast and affordable with 200K context. Great for simple tasks and high volume.",
        "tier": "direct", "category": "fast", "is_featured": False, "is_default": True,
        "requires_plan": None, "tags": ["fast", "affordable", "default"],
    },
    {
        "provider": "anthropic", "model_id": "claude-3-5-sonnet-20241022",
        "display_name": "Claude 3.5 Sonnet", "model_family": "claude-3.5",
        "context_window": 200000, "max_output_tokens": 8192,
        "input_cost_per_1k_tokens": 0.003, "output_cost_per_1k_tokens": 0.015,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "creativity": "very good"},
        "recommended_for": ["code_review", "analysis", "complex_reasoning", "general_tasks"],
        "description": "Previous-gen Sonnet. Strong across all tasks with 200K context.",
        "tier": "direct", "category": "balanced", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["reliable", "popular"],
    },
    {
        "provider": "anthropic", "model_id": "claude-3-opus-20240229",
        "display_name": "Claude 3 Opus", "model_family": "claude-3",
        "context_window": 200000, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.015, "output_cost_per_1k_tokens": 0.075,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "creativity": "excellent"},
        "recommended_for": ["complex_reasoning", "research", "analysis"],
        "description": "Most capable Claude 3 model. Best for complex, nuanced tasks.",
        "tier": "direct", "category": "premium", "is_featured": False, "is_default": False,
        "requires_plan": "enterprise", "tags": ["premium", "most-capable"],
    },

    # ── Google ─────────────────────────────────────────────────────────
    {
        "provider": "google", "model_id": "gemini-2.0-flash",
        "display_name": "Gemini 2.0 Flash", "model_family": "gemini-2",
        "context_window": 1048576, "max_output_tokens": 8192,
        "input_cost_per_1k_tokens": 0.000075, "output_cost_per_1k_tokens": 0.0003,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "good", "coding": "good", "analysis": "good", "creativity": "moderate"},
        "recommended_for": ["document_processing", "simple_tasks", "high_volume", "general_tasks"],
        "description": "Extremely cheap with 1M token context. Ideal for document processing.",
        "tier": "direct", "category": "fast", "is_featured": True, "is_default": True,
        "requires_plan": None, "tags": ["cheapest", "huge-context", "default"],
    },
    {
        "provider": "google", "model_id": "gemini-1.5-pro",
        "display_name": "Gemini 1.5 Pro", "model_family": "gemini-1.5",
        "context_window": 2097152, "max_output_tokens": 8192,
        "input_cost_per_1k_tokens": 0.00125, "output_cost_per_1k_tokens": 0.005,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "very good", "coding": "good", "analysis": "very good", "creativity": "good"},
        "recommended_for": ["document_processing", "analysis", "research"],
        "description": "2M token context — largest available. Great for processing entire codebases.",
        "tier": "direct", "category": "balanced", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["largest-context", "multimodal"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # TIER 2 — OPENROUTER AGGREGATOR
    # ═══════════════════════════════════════════════════════════════════

    # ── Fast & Cheap ───────────────────────────────────────────────────
    {
        "provider": "openrouter", "model_id": "meta-llama/llama-3.1-8b-instruct",
        "display_name": "Llama 3.1 8B Instruct", "model_family": "llama-3",
        "context_window": 131072, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.000055, "output_cost_per_1k_tokens": 0.000055,
        "supports_functions": True, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "moderate", "coding": "moderate", "analysis": "moderate", "creativity": "moderate"},
        "recommended_for": ["simple_tasks", "content_generation", "routing"],
        "description": "Meta's lightweight open-source model. Very cheap for simple agent tasks.",
        "tier": "aggregator", "category": "fast", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "meta", "cheap"],
        "external_id": "meta-llama/llama-3.1-8b-instruct",
    },
    {
        "provider": "openrouter", "model_id": "mistralai/mistral-7b-instruct",
        "display_name": "Mistral 7B Instruct", "model_family": "mistral",
        "context_window": 32768, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.000055, "output_cost_per_1k_tokens": 0.000055,
        "supports_functions": False, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "moderate", "coding": "moderate", "analysis": "moderate", "creativity": "moderate"},
        "recommended_for": ["simple_tasks", "content_generation"],
        "description": "Mistral's efficient small model. Solid general performance at very low cost.",
        "tier": "aggregator", "category": "fast", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "mistral", "cheap"],
        "external_id": "mistralai/mistral-7b-instruct",
    },
    {
        "provider": "openrouter", "model_id": "qwen/qwen-2.5-7b-instruct",
        "display_name": "Qwen 2.5 7B Instruct", "model_family": "qwen",
        "context_window": 131072, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.000054, "output_cost_per_1k_tokens": 0.000054,
        "supports_functions": True, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "moderate", "coding": "good", "analysis": "moderate", "creativity": "moderate"},
        "recommended_for": ["simple_tasks", "coding"],
        "description": "Alibaba's efficient model with strong coding ability and 128K context.",
        "tier": "aggregator", "category": "fast", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "alibaba", "cheap"],
        "external_id": "qwen/qwen-2.5-7b-instruct",
    },

    # ── Balanced ───────────────────────────────────────────────────────
    {
        "provider": "openrouter", "model_id": "meta-llama/llama-3.1-70b-instruct",
        "display_name": "Llama 3.1 70B Instruct", "model_family": "llama-3",
        "context_window": 131072, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.00052, "output_cost_per_1k_tokens": 0.00052,
        "supports_functions": True, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "good", "coding": "good", "analysis": "good", "creativity": "good"},
        "recommended_for": ["general_tasks", "content_generation", "analysis"],
        "description": "Meta's flagship open-source model. Strong all-round performance at fraction of GPT-4 cost.",
        "tier": "aggregator", "category": "balanced", "is_featured": True, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "meta", "popular"],
        "external_id": "meta-llama/llama-3.1-70b-instruct",
    },
    {
        "provider": "openrouter", "model_id": "deepseek/deepseek-chat",
        "display_name": "DeepSeek Chat", "model_family": "deepseek",
        "context_window": 131072, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.00014, "output_cost_per_1k_tokens": 0.00028,
        "supports_functions": True, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "very good", "coding": "excellent", "analysis": "very good", "creativity": "good"},
        "recommended_for": ["coding", "analysis", "general_tasks"],
        "description": "Excellent reasoning and coding at remarkably low cost. Strong open-source contender.",
        "tier": "aggregator", "category": "balanced", "is_featured": True, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "deepseek", "value"],
        "external_id": "deepseek/deepseek-chat",
    },
    {
        "provider": "openrouter", "model_id": "mistralai/mixtral-8x22b-instruct",
        "display_name": "Mixtral 8x22B Instruct", "model_family": "mixtral",
        "context_window": 65536, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.0009, "output_cost_per_1k_tokens": 0.0009,
        "supports_functions": True, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "good", "coding": "good", "analysis": "good", "creativity": "good"},
        "recommended_for": ["general_tasks", "content_generation"],
        "description": "Mistral's MoE architecture — strong general performance, efficient inference.",
        "tier": "aggregator", "category": "balanced", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "mistral", "moe"],
        "external_id": "mistralai/mixtral-8x22b-instruct",
    },

    # ── Coding Specialists ─────────────────────────────────────────────
    {
        "provider": "openrouter", "model_id": "deepseek/deepseek-coder",
        "display_name": "DeepSeek Coder", "model_family": "deepseek",
        "context_window": 131072, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.00014, "output_cost_per_1k_tokens": 0.00028,
        "supports_functions": True, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "good", "coding": "excellent", "analysis": "good", "creativity": "moderate"},
        "recommended_for": ["coding", "code_review", "code_generation"],
        "description": "Purpose-built for code generation and understanding. Top open-source coding model.",
        "tier": "aggregator", "category": "coding", "is_featured": True, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "coding", "specialist"],
        "external_id": "deepseek/deepseek-coder",
    },
    {
        "provider": "openrouter", "model_id": "qwen/qwen-2.5-coder-32b-instruct",
        "display_name": "Qwen 2.5 Coder 32B", "model_family": "qwen",
        "context_window": 131072, "max_output_tokens": 4096,
        "input_cost_per_1k_tokens": 0.00018, "output_cost_per_1k_tokens": 0.00018,
        "supports_functions": True, "supports_vision": False, "supports_streaming": True,
        "capabilities": {"reasoning": "good", "coding": "excellent", "analysis": "good", "creativity": "moderate"},
        "recommended_for": ["coding", "code_review", "code_generation"],
        "description": "Alibaba's coding specialist. Competes with GPT-4 on code tasks at much lower cost.",
        "tier": "aggregator", "category": "coding", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["open-source", "coding", "alibaba"],
        "external_id": "qwen/qwen-2.5-coder-32b-instruct",
    },

    # ── Premium (via OpenRouter) ───────────────────────────────────────
    {
        "provider": "openrouter", "model_id": "openai/gpt-4o",
        "display_name": "GPT-4o (via OpenRouter)", "model_family": "gpt-4o",
        "context_window": 128000, "max_output_tokens": 16384,
        "input_cost_per_1k_tokens": 0.0025, "output_cost_per_1k_tokens": 0.01,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "excellent", "coding": "excellent", "analysis": "excellent", "creativity": "good"},
        "recommended_for": ["code_review", "analysis", "complex_reasoning"],
        "description": "OpenAI GPT-4o accessed through OpenRouter. Same model, aggregator pricing.",
        "tier": "aggregator", "category": "premium", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["openai", "premium", "aggregator"],
        "external_id": "openai/gpt-4o",
    },
    {
        "provider": "openrouter", "model_id": "google/gemini-pro-1.5",
        "display_name": "Gemini 1.5 Pro (via OpenRouter)", "model_family": "gemini-1.5",
        "context_window": 2097152, "max_output_tokens": 8192,
        "input_cost_per_1k_tokens": 0.00125, "output_cost_per_1k_tokens": 0.005,
        "supports_functions": True, "supports_vision": True, "supports_streaming": True,
        "capabilities": {"reasoning": "very good", "coding": "good", "analysis": "very good", "creativity": "good"},
        "recommended_for": ["document_processing", "analysis", "research"],
        "description": "Google Gemini 1.5 Pro via OpenRouter. 2M token context window.",
        "tier": "aggregator", "category": "balanced", "is_featured": False, "is_default": False,
        "requires_plan": "pro", "tags": ["google", "largest-context", "aggregator"],
        "external_id": "google/gemini-pro-1.5",
    },
]


def upgrade() -> None:
    # Delete any existing seeded models to avoid duplicates on re-run
    conn = op.get_bind()
    model_ids = [m["model_id"] for m in MODELS]
    for mid in model_ids:
        conn.execute(
            sa.text("DELETE FROM llm_models WHERE model_id = :mid"),
            {"mid": mid},
        )

    # Insert all models
    op.bulk_insert(llm_models, [
        {
            "provider": m["provider"],
            "model_id": m["model_id"],
            "display_name": m["display_name"],
            "model_family": m["model_family"],
            "capabilities": m["capabilities"],
            "context_window": m["context_window"],
            "max_output_tokens": m["max_output_tokens"],
            "supports_functions": m["supports_functions"],
            "supports_vision": m["supports_vision"],
            "supports_streaming": m["supports_streaming"],
            "input_cost_per_1k_tokens": m["input_cost_per_1k_tokens"],
            "output_cost_per_1k_tokens": m["output_cost_per_1k_tokens"],
            "description": m["description"],
            "status": "active",
            "recommended_for": m["recommended_for"],
            "default_temperature": 0.7,
            "tier": m["tier"],
            "tags": m["tags"],
            "category": m["category"],
            "is_featured": m["is_featured"],
            "is_default": m["is_default"],
            "requires_plan": m.get("requires_plan"),
            "external_id": m.get("external_id"),
        }
        for m in MODELS
    ])


def downgrade() -> None:
    conn = op.get_bind()
    model_ids = [m["model_id"] for m in MODELS]
    for mid in model_ids:
        conn.execute(
            sa.text("DELETE FROM llm_models WHERE model_id = :mid"),
            {"mid": mid},
        )
