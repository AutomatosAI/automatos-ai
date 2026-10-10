"""
Single source of truth for LLM defaults across the orchestrator.

The REAL source of truth is Settings > Orchestrator (the Auto agent row).
These defaults are only used:
  - when seeding a new Auto agent
  - as fallbacks when agents have no model_config
  - for agent creation via platform tools

Change here → changes everywhere.
"""

DEFAULT_LLM_PROVIDER = "openrouter"
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash"

# Image-generation default for blog covers and other on-demand image needs.
# Overridable per-deployment via env BLOG_COVER_MODEL or system_settings
# (category=content_creation, key=blog_cover_model). See config.BLOG_COVER_MODEL.
DEFAULT_IMAGE_GEN_MODEL = "google/gemini-3-pro-image-preview"

# Last-resort output-token budget — used only when neither the agent's configured
# Max Output Tokens nor the selected model's registry ceiling is available. The
# per-agent setting is the source of truth; the model's own max_output_tokens is
# the preferred fallback. This is the single named default; there are no other
# hardcoded token numbers in the agent/coordinator budget path.
DEFAULT_MAX_OUTPUT_TOKENS = 8000


def get_default_model_config() -> dict:
    """Return a fresh default model_config dict."""
    return {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_id": DEFAULT_LLM_MODEL,
        "temperature": 0.7,
        "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "fallback_model_id": None,
    }


# The cross-model verifier's defaults (executor family → verifier model), used when
# neither coordination.verifier_model_mapping nor its env names one. Read through
# core/llm/vendor_fit.py, which picks the pair for the System LLM's vendor.
VERIFIER_MODEL_MAPPING = (
    "anthropic=openai/gpt-4o-mini,openai=anthropic/claude-haiku-4-5,"
    "google=openai/gpt-4o-mini,deepseek=openai/gpt-4o-mini,meta=openai/gpt-4o-mini"
)
VERIFIER_FALLBACK_MODEL = "openai/gpt-4o-mini"
# P256-FIX-T1: with Anthropic as the System LLM the verifier is a Claude model other
# than the executor's (a tier key, ``sonnet=``, is tried before the family).
ANTHROPIC_VERIFIER_MODEL_MAPPING = "sonnet=claude-haiku-5-5,anthropic=claude-sonnet-5-5"
ANTHROPIC_VERIFIER_FALLBACK_MODEL = "claude-sonnet-5-5"
