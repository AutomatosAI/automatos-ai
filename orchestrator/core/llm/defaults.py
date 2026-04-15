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


def get_default_model_config() -> dict:
    """Return a fresh default model_config dict."""
    return {
        "provider": DEFAULT_LLM_PROVIDER,
        "model_id": DEFAULT_LLM_MODEL,
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "fallback_model_id": None,
    }
