"""The settings every LLM tier carries (PRD-136): the per-tier blocks ``seed_system_settings``
seeds for ``orchestrator_llm`` and ``system_llm``.

Every tier exposes the same shape, so a field added here reaches both tiers. Each
builder takes the tier's ``category`` and its ``defaults`` (one value per key) and
returns NEW dicts. The description is the tooltip the Settings page shows.
"""
from __future__ import annotations


def tier_route_settings(category: str, defaults: dict) -> list[dict]:
    """Provider and model: which LLM serves the tier."""
    return [
        {
            "category": category,
            "key": "provider",
            "default_value": defaults["provider"],
            "value_type": "string",
            "description": (
                "Which LLM provider serves this tier. OpenRouter is recommended — "
                "one credential covers 400+ models across vendors. Direct keys for "
                "OpenAI, Anthropic, Google and others work too."
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
    ]


def tier_sampling_settings(category: str, defaults: dict) -> list[dict]:
    """Temperature, reply length and nucleus sampling."""
    return [
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
    ]


def tier_penalty_settings(category: str, defaults: dict) -> list[dict]:
    """Repetition and novelty penalties."""
    return [
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
    ]


def tier_call_settings(category: str, defaults: dict) -> list[dict]:
    """How long to wait for one LLM call, and how often to retry it."""
    return [
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


__all__ = ["tier_call_settings", "tier_penalty_settings", "tier_route_settings", "tier_sampling_settings"]
