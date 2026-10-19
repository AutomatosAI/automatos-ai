"""platform_install_model's ActionDefinition (left actions_marketplace.py, 19 Oct 2026).

The tool takes OpenRouter's vendor/model id, so its example is the platform default's
OpenRouter id: with the default on Anthropic (``claude-haiku-5-5``) that is the
dotted twin ``anthropic/claude-haiku-5.5``, never the direct id OpenRouter does not
list.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_model_install_action(registry: ActionRegistry) -> None:
    """Register platform_install_model."""
    from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
    from core.llm.providers import openrouter_model_id

    example = openrouter_model_id(DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL)
    registry.register(ActionDefinition(
        name="platform_install_model",
        description=(
            "Install an LLM model from OpenRouter catalog. "
            f"Use OpenRouter's vendor/model id, like '{example}'. "
            "Idempotent — re-installing reactivates inactive installs. "
            "Auto-creates registry entry from OpenRouter cache if needed."
        ),
        category="marketplace",
        parameters={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": f"OpenRouter model ID (e.g. '{example}').",
                },
            },
            "required": ["model_id"],
        },
        permission_level="write",
        tags=["marketplace", "models", "llm", "install", "enable"],
        examples=[
            f"install {example}",
            "add model google/gemini-2.5-pro",
            "enable openai/gpt-4o",
        ],
        accepts=("provider",),
    ))
