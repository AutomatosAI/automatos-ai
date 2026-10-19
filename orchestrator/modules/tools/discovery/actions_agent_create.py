"""platform_create_agent's ActionDefinition (left ``register_agents_actions``, 19 Oct 2026).

Omitting ``model_id`` gives the new agent the platform default
(``core/llm/new_agent_model``: system_settings ``orchestrator_llm``, set in Settings >
Orchestrator), not ``core/llm/defaults``' last-resort pair, which the copy used to
name as "the default model". ``actions_agents.takes_the_runtime`` registers it
through the same registry as the other agent actions, so it still gains the runtime
properties.
"""

from typing import Any, Dict

from .action_registry import ActionDefinition, ActionRegistry

OMIT_MODEL_ID = "Omit it for the platform's default model (Settings > Orchestrator)."


def _agent_properties(example_model: str) -> Dict[str, Any]:
    """What the agent is: name, type, purpose, model and how it speaks."""
    return {
        "name": {
            "type": "string",
            "description": "Name for the new agent.",
        },
        "agent_type": {
            "type": "string",
            "enum": ["chatbot", "worker", "researcher", "coder"],
            "description": "Type of agent to create. Defaults to 'chatbot'.",
        },
        "description": {
            "type": "string",
            "description": "Brief description of the agent's purpose.",
        },
        "model_id": {
            "type": "string",
            "description": (
                "LLM model ID, as platform_list_workspace_models lists it "
                f"(e.g. '{example_model}'). {OMIT_MODEL_ID}"
            ),
        },
        "system_prompt": {
            "type": "string",
            "description": (
                "Custom system prompt that defines the agent's persona, behaviour, "
                "and constraints. This is the instruction text the agent sees at the "
                "start of every conversation."
            ),
        },
        "temperature": {
            "type": "number",
            "description": (
                "Sampling temperature (0.0–2.0). Lower values are more deterministic, "
                "higher values are more creative. Defaults to 0.7."
            ),
        },
    }


def _org_properties() -> Dict[str, Any]:
    """Where the agent sits: tags, team, title and manager."""
    return {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for categorisation (e.g. ['support', 'customer-facing']).",
        },
        "team": {
            "type": "string",
            "description": "Department/team name (e.g. 'Engineering & DevOps', 'Growth & Marketing').",
        },
        "job_title": {
            "type": "string",
            "description": "Human-readable role title (e.g. 'Engineering Reliability & Security Lead').",
        },
        "reports_to_id": {
            "type": "integer",
            "description": "Agent ID of the manager this agent reports to (org hierarchy).",
        },
    }


def register_create_agent_action(registry: ActionRegistry) -> None:
    """Register platform_create_agent."""
    from core.llm.defaults import DEFAULT_LLM_MODEL

    registry.register(ActionDefinition(
        name="platform_create_agent",
        description=(
            "Create a new agent in the workspace. Requires a name and agent type. "
            "Optionally accepts description, model, system prompt, temperature, tags, "
            "team, job_title, and reports_to_id. "
            "Use when the user asks to create, add, or set up a new agent."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {**_agent_properties(DEFAULT_LLM_MODEL), **_org_properties()},
            "required": ["name"],
        },
        permission_level="write",
        promoted=True,
        requires_confirmation=False,
        tags=["agents", "create", "write"],
        examples=[
            "create an agent called DevOps Bot",
            "make a new researcher agent",
            "create a support agent using claude sonnet with a helpful persona",
        ],
        accepts=("model", "provider"),
    ))


__all__ = ["OMIT_MODEL_ID", "register_create_agent_action"]
