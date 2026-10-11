"""Diagnostics ActionDefinitions: what went wrong in this workspace, grouped by cause.

Auto's manual named ``platform_get_workspace_errors`` from August and no tool existed,
so a report of failures could not say why in one call (Auto's wishlist, 9 Oct 2026).
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_diagnostics_actions(registry: ActionRegistry) -> None:
    """Register the workspace-safe diagnostics actions."""

    registry.register(ActionDefinition(
        name="platform_get_workspace_errors",
        description=(
            "What went wrong in THIS workspace, grouped by cause: failed cards, failed LLM calls and "
            "failed tool runs, ranked by how often each cause happened (out of credit, timed out, "
            "rate-limited, a key refused, a tool missing, the model refused, not found, other), with "
            "examples to open by card number or id. Use it to answer 'why did these fail?' in one call."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Days to look back, 1 to 14. Defaults to 1.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["diagnostics", "errors", "failures", "health", "why"],
        examples=[
            "why did so many cards fail yesterday?",
            "what's going wrong in my workspace?",
            "group the failures by cause",
        ],
    ))
