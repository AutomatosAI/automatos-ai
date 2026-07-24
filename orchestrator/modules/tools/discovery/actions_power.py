"""Power-mode ActionDefinition — the per-workspace compute/quality tier (PRD-142 Wave 4, W4-S5).

Exposes the workspace power-mode knob (``workspace.settings['power_mode']``) as a
platform action. A Mission run inherits this default when its run_config doesn't
pin a mode (see ``coordinator_service._workspace_power_mode_default``). This is the
action HARNESS's ``power_mode_upgrade`` / ``power_mode_downgrade`` prescription
applies, and a tool Auto can use to tune a workspace's cost/quality trade-off.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_power_actions(registry: ActionRegistry) -> None:
    """Register the power-mode platform tool."""

    registry.register(ActionDefinition(
        name="platform_set_power_mode",
        description=(
            "Set the workspace's default power mode — the cost/quality tier each "
            "Mission run inherits unless it pins its own. 'light' is cheapest "
            "(small token + tool-iteration budget, system LLM); 'standard' is the "
            "balanced default; 'max' is highest-quality (large budget, orchestrator "
            "LLM). Use to dial a workspace's spend vs. quality trade-off."
        ),
        category="configuration",
        parameters={
            "type": "object",
            "properties": {
                "power_mode": {
                    "type": "string",
                    "enum": ["light", "standard", "max"],
                    "description": "The compute/quality tier: 'light', 'standard', or 'max'.",
                },
            },
            "required": ["power_mode"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["power_mode", "configuration", "cost", "quality"],
        examples=[
            "set the workspace to max power mode",
            "switch to light power mode to save cost",
            "use standard power mode",
            "dial up quality for this workspace",
        ],
    ))

    # PRD-143 S10: the read side of the knob — setup-surface gap-fill.
    registry.register(ActionDefinition(
        name="platform_get_power_mode",
        description=(
            "Read the workspace's default power mode — the cost/quality tier a "
            "Mission run inherits unless it pins its own ('light', 'standard' or "
            "'max'). Reports whether the value is a stored workspace setting or "
            "the platform default. Use before changing the dial or when "
            "explaining a workspace's spend/quality behaviour."
        ),
        category="configuration",
        parameters={
            "type": "object",
            "properties": {},
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["power_mode", "configuration", "cost", "quality"],
        examples=[
            "what power mode is this workspace on?",
            "check the current power mode",
            "is the workspace running light, standard or max?",
        ],
    ))
