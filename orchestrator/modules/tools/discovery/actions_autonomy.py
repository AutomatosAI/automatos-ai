"""Autonomy-level platform tools.

Two tools that let the workspace owner read and set how far Auto runs
unsupervised. The level lives on ``workspace.settings.autonomy`` and is read
at execute() time by PlatformActionExecutor:

    standard  — admin_only actions need an admin; requires_confirmation actions ask.
    full      — Auto runs as admin and the confirmation gate is skipped.

Both tools are ``promoted`` so Auto always sees the dial and can answer
"what can you do?" / "go full autonomy" without category routing. ``set`` is
``admin_only`` so only the workspace owner (not a self-directing sub-agent at
standard) can turn the dial up.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_autonomy_actions(registry: ActionRegistry) -> None:
    """Register the autonomy get/set tools."""

    registry.register(ActionDefinition(
        name="platform_get_autonomy_level",
        description=(
            "Read this workspace's autonomy level — 'standard' (Auto asks before "
            "writes that require confirmation and can't run admin-only tools) or "
            "'full' (Auto runs as admin and executes without asking). Use to answer "
            "'what's my autonomy level?' or before proposing to change it."
        ),
        category="settings",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["settings", "autonomy", "permissions"],
        examples=[
            "what autonomy level am I on?",
            "is Auto running supervised or full?",
            "how much can Auto do without asking?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_set_autonomy_level",
        description=(
            "Set this workspace's autonomy level. 'full' makes Auto run as admin and "
            "skips the confirmation gate — writes and the destructive deletes run "
            "without asking. 'standard' restores the supervised default. "
            "Workspace-scoped: rate limits and the agent-hierarchy permission check "
            "still apply. Only the workspace owner can change this."
        ),
        category="settings",
        parameters={
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["standard", "full"],
                    "description": (
                        "'full' = run unsupervised (admin + no confirmation); "
                        "'standard' = supervised default."
                    ),
                },
            },
            "required": ["level"],
        },
        permission_level="write",
        admin_only=True,
        promoted=True,
        tags=["settings", "autonomy", "permissions"],
        examples=[
            "go full autonomy",
            "run Auto unsupervised",
            "set autonomy back to standard",
        ],
    ))
