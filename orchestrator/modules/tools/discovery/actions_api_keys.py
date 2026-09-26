"""SDK API-key ActionDefinitions (PRD-143 S11 — administration surface).

List and revoke the workspace's SDK keys via ApiKeyService — the same
service layer as ``api/api_keys.py``. Revoke is ``destructive`` + confirmed
(it cuts off whatever integration uses the key) and, since F151, an owner's
or admin's. There is no create tool: a new key's full value exists only in
its create response, so an owner or admin creates keys in Settings → Widget
SDK → API keys, which shows it once, and it never enters the LLM context.
BYOK provider keys are not exposed either, for the same reason.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_api_keys_actions(registry: ActionRegistry) -> None:
    """Register the SDK API-key administration tools."""

    registry.register(ActionDefinition(
        name="platform_list_api_keys",
        description=(
            "List the workspace's SDK API keys — name, masked prefix, type "
            "(public/server), permission scopes, expiry and last-used time. "
            "Keys are always masked; the full key only exists at creation. "
            "Use before revoking a key or when auditing API access. Auto "
            "cannot create a key: an owner or admin creates one in Settings → "
            "Widget SDK → API keys, which shows the full key once."
        ),
        category="api_keys",
        parameters={
            "type": "object",
            "properties": {},
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["api_keys", "sdk", "credentials", "security"],
        examples=[
            "list our API keys",
            "what SDK keys exist for this workspace?",
            "show API key usage",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_revoke_api_key",
        description=(
            "Revoke (deactivate) an SDK API key — any integration using it "
            "stops working immediately. Use platform_list_api_keys first to "
            "get the key id."
        ),
        category="api_keys",
        parameters={
            "type": "object",
            "properties": {
                "key_id": {
                    "type": "string",
                    "description": "The key id (UUID) from platform_list_api_keys.",
                },
            },
            "required": ["key_id"],
        },
        permission_level="destructive",
        requires_confirmation=True,
        admin_only=True,  # F151: REST: workspace:manage
        tags=["api_keys", "sdk", "credentials", "security", "revoke"],
        examples=[
            "revoke the old CI key",
            "disable that leaked API key",
        ],
    ))
