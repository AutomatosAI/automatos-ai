"""SDK API-key ActionDefinitions (PRD-143 S11 — administration surface).

Operator tier by design (the Rev 2 inversion). List/create/revoke the
workspace's SDK keys via ApiKeyService — the same service layer as
``api/api_keys.py``. Revoke is ``destructive`` + confirmed (it cuts off
whatever integration uses the key). BYOK provider keys are deliberately
not exposed: raw provider secrets must never transit the LLM context.
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
            "Use before revoking a key or when auditing API access."
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
        name="platform_create_api_key",
        description=(
            "Create a new SDK API key for the workspace. 'server' keys are "
            "for backend integrations; 'public' keys are for browser widgets "
            "and require allowed_domains. The full key is returned exactly "
            "once — it cannot be retrieved again. Permission scopes follow "
            "the SDK catalogue (chat, blog, documents:read, agents:execute, …)."
        ),
        category="api_keys",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable key name (e.g. 'CI deploy key').",
                },
                "key_type": {
                    "type": "string",
                    "enum": ["server", "public"],
                    "description": "Key type — 'server' (backend) or 'public' (browser, needs allowed_domains).",
                },
                "permissions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Permission scopes to grant (e.g. ['chat', 'documents:read']).",
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Origins allowed to use the key — required for public keys.",
                },
            },
            "required": ["name"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["api_keys", "sdk", "credentials", "integrations"],
        examples=[
            "create a server API key for our CI",
            "make a public widget key for acme.com",
            "generate an SDK key with chat permission",
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
        tags=["api_keys", "sdk", "credentials", "security", "revoke"],
        examples=[
            "revoke the old CI key",
            "disable that leaked API key",
        ],
    ))
