"""Widget-config ActionDefinitions — the embedded-widget surface (PRD-143 S10).

Exposes the public widget-config slice of ``workspace.settings`` (the keys
``api.widgets.config.PUBLIC_WIDGET_CONFIG_KEYS`` serves to browser widgets:
proactive engagement, cart-idle popup, callback form). The handler validates
keys against that whitelist at call time — it is the single source of truth,
never duplicated here.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_widgets_actions(registry: ActionRegistry) -> None:
    """Register the widget-config platform tools."""

    registry.register(ActionDefinition(
        name="platform_get_widget_config",
        description=(
            "Read the workspace's public widget configuration — the settings the "
            "embedded site widget actually receives (proactive engagement, "
            "cart-idle popup, callback form). Also returns which keys are "
            "configurable. Use before changing widget behaviour or to explain "
            "why the widget is (not) doing something on the merchant's site."
        ),
        category="widgets",
        parameters={
            "type": "object",
            "properties": {},
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["widgets", "configuration", "setup", "site"],
        examples=[
            "what's the widget config for this workspace?",
            "is the cart-idle popup enabled?",
            "show the proactive engagement settings",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_widget_config",
        description=(
            "Set one public widget-config key for the workspace (e.g. enable the "
            "cart-idle popup, tune proactive-engagement delays, configure the "
            "callback form). Only whitelisted widget keys can be written — "
            "anything else is refused. The embedded widget picks the change up "
            "on its next config fetch."
        ),
        category="widgets",
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "The widget-config key to set — one of the public whitelist "
                        "(currently 'widget_proactive', 'cart_idle', 'callback'; "
                        "platform_get_widget_config returns the live list)."
                    ),
                },
                "config": {
                    "type": "object",
                    "description": "The config object to store under that key (e.g. {'enabled': true, 'idle_seconds': 30}).",
                },
            },
            "required": ["key", "config"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["widgets", "configuration", "setup", "site"],
        examples=[
            "enable the cart idle popup",
            "set the widget proactive delay to 10 seconds",
            "configure the callback form destinations",
        ],
    ))
