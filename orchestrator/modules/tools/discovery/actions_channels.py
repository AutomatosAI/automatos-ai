"""Channel ActionDefinitions — the messaging-channel surface (PRD-143 S10).

Exposes ``channel_connections`` (Telegram, Slack, Discord, WhatsApp, ...) to
Auto: list, connect (driver-mediated verify + webhook install / polling),
configure, start and stop. Connect/configure carry credentials in ``config``;
the handler delegates to the canonical ``api.channels`` flow so behaviour
never drifts from the dashboard's.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_channels_actions(registry: ActionRegistry) -> None:
    """Register the channel platform tools."""

    registry.register(ActionDefinition(
        name="platform_list_channels",
        description=(
            "List the workspace's messaging-channel connections (Telegram, Slack, "
            "Discord, Teams, WhatsApp, ...) with platform, live status, mode "
            "(webhook/polling), default agent and message counts. Use to see "
            "which channels are wired up before connecting or troubleshooting one."
        ),
        category="channels",
        parameters={
            "type": "object",
            "properties": {},
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["channels", "integrations", "messaging", "setup"],
        examples=[
            "which channels are connected?",
            "list the messaging channels",
            "is telegram connected for this workspace?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_connect_channel",
        description=(
            "Connect a messaging channel to the workspace. Creates the "
            "connection, verifies the credentials via the platform driver, then "
            "installs the inbound webhook or starts polling. Each platform "
            "needs specific config fields (e.g. telegram: bot_token; slack: "
            "bot_token + signing_secret; whatsapp: phone_number_id + "
            "access_token). Use when the user wants Auto reachable on a new channel."
        ),
        category="channels",
        parameters={
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": (
                        "The messaging platform: telegram, slack, discord, teams, "
                        "google_chat, signal, imessage, irc, matrix, line, whatsapp or webhook."
                    ),
                },
                "config": {
                    "type": "object",
                    "description": "Platform-specific credentials/config (e.g. {'bot_token': '...'}).",
                },
                "default_agent_id": {
                    "type": "string",
                    "description": "Optional agent to route this channel's messages to.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["webhook", "polling"],
                    "description": "Optional connectivity mode override; defaults to the driver's preference.",
                },
            },
            "required": ["platform", "config"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["channels", "integrations", "messaging", "setup", "connect"],
        examples=[
            "connect telegram with this bot token",
            "set up a slack channel connection",
            "wire whatsapp into the workspace",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_configure_channel",
        description=(
            "Update an existing channel connection — replace its credentials/config "
            "and/or change which agent handles its messages. Use platform_list_channels "
            "first to get the channel_id."
        ),
        category="channels",
        parameters={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The channel connection id (from platform_list_channels).",
                },
                "config": {
                    "type": "object",
                    "description": "New platform-specific credentials/config to store.",
                },
                "default_agent_id": {
                    "type": "string",
                    "description": "Agent to route this channel's messages to.",
                },
            },
            "required": ["channel_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["channels", "integrations", "messaging", "configure"],
        examples=[
            "rotate the telegram bot token",
            "point the slack channel at the support agent",
            "update the whatsapp channel config",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_start_channel",
        description=(
            "Start the adapter for a channel connection so it begins receiving "
            "messages (marks it active). Use after connecting or to resume a "
            "stopped channel."
        ),
        category="channels",
        parameters={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The channel connection id (from platform_list_channels).",
                },
            },
            "required": ["channel_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["channels", "integrations", "messaging", "lifecycle"],
        examples=[
            "start the telegram channel",
            "resume the discord connection",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_stop_channel",
        description=(
            "Stop the adapter for a channel connection so it stops receiving "
            "messages (marks it inactive). The connection and its config are kept."
        ),
        category="channels",
        parameters={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The channel connection id (from platform_list_channels).",
                },
            },
            "required": ["channel_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["channels", "integrations", "messaging", "lifecycle"],
        examples=[
            "pause the slack channel",
            "stop the telegram connection",
        ],
    ))
