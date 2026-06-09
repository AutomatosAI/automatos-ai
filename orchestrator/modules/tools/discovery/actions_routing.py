"""Routing ActionDefinitions — workspace routing rules (PRD-142 Wave 4, W4-S6).

Wraps the existing ``routing_rules`` table (PRD-55), which the UniversalRouter
reads at Tier 2a (`core/routing/engine.py`) to direct inbound messages to an
agent or playbook. This is the platform action HARNESS's ``routing_rule_add``
prescription applies, and a tool Auto can use to set up channel/intent routing.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_routing_actions(registry: ActionRegistry) -> None:
    """Register routing-rule platform tools."""

    registry.register(ActionDefinition(
        name="platform_create_routing_rule",
        description=(
            "Create a workspace routing rule that directs inbound messages to a "
            "specific agent or playbook. Match by source_pattern (a substring/keyword "
            "in the message text) and/or source_channel (e.g. 'telegram', 'slack'), "
            "optionally narrowed by intent_keywords; route to target_agent_id OR "
            "target_workflow_id (provide one). Higher-priority rules win. Use to set "
            "up channel-based or keyword-based routing for a workspace."
        ),
        category="routing",
        parameters={
            "type": "object",
            "properties": {
                "source_pattern": {
                    "type": "string",
                    "description": "Substring/keyword to match in the message text (optional, but provide this or source_channel).",
                },
                "source_channel": {
                    "type": "string",
                    "description": "Channel to match, e.g. 'telegram', 'slack', 'whatsapp' (optional, but provide this or source_pattern).",
                },
                "intent_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional intent keywords that further narrow the match.",
                },
                "target_agent_id": {
                    "type": "integer",
                    "description": "Agent to route matching messages to (provide this OR target_workflow_id).",
                },
                "target_workflow_id": {
                    "type": "integer",
                    "description": "Playbook to route matching messages to (provide this OR target_agent_id).",
                },
                "priority": {
                    "type": "integer",
                    "description": "Higher priority rules are evaluated first. Default 0.",
                },
            },
            "required": [],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["routing", "rules", "channels", "configuration"],
        examples=[
            "route telegram messages to the support agent",
            "send anything mentioning 'invoice' to the billing playbook",
            "set up channel routing for slack",
            "create a routing rule",
        ],
    ))
