"""Field ActionDefinitions — PRD-108 Memory Field (shared semantic context)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_field_actions(registry: ActionRegistry) -> None:
    """Register shared field tools for mission agents."""

    registry.register(ActionDefinition(
        name="platform_field_query",
        description=(
            "Query the shared mission field for relevant context from other agents. "
            "Use this to find what other agents have discovered, analyzed, or produced "
            "during the current mission. Returns ranked results by relevance — "
            "patterns that resonate with your query surface first, stale patterns fade. "
            "Always query the field before starting work to see what's already known."
        ),
        category="field",
        promoted=True,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you're looking for — e.g. 'research findings about EU AI Act', 'analysis of competitor pricing'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
        tags=["field", "mission", "shared-context", "coordination", "memory"],
        examples=[
            "what have the other agents found so far",
            "check the field before I start researching",
            "find prior analysis on competitor pricing from the team",
            "see what's already known about the EU AI Act",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_field_inject",
        description=(
            "Share a finding, analysis, or conclusion with other mission agents by "
            "injecting it into the shared field. Other agents will see your contribution "
            "when they query the field — important findings resonate and surface first. "
            "Use this for key discoveries, intermediate results, and conclusions."
        ),
        category="field",
        promoted=True,
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Short label for this pattern — e.g. 'finding_1', 'competitor_analysis', 'risk_assessment'.",
                },
                "value": {
                    "type": "string",
                    "description": "The content to share — research finding, analysis result, conclusion, etc.",
                },
                "strength": {
                    "type": "number",
                    "description": "How important this is (0.0-1.0, default 1.0). Use lower values for uncertain findings.",
                    "default": 1.0,
                },
            },
            "required": ["key", "value"],
        },
        tags=["field", "mission", "share", "findings", "coordination"],
        examples=[
            "share this finding with the rest of the team",
            "record my analysis so other agents can see it",
            "post this conclusion to the shared field",
            "let the other agents know what I discovered",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_field_stability",
        description=(
            "Check how converged the shared field is — are agents reaching consensus? "
            "High stability means patterns are reinforcing each other. "
            "Low stability means the field is still evolving. "
            "Useful for coordinators to decide when enough research has been done."
        ),
        category="field",
        parameters={
            "type": "object",
            "properties": {},
        },
        tags=["field", "mission", "convergence", "consensus", "coordination"],
        examples=[
            "has the team reached consensus yet",
            "is the field converged enough to stop researching",
            "how stable are the agents' findings",
            "are we done gathering research",
        ],
    ))
