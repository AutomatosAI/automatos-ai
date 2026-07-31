"""Business-intake ActionDefinitions (PRD-222 W1S8) — Auto's scan + status tools.

Two tools expose the existing PRD-130 intake pipeline (Firecrawl scan → scrape →
RAG ingest → knowledge graph → profile) to Auto so onboarding can learn a user's
business conversationally instead of through the wizard UI (which W2 retires).
Schema truth from birth (US-011 rule): each tool's ``required`` names exactly the
param its handler hard-fails without.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_intake_actions(registry: ActionRegistry) -> None:
    """Register the intake scan + status tools (PRD-222 W1S8)."""

    registry.register(ActionDefinition(
        name="platform_scan_business_site",
        description=(
            "Start the business-intake pipeline (Firecrawl scan -> scrape -> RAG "
            "ingest -> knowledge graph -> profile) for the caller's workspace from "
            "a website domain. Returns {profile_id, started:true} the moment the "
            "background pipeline is launched — poll platform_get_intake_status with "
            "that profile_id to follow it. If web scanning is NOT configured on this "
            "deployment, returns {configured:false, alternatives:'doc upload / "
            "conversation'} — offer those instead; this is an honest result, never "
            "an error to retry."
        ),
        category="onboarding",
        parameters={
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "The business website domain to scan, e.g. 'acme.com'.",
                },
            },
            "required": ["domain"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["onboarding", "intake", "write", "auto"],
        examples=[
            "scan the user's business site acme.com",
            "kick off intake for example.co.uk",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_intake_status",
        description=(
            "Get the current stage and a summary of a business-intake run started "
            "by platform_scan_business_site. Provide the profile_id it returned. "
            "Returns {stage, domain, archetype, pages_found, pages_selected, ...}. "
            "A profile_id that does not belong to the caller's workspace is refused."
        ),
        category="onboarding",
        parameters={
            "type": "object",
            "properties": {
                "profile_id": {
                    "type": "string",
                    "description": "The intake profile_id returned by platform_scan_business_site.",
                },
            },
            "required": ["profile_id"],
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["onboarding", "intake", "read", "auto"],
        examples=[
            "check the intake progress for profile <id>",
            "what stage is the business scan at",
        ],
    ))
