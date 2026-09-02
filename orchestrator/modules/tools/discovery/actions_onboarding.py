"""Onboarding ActionDefinitions (PRD-222 W1S3) — the platform_update_onboarding tool.

The ONLY way the Auto-led onboarding spine advances. State never moves as a
prompt side effect (FR-4): Auto records every stage advance and every segment
answer through this auditable tool call. Schema truth from birth (US-011 rule):
both params are optional (each has a handler default), so ``required`` is ``[]``;
the at-least-one rule is stated in the description AND enforced by the handler.
"""

from .action_registry import ActionDefinition, ActionRegistry

# The onboarding spine's tool surface (PRD-222, live-test 2026-08-29): while a
# workspace is mid-onboarding these actions must SURVIVE semantic narrowing —
# the OnboardingSection instructs Auto to call them by name, but the top-K
# ranking keys on the user's latest text, and onboarding turns are exactly the
# ones with no tool-shaped text ("Yes please."). Ranked out ⇒ Auto narrates the
# action it cannot take and the flow dead-ends. tool_router folds this list
# into the dispatcher enum (the PRD-221 page-prior mechanism) while
# ``onboarding_state.is_onboarding_active`` holds. Every name is guarded
# against the registry by test_prd222_onboarding_tool_prior.
ONBOARDING_PRIOR_ACTIONS = [
    "platform_update_onboarding",      # the spine — every stage advance
    "platform_scan_business_site",     # teach: site scan
    "platform_search_packages",        # proposal: match a package (PRD-230)
    "platform_install_package",        # building: install the accepted package
    "platform_install_marketplace_agent",  # building: custom-design staffing
    "platform_submit_report",          # powerup: the onboarding summary Deliverable
]

# Stages Auto may advance TO. Deliberately excludes ``not_started`` (the initial
# state — you never advance back to it); mirrors services.onboarding_state.
_ADVANCE_TARGETS = [
    "questions",
    "teach",
    "proposal",
    "building",
    "boom",
    "powerup",
    "completed",
    "skipped",
]


def register_onboarding_actions(registry: ActionRegistry) -> None:
    """Register the onboarding-spine tool (PRD-222 W1S3)."""

    # PRD-222 W2·S2 (US-025): the plan field accepts only ASSIGNABLE tiers,
    # sourced from the config-driven tier contract (PLAN_TIERS). 'enterprise' is
    # coming-soon and never appears here (nor is it accepted by the handler).
    from services.plan_tiers import assignable_tiers

    _assignable_plans = sorted(assignable_tiers().keys())

    registry.register(ActionDefinition(
        name="platform_update_onboarding",
        description=(
            "Advance the workspace's Auto-led onboarding spine, record the user's "
            "segment answers, and/or set the plan the user accepted at the "
            "proposal. This is the ONLY way onboarding state moves — never assume a "
            "stage changed, call this after each step. Provide advance_to (the next "
            "stage), segment (business/goal/comfort/team_size), or plan (the "
            "accepted tier) — "
            "AT LEAST ONE is required; passing none returns a clear error. Stages "
            "move forward only: questions -> teach -> proposal -> building -> boom "
            "-> powerup -> completed; 'skipped' ends the flow from any stage. A "
            "backward or repeat transition returns an error, not a crash. Only "
            "assignable tiers (basic/pro/business) are accepted for plan — "
            "'enterprise' is coming soon and is rejected. Returns the updated "
            "{stage, trial}."
        ),
        category="onboarding",
        parameters={
            "type": "object",
            "properties": {
                "advance_to": {
                    "type": "string",
                    "enum": _ADVANCE_TARGETS,
                    "description": (
                        "The next onboarding stage to move to. Forward-only; "
                        "'skipped' ends the flow. Omit to only record segment answers."
                    ),
                },
                "segment": {
                    "type": "object",
                    "properties": {
                        "business": {
                            "type": "string",
                            "description": "What the user's business is.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "The first thing they want Auto to handle.",
                        },
                        "comfort": {
                            "type": "string",
                            "description": (
                                "AI comfort level — sets Auto's register, e.g. "
                                "'novice' .. 'very technical'."
                            ),
                        },
                        "team_size": {
                            "type": "integer",
                            "description": (
                                "How many people are on the user's team (seats), if "
                                "stated. Drives the plan recommendation — solo -> "
                                "Basic, a small team -> Pro, an org -> Business. Omit "
                                "if the user hasn't said."
                            ),
                        },
                    },
                    "description": (
                        "The onboarding answers as a JSON object — business, goal, "
                        "comfort, and optional team_size. Pass an object, never a "
                        "string. Any subset may be supplied; keys are merged into "
                        "onboarding state. Omit to only advance."
                    ),
                },
                "plan": {
                    "type": "string",
                    "enum": _assignable_plans,
                    "description": (
                        "The plan tier the user accepted at the proposal stage — one "
                        "of basic/pro/business. Writes plan + plan_limits (auditable, "
                        "FR-4). 'enterprise' is coming soon and is rejected. Omit "
                        "unless the user explicitly accepted a plan."
                    ),
                },
            },
            # All optional (each has a handler default); the at-least-one rule is
            # documented above and enforced by the handler — never in required[].
            "required": [],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["onboarding", "write", "auto"],
        examples=[
            "record that the user runs a barber shop and wants appointment booking",
            "advance onboarding to the proposal stage",
            "skip onboarding — the user wants to explore on their own",
            "mark onboarding completed",
        ],
    ))
