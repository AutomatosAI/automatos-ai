"""Onboarding ActionDefinitions (PRD-222 W1S3) — the platform_update_onboarding tool.

The ONLY way the Auto-led onboarding spine advances. State never moves as a
prompt side effect (FR-4): Auto records every stage advance and every segment
answer through this auditable tool call. Schema truth from birth (US-011 rule):
both params are optional (each has a handler default), so ``required`` is ``[]``;
the at-least-one rule is stated in the description AND enforced by the handler.
"""

from .action_registry import ActionDefinition, ActionRegistry

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

    registry.register(ActionDefinition(
        name="platform_update_onboarding",
        description=(
            "Advance the workspace's Auto-led onboarding spine and/or record the "
            "user's three segment answers. This is the ONLY way onboarding state "
            "moves — never assume a stage changed, call this after each step. "
            "Provide advance_to (the next stage) OR segment (business/goal/comfort) "
            "— AT LEAST ONE is required; passing neither returns a clear error. "
            "Stages move forward only: questions -> teach -> proposal -> building "
            "-> boom -> powerup -> completed; 'skipped' ends the flow from any "
            "stage. A backward or repeat transition returns an error, not a crash. "
            "Returns the updated {stage, trial}."
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
                    },
                    "description": (
                        "The three onboarding answers. Any subset may be supplied; "
                        "keys are merged into onboarding state. Omit to only advance."
                    ),
                },
            },
            # Both optional (each has a handler default); the at-least-one rule is
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
