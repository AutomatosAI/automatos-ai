"""The Brand designer's proposal card and its save (PRD-255 US-014, FR-11).

``platform_propose_brand_kit`` files a card the owner approves (the proposal, what it
changes and the Brand Board drawn from it, not saved); ``platform_save_approved_brand_kit``
saves the proposal the owner approved, through ``platform_update_brand_kit``'s own handler.
Handlers: ``handlers_brand_proposals.py``. Both work only the caller's own ticket.
"""

import copy

from .action_registry import ActionDefinition, ActionRegistry
from .actions_brand_kit_update import _PARAMETERS as KIT_FIELDS

PROPOSE_DESCRIPTION = (
    "Propose a brand kit change to the owner on a card they approve: the kit fields to change "
    "(the ones platform_update_brand_kit takes) and one line on why. The proposal is checked as "
    "the kit checks it, the Brand Board is drawn from it into your ticket's folder WITHOUT saving "
    "it, and a question card goes on your ticket listing what changes, linking the board, with "
    "the options Approve and Revise. Nothing is saved. Any answer other than Approve is a "
    "revision: revise and propose again. Only on the ticket you are working. Every call puts a "
    "real card in front of the owner: there is no test mode, and a why that says the card is a "
    "probe or a test is refused (F365). A proposal the kit refuses comes back with its reasons and asks nothing."
)


def register_brand_proposal_actions(registry: ActionRegistry) -> None:
    """Register platform_propose_brand_kit and platform_save_approved_brand_kit."""
    registry.register(ActionDefinition(
        name="platform_propose_brand_kit",
        description=PROPOSE_DESCRIPTION,
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "brand_kit": {
                    "type": "object",
                    "description": "The kit fields to change, as platform_update_brand_kit takes them.",
                    "properties": copy.deepcopy(KIT_FIELDS["properties"]),
                },
                "why": {"type": "string", "description": "One line on why: what the logo told you and what this fixes."},
            },
            "required": ["brand_kit"],
        },
        permission_level="write",
        tags=["documents", "brand", "brand kit", "proposal", "approval", "brand board", "design"],
        examples=[
            "propose the new brand kit to the owner",
            "send the owner the palette proposal to approve",
            "ask the owner to approve a less orange accent",
        ],
    ))
    registry.register(ActionDefinition(
        name="platform_save_approved_brand_kit",
        description=(
            "Save the brand kit proposal the owner approved on your ticket's proposal card, exactly as "
            "they saw it, through platform_update_brand_kit's checks. It takes no fields. Refused while "
            "the owner has not answered, when they asked for a revision, when the kit changed after you "
            "proposed, and when it is already saved."
        ),
        category="documents",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="write",
        tags=["documents", "brand", "brand kit", "save", "approved", "proposal"],
        examples=[
            "the owner approved the brand kit proposal, save it",
            "apply the approved palette to the brand kit",
            "save the kit change the owner signed off",
        ],
    ))


__all__ = ["register_brand_proposal_actions"]
