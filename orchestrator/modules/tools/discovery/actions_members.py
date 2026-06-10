"""Workspace-member ActionDefinitions (PRD-143 S11 — administration surface).

Operator tier by design: Rev 2 opens member administration to Auto
deliberately, protected by gates-and-logs — role-set and remove are
``destructive`` with ``requires_confirmation=True`` (the executor's
destructive backstop), every handler is workspace-scoped, and invitations
go through the canonical flow extracted from the team router.
"""

from .action_registry import ActionDefinition, ActionRegistry

_ASSIGNABLE_ROLES = ["admin", "editor", "viewer"]


def register_members_actions(registry: ActionRegistry) -> None:
    """Register the workspace-member administration tools."""

    registry.register(ActionDefinition(
        name="platform_list_members",
        description=(
            "List the workspace's active members — email, name, role "
            "(owner/admin/editor/viewer), join date and member_id. Use before "
            "inviting, removing or changing a member's role, or when the user "
            "asks who is in the workspace."
        ),
        category="team",
        parameters={
            "type": "object",
            "properties": {},
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["team", "members", "roles", "workspace"],
        examples=[
            "who is in this workspace?",
            "list the team members",
            "show members and their roles",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_invite_member",
        description=(
            "Invite a new member to the workspace by email. Sends the "
            "invitation email (via Clerk) and respects the plan's member "
            "limit. Roles: admin, editor or viewer. Use when the user asks to "
            "add someone to the team."
        ),
        category="team",
        parameters={
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Email address to invite.",
                },
                "role": {
                    "type": "string",
                    "enum": _ASSIGNABLE_ROLES,
                    "description": "Workspace role for the invitee (default: viewer).",
                },
            },
            "required": ["email"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["team", "members", "invite", "onboarding"],
        examples=[
            "invite jane@acme.com to the workspace",
            "add a new editor to the team",
            "send an invite to my colleague",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_set_member_role",
        description=(
            "Change a workspace member's role to admin, editor or viewer. The "
            "owner's role can never be changed. Use platform_list_members "
            "first to get the member_id. Role changes alter what the member "
            "can do — treated as destructive and confirmed before running."
        ),
        category="team",
        parameters={
            "type": "object",
            "properties": {
                "member_id": {
                    "type": "integer",
                    "description": "The member_id from platform_list_members.",
                },
                "role": {
                    "type": "string",
                    "enum": _ASSIGNABLE_ROLES,
                    "description": "The new workspace role.",
                },
            },
            "required": ["member_id", "role"],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["team", "members", "roles", "permissions"],
        examples=[
            "make John an admin",
            "demote that member to viewer",
            "change Jane's role to editor",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_remove_member",
        description=(
            "Remove a member from the workspace (deactivates their "
            "membership). The owner can never be removed. Use "
            "platform_list_members first to get the member_id."
        ),
        category="team",
        parameters={
            "type": "object",
            "properties": {
                "member_id": {
                    "type": "integer",
                    "description": "The member_id from platform_list_members.",
                },
            },
            "required": ["member_id"],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["team", "members", "remove", "offboarding"],
        examples=[
            "remove John from the workspace",
            "take that contractor off the team",
        ],
    ))
