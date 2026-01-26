"""
Capability Taxonomy
===================

Defines the standardized capability tags for action classification.
This taxonomy enables deterministic intent-to-action filtering at runtime.

Design Principles:
- Hierarchical: category.action format (e.g., message.send)
- Exhaustive: Covers 90%+ of common use cases
- Extensible: Easy to add new capabilities
- Separate: Capabilities are semantic affordances, NOT risk levels
"""

from typing import Dict, Set, List


# =============================================================================
# CAPABILITY TAXONOMY
# =============================================================================
# Format: "category.action": "Human-readable description"

CAPABILITY_TAXONOMY: Dict[str, str] = {
    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------
    "message.send": "Send messages, posts, or notifications",
    "message.read": "Read or list messages",
    "message.reply": "Reply to existing messages or threads",
    "message.edit": "Edit existing messages",
    "message.delete": "Delete messages",
    "message.react": "Add reactions to messages",
    "message.schedule": "Schedule messages for later",

    # -------------------------------------------------------------------------
    # Channels/Spaces/Rooms
    # -------------------------------------------------------------------------
    "channel.list": "List channels, rooms, or spaces",
    "channel.read": "Get channel information or details",
    "channel.join": "Join or leave channels",
    "channel.create": "Create new channels",
    "channel.archive": "Archive or close channels",
    "channel.manage": "Modify channel settings, topic, or description",
    "channel.invite": "Invite users to channels",
    "channel.delete": "Delete channels permanently",

    # -------------------------------------------------------------------------
    # Users/Members
    # -------------------------------------------------------------------------
    "user.lookup": "Find or get user information",
    "user.list": "List users or team members",
    "user.invite": "Invite users to workspace or team",
    "user.remove": "Remove users from workspace or team",
    "user.manage": "Modify user roles or permissions",
    "user.status": "Get or set user status/presence",

    # -------------------------------------------------------------------------
    # Files/Documents
    # -------------------------------------------------------------------------
    "file.read": "Read or download files",
    "file.list": "List files in a location",
    "file.write": "Create or upload files",
    "file.update": "Modify existing files",
    "file.delete": "Delete files",
    "file.share": "Share files with others",
    "file.move": "Move or rename files",

    # -------------------------------------------------------------------------
    # Data/Records (Generic CRUD)
    # -------------------------------------------------------------------------
    "data.query": "Search or filter records",
    "data.read": "Read specific records",
    "data.create": "Create new records",
    "data.update": "Modify existing records",
    "data.delete": "Delete records",
    "data.export": "Export data to external format",
    "data.import": "Import data from external source",

    # -------------------------------------------------------------------------
    # Code/Repositories (GitHub, GitLab, Bitbucket)
    # -------------------------------------------------------------------------
    "repo.read": "Read repository information",
    "repo.list": "List repositories",
    "repo.create": "Create new repositories",
    "repo.update": "Modify repository settings",
    "repo.delete": "Delete repositories",
    "repo.fork": "Fork repositories",

    "branch.read": "Read branch information",
    "branch.create": "Create branches",
    "branch.delete": "Delete branches",
    "branch.merge": "Merge branches",

    "commit.read": "Read commit information",
    "commit.list": "List commits",
    "commit.create": "Create commits",

    "issue.read": "Read issues or tickets",
    "issue.list": "List issues",
    "issue.create": "Create issues",
    "issue.update": "Update issues",
    "issue.close": "Close issues",
    "issue.delete": "Delete issues",
    "issue.comment": "Comment on issues",
    "issue.assign": "Assign issues to users",
    "issue.label": "Add or remove labels",

    "pr.read": "Read pull requests",
    "pr.list": "List pull requests",
    "pr.create": "Create pull requests",
    "pr.update": "Update pull requests",
    "pr.merge": "Merge pull requests",
    "pr.close": "Close pull requests",
    "pr.review": "Review or approve pull requests",
    "pr.comment": "Comment on pull requests",

    "release.read": "Read release information",
    "release.create": "Create releases",
    "release.update": "Update releases",
    "release.delete": "Delete releases",

    # -------------------------------------------------------------------------
    # Calendar/Events
    # -------------------------------------------------------------------------
    "event.read": "Read calendar events",
    "event.list": "List calendar events",
    "event.create": "Create calendar events",
    "event.update": "Modify calendar events",
    "event.delete": "Delete calendar events",
    "event.rsvp": "Respond to event invitations",

    # -------------------------------------------------------------------------
    # Email
    # -------------------------------------------------------------------------
    "email.read": "Read emails",
    "email.list": "List emails",
    "email.send": "Send emails",
    "email.reply": "Reply to emails",
    "email.forward": "Forward emails",
    "email.delete": "Delete emails",
    "email.label": "Label or categorize emails",
    "email.archive": "Archive emails",
    "email.draft": "Create email drafts",

    # -------------------------------------------------------------------------
    # Tasks/Projects
    # -------------------------------------------------------------------------
    "task.read": "Read task details",
    "task.list": "List tasks",
    "task.create": "Create tasks",
    "task.update": "Update tasks",
    "task.complete": "Mark tasks complete",
    "task.delete": "Delete tasks",
    "task.assign": "Assign tasks to users",

    "project.read": "Read project details",
    "project.list": "List projects",
    "project.create": "Create projects",
    "project.update": "Update projects",
    "project.delete": "Delete projects",

    # -------------------------------------------------------------------------
    # Notifications/Alerts
    # -------------------------------------------------------------------------
    "notification.send": "Send notifications or alerts",
    "notification.list": "List notifications",
    "notification.read": "Mark notifications as read",
    "notification.configure": "Configure notification settings",

    # -------------------------------------------------------------------------
    # Webhooks/Triggers
    # -------------------------------------------------------------------------
    "webhook.create": "Create webhooks",
    "webhook.list": "List webhooks",
    "webhook.update": "Update webhooks",
    "webhook.delete": "Delete webhooks",
    "webhook.trigger": "Manually trigger webhooks",

    # -------------------------------------------------------------------------
    # Auth/Admin (High Risk)
    # -------------------------------------------------------------------------
    "auth.token": "Generate or manage access tokens",
    "auth.revoke": "Revoke access or permissions",
    "auth.oauth": "Manage OAuth connections",

    "admin.settings": "Modify workspace or organization settings",
    "admin.billing": "Access or modify billing information",
    "admin.audit": "Access audit logs",
    "admin.users": "Manage user accounts administratively",

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------
    "search.messages": "Search messages",
    "search.files": "Search files",
    "search.users": "Search users",
    "search.general": "General search across resources",

    # -------------------------------------------------------------------------
    # Analytics/Reporting
    # -------------------------------------------------------------------------
    "analytics.read": "Read analytics data",
    "analytics.export": "Export analytics reports",

    # -------------------------------------------------------------------------
    # Spreadsheets (Google Sheets, Excel, Airtable)
    # -------------------------------------------------------------------------
    "sheet.read": "Read spreadsheet data",
    "sheet.write": "Write to spreadsheets",
    "sheet.create": "Create spreadsheets",
    "sheet.update": "Update spreadsheet structure",
    "sheet.delete": "Delete spreadsheets",

    # -------------------------------------------------------------------------
    # Database Operations
    # -------------------------------------------------------------------------
    "database.query": "Query database",
    "database.insert": "Insert records",
    "database.update": "Update records",
    "database.delete": "Delete records",
    "database.schema": "Modify database schema",
}


# =============================================================================
# RISK CLASSIFICATION
# =============================================================================
# These capabilities are inherently destructive or high-risk

DESTRUCTIVE_CAPABILITIES: Set[str] = {
    # Deletion operations
    "message.delete",
    "channel.delete",
    "channel.archive",
    "file.delete",
    "data.delete",
    "repo.delete",
    "branch.delete",
    "issue.delete",
    "release.delete",
    "event.delete",
    "email.delete",
    "task.delete",
    "project.delete",
    "webhook.delete",
    "sheet.delete",
    "database.delete",
    "database.schema",

    # Auth/Admin operations
    "auth.revoke",
    "admin.settings",
    "admin.billing",
    "admin.users",
    "user.remove",
    "user.manage",
}

REQUIRES_CONFIRMATION: Set[str] = {
    # All destructive capabilities require confirmation
    *DESTRUCTIVE_CAPABILITIES,

    # Plus these high-impact operations
    "repo.create",      # Creating repos is significant
    "pr.merge",         # Merging code changes
    "branch.merge",     # Merging branches
    "email.send",       # Sending emails externally
    "notification.send", # Sending notifications
    "auth.token",       # Creating tokens
    "webhook.create",   # Creating webhooks
}


# =============================================================================
# INTENT TO CAPABILITIES MAPPING
# =============================================================================
# Maps user intent keywords to capability tags for runtime filtering

INTENT_TO_CAPABILITIES: Dict[str, List[str]] = {
    # -------------------------------------------------------------------------
    # Sending/Creating content
    # -------------------------------------------------------------------------
    "send": ["message.send", "email.send", "notification.send"],
    "post": ["message.send", "message.reply"],
    "notify": ["message.send", "notification.send"],
    "tell": ["message.send"],
    "dm": ["message.send"],
    "write": ["message.send", "file.write", "sheet.write"],
    "compose": ["email.send", "email.draft", "message.send"],
    "draft": ["email.draft"],

    # -------------------------------------------------------------------------
    # Reading/Fetching content
    # -------------------------------------------------------------------------
    "read": ["message.read", "file.read", "email.read", "data.read"],
    "get": ["message.read", "file.read", "data.read", "data.query"],
    "fetch": ["message.read", "file.read", "data.read"],
    "show": ["message.read", "file.list", "data.query", "channel.list"],
    "list": ["message.read", "file.list", "channel.list", "user.list", "task.list"],
    "find": ["search.general", "data.query", "user.lookup"],
    "search": ["search.general", "search.messages", "search.files"],
    "lookup": ["user.lookup", "data.query"],

    # -------------------------------------------------------------------------
    # Creating resources
    # -------------------------------------------------------------------------
    "create": ["data.create", "file.write", "issue.create", "task.create", "channel.create"],
    "add": ["data.create", "issue.create", "task.create", "user.invite"],
    "new": ["data.create", "file.write", "issue.create", "channel.create"],
    "make": ["data.create", "channel.create", "task.create"],
    "open": ["issue.create", "pr.create"],
    "file": ["issue.create"],

    # -------------------------------------------------------------------------
    # Updating/Modifying
    # -------------------------------------------------------------------------
    "update": ["data.update", "issue.update", "task.update", "file.update"],
    "edit": ["data.update", "message.edit", "file.update"],
    "modify": ["data.update", "file.update", "channel.manage"],
    "change": ["data.update", "channel.manage"],
    "set": ["channel.manage", "user.status"],
    "rename": ["file.move", "channel.manage"],

    # -------------------------------------------------------------------------
    # Deleting/Removing
    # -------------------------------------------------------------------------
    "delete": ["data.delete", "file.delete", "message.delete"],
    "remove": ["data.delete", "file.delete", "user.remove"],
    "archive": ["channel.archive", "email.archive"],
    "close": ["issue.close", "pr.close", "channel.archive"],

    # -------------------------------------------------------------------------
    # Code/Development specific
    # -------------------------------------------------------------------------
    "commit": ["commit.create", "commit.read"],
    "push": ["commit.create", "branch.create"],
    "pull": ["pr.read", "pr.list"],
    "merge": ["pr.merge", "branch.merge"],
    "branch": ["branch.create", "branch.read", "branch.list"],
    "review": ["pr.review"],
    "approve": ["pr.review"],
    "comment": ["issue.comment", "pr.comment"],
    "assign": ["issue.assign", "task.assign"],
    "label": ["issue.label", "email.label"],

    # -------------------------------------------------------------------------
    # Communication specific
    # -------------------------------------------------------------------------
    "reply": ["message.reply", "email.reply"],
    "forward": ["email.forward"],
    "react": ["message.react"],
    "invite": ["user.invite", "channel.invite"],
    "join": ["channel.join"],
    "leave": ["channel.join"],

    # -------------------------------------------------------------------------
    # Calendar/Scheduling
    # -------------------------------------------------------------------------
    "schedule": ["event.create", "message.schedule"],
    "meeting": ["event.create", "event.read"],
    "calendar": ["event.list", "event.read"],
    "event": ["event.create", "event.read"],
    "rsvp": ["event.rsvp"],

    # -------------------------------------------------------------------------
    # Spreadsheet/Data
    # -------------------------------------------------------------------------
    "spreadsheet": ["sheet.read", "sheet.write"],
    "sheet": ["sheet.read", "sheet.write"],
    "cell": ["sheet.read", "sheet.write"],
    "row": ["sheet.read", "sheet.write", "data.create"],
    "column": ["sheet.read"],
    "export": ["data.export", "analytics.export"],
    "import": ["data.import"],
}


def get_capabilities_for_intent(intent: str) -> List[str]:
    """
    Extract capability tags from a user intent string.

    Args:
        intent: User's intent text (e.g., "send a message to #general")

    Returns:
        List of matching capability tags
    """
    intent_lower = intent.lower()
    matched_caps: Set[str] = set()

    for keyword, capabilities in INTENT_TO_CAPABILITIES.items():
        if keyword in intent_lower:
            matched_caps.update(capabilities)

    # If no matches, return general capabilities
    if not matched_caps:
        return ["data.query", "search.general"]

    return list(matched_caps)


def is_destructive_capability(capability: str) -> bool:
    """Check if a capability is considered destructive."""
    return capability in DESTRUCTIVE_CAPABILITIES


def requires_confirmation_capability(capability: str) -> bool:
    """Check if a capability requires user confirmation."""
    return capability in REQUIRES_CONFIRMATION


def get_risk_level(capabilities: List[str]) -> str:
    """
    Determine risk level based on capabilities.

    Returns:
        "dangerous" if any destructive capability
        "cautious" if any requires confirmation
        "safe" otherwise
    """
    for cap in capabilities:
        if cap in DESTRUCTIVE_CAPABILITIES:
            return "dangerous"

    for cap in capabilities:
        if cap in REQUIRES_CONFIRMATION:
            return "cautious"

    return "safe"
