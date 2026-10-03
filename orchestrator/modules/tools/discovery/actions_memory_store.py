"""platform_store_memory, registered by register_all_actions right after the
workspace actions."""

from .action_registry import ActionDefinition, ActionRegistry

_STORE_MEMORY_PARAMETERS = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": "The information to remember.",
        },
        "type": {
            "type": "string",
            "enum": [
                "tool_outcome", "task_learning", "playbook_pattern",
                "user_fact", "business_fact", "preference", "procedure",
                "decision", "open_loop", "thread_summary",
            ],
            "description": (
                "What kind of fact this is. Use 'decision' for choices made "
                "(with the why), 'open_loop' for unresolved follow-ups. "
                "Default when omitted: business_fact."
            ),
        },
        "importance": {
            "type": "number",
            "description": "0.0-1.0 how load-bearing this fact is (0.8+ = critical). Default 0.5.",
        },
        "scope": {
            "type": "string",
            "enum": ["private", "workspace"],
            "description": (
                "Sharing override. Default: private for user_fact/preference "
                "(visible only to the current user), workspace for everything else."
            ),
        },
        "pinned": {
            "type": "boolean",
            "description": "Pin this memory so recall always ranks it highly.",
        },
        "source_type": {
            "type": "string",
            "enum": ["platform_verified", "claude_reports", "current_status", "inference"],
            "description": (
                "Provenance: platform_verified (queried + confirmed via tools), "
                "claude_reports (the assistant's claim, unverified), current_status "
                "(transient state read from a live source), inference (pattern-based)."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 confidence in the claim. 1.0 = verified.",
        },
        "evidence_uri": {
            "type": "string",
            "description": "Optional pointer to the source — workspace file path, report id, run id, etc.",
        },
    },
    "required": ["content"],
}


def register_store_memory_action(registry: ActionRegistry) -> None:
    """Register platform_store_memory (first-class: promoted)."""
    registry.register(ActionDefinition(
        name="platform_store_memory",
        description=(
            "Store a curated fact in workspace long-term memory for future conversations. "
            "Use for: user facts, confirmed decisions, workspace patterns, user corrections. "
            "Do NOT use for: task artifacts, raw tool outputs, volatile data — and NEVER "
            "secrets, credentials, passwords, API keys, card or bank numbers (such content "
            "is refused by the exclusion policy). "
            "Keep under 200 chars. For searching stored memories, use platform_search_memory.\n\n"
            "Set `source_type` honestly so future readers can tell platform_verified facts "
            "from claude_reports / current_status / inference; when unsure it is 'inference'.\n\n"
            "Set `type` from the taxonomy (decision / open_loop / preference / "
            "user_fact / business_fact / procedure / ...). Sharing defaults split by type: "
            "user_fact and preference are private to the current user; everything else is "
            "workspace-shared. Override per memory with `scope`."
        ),
        category="memory",
        parameters=_STORE_MEMORY_PARAMETERS,
        permission_level="write",
        promoted=True,
        requires_confirmation=False,
        tags=["memory", "store", "write", "provenance"],
        examples=[
            "remember that our deploy day is Thursday",
            "remember we decided to ship the pilot without SSO (type=decision)",
            "store with source_type=platform_verified after running the check",
        ],
    ))
