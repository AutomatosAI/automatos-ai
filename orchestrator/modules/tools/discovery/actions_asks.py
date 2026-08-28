"""PRD-225 — the agent-facing ask tool: ``platform_ask_human``.

An executing lane (board task, playbook run, tool call) that can only move with a
human decision raises a short markdown question. The tool PARKS the subject and
returns immediately — it never idle-waits for the answer (PRD-226's doctrine
tells agents to park-and-move-on; this is the mechanism). The answer arrives
later through the Questions tab or a Telegram reply and resumes the work.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_asks_actions(registry: ActionRegistry) -> None:
    """Register the human-question platform action."""

    registry.register(ActionDefinition(
        name="platform_ask_human",
        description=(
            "Ask a human a question when the work genuinely cannot proceed without a "
            "human decision, credential, or approval — then STOP and move on to other "
            "work. This PARKS the subject (the board task / playbook run / tool call you "
            "name) and returns immediately with an ask id; it does NOT wait for the "
            "answer. Keep the question a short markdown decision (a sentence or two, "
            "options if there are discrete choices) — never paste a report. Use it "
            "sparingly: prefer deciding yourself when you reasonably can."
        ),
        category="governance",
        parameters={
            "type": "object",
            "properties": {
                "subject_type": {
                    "type": "string",
                    "enum": ["board_task", "playbook_run", "tool_call"],
                    "description": "What the answer unblocks — the kind of work parked.",
                },
                "subject_id": {
                    "type": "string",
                    "description": (
                        "Id of the subject to park (the board task id, playbook run id, "
                        "or tool-call id). Must belong to your workspace."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "The ask, as short markdown. A decision, not a status report."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional discrete choices, rendered as buttons. Free text is "
                        "always allowed in addition."
                    ),
                },
                "expires_hours": {
                    "type": "integer",
                    "description": (
                        "Optional: hours until the ask lapses if unanswered. Omit for "
                        "the workspace default."
                    ),
                },
            },
            "required": ["subject_type", "subject_id", "question"],
        },
        permission_level="write",
        tags=["human", "question", "ask", "approval", "blocked", "clarification"],
        examples=[
            "ask the human which vendor to use before I place the order",
            "I need the API credentials to continue — ask the operator",
            "park this task and ask whether to ship variant A or B",
        ],
    ))
