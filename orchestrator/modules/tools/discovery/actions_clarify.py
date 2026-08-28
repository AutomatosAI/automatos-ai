"""PRD-229 — the executing-agent clarification tool: ``ask_orchestrator``.

A worker agent that hits ambiguity mid-run asks the orchestrator ONE question.
Auto answers it from the work's own retrievable context (upstream results, the
mission field, memory, the intake corpus, fleet state) with citations, or — when
it cannot, or the question is a governance decision — the ladder escalates it to
a human (US-003) and parks the task. The ladder is vertical (worker →
orchestrator → human); this tool never talks to another agent.

Exposed ONLY in the TASK_EXECUTION surface (see modules/context/modes.py
``EXECUTION_ONLY_TOOLS`` / ``excluded_tool_names``): a chat user is talking to
Auto directly, so the tool is stripped from the chat surface and catalog.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_clarify_actions(registry: ActionRegistry) -> None:
    """Register the mid-run clarification platform action."""

    registry.register(ActionDefinition(
        # Registered name carries the ``platform_`` prefix: the resolver routes
        # every action by namespace (``platform_*`` → PlatformActionExecutor,
        # ``workspace_*`` → worker) and a third prefix routes to neither dispatch
        # path (tests/test_tool_reachability.py namespace invariant). The Python
        # handler/import stays ``ask_orchestrator``; only the registered string,
        # the executor dispatch-key, the caller-binding guard and the
        # EXECUTION_ONLY_TOOLS entry carry the prefix. Sibling: platform_ask_human.
        name="platform_ask_orchestrator",
        description=(
            "Ask the orchestrator (Auto) a question when you hit genuine ambiguity "
            "mid-task and cannot decide well on your own. Auto answers from the "
            "work's own context (the mission plan, upstream results, memory, the "
            "knowledge corpus, the live floor) with sources, or — if it cannot, or "
            "the question is a governance decision (something destructive, spending "
            "real money, or a change of scope) — it escalates to a human and PARKS "
            "your task so you stop cleanly. Ask a single focused question; prefer "
            "deciding yourself when you reasonably can. Set 'category' when the "
            "question is a governance decision."
        ),
        category="governance",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The one clarification you need to proceed, as a short, "
                        "specific question — not a status report."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": ["destructive", "spend", "scope"],
                    "description": (
                        "Optional: mark the question as a governance decision so it "
                        "goes straight to a human — a destructive operation, spending "
                        "real money, or a change of scope."
                    ),
                },
            },
            "required": ["question"],
        },
        permission_level="write",
        tags=["clarification", "ask", "question", "orchestrator", "blocked"],
        examples=[
            "ask the orchestrator which data source is canonical for revenue",
            "I'm not sure which of two upstream outputs to use — ask Auto",
            "confirm whether deleting the old records is in scope before I proceed",
        ],
    ))
