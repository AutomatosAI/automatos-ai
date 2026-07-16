"""Watch ActionDefinitions -- PRD-204 S9 (create, list, get, cancel).

CONTRACT GUARD (onboarding-wall lesson, PRD memory): ``required[]`` lists
ONLY fields the handler cannot default. Every optional field here has a
handler-side default -- a required field that the handler defaults
dead-ends the LLM. test_prd204_watch_tools locks this.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_watch_actions(registry: ActionRegistry) -> None:
    """Register watch actions (PRD-204 S9)."""

    registry.register(ActionDefinition(
        name="platform_create_watch",
        description=(
            "Put a launched unit of work under supervision. A watch follows a "
            "mission or playbook execution to a verdict: it detects the terminal "
            "state, scores the output against the success criteria (0-1, "
            "displayed x10), takes bounded corrective action when the policy "
            "allows, and notifies the user. Use after launching work the user "
            "cares about, or when the user says 'keep an eye on it', 'make sure "
            "it gets done', 'check the result'."
        ),
        category="watches",
        parameters={
            "type": "object",
            "properties": {
                "target_type": {
                    "type": "string",
                    "enum": ["mission", "playbook_execution", "scheduled_playbook"],
                    "description": (
                        "What kind of thing to watch: a mission (run UUID), a "
                        "playbook execution (execution_id like 'exec-...'), or a "
                        "scheduled playbook (playbook id -- watches the schedule "
                        "for missed/benched runs)."
                    ),
                },
                "target_id": {
                    "type": "string",
                    "description": (
                        "The id of the target: mission UUID, execution_id, or "
                        "playbook id."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Short human label (default: derived from the target).",
                },
                "success_criteria": {
                    "type": "string",
                    "description": (
                        "What 'good' means for this work, in the user's words -- "
                        "the run output is scored against this (default: derived "
                        "from the target's goal/name)."
                    ),
                },
                "quality_threshold": {
                    "type": "number",
                    "description": (
                        "Pass bar on the 0-1 internal scale (default 0.8, "
                        "displayed as 8/10)."
                    ),
                },
                "policy": {
                    "type": "string",
                    "enum": [
                        "run_and_report",
                        "score_and_improve",
                        "watch_change",
                        "persistent",
                    ],
                    "description": (
                        "Decision profile (default run_and_report). "
                        "run_and_report: score + notify + close, no actions. "
                        "score_and_improve: below-threshold triggers one "
                        "diagnose+tweak+rerun cycle. watch_change: compare a "
                        "rerun against the prior run and report the delta. "
                        "persistent: recurring supervision of a scheduled "
                        "playbook until the deadline."
                    ),
                },
                "deadline_hours": {
                    "type": "number",
                    "description": (
                        "Give up after this many hours without a verdict "
                        "(default: no deadline)."
                    ),
                },
                "action_budget": {
                    "type": "integer",
                    "description": (
                        "Max corrective actions before escalating to a human "
                        "(default 2)."
                    ),
                },
            },
            "required": ["target_type", "target_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["watches", "write", "supervision", "quality", "follow-up"],
        examples=[
            "keep an eye on that mission and tell me how it went",
            "watch this playbook run and make sure the output is good",
            "make sure the nightly report actually runs",
            "supervise the launch and improve it if the result is weak",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_watches",
        description=(
            "List watches in this workspace -- id, title, status "
            "(watching/acting/awaiting_approval/needs_attention/passed/failed/"
            "escalated/expired/cancelled), score, target. Live watches by "
            "default. For one watch's full timeline use platform_get_watch."
        ),
        category="watches",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by exact status (omit for all live).",
                },
                "watch_type": {
                    "type": "string",
                    "enum": ["mission", "playbook_execution", "scheduled_playbook"],
                    "description": "Filter by watch type.",
                },
                "include_closed": {
                    "type": "boolean",
                    "description": "Include closed watches (default false).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20, max 50).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["watches", "read", "list", "status"],
        examples=[
            "what are you watching right now?",
            "list my watches",
            "any watches that need attention?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_watch",
        description=(
            "Full detail of ONE watch: status, score/verdict, action budget, "
            "target lineage (reruns/replans it followed), and the recent event "
            "timeline."
        ),
        category="watches",
        parameters={
            "type": "object",
            "properties": {
                "watch_id": {
                    "type": "string",
                    "description": "The watch UUID.",
                },
            },
            "required": ["watch_id"],
        },
        permission_level="read",
        tags=["watches", "read", "details", "verdict"],
        examples=[
            "how is that watch doing?",
            "show me the watch verdict",
            "what happened on the report watch?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_cancel_watch",
        description=(
            "Cancel a live watch (stop supervising -- the watched work itself "
            "keeps running). Closed watches cannot be cancelled."
        ),
        category="watches",
        parameters={
            "type": "object",
            "properties": {
                "watch_id": {
                    "type": "string",
                    "description": "The watch UUID to cancel.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why (recorded on the watch timeline).",
                },
            },
            "required": ["watch_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["watches", "write", "cancel"],
        examples=[
            "stop watching that mission",
            "cancel the watch on the nightly report",
        ],
    ))
