"""What a session is told: the session rules, the system prompt and the ticket file.

Moved out of ``session.py`` (PRD-253 Wave P), which supervises the process; this
module only builds text. The system prompt is stable per agent — no ids, dates or
counters (the prompt-cache invariant) — so anything that changes from one claim to
the next, such as a Plan turn, goes in the ticket file instead.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .allowlist import session_deliverables_dir

# Stable per agent — no ids, dates or counters (the prompt-cache invariant).
# PRD-245 S0.6: the session is told what it can and cannot reach, and how to ask.
SESSION_RULES = (
    "The ticket you are working is described in the file named in your first message; "
    "read it fully before acting.\n"
    "Rules of the session: work only inside the directory you were started in; "
    "never push, publish or open pull requests — the manager integrates your work; "
    "keep changes scoped to the ticket's OBJECTIVE and BOUNDARIES; when you are done, "
    "reply with a concise summary of what changed, what you verified, and anything left open.\n"
    "Tools in this session: file tools work only inside the working folder and the ticket "
    "folder; Bash runs an allowlist of read, build and test verbs, and anything else may be held "
    "for the operator. The Automatos tools you have are listed earlier in this prompt, under "
    "\"Tools in this session\" — that list is the truth, and it is the only place to read it. "
    "A platform tool your skills name that is NOT on that list does not exist here: do not call "
    "it and do not wait for it.\n"
    "To ask a question: use the ask_human tool if you have it — your ticket parks when your turn "
    "ends and picks up again with the answer. Without it, state the question in your final "
    "message and end the turn. Never wait for an answer inside the session.\n"
)

# PRD-253 Wave P — Plan on a CLI with no plan mode of its own: the session explores
# read-only and its FINAL MESSAGE is the plan. The turn's end takes it to the
# operator as the Plan card; approving it resumes this same session to do the work.
PLAN_TURN_SECTION = (
    "\n## Plan mode\n"
    "This turn is for planning only. Explore read-only — read files, search, run read-only "
    "commands — and make no changes: no edits, no new files, no commits. Edits are refused "
    "until the operator approves your plan.\n"
    "End your turn with the plan as your final message: what you will change, in which files, "
    "in what order, and how you will verify it. The operator reads it and approves it, or sends "
    "you feedback; either way this same session continues afterwards.\n"
)


def build_system_prompt(ticket: Dict[str, Any], cli_label: str = "Claude Code") -> str:
    """Stable per agent: no ids, no dates, no counters (prompt-cache invariant).

    PRD-239 S1: the backend renders the agent's soul — description, persona and
    skills — as ``system_prompt`` on the ticket (stable per agent); it sits
    between the introduction and the session rules. Without it the prompt is
    exactly the name and the rules, as before.
    """
    name = ticket.get("agent_name") or "an Automatos agent"
    intro = f"You are {name}, working as a supervised {cli_label} session managed by Automatos.\n"
    soul = ticket.get("system_prompt")
    soul = soul.strip() if isinstance(soul, str) else ""
    if soul:
        return intro + "\n" + soul + "\n\n" + SESSION_RULES
    return intro + SESSION_RULES


def build_ticket_file(ticket: Dict[str, Any], default_root: Optional[str] = None, plan_turn: bool = False) -> str:
    """The dispatch contract. With the host's default root known, the ticket names
    its own deliverables folder (PRD-245 S0.7); without one there is no such line.
    A Plan turn (PRD-253 Wave P) says so at the end."""
    folder = session_deliverables_dir(default_root, str(ticket.get("task_id")))
    deliverables = f"\nDeliverables: save any file you produce under {folder}/\n" if folder else ""
    return (
        f"# Ticket #{ticket.get('task_id')} — {ticket.get('title') or ''}\n\n"
        f"{ticket.get('prompt') or ''}\n"
        f"{deliverables}"
        f"{PLAN_TURN_SECTION if plan_turn else ''}"
    )


__all__ = ["PLAN_TURN_SECTION", "SESSION_RULES", "build_system_prompt", "build_ticket_file"]
