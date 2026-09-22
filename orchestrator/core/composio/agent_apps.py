"""Which Composio apps an agent may use — one rule for what it can see and what
it can execute (F040).

An agent's app assignments decide: the ones switched on are its apps. An agent
with no assignment rows at all was never configured, and inherits every app its
workspace has connected (850+ tools; nobody assigns them one by one). An agent
whose assignments are all switched OFF was configured — to nothing — and
inherits nothing.

Before this, every site asked "does the agent have an ACTIVE assignment?" and
inherited when it did not, so switching an agent's last app off handed it every
app instead. Night 1: OPS had Gmail and Calendar switched off, and a ticket of
its sent mail through Gmail — the toggle that hid Gmail from the assistant did
not stop an agent using it. Six copies of the rule (the executor, the tool
registry, the tool router, the routing engine, the tool and hint services) now
ask this module.
"""
from __future__ import annotations

from typing import Any, Set, Tuple

from sqlalchemy.orm import Session


def assignment_state(db: Session, agent_id: Any) -> Tuple[Set[str], Set[str]]:
    """``(on, off)`` — the agent's app names, upper-cased, of every app type."""
    from core.models.composio_cache import AgentAppAssignment

    rows = (
        db.query(AgentAppAssignment.app_name, AgentAppAssignment.is_active)
        .filter(AgentAppAssignment.agent_id == agent_id)
        .all()
    )
    on: Set[str] = set()
    off: Set[str] = set()
    for name, active in rows:
        key = str(name or "").upper().strip()
        if key:
            (on if active else off).add(key)
    return on, off - on


def inherits_workspace_apps(db: Session, agent_id: Any) -> bool:
    """True only for an agent with no app assignments at all — switched on or off."""
    on, off = assignment_state(db, agent_id)
    return not on and not off


def switched_off(db: Session, agent_id: Any, app_name: Any) -> bool:
    """The operator switched this app off for this agent."""
    _, off = assignment_state(db, agent_id)
    return str(app_name or "").upper().strip() in off
