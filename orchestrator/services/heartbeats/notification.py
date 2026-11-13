"""The pieces of ``HeartbeatService._dispatch_heartbeat_notification`` (PRD-128).

Moved out unchanged in what they do, so the dispatcher fits the 50-line rule once it learned
to send nothing for a Claude Code agent's tick that only queued its session (13 Nov 2026,
``services.heartbeats.session_ticks``): the message a heartbeat's findings make, who the
heartbeat is from (with its ``report_to``), and the workspace-level fan-out through
``NotificationDispatcher``. The dispatcher itself, its triage shadow and its ``report_to``
routes stay in ``services.heartbeat_service``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MESSAGE_MAX_CHARS = 2000
DETAIL_MAX_CHARS = 300
NO_FINDINGS = "No findings."
LLM_ANALYSIS = "llm_analysis"
AGENT_SOURCE = "agent"
EVENT_TYPE = "heartbeat_complete"
LINK_TYPE = "heartbeat"


@dataclass(frozen=True)
class HeartbeatSource:
    """Who a heartbeat is from: the agent (or the orchestrator), its title and heartbeat block."""

    agent_id: Optional[int]
    agent_name: Optional[str]
    title: str
    hb_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def report_to(self) -> str:
        return str(self.hb_config.get("report_to") or "").strip().lower()


def message_of(findings: Optional[Sequence[Dict[str, Any]]]) -> str:
    """The model's analysis when there is one, else every finding's detail, else "No findings."."""
    items = list(findings or [])
    analysis = next((f for f in items if f.get("check") == LLM_ANALYSIS), None)
    message = ((analysis or {}).get("detail") or "")[:MESSAGE_MAX_CHARS]
    if message:
        return message
    details = [(f.get("detail") or "")[:DETAIL_MAX_CHARS] for f in items if f.get("detail")]
    return "\n".join(details) if details else NO_FINDINGS


def source_of(result: Dict[str, Any]) -> HeartbeatSource:
    """The heartbeat's source, read on a session of its own (run it on a worker thread)."""
    if result.get("source_type", "orchestrator") != AGENT_SOURCE:
        return HeartbeatSource(agent_id=None, agent_name="Orchestrator", title="Orchestrator Heartbeat")
    try:
        agent_id: Optional[int] = int(result.get("source_id"))
    except (TypeError, ValueError):
        agent_id = None
    agent_name, hb_config = _agent_heartbeat(agent_id) if agent_id is not None else (None, {})
    return HeartbeatSource(agent_id=agent_id, agent_name=agent_name, title=f"{agent_name or 'Agent'} Heartbeat",
                           hb_config=hb_config)


def _agent_heartbeat(agent_id: int) -> Tuple[str, Dict[str, Any]]:
    from core.database.database import SessionLocal
    from core.models import Agent

    db = SessionLocal()
    try:
        agent = db.query(Agent).get(agent_id)
        name = agent.name if agent else f"agent-{agent_id}"
        hb_config = ((agent.configuration or {}).get("heartbeat", {}) or {}) if agent and agent.configuration else {}
        return name, hb_config
    finally:
        db.close()


async def dispatch_to_workspace(*, workspace_id: str, source: HeartbeatSource, message: str,
                                link_id: Optional[Any], status: str) -> None:
    """The workspace's own channels (in_app / telegram / slack / webhook / silent), on a session
    of its own so the in-app row commits apart from the caller's. Raises what the dispatch raised."""
    from core.database.database import SessionLocal
    from core.services.notification_dispatcher import NotificationDispatcher

    db = SessionLocal()
    try:
        dispatched = await NotificationDispatcher(db, workspace_id).dispatch(
            event_type=EVENT_TYPE, title=source.title, message=message, link_type=LINK_TYPE,
            link_id=str(link_id) if link_id is not None else None,
            agent_id=source.agent_id, agent_name=source.agent_name, status=status,
        )
        db.commit()
        logger.info("[Heartbeat] Dispatched heartbeat_complete ws=%s → %s", workspace_id, dispatched.get("dispatched_to"))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


__all__ = ["HeartbeatSource", "dispatch_to_workspace", "message_of", "source_of"]
