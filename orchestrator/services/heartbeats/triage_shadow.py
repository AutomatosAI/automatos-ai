"""The decision engine's shadow read of a heartbeat (PRD-248 S5).

Moved out of :mod:`services.heartbeat_service` (PRD-268 US-C04a) so that file can
take the check-in bridge without growing. ``heartbeat_service`` re-exports it as
``_shadow_heartbeat_triage``, the name ``_dispatch_heartbeat_notification`` calls.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def shadow_heartbeat_triage(
    *,
    workspace_id: Any,
    link_id: Any,
    agent_id: Any,
    agent_name: Optional[str],
    title: str,
    message: str,
    status: str,
    source_type: str,
    platform_action: str,
) -> None:
    """PRD-248 S5 (shadow only): the decision engine says whether the owner
    should act on this heartbeat today and how severe it is, logged beside the
    route the platform takes. Lazy, off by default, fail-open."""
    try:
        from core.llm.decisions import MODE_OFF, get_decision_engine, judgements

        engine = get_decision_engine()
        if engine.dials().report_triage_mode == MODE_OFF:
            return
        engine.shadow(
            judgements.shadow_report_triage(
                engine,
                workspace_id=workspace_id,
                kind=f"heartbeat:{source_type}",
                subject_id=link_id,
                title=title,
                summary=message,
                status=status,
                agent_name=agent_name,
                agent_id=agent_id,
                report_type=None,
                platform_action=platform_action,
            ),
            purpose=judgements.PURPOSE_REPORT,
        )
    except Exception:  # noqa: BLE001 — never into a heartbeat
        logger.debug("[decision] heartbeat triage shadow skipped", exc_info=True)
