"""#942: an agent's session tool groups, on the agent routes (no routes of its own).

* ``PUT /api/agents/{id}`` (and the create routes) refuse an unknown group id in
  ``configuration.session_tool_groups`` with a 422 (``reject_unknown_tool_groups``,
  called from ``api.agents._reject_invalid_runtime``).
* ``GET /api/agents/{id}`` carries ``session_tool_groups`` (``AgentDetailResponse``)
  and, beside the skill gaps, the workspace's: a connected database, a built
  Knowledge Graph … that the agent's groups leave out. ``?groups=data,graph``
  previews other groups for that one response (empty = core only), and the gaps
  follow it.
* ``GET /api/marketplace/items/{id}?type=agent`` carries, in ``dependencies``, what a
  CLI-session copy of the agent could not call (PRD-256 F430, ``with_session_tool_gaps``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from core.models import AgentResponse
from services.session_capability_gaps import workspace_gaps
from services.session_tool_groups import (
    DEFAULT_SESSION_TOOL_GROUPS,
    UnknownToolGroup,
    check_configuration,
    effective_groups,
    groups_payload,
    parse_groups,
    tools_for_groups,
)

logger = logging.getLogger(__name__)

UNPROCESSABLE = 422
SKILL_GAP_KIND = "skill"


class AgentDetailResponse(AgentResponse):
    """One agent, with the session tool groups the agent page switches."""

    session_tool_groups: Optional[Dict[str, Any]] = None


def reject_unknown_tool_groups(configuration: Any) -> None:
    """422 naming the unknown ids and the valid ones; absent passes."""
    try:
        check_configuration(configuration)
    except UnknownToolGroup as err:
        raise HTTPException(status_code=UNPROCESSABLE, detail=str(err)) from err


def requested_groups(raw: Optional[str]) -> Optional[List[str]]:
    """``?groups=``: ``None`` when absent, ``[]`` when empty (core only), else
    the ids in catalogue order. 422 on an unknown id."""
    if raw is None:
        return None
    try:
        return parse_groups(raw)
    except UnknownToolGroup as err:
        raise HTTPException(status_code=UNPROCESSABLE, detail=str(err)) from err


def skill_gaps(agent: Any, groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The tools the agent's skills name that its sessions would not have, given
    ``groups`` (else its own). The prompt's own computation, against the list."""
    from services.cli_session_prompt import session_tool_gaps

    enabled = effective_groups(agent) if groups is None else groups
    return session_tool_gaps(agent, tools_for_groups(enabled)) or []


async def agent_detail(
    base: AgentResponse, agent: Any, db: Any, preview: Optional[List[str]],
    skill_gap_list: Optional[List[Dict[str, Any]]],
) -> AgentDetailResponse:
    """``base`` with the groups payload, and the workspace's gaps after the skills'."""
    enabled = effective_groups(agent) if preview is None else preview
    gaps = skill_gap_list
    if gaps is not None:
        gaps = gaps + await workspace_gaps(db, agent.workspace_id, enabled)
    return AgentDetailResponse(**{
        **base.model_dump(),
        "session_tool_gaps": gaps,
        "session_tool_groups": groups_payload(agent, preview),
    })


def with_session_tool_gaps(detail: Any, agent: Any) -> Any:
    """F430: a marketplace agent's detail with ``dependencies.session_tool_gaps``: what its
    skills name that a CLI-session copy on the default groups could not call (an install
    is an API agent; a copy switched to a CLI session gets the default groups)."""
    try:
        gaps = [{**gap, "kind": SKILL_GAP_KIND} for gap in skill_gaps(agent, list(DEFAULT_SESSION_TOOL_GROUPS))]
    except Exception:  # noqa: BLE001 — logged; the detail renders without the gaps
        logger.exception("[F430] session tool gaps unavailable for marketplace agent %s", getattr(agent, "id", "?"))
        return detail
    return detail.model_copy(update={"dependencies": {**detail.dependencies, "session_tool_gaps": gaps}})
