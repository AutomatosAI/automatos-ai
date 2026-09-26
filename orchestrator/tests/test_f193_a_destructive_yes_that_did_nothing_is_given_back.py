"""F193 (b) — a destructive yes that did nothing is given back.

A destructive approval is single-use: the gate spends it before the call runs.
When the call then did nothing — the handler failed softly ("not found"), or a
check after the gate (the hierarchy, the rate limit) refused it — the yes was
spent anyway and the owner was asked again for what was never done. It is now
given back, the same rule as publishing (F179); a call that ran still spends it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

LANE = {"mission_id": "m-1"}


@pytest.fixture
def cafe(db_session, seed_workspace):
    return NS(db=db_session, ws=UUID(seed_workspace()))


def _approved(cafe, action, params):
    from core.services.approval_grants import grant_grant
    from modules.tools.execution.tool_grants import issue_tool_grant

    grant = issue_tool_grant(cafe.db, cafe.ws, action=action, params=params, permission_level="destructive",
                             description="Delete it", caller_context=dict(LANE))
    grant_grant(grant, granted_by="user:1")
    cafe.db.flush()
    return grant


def _run(cafe, action, params, handler):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: False
    executor._handlers[action] = handler
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, dict(params), dict(LANE)))


def test_a_yes_whose_handler_did_nothing_is_given_back_and_still_covers_one_run(cafe):
    params = {"memory_id": "mem-9"}
    grant = _approved(cafe, "platform_delete_memory", params)

    failed = _run(cafe, "platform_delete_memory", params, AsyncMock(return_value={"success": False, "error": "Memory not found"}))
    assert failed["success"] is False and grant.status == "granted"          # night: 'revoked'

    ran = _run(cafe, "platform_delete_memory", params, AsyncMock(return_value={"success": True}))
    assert ran["approved_via_grant_id"] == grant.id and grant.status == "revoked"


def test_a_yes_refused_after_the_gate_is_given_back(cafe):
    """platform_delete_playbook is hierarchy-checked after the gate; a lane with no actor is refused."""
    params = {"playbook_id": 103}
    grant = _approved(cafe, "platform_delete_playbook", params)
    handler = AsyncMock(return_value={"success": True})

    refused = _run(cafe, "platform_delete_playbook", params, handler)

    assert refused.get("permission_denied") is True and handler.await_count == 0
    assert grant.status == "granted"
