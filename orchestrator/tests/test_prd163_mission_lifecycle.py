"""PRD-163 S1 — mission lifecycle tools: reachability + attribution helper.

Pure layer (CI runs the integration drive separately). Confirms the 6 lifecycle
tools are registered, write-tier, and wired into the executor, and that the Q56
attribution helper prefers the chatting user over the agent.
"""

from __future__ import annotations

import os
import sys
import types

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402

from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.actions_missions import register_mission_actions  # noqa: E402

_LIFECYCLE = [
    "platform_approve_mission",
    "platform_reject_mission",
    "platform_pause_mission",
    "platform_resume_mission",
    "platform_cancel_mission",
    "platform_replan_mission",
]


def _registry() -> ActionRegistry:
    reg = ActionRegistry()
    register_mission_actions(reg)
    return reg


class TestReachability:
    @pytest.mark.parametrize("name", _LIFECYCLE)
    def test_registered_write_tier(self, name):
        action = _registry().get(name)
        assert action is not None, f"{name} not registered"
        assert action.permission_level == "write"
        assert action.workspace_scoped is True
        # mission_id is always required
        assert "mission_id" in action.parameters.get("required", [])

    @pytest.mark.parametrize("name", _LIFECYCLE)
    def test_handler_wired_in_executor(self, name):
        from modules.tools.discovery.platform_executor import PlatformActionExecutor

        ex = PlatformActionExecutor(db=None, workspace_id=None)
        assert name in ex._handlers
        assert callable(ex._handlers[name])


class TestQ56Attribution:
    def test_actor_prefers_chatting_user(self):
        from modules.tools.discovery.handlers_missions import _actor

        assert _actor({"_created_by": "user_abc", "_agent_id": 7}) == "user_abc"
        assert _actor({"_agent_id": 7}) == "7"
        assert _actor({}) == "agent"


class TestValidationNoDB:
    @pytest.mark.asyncio
    async def test_lifecycle_requires_mission_id(self):
        from modules.tools.discovery.handlers_missions import (
            approve_mission, pause_mission, cancel_mission,
        )

        for handler in (approve_mission, pause_mission, cancel_mission):
            res = await handler(db=None, workspace_id="ws", params={})
            assert res["success"] is False and "mission_id" in res["error"]

    @pytest.mark.asyncio
    async def test_invalid_mission_id_fails_cleanly(self):
        from modules.tools.discovery.handlers_missions import resume_mission

        res = await resume_mission(db=None, workspace_id="ws", params={"mission_id": "not-a-uuid"})
        assert res["success"] is False
