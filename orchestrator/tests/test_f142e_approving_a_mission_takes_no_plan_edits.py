"""F142 (e) — approving a mission advertises no plan edits it never applied.

platform_approve_mission offered ``modifications`` (task_overrides,
agent_overrides, notes), but the handler never read them and the approval never
applied them (api/missions.py, PRD-163 S4). Auto could send agent_overrides and
believe it had pinned the owner's staff when it had not (night 4). The parameter
is gone; plan edits go through platform_update_mission_plan.
"""
from __future__ import annotations


def test_approving_takes_only_the_mission():
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    approve = registry.get("platform_approve_mission")
    assert "modifications" not in approve.parameters["properties"]
    assert "task_edits" in registry.get("platform_update_mission_plan").parameters["properties"]
