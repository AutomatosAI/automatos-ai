"""F271 (night 7b): Auto's update tool keeps the schedule rules the playbook routes keep.

The owner saved a timer as {type: 'none'} to turn it off; nothing changed and it fired
again. The routes refuse 'none' now, saying how to turn a timer off (#911). Auto's
platform_update_playbook stored any schedule it was given.
"""
from __future__ import annotations

import asyncio
from uuid import UUID, uuid4


def _playbook(db_session, seed_workspace):
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())
    playbook = WorkflowTemplate(template_id=f"f271-{uuid4().hex[:8]}", name="Weekly Instagram posts", workspace_id=ws,
                                description="Captions.", template_definition={"steps": []}, steps=[], created_by="f271",
                                schedule_config={"type": "cron", "cron_expression": "25 20 * * *",
                                                 "timezone": "Europe/London", "enabled": True})
    db_session.add(playbook)
    db_session.flush()
    return ws, playbook


def test_a_schedule_type_the_routes_refuse_is_refused_saying_how_to_turn_a_timer_off(db_session, seed_workspace):
    from modules.tools.discovery.handlers_playbooks import update_playbook

    ws, playbook = _playbook(db_session, seed_workspace)
    out = asyncio.run(update_playbook(db_session, ws, {"playbook_id": playbook.id, "schedule_config": {"type": "none"}}))

    assert out["success"] is False and "there is no 'none'" in out["error"] and '"enabled": false' in out["error"]
    db_session.refresh(playbook)
    assert playbook.schedule_config["type"] == "cron"                       # nothing was saved


def test_a_timer_turned_off_the_right_way_is_saved(db_session, seed_workspace):
    from modules.tools.discovery.handlers_playbooks import update_playbook

    ws, playbook = _playbook(db_session, seed_workspace)
    off = {**playbook.schedule_config, "enabled": False}
    out = asyncio.run(update_playbook(db_session, ws, {"playbook_id": playbook.id, "schedule_config": off}))

    assert out["success"] is True
    db_session.refresh(playbook)
    assert playbook.schedule_config["enabled"] is False
