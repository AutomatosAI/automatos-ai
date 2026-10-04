"""F266 (night 7b): Auto set the owner's timer in UTC and couldn't change it.

The owner, in Bristol, asked for "every day at 20:25". Auto called
platform_schedule_playbook(timezone="UTC"); the owner's other timers are on UK time.
Two follow-ups sent {frequency, time, requires_approval} and {playbook_id: "Weekly
Instagram posts", cron_schedule, wait_for_approval}, and both were refused.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

UK = "Europe/London"


@pytest.fixture
def roastery(db_session, seed_workspace):
    """The owner's playbooks: two on UK-time timers, and the one Auto is asked to time."""
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())

    def playbook(name, schedule=None):
        made = WorkflowTemplate(template_id=f"f266-{uuid4().hex[:8]}", name=name, workspace_id=ws,
                                description=name, template_definition={"steps": []}, steps=[], created_by="f266",
                                schedule_config=schedule)
        db_session.add(made)
        db_session.flush()
        return made

    for name in ("Monday Stock Report", "Tom's Monday Dispatch Checklist"):
        playbook(name, {"type": "cron", "cron_expression": "0 7 * * 1", "timezone": UK, "enabled": True})
    weekly = playbook("Weekly Instagram posts")
    return NS(db=db_session, ws=ws, weekly=weekly)


def _schedule(roastery, **params):
    from modules.tools.discovery.handlers_playbooks import schedule_playbook

    return asyncio.run(schedule_playbook(roastery.db, roastery.ws, {"cron_expression": "25 20 * * *", **params}))


def test_a_workspace_with_no_heartbeat_zone_takes_the_zone_its_timers_use(roastery):
    from services.playbook_scheduler import default_schedule_zone

    assert default_schedule_zone(roastery.db, roastery.ws) == UK


def test_a_utc_the_owner_never_said_is_the_workspaces_zone(roastery):
    out = _schedule(roastery, playbook_id=roastery.weekly.id, timezone="UTC")      # night 7b's call

    assert out["success"] is True and out["schedule_config"]["timezone"] == UK
    assert out["timezone_kept"] == UK and "which the owner never said" in out["message"]


def test_a_utc_the_owner_said_is_kept(roastery, monkeypatch):
    from core.llm.usage_context import LANE_CHAT, usage_scope

    monkeypatch.setattr("modules.tools.discovery.handlers_board_task_review.owner_words",
                        lambda db, ws, chat_id: ["Every day at 20:25 UTC please, it's for the US shop."])
    with usage_scope(request_type=LANE_CHAT, execution_id=f"chat:{uuid4()}"):
        out = _schedule(roastery, playbook_id=roastery.weekly.id, timezone="UTC")
    assert out["schedule_config"]["timezone"] == "UTC" and "timezone_kept" not in out


def test_the_second_follow_up_reaches_the_timer(roastery):
    """{playbook_id: name, cron_schedule, wait_for_approval}: night 7b's third try."""
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.unified_executor import (
        _fill_required_from_aliases, map_optional_aliases, undeclared_params_refusal,
    )

    action = get_action_registry().get("platform_schedule_playbook")
    sent = {"playbook_id": "Weekly Instagram posts", "cron_schedule": "25 20 * * *", "timezone": UK,
            "wait_for_approval": True}
    params = map_optional_aliases("platform_schedule_playbook", action,
                                  _fill_required_from_aliases(sent, ["cron_expression"]), "t")
    params = {k: v for k, v in params.items() if k != "cron_schedule"}
    assert undeclared_params_refusal("platform_schedule_playbook", action, params, "t") is None

    out = _schedule(roastery, **{k: v for k, v in params.items() if k != "cron_expression"})
    roastery.db.refresh(roastery.weekly)
    assert out["success"] is True and out["playbook_id"] == roastery.weekly.id
    assert roastery.weekly.schedule_config["timezone"] == UK
    assert roastery.weekly.execution_config["wait_for_me"] is True


def test_a_time_sent_as_words_is_told_the_cron_to_send():
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.unified_executor import REFUSED_CALL_IS_YOURS, undeclared_params_refusal

    action = get_action_registry().get("platform_schedule_playbook")
    refused = undeclared_params_refusal("platform_schedule_playbook", action, {
        "playbook_name": "Weekly Instagram posts", "frequency": "daily", "time": "20:25", "timezone": UK}, "t")
    assert "'25 20 * * *'" in refused and REFUSED_CALL_IS_YOURS in refused
