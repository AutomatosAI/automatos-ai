"""F323 (night 9b): every agent run, and Auto, knows today's date in the owner's zone.

On 4 October 2026 the Business Analyst read "this summer" as 2024 (#1984 run 1: "no
data available for June–August 2024"), and Auto searched the web for "current date"
(chat 32bb645f). The card, mission-step and playbook-step prompts carried only
"Current UTC time: …Z", in the section the budget drops first; Auto's short chat
path carried no date at all.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

# 14:23 UTC on Sunday 4 October 2026 is 15:00 in London, to the hour.
NIGHT_9B = datetime(2026, 10, 4, 14, 23, tzinfo=timezone.utc)
IN_LONDON = "Today is Sunday 4 October 2026, 15:00 in Europe/London."


@pytest.fixture
def roastery(db_session, seed_workspace):
    ws = UUID(seed_workspace())

    def zone(name):
        db_session.execute(text("UPDATE workspaces SET settings = CAST(:s AS json) WHERE id = CAST(:w AS uuid)"),
                           {"s": '{"orchestrator": {"heartbeat": {"timezone": "%s"}}}' % name, "w": str(ws)})
        db_session.expire_all()
    return NS(db=db_session, ws=ws, zone=zone)


def test_the_date_is_said_in_words_in_the_workspaces_zone(roastery):
    from services.todays_date import today_line

    roastery.zone("Europe/London")
    assert today_line(roastery.db, roastery.ws, NIGHT_9B) == IN_LONDON


def test_a_workspace_with_no_zone_or_an_unknown_one_gets_utc(roastery):
    from services.todays_date import today_line

    assert today_line(roastery.db, roastery.ws, NIGHT_9B) == "Today is Sunday 4 October 2026, 14:00 in UTC."
    roastery.zone("Bristol")
    assert today_line(roastery.db, roastery.ws, NIGHT_9B).endswith("14:00 in UTC.")
    assert today_line(None, None, NIGHT_9B).endswith("14:00 in UTC.")


def test_every_agent_runs_prompt_leads_its_time_with_todays_date(roastery):
    """Cards and mission steps (task execution) and playbook steps (recipe) all render it."""
    from modules.context.sections.base import SectionContext
    from modules.context.sections.datetime_context import DatetimeContextSection
    from services.todays_date import today_line

    roastery.zone("Europe/London")
    ctx = SectionContext(agent=None, workspace_id=str(roastery.ws), db_session=roastery.db,
                         context_mode="task_execution")
    line = asyncio.run(DatetimeContextSection().render(ctx))

    assert line.startswith(today_line(roastery.db, roastery.ws)[:-len(" in Europe/London.")])
    assert "in Europe/London. Current UTC time: " in line


def test_the_budget_never_drops_the_date():
    """It was the first section dropped when a prompt ran over its budget."""
    from modules.context.budget import RenderedSection, TokenBudget, TokenBudgetManager
    from modules.context.sections.datetime_context import DatetimeContextSection

    section = DatetimeContextSection()
    today = RenderedSection(name=section.name, priority=section.priority, content=IN_LONDON, token_estimate=20)
    catalog = RenderedSection(name="platform_actions", priority=5, content="x " * 900, token_estimate=900)
    kept, dropped = TokenBudgetManager().allocate([catalog, today], TokenBudget(total=500, reserved_for_response=0,
                                                                                  reserved_for_messages=0))

    assert [s.name for s in kept] == [section.name] and dropped == ["platform_actions"]


def test_autos_short_chat_path_says_todays_date():
    from consumers.chatbot.atom_prompt import atom_system_prompt

    prompt = atom_system_prompt(NS(name="Auto", description="", persona=None), identity=" You're talking to Sam.",
                                memory_block="", facts="")
    today = datetime.now(timezone.utc)
    assert f"Today is {today:%A} {today.day} {today:%B %Y}" in prompt
    assert " in UTC. You're talking to Sam. Read the conversation" in prompt
