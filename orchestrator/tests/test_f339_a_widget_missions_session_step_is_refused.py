"""F339: a widget-born mission's step for a session (CLI) agent is refused, through the real prepare step.

F155 refuses such a step in ``_task_io`` when the prepared task's ``origin`` says the
mission was started on a widget turn. ``_prepare_task``'s session-agent branch returned
its dict without ``origin``, so ``widget_born(None)`` was False and the step would have
run on a Claude Code session, which the widget key's restrictions cannot reach. The
F155 test built the prepared dict by hand, with ``origin``, and never saw the gap.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from services.coordinator_service import WIDGET_SESSION_REFUSAL, CoordinatorService

WIDGET_ORIGIN = {"origin_surface": "widget", "origin_scopes": ["chat", "documents:read"],
                 "origin_team": "franchise-a"}


def _prepared(run_config):
    service = CoordinatorService.__new__(CoordinatorService)
    service._field = None
    task = MagicMock(id=uuid4(), task_type="llm_generation", input_context=None, attachment_ids=None)
    run = MagicMock(id=uuid4(), workspace_id=uuid4(), goal="g", config={**run_config, "power_mode": "standard"})
    with patch("services.coordinator_service.MissionDispatcher") as dispatcher, \
            patch("services.coordinator_service._get_power_mode_caps", return_value={}), \
            patch("services.cli_ticket_lane.is_cli_agent", return_value=True), \
            patch("services.step_files.earlier_step_files", return_value=[]), \
            patch.dict("sys.modules", {"modules.agents.factory.agent_factory": MagicMock()}), \
            patch.object(service, "_collect_upstream_digest_rows", return_value=[]):
        dispatcher.build_task_prompt.return_value = "do the step"
        prepared = asyncio.run(service._prepare_task(MagicMock(), run, task, agent_id=7))
    return service, prepared


def test_a_session_steps_prepared_task_carries_the_missions_origin():
    _, prepared = _prepared(WIDGET_ORIGIN)
    assert prepared["cli_agent"] is True
    assert prepared["origin"]["origin_surface"] == "widget"


def test_a_widget_born_missions_session_step_is_refused_and_files_no_ticket():
    service, prepared = _prepared(WIDGET_ORIGIN)
    service._run_cli_ticket = AsyncMock(return_value={"status": "success"})
    assert asyncio.run(service._task_io(prepared)) == {"status": "error", "error": WIDGET_SESSION_REFUSAL}
    service._run_cli_ticket.assert_not_awaited()


def test_an_owners_missions_session_step_still_runs():
    service, prepared = _prepared({})
    service._run_cli_ticket = AsyncMock(return_value={"status": "success"})
    async def as_it_is(work, _run_id):
        return await work

    with patch("modules.coordination.mission_cancel.until_mission_cancelled", new=as_it_is):
        assert asyncio.run(service._task_io(prepared)) == {"status": "success"}
    service._run_cli_ticket.assert_awaited_once()
