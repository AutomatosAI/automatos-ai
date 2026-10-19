"""PRD-263 US-018 — a wait for the owner's click is logged as "asked" and is no routing failure.

An owner-only call from a person's chat does not run: it returns an ask
(``requires_confirmation: True``, ``success: False``, owner_only._ask). At 697ca7d6e the
tool router folded that ask into the routing graph as a failure signal (the GraphRouter
then penalises the tool through ``fails_for_intent``), and telemetry logged it as
``error``. Now the row's status is ``asked`` and the router sends no signal for it; a
refusal and a success are signalled as before. The recorder itself is unchanged.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from modules.tools.execution import telemetry

UPDATE_AGENT = "platform_update_agent"
PARAMS = {"agent_id": 12, "name": "Scout", "model": "claude-sonnet-5-5"}
GRANT_ID = 42


def _with_a_card(db, workspace_id, *, ask, **_kwargs):
    """tool_grants.attach_ask_grant with its grant issued."""
    return {**ask, "grant_id": GRANT_ID, "requires_approval": True}


def _owner_only_ask():
    """The ask owner_only raises for update_agent from a person's chat."""
    from modules.tools.discovery import owner_only

    chat = {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}
    with patch("modules.tools.execution.tool_grants.attach_ask_grant", new=_with_a_card):
        act = f"{owner_only.VERBS[UPDATE_AGENT]} 'Scout' (agent #12)"
        return owner_only._ask(None, "ws", UPDATE_AGENT, PARAMS, chat, act=act, what="'Scout' (agent #12)",
                               asked=f"{act}.")


class _Recorder:
    """The routing graph's batched recorder: keeps every signal it is given."""

    def __init__(self):
        self.signals = []

    def record(self, signal):
        self.signals.append(signal)


def _routed(result, recorder):
    """The tool router's envelope for a call that returned ``result``."""
    from modules.tools.tool_router import ToolRouter

    with patch("modules.tools.tool_router.execute_tool", new=AsyncMock(return_value=result)), \
            patch("modules.tools.tool_router.get_tool_signal_recorder", return_value=recorder):
        return asyncio.run(ToolRouter().execute_and_format(UPDATE_AGENT, dict(PARAMS), agent_id=1))


def _logged(result):
    """The tool_execution_logs row write_telemetry builds for ``result``."""
    db = MagicMock()
    asyncio.run(telemetry.write_telemetry(
        session_factory=lambda: db, tool_name=UPDATE_AGENT, parameters=dict(PARAMS), agent_id=1,
        workspace_id=uuid4(), result=result, execution_time_ms=5,
    ))
    assert db.add.call_count == 1 and db.commit.call_count == 1
    return db.add.call_args[0][0]


def test_an_owner_only_ask_is_logged_as_asked_not_error():
    ask = _owner_only_ask()
    assert ask["requires_confirmation"] is True and ask["success"] is False

    row = _logged(ask)

    assert row.status == telemetry.ASKED_STATUS == "asked"
    assert row.error_message is None and row.error_code is None


def test_a_refusal_and_a_success_keep_their_statuses():
    refused = _logged({"success": False, "error": "Agent 12 is not in this workspace"})
    assert (refused.status, refused.error_message) == ("error", "Agent 12 is not in this workspace")
    assert _logged({"success": True, "data": {"id": 12}}).status == "success"


def test_an_owner_only_ask_fires_no_routing_failure_signal():
    recorder = _Recorder()

    envelope = _routed(_owner_only_ask(), recorder)

    assert envelope["success"] is False and envelope["raw_result"]["requires_confirmation"] is True
    assert recorder.signals == []


def test_the_row_and_the_router_read_an_ask_inside_an_envelope_alike():
    """One reading of an ask (card_raised.is_waiting): an ask one envelope down (an
    executor's ``data``) is logged as asked AND sends no signal, so the two never split."""
    nested = {"success": False, "data": _owner_only_ask()}
    recorder = _Recorder()

    row = _logged(nested)
    _routed(nested, recorder)

    assert (row.status, row.error_message) == (telemetry.ASKED_STATUS, None)
    assert recorder.signals == []


def test_a_refused_call_and_a_done_call_are_still_signalled():
    refused, done = _Recorder(), _Recorder()

    _routed({"success": False, "error": "Agent 12 is not in this workspace"}, refused)
    _routed({"success": True, "data": {"id": 12}}, done)

    assert [(s.action_name, s.success) for s in refused.signals] == [(UPDATE_AGENT, False)]
    assert [(s.action_name, s.success) for s in done.signals] == [(UPDATE_AGENT, True)]


def test_a_signal_recorded_outside_a_router_call_goes_out_at_once():
    from modules.tools.tool_router import ToolRouter

    recorder = _Recorder()
    with patch("modules.tools.tool_router.get_tool_signal_recorder", return_value=recorder):
        ToolRouter._record_tool_signal(UPDATE_AGENT, False, 1, None, None, dict(PARAMS))

    assert [(s.action_name, s.success) for s in recorder.signals] == [(UPDATE_AGENT, False)]


def test_every_reader_leaves_out_the_literal_telemetry_writes():
    """WIRE PROTOCOL: the status and the sources are one literal each, everywhere."""
    from core.services import edge_builder_logs
    from services import harness_metrics, slo_metrics

    assert edge_builder_logs.ASKED_STATUS == slo_metrics.ASKED_STATUS == telemetry.ASKED_STATUS
    assert telemetry.ASKED_STATUS not in harness_metrics.TOOL_STATUSES
    assert set(edge_builder_logs.READ_SOURCES) == {"production", telemetry.SYNTHETIC_SIGNAL_SOURCE}
