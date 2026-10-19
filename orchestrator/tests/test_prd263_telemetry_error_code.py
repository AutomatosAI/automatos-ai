"""PRD-263 US-018 — a failed tool result's own error_code is kept on its log row.

Tool results carry an expected failure's code as ``error_code`` (modules/tools/README.md,
"Result shape"); tool_router's refusal of a Composio call while integrations are off
writes ``integrations_unavailable``. At 697ca7d6e telemetry dropped it, so
tool_execution_logs.error_code was always NULL and HARNESS could tell a broken
connection from a bad tool only by the message. Now an error row keeps the code, cut to
the column's 100 characters; any other row, and a failure without a code, logs NULL.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

from modules.tools.execution import telemetry


def _logged(result, tool_name="composio_execute", parameters=None):
    db = MagicMock()
    asyncio.run(telemetry.write_telemetry(
        session_factory=lambda: db, tool_name=tool_name,
        parameters=parameters if parameters is not None else {"action": "GMAIL_SEND_EMAIL"},
        agent_id=3, workspace_id=uuid4(), result=result, execution_time_ms=7,
    ))
    return db.add.call_args[0][0]


def test_a_failed_result_with_an_error_code_logs_that_code():
    from modules.tools import tool_router

    result = tool_router._integrations_unavailable_result("composio_execute", "trace-1")
    assert result["error_code"] == tool_router.INTEGRATIONS_UNAVAILABLE == "integrations_unavailable"

    row = _logged(result)

    assert (row.status, row.error_code, row.app_name) == ("error", "integrations_unavailable", "GMAIL")
    assert row.error_message == result["error"]


def test_a_failed_result_without_an_error_code_logs_null():
    row = _logged({"success": False, "error": "channel_not_found"})
    assert (row.status, row.error_code) == ("error", None)


def test_a_long_code_is_cut_to_the_column():
    row = _logged({"success": False, "error": "boom", "error_code": "x" * 150})
    assert row.error_code == "x" * telemetry.ERROR_CODE_MAX_CHARS
    assert telemetry.ERROR_CODE_MAX_CHARS == 100


def test_only_an_error_row_carries_a_code():
    assert _logged({"success": True, "data": {}, "error_code": "stale"}).error_code is None
    asked = _logged({"success": False, "requires_confirmation": True, "error_code": "requires_confirmation"},
                    tool_name="platform_update_agent", parameters={"agent_id": 1})
    assert (asked.status, asked.error_code) == ("asked", None)
