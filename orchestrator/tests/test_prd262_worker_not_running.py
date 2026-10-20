"""PRD-262 US-040: without the workspace worker, a clear "not running" error, never a 500.

The core stack (``docker-compose.core.yml``, ``make up CORE=1``) runs five containers and
leaves the workspace worker out. Every call the backend makes to it then fails to
connect: compose's DNS has no ``workspace-worker`` to resolve, or nothing listens. This
suite drives the real ``WorkspaceClient`` against a transport that refuses every
connection, and against a local port nothing listens on, and checks what reaches:

* the client: ``Workspace worker not running or unreachable (<action>)``, status 503, the
  transport's own error kept for the log only; the local edition adds how to start it. A
  connected call that ran out of time says it timed out instead;
* the workspace endpoints (``api/workspace_files.py``): a 503 with that message, including
  the canvas commit routes, which used to read the failed call as "no changes" or "no
  session";
* the agent workspace tools (``exec_workspace.execute_workspace_action``): a failed result
  carrying the same message, never an exception.

No database, no Docker: the routes are called as functions, as test_f333 does.
"""
from __future__ import annotations

import asyncio
import logging
import socket
from types import SimpleNamespace as NS
from uuid import UUID

import httpx
import pytest
from fastapi import HTTPException

from api import workspace_files
from core import workspace_client as wc
from modules.tools.execution import exec_workspace

WS = "00000000-0000-0000-0000-0000000000c1"
DNS_FAILURE = "[Errno -2] Name or service not known"


def _refused(request: httpx.Request) -> httpx.Response:
    """What httpx raises when compose's DNS has no workspace-worker: a connect error."""
    raise httpx.ConnectError(DNS_FAILURE, request=request)


@pytest.fixture
def no_worker(monkeypatch):
    """The backend's shared worker client, on a transport that never connects."""
    monkeypatch.setattr(wc.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(wc.config, "WORKER_INTERNAL_URL", "http://workspace-worker:8081")
    monkeypatch.setattr(wc.config, "WORKER_INTERNAL_TOKEN", "")
    monkeypatch.setattr(wc, "_client", httpx.AsyncClient(transport=httpx.MockTransport(_refused)))


def _run(work):
    async def run():
        try:
            return await work()
        finally:
            await wc._get_client().aclose()
    return asyncio.run(run())


def _assert_not_running(message: str, action: str) -> None:
    assert message.startswith(f"{wc.WORKER_NOT_RUNNING} ({action})"), message
    assert DNS_FAILURE not in message, "the transport's error goes to the log, not to the caller"


# --------------------------------------------------------------------------- the client


@pytest.mark.parametrize("action, call", [
    ("list_dir", lambda client: client.list_dir(".")),
    ("read_file", lambda client: client.read_file("reports/q3.md")),
    ("write_file", lambda client: client.write_file("reports/q3.md", "# Q3")),
    ("exec_command", lambda client: client.exec_command("ls")),
    ("canvas_session_start", lambda client: client.canvas_session_start()),
    ("git", lambda client: client.git("status")),
])
def test_every_call_says_the_worker_is_not_running(no_worker, action, call, caplog):
    with caplog.at_level(logging.ERROR, logger=wc.logger.name):
        result = _run(lambda: call(wc.WorkspaceClient(WS)))

    assert result["success"] is False and result["status_code"] == wc.WORKER_UNAVAILABLE == 503
    _assert_not_running(result["error"], action)
    assert DNS_FAILURE in caplog.text, "the detail is logged"


def test_the_local_edition_says_how_to_start_it_and_the_hosted_one_does_not(no_worker, monkeypatch):
    local = _run(lambda: wc.WorkspaceClient(WS).list_dir("."))
    assert local["error"].endswith(wc.WORKER_START_HINT)
    assert "docker compose up -d workspace-worker" in local["error"] and "CORE=1" in local["error"]

    monkeypatch.setattr(wc.config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(wc, "_client", httpx.AsyncClient(transport=httpx.MockTransport(_refused)))
    hosted = _run(lambda: wc.WorkspaceClient(WS).list_dir("."))
    assert hosted["error"] == f"{wc.WORKER_NOT_RUNNING} (list_dir)"


def test_a_port_nothing_listens_on_is_not_running(monkeypatch):
    """A real connection to a closed local port, as when the worker's container is gone."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(wc.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(wc.config, "WORKER_INTERNAL_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(wc.config, "WORKER_INTERNAL_TOKEN", "")
    monkeypatch.setattr(wc, "_client", None)

    result = _run(lambda: wc.WorkspaceClient(WS).list_dir("."))

    assert result["status_code"] == 503
    _assert_not_running(result["error"], "list_dir")


def test_no_connection_in_time_is_not_running_but_a_slow_answer_timed_out():
    request = httpx.Request("GET", "http://workspace-worker:8081/health")
    unreached = wc._connection_error("list_dir", httpx.ConnectTimeout("timed out", request=request))
    _assert_not_running(unreached["error"], "list_dir")

    slow = wc._connection_error("exec_command", httpx.ReadTimeout("timed out", request=request))
    assert slow == {"success": False, "error": "Workspace worker timed out (exec_command)", "status_code": 503}


# --------------------------------------------------------------------------- the endpoints


def _route(call):
    with pytest.raises(HTTPException) as raised:
        _run(call)
    return raised.value


CTX = NS(workspace_id=WS)


@pytest.mark.parametrize("action, call", [
    ("list_dir", lambda: workspace_files.list_files(workspace_id=WS, path=".", ctx=CTX)),
    ("read_file", lambda: workspace_files.get_file_content(workspace_id=WS, path="reports/q3.md", ctx=CTX)),
    ("download_file", lambda: workspace_files.get_file_raw(workspace_id=WS, path="reports/q3.pdf", ctx=CTX)),
    ("canvas_session_start", lambda: workspace_files.start_canvas_session(workspace_id=WS, ctx=CTX)),
    ("canvas_session_status", lambda: workspace_files.get_canvas_session_status(workspace_id=WS, ctx=CTX)),
])
def test_the_workspace_endpoints_answer_503_with_the_message(no_worker, action, call):
    error = _route(call)

    assert error.status_code == 503, "a 503 the UI can show, never a 500"
    _assert_not_running(error.detail, action)


def test_the_commit_preview_no_longer_reads_a_missing_worker_as_no_changes(no_worker):
    error = _route(lambda: workspace_files.preview_canvas_commit(workspace_id=WS, cwd="repos/app", ctx=CTX))

    assert error.status_code == 503
    _assert_not_running(error.detail, "canvas_session_status")


def test_a_commit_no_longer_reads_a_missing_worker_as_no_session(no_worker):
    body = workspace_files.CanvasCommitRequest(message="Edit the readme", cwd="repos/app")
    error = _route(lambda: workspace_files.commit_canvas_session(workspace_id=WS, body=body, ctx=CTX))

    assert error.status_code == 503, "409 'No canvas session to commit' was the wrong reason"
    _assert_not_running(error.detail, "canvas_session_status")


def test_a_running_worker_without_a_session_still_previews():
    """The worker's own 404 (no session) is not the missing worker: the preview goes on."""
    workspace_files._raise_if_worker_unavailable({"success": False, "error": "No session", "status_code": 404})
    workspace_files._raise_if_worker_unavailable({"success": True})


# --------------------------------------------------------------------------- the agent tools


@pytest.mark.parametrize("tool, parameters, action", [
    ("workspace_read_file", {"path": "reports/q3.md"}, "read_file"),
    ("workspace_write_file", {"path": "reports/q3.md", "content": "# Q3"}, "write_file"),
    ("workspace_list_dir", {"path": "."}, "list_dir"),
    ("workspace_grep", {"pattern": "revenue"}, "grep"),
    ("workspace_exec", {"command": "ls"}, "exec_command"),
])
def test_the_agent_workspace_tools_return_the_message(no_worker, tool, parameters, action):
    result = _run(lambda: exec_workspace.execute_workspace_action(
        None, tool, parameters, workspace_id=UUID(WS), trace_id="t-us040"))

    assert result["success"] is False
    _assert_not_running(result["error"], action)
