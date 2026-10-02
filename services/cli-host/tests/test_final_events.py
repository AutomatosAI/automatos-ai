"""PRD-253 Wave P — a session's result never overtakes its last events.

The Plan card rides on the final flush: the backend files it from the turn's
``PlanReady`` event, and the result that follows parks the ticket on it. Before,
a backend restarting at that moment dropped the batch with a warning and the
result still landed — the ticket finished, and nobody was asked about its plan.
"""
from __future__ import annotations

import queue
from types import SimpleNamespace as NS

from automatos_cli_host.api import BackendError
from automatos_cli_host.config import HostConfig
from automatos_cli_host.host import Host


class _Api:
    """Records what the host sends; the first ``fail`` events posts raise ``status``."""

    def __init__(self, fail=0, status=503):
        self.calls = []
        self.fail = fail
        self.status = status

    def events(self, host_id, task_id, events):
        self.calls.append(("events", task_id, [e["event"] for e in events]))
        if self.fail:
            self.fail -= 1
            raise BackendError(self.status, "the backend is restarting", "http://backend/events")
        return {"status": "in_progress", "control": []}

    def result(self, host_id, task_id, payload):
        self.calls.append(("result", task_id, payload["status"]))
        return {"applied": True, "status": "blocked"}


def _host(short_tmp, api):
    host = Host(HostConfig(url="http://127.0.0.1:9", state_dir=short_tmp / "state", allow_dirs=[short_tmp / "ws"]))
    host.api = api
    return host


def _finished(host, task_id="7"):
    """A session that just ended with its plan still in the queue, and its result waiting."""
    events = queue.Queue()
    for name in ("Stop", "PlanReady"):
        events.put({"event": name})
    host.pending_results[task_id] = {"attempt": 1, "status": "success"}
    return NS(events=events)


def test_the_result_waits_until_the_backend_takes_the_last_events(short_tmp):
    api = _Api(fail=2)
    host = _host(short_tmp, api)
    host._flush_session_events("h1", "7", _finished(host))      # the backend is down: kept, not dropped
    host._retry_results("h1")                                    # still down: the result waits with them
    assert [c[0] for c in api.calls] == ["events", "events"]
    assert host.final_events["7"] and host.pending_results["7"]
    host._retry_results("h1")                                    # back: the events, THEN the result
    assert api.calls[-2:] == [("events", 7, ["Stop", "PlanReady"]), ("result", 7, "success")]
    assert host.final_events == {} and host.pending_results == {}


def test_the_last_events_go_once(short_tmp):
    api = _Api()
    host = _host(short_tmp, api)
    host._flush_session_events("h1", "7", _finished(host))
    host._retry_results("h1")
    host._retry_results("h1")
    assert api.calls == [("events", 7, ["Stop", "PlanReady"]), ("result", 7, "success")]


def test_a_batch_the_backend_refuses_does_not_hold_the_result_back(short_tmp):
    """A 4xx is the backend's answer, not an outage: retrying cannot change it."""
    api = _Api(fail=1, status=404)
    host = _host(short_tmp, api)
    host._flush_session_events("h1", "7", _finished(host))
    host._retry_results("h1")
    assert [c[0] for c in api.calls] == ["events", "result"]
    assert host.final_events == {} and host.pending_results == {}


def test_a_result_with_no_events_left_is_sent_straight_away(short_tmp):
    api = _Api()
    host = _host(short_tmp, api)
    host.pending_results["9"] = {"attempt": 1, "status": "usage_limit"}     # released, never spawned
    host._retry_results("h1")
    assert api.calls == [("result", 9, "usage_limit")]
