"""PRD-268 US-C04a (K3) — room in ``services/heartbeat_service.py`` for the bridge.

The decision engine's shadow read of a heartbeat (PRD-248 S5) moved to
``services/heartbeats/triage_shadow.py``. ``heartbeat_service`` re-exports it under the
name ``_dispatch_heartbeat_notification`` calls, so the dispatcher stays byte-identical:

* both paths resolve to one function object;
* the dispatcher's own globals resolve the call to it;
* through the new path, the dial off never reaches the engine and the dial on shadows
  one report triage.

PURE tests: a fake engine, no HTTP, no database.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List

WS = "11111111-1111-1111-1111-111111111111"
_CALL = dict(
    workspace_id=WS, link_id=1, agent_id=1, agent_name="A", title="t", message="m",
    status="ok", source_type="agent", platform_action="report_to=auto",
)


class _Engine:
    def __init__(self, mode: str):
        self._dials = SimpleNamespace(report_triage_mode=mode)
        self.purposes: List[str] = []

    def dials(self):
        return self._dials

    def shadow(self, coro, purpose="shadow"):
        self.purposes.append(purpose)
        coro.close()
        return True


def _install(monkeypatch, engine):
    import core.llm.decisions as pkg

    monkeypatch.setattr(pkg, "get_decision_engine", lambda: engine)


def test_the_moved_function_resolves_from_both_paths():
    from services import heartbeat_service as hb
    from services.heartbeats import triage_shadow

    assert hb._shadow_heartbeat_triage is triage_shadow.shadow_heartbeat_triage
    assert triage_shadow.shadow_heartbeat_triage.__module__ == "services.heartbeats.triage_shadow"


def test_the_dispatcher_calls_the_moved_function():
    from services.heartbeat_service import HeartbeatService
    from services.heartbeats.triage_shadow import shadow_heartbeat_triage

    dispatcher_globals = HeartbeatService._dispatch_heartbeat_notification.__globals__
    assert dispatcher_globals["_shadow_heartbeat_triage"] is shadow_heartbeat_triage


def test_the_moved_function_shadows_only_when_the_dial_is_on(monkeypatch):
    from core.llm.decisions import MODE_OFF, MODE_SHADOW, judgements
    from services.heartbeats.triage_shadow import shadow_heartbeat_triage

    off = _Engine(MODE_OFF)
    _install(monkeypatch, off)
    assert shadow_heartbeat_triage(**_CALL) is None
    assert off.purposes == []

    on = _Engine(MODE_SHADOW)
    _install(monkeypatch, on)
    shadow_heartbeat_triage(**_CALL)
    assert on.purposes == [judgements.PURPOSE_REPORT]


def test_a_failing_engine_never_reaches_the_heartbeat(monkeypatch):
    from services.heartbeats.triage_shadow import shadow_heartbeat_triage

    def _boom():
        raise RuntimeError("engine down")

    import core.llm.decisions as pkg

    monkeypatch.setattr(pkg, "get_decision_engine", _boom)
    assert shadow_heartbeat_triage(**_CALL) is None
