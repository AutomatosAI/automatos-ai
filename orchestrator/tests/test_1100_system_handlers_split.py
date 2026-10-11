"""#1100 follow-up: the three dashboard handlers the CPU fix touched were split.

``get_system_health`` (166 lines), ``get_system_metrics`` (155) and
``get_efficiency_score`` (63) failed the changed-lines length rule, so their work
moved to ``api.system_health_checks``, ``api.system_metrics_report`` and
``api.dashboard_efficiency``. Two old faults went with the move:

- the database check ran ``db.execute("SELECT 1")``, which SQLAlchemy 2 refuses
  before it reaches Postgres, so the database always read "unhealthy" and the
  system "degraded";
- the ``system_metrics`` history table was dropped by migration 128a785a7681, so
  ``/api/system/metrics?timeRange=…`` failed on a migrated database. The history
  now falls back to the current reading, as its docstring said.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from api import dashboard_efficiency as de
from api import system_health_checks as shc
from api import system_metrics_report as smr

SNAPSHOT = {"cpu": {"average_usage": 12.345}, "memory": {"percent": 40.0}, "disk": {"percent": 70.5}}


# --- the efficiency score ----------------------------------------------------------------


@pytest.mark.parametrize("score, grade", [
    (95, ("A", "green")), (90, ("A", "green")), (89.9, ("B", "blue")), (80, ("B", "blue")),
    (70, ("C", "yellow")), (69, ("D", "red")), (0, ("D", "red")),
])
def test_the_grade_bands(score, grade):
    assert de.grade_for(score) == grade


def test_the_score_weighs_cpu_memory_agents_and_completions():
    report = de.efficiency_report(cpu_usage=50, memory_percent=50, agent_efficiency=100, workflow_efficiency=50)
    # 60 * 0.3 + 55 * 0.25 + 100 * 0.25 + 50 * 0.2 = 66.75
    assert (report["score"], report["grade"], report["color"]) == (67, "D", "red")
    assert report["breakdown"] == {"cpu_efficiency": 60.0, "memory_efficiency": 55.0,
                                   "agent_efficiency": 100.0, "workflow_efficiency": 50.0}
    assert de.efficiency_report(100, 100, 100, 100)["breakdown"]["cpu_efficiency"] == 100   # capped


def test_the_efficiency_route_on_an_empty_workspace(db_session, seed_workspace):
    from api.analytics_real import get_efficiency_score

    ctx = NS(workspace_id=UUID(seed_workspace()))
    out = asyncio.run(get_efficiency_score(ctx=ctx, db=db_session))
    assert out["breakdown"]["agent_efficiency"] == 0 and out["breakdown"]["workflow_efficiency"] == 0
    assert out["grade"] in {"A", "B", "C", "D"} and "error" not in out


# --- the health checks -------------------------------------------------------------------


def test_the_database_check_reaches_postgres(db_session):
    assert shc._database(db_session)() == ("healthy", {"connection": "active"})


def test_a_failing_check_is_unhealthy_with_its_failure_metrics_and_is_logged(caplog):
    def boom():
        raise RuntimeError("redis://user:secret@host refused")

    component = shc._check("redis", boom, {"ping": "failed", "error": shc.CHECK_FAILED})
    assert (component.name, component.status) == ("redis", "unhealthy")
    assert component.metrics == {"ping": "failed", "error": shc.CHECK_FAILED}     # never the exception text
    assert "the redis check failed" in caplog.text


def test_the_overall_status_is_degraded_when_any_component_is_not_healthy():
    def component(status):
        return shc.ComponentHealth(name="x", status=status, last_check=datetime.now())

    assert shc.overall_status([component("healthy"), component("healthy")]) == "healthy"
    assert shc.overall_status([component("healthy"), component("unhealthy")]) == "degraded"


def test_the_legacy_chunk_count_is_a_number(db_session):
    assert isinstance(shc._chunk_count(db_session), int)


def test_the_health_route_puts_the_checks_together(monkeypatch, db_session):
    import api.system as system
    import services.capability_report as capability_report

    healthy = shc.ComponentHealth(name="database", status="healthy", last_check=datetime.now())
    monkeypatch.setattr(system, "component_health", lambda db: [healthy])
    monkeypatch.setattr(capability_report, "onboarding_capabilities", lambda db, workspace_id=None: {"llm": True})
    out = asyncio.run(system.get_system_health(ctx=NS(workspace_id=None), db=db_session))
    assert out.overall_status == "healthy" and out.components == [healthy]
    assert set(out.system_metrics) == {"cpu_usage", "memory_usage", "memory_available", "disk_usage", "disk_free"}
    assert out.capabilities == {"llm": True}


# --- the metrics report ------------------------------------------------------------------


@pytest.fixture
def main_stats(monkeypatch):
    stats = {"/a": {"call_count": 10, "avg_time": 2.0}, "/b": {"call_count": 14, "avg_time": 4.0}}
    monkeypatch.setitem(sys.modules, "main", NS(api_call_stats=stats))


def _no_history_table(monkeypatch):
    monkeypatch.setattr(smr, "inspect", lambda bind: NS(has_table=lambda name: False))


def test_with_no_history_table_each_chart_is_the_current_reading(monkeypatch, main_stats):
    _no_history_table(monkeypatch)
    out = smr.history(NS(get_bind=lambda: None), "7d", SNAPSHOT)
    assert [p["value"] for p in out["cpu_usage"]] == [12.35]
    assert [p["value"] for p in out["memory_usage"]] == [40.0] and [p["value"] for p in out["disk_usage"]] == [70.5]
    assert out["aggregated"] == {"cpu_average": 12.35, "memory_average": 40.0, "disk_average": 70.5,
                                 "api_calls_total": 24, "response_time_average": 3.0}
    assert [p["value"] for p in out["api_calls"]] == [24]


def test_stored_history_is_read_and_averaged(monkeypatch, main_stats):
    monkeypatch.setattr(smr, "inspect", lambda bind: NS(has_table=lambda name: True))
    rows = [(datetime(2026, 10, 9, 10), 10.0), (datetime(2026, 10, 9, 11), 20.0)]
    db = NS(get_bind=lambda: None, execute=lambda sql, params: NS(fetchall=lambda: rows))
    out = smr.history(db, "24h", SNAPSHOT)
    assert [p["value"] for p in out["cpu_usage"]] == [10.0, 20.0]
    assert out["aggregated"]["cpu_average"] == 15.0
    assert [p["value"] for p in out["api_calls"]] == [12, 12]      # 24 calls over two points


def test_api_stats_that_cannot_be_read_give_empty_series(monkeypatch):
    monkeypatch.setitem(sys.modules, "main", NS())      # no api_call_stats
    assert smr._api_calls(24, 3) == {"total": 0, "avg_time": 0, "calls": [], "response_time": []}


def test_the_analytics_sections_are_zeros_when_the_engine_fails(monkeypatch):
    import core.services.analytics_engine as engine_module

    def broken(db):
        raise RuntimeError("no engine")

    monkeypatch.setattr(engine_module, "AnalyticsEngine", broken)
    context, learning = asyncio.run(smr.analytics_sections(None))
    assert context == smr.EMPTY_CONTEXT and learning == smr.EMPTY_LEARNING


def test_the_metrics_route_is_the_snapshot_and_sections_and_history_only_when_asked(monkeypatch, main_stats):
    import api.system as system

    async def sections(db):
        return {"tokens_saved": 1}, {"total_memories": 2}

    monkeypatch.setattr(system, "analytics_sections", sections)
    _no_history_table(monkeypatch)
    db = NS(get_bind=lambda: None)
    now = asyncio.run(system.get_system_metrics(ctx=None, db=db, timeRange=None))
    assert {"cpu", "memory", "swap", "disk", "network"} <= set(now) and "cpu_usage" not in now
    assert now["context_optimization"] == {"tokens_saved": 1} and now["learning"] == {"total_memories": 2}
    assert len(now["cpu"]["usage_percent"]) == now["cpu"]["count"]
    with_history = asyncio.run(system.get_system_metrics(ctx=None, db=db, timeRange="1h"))
    assert {"cpu_usage", "memory_usage", "disk_usage", "api_calls", "response_time", "aggregated"} <= set(with_history)
