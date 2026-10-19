"""In-memory evidence for the change registry's detectors (PRD-263 US-049, US-050).

``agent_row`` builds one read-model row (every ``harness_metrics.AGENT_ROW_FIELDS`` key)
with the heartbeat figures given; ``evidence`` wraps rows and DIAGNOSE's issues the way
``HarnessService._phase_prescribe`` does, with the thresholds from config.py. The judged
week is 2026-W42 and the one reference week W41. Shared by
test_prd263_harness_changes_registry.py and test_prd263_harness_heartbeat_detector.py.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping
from uuid import uuid4

from services import harness_changes as hc
from services import harness_metrics as hm

WEEK = "2026-W42"
REF_WEEK = "2026-W41"


def agent_row(agent_id: int, name: str, *, week: str = WEEK, enabled: bool = True, interval: int = 60,
              met: int = 0, missed: int = 0, unknown: int = 0) -> Dict[str, Any]:
    """One agent-week: every count 0 except the heartbeat runs, met, missed and unknown."""
    counts = {field: 0 for field in hm.AGENT_ROW_FIELDS}
    return {
        **counts, "agent_id": agent_id, "agent_name": name, "model_id": None, "iso_week": week,
        "heartbeat_enabled": enabled, "heartbeat_interval_minutes": interval,
        "heartbeat_runs": met + missed + unknown, "heartbeat_objective_met": met,
        "heartbeat_objective_missed": missed, "heartbeat_objective_unknown": unknown,
    }


def evidence(*rows: Mapping[str, Any], issues: Iterable[Mapping[str, Any]] = ()) -> hc.Evidence:
    """A run's evidence: these agent rows, DIAGNOSE's ``issues``, and config.py's thresholds."""
    metrics = hm.HarnessMetrics(workspace_id=str(uuid4()), window=hm.build_window(WEEK, 1),
                                agent_rows=tuple(dict(row) for row in rows), tool_rows=())
    return hc.Evidence(metrics, {"issues": [dict(issue) for issue in issues]}, hc.settings_from_config())


def reconnect_issue(agent_id: int, name: str, app: str = "GMAIL") -> Dict[str, Any]:
    """A reconnect_app issue as harness_diagnose.tool_failure_issues returns one."""
    return {
        "agent_id": str(agent_id), "agent_name": name, "root_cause": "reconnect_app", "app_name": app,
        "detail": f"reconnect {app}: 20 of its 20 calls this week failed on {app}'s sign-in or connection",
        "calls": 20, "failed_calls": 20, "reconnect_calls": 20,
    }
