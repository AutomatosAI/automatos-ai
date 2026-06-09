"""PRD-142 Wave 4 (W4-S10): HARNESS learns from tool failures (the cross-link).

DIAGNOSE reads the tool-routing fails_for_intent affinities (Role-1 learning,
PRD-138/139) and surfaces a SUSTAINED failure as an inefficiency issue; PRESCRIBE
turns it into a QUEUED (risk 3, human-reviewed) tool_assignment_remove — but ONLY
when the failing action cleanly maps to a removable Composio app. The tool hot path
often records the `composio_execute` meta-tool rather than the specific app action,
so a fuzzy signal must never auto-yank a tool: no clean app mapping -> surfaced as a
diagnosis, no removal proposed. Learning-only: reads ONLY tool_routing_affinities.

Dummy POSTGRES_* + the apscheduler stub let the harness_service import chain load.
"""
import asyncio
import os
import sys
import types
from uuid import UUID

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()

from services.harness_service import HarnessService  # noqa: E402

_WS = UUID("00000000-0000-0000-0000-000000000001")


class _FakeAffinity:
    def __init__(self, agent_id, action_name, sample_count):
        self.agent_id = agent_id
        self.action_name = action_name
        self.sample_count = sample_count


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, _model):
        return _FakeQuery(self._rows)


def _tool_failure_issue(action_name, app_name, agent_id=7, agent_name="SCOUT"):
    return {
        "agent_id": str(agent_id),
        "agent_name": agent_name,
        "root_cause": "tool_failure",
        "detail": f"tool '{action_name}' failed 9x",
        "tool_failure_spec": {
            "agent_id": agent_id, "action_name": action_name,
            "app_name": app_name, "sample_count": 9,
        },
    }


# --- the app-mapping guard --------------------------------------------------

def test_derive_app_name_maps_composio_action():
    assert HarnessService._derive_app_name("GMAIL_SEND_EMAIL") == "GMAIL"


def test_derive_app_name_none_for_meta_and_platform_tools():
    assert HarnessService._derive_app_name("composio_execute") is None      # meta-tool
    assert HarnessService._derive_app_name("platform_create_routing_rule") is None
    assert HarnessService._derive_app_name("SINGLEWORD") is None            # no underscore
    assert HarnessService._derive_app_name("") is None


# --- the diagnose cross-link ------------------------------------------------

def test_diagnose_surfaces_sustained_tool_failure():
    svc = HarnessService()
    db = _FakeDB([_FakeAffinity(7, "GMAIL_SEND_EMAIL", 9)])
    issues = svc._diagnose_tool_failures(_WS, db, [{"id": 7, "name": "SCOUT"}])
    assert len(issues) == 1
    iss = issues[0]
    assert iss["root_cause"] == "tool_failure"
    assert iss["agent_name"] == "SCOUT"
    assert iss["tool_failure_spec"]["app_name"] == "GMAIL"
    assert iss["tool_failure_spec"]["agent_id"] == 7


def test_diagnose_is_best_effort_never_raises():
    svc = HarnessService()

    class _BoomDB:
        def query(self, *a, **k):
            raise RuntimeError("db down")

    # A telemetry read must never break the weekly tick.
    assert svc._diagnose_tool_failures(_WS, _BoomDB(), []) == []


# --- the prescribe side (queued, guarded) -----------------------------------

def test_prescribe_queues_removal_for_clean_app():
    svc = HarnessService()
    rxs, seq = svc._prescribe_tool_removals(
        [_tool_failure_issue("GMAIL_SEND_EMAIL", "GMAIL")], set(), "2026-06-09", 0
    )
    assert len(rxs) == 1
    rx = rxs[0]
    assert rx["change_type"] == "tool_assignment_remove"
    assert rx["risk_score"] == 3            # QUEUED for human review, never auto-applied
    assert rx["target_id"] == 7
    assert rx["proposed_value"] == {"app_name": "GMAIL"}
    assert seq == 1


def test_prescribe_skips_when_no_clean_app_mapping():
    svc = HarnessService()
    rxs, seq = svc._prescribe_tool_removals(
        [_tool_failure_issue("composio_execute", None)], set(), "2026-06-09", 0
    )
    assert rxs == []   # surfaced as a diagnosis, but never an auto-removal on a fuzzy signal
    assert seq == 0


def test_prescribe_respects_rejected_signature():
    svc = HarnessService()
    rxs, _seq = svc._prescribe_tool_removals(
        [_tool_failure_issue("GMAIL_SEND_EMAIL", "GMAIL")],
        {"tool_remove:SCOUT:GMAIL"}, "2026-06-09", 0,
    )
    assert rxs == []   # a previously-rejected removal is not re-proposed


def test_prescribe_ignores_non_tool_failure_issues():
    svc = HarnessService()
    rxs, seq = svc._prescribe_tool_removals(
        [{"root_cause": "auto_applied_regression"}], set(), "2026-06-09", 5
    )
    assert rxs == []
    assert seq == 5
