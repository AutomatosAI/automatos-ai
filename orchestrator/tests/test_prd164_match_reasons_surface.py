"""PRD-164 S2 — match reasons persist on the plan and reach the approval card.

Three pure seams, no DB:

  * ``annotate_plan_with_matches`` (services.coordinator_service) mirrors the
    per-task match annotation into the ``run.plan`` snapshot immutably — the
    snapshot is what the create-mission tool result (and so the approval card)
    is built from;
  * ``_plan_task_summary`` (handlers_missions) carries ``match_agent`` /
    ``match_reason`` into the tool result's tasks array;
  * ``ToolResultFormatter.format_for_frontend`` passes those task fields
    through to the ``mission_approval`` card payload (PRD-163 S4 card).
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys as _sys

# Dummy POSTGRES_* satisfies the config chain (blessed pattern, see
# tests/test_harness_self_management.py) — the port points at nothing so the
# modules.tools import chain's fail-soft DB connect refuses instantly instead
# of hanging on a wedged local proxy. CI exports real POSTGRES_* so these
# setdefaults no-op there. Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which
# pulls modules.rag's ingestion chain (camelot at module top). Stub the missing
# *leaf* only when truly absent — never the modules.rag package.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from modules.tools.discovery.handlers_missions import _plan_task_summary  # noqa: E402
from modules.tools.formatting.result_formatter import ToolResultFormatter  # noqa: E402
from services.coordinator_service import annotate_plan_with_matches  # noqa: E402


def _plan():
    return {
        "tasks": [
            {"temp_id": "t1", "title": "Research pricing", "agent_role": "research",
             "sequence_number": 1},
            {"temp_id": "t2", "title": "Write summary", "agent_role": "writer",
             "sequence_number": 2},
        ],
        "dependencies": [{"from": "t1", "to": "t2"}],
    }


def _match(agent_name="SCOUT", reason="Strong role match for 'research'",
           agent_id=1, is_override=False):
    return {"agent_id": agent_id, "agent_name": agent_name, "score": 0.81,
            "reason": reason, "is_override": is_override}


# ---------------------------------------------------------------------------
# annotate_plan_with_matches — plan snapshot mirror (immutably)
# ---------------------------------------------------------------------------


def test_annotate_plan_mirrors_match_fields_by_sequence():
    plan = _plan()
    out = annotate_plan_with_matches(plan, {1: _match()})

    t1, t2 = out["tasks"]
    assert t1["match_agent"] == "SCOUT"
    assert t1["match_reason"] == "Strong role match for 'research'"
    assert t1["match_agent_id"] == 1
    assert t1["match_is_override"] is False
    assert "match_agent" not in t2  # unmatched task untouched

    # Immutability: the input plan dict and its tasks were not mutated.
    assert "match_agent" not in plan["tasks"][0]
    assert out is not plan


def test_annotate_plan_handles_empty_inputs():
    assert annotate_plan_with_matches(None, {}) == {"tasks": []}
    plan = _plan()
    out = annotate_plan_with_matches(plan, {})
    assert out["tasks"] == plan["tasks"]


# ---------------------------------------------------------------------------
# _plan_task_summary — the tool-result tasks array the card is built from
# ---------------------------------------------------------------------------


def test_plan_task_summary_carries_match_fields_when_present():
    plan = annotate_plan_with_matches(_plan(), {1: _match()})
    summary = _plan_task_summary(plan["tasks"])

    assert summary[0]["title"] == "Research pricing"
    assert summary[0]["agent_role"] == "research"
    assert summary[0]["sequence"] == 1
    assert summary[0]["match_agent"] == "SCOUT"
    assert summary[0]["match_reason"] == "Strong role match for 'research'"
    # No match annotation → no match keys (payload stays lean).
    assert "match_reason" not in summary[1] and "match_agent" not in summary[1]


def test_plan_task_summary_caps_at_ten_tasks():
    tasks = [{"title": f"T{i}", "agent_role": "research", "sequence_number": i}
             for i in range(1, 15)]
    assert len(_plan_task_summary(tasks)) == 10


# ---------------------------------------------------------------------------
# Approval card payload — reasons visible to the card (AC3)
# ---------------------------------------------------------------------------


def test_approval_card_payload_includes_match_reasons():
    result = {
        "success": True, "mission_id": "abc-123", "state": "awaiting_approval",
        "awaiting_approval": True, "goal": "research X", "task_count": 2,
        "tasks": _plan_task_summary(
            annotate_plan_with_matches(_plan(), {1: _match(), 2: _match(
                agent_name="SCRIBE", reason="Strong role match for 'writer'",
                agent_id=2)})["tasks"]
        ),
    }
    fd = ToolResultFormatter.format_for_frontend(result, "platform_create_mission")
    card = fd["mission_approval"]
    assert card["mission_id"] == "abc-123"
    assert card["tasks"][0]["match_reason"] == "Strong role match for 'research'"
    assert card["tasks"][0]["match_agent"] == "SCOUT"
    assert card["tasks"][1]["match_reason"] == "Strong role match for 'writer'"
