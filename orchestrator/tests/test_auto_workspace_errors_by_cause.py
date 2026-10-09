"""Auto's wishlist (9 Oct): "79 failed" had no "why" in one call.

Auto's skill listed four observability tools (``platform_get_workspace_errors``,
``platform_get_my_slow_calls``, ``platform_get_cost_anomalies``, ``platform_get_trace``)
and none was registered, so every call to one failed. The errors tool is now real:
failed cards, LLM calls and tool runs in THIS workspace, grouped by cause. The three
others left the skill (automatos-skills v2.3.3), and a guard keeps the skill from
naming a tool the platform does not register.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery import failure_causes as fc
from modules.tools.discovery import handlers_diagnostics as hd

SEED = Path(__file__).resolve().parents[1] / "core" / "seeds" / "platform-management-skill.md"
CREDIT = "Error code: 402 - {'error': {'message': 'This request requires more credits'}}"
NOW = datetime(2026, 10, 9, 12, 0, 0)


def _failure(message, *, source="card", ref="#1", minutes_ago=0, agent_id=None):
    return fc.Failure(source=source, ref=ref, message=message, at=NOW - timedelta(minutes=minutes_ago),
                      agent_id=agent_id)


@pytest.mark.parametrize("message, cause", [
    (CREDIT, fc.CAUSE_OUT_OF_CREDIT),
    ("Request timed out after 120s", fc.CAUSE_TIMEOUT),
    ("timeout", fc.CAUSE_TIMEOUT),
    ("Error code: 429 - too many requests", fc.CAUSE_RATE_LIMITED),
    ("rate_limited", fc.CAUSE_RATE_LIMITED),
    ("Error code: 401 - invalid x-api-key", fc.CAUSE_AUTH),
    ("Unknown tool: platform_get_trace", fc.CAUSE_TOOL_MISSING),
    ("Gmail is not connected for this workspace", fc.CAUSE_TOOL_MISSING),
    ("The model declined to write this", fc.CAUSE_REFUSED),
    ("Document 1547 does not exist", fc.CAUSE_NOT_FOUND),
    ("KeyError: 'title'", fc.CAUSE_OTHER),
    ("", fc.CAUSE_OTHER),
    (None, fc.CAUSE_OTHER),
])
def test_each_failure_goes_under_the_cause_its_text_points_to(message, cause):
    assert fc.classify(message) == cause


def test_out_of_credit_wins_over_the_status_code_rules():
    # A credit refusal is a 402 that also says "requires more credits"; it is never "other".
    assert fc.classify(CREDIT + " (rate limit headers attached)") == fc.CAUSE_OUT_OF_CREDIT


def test_causes_are_ranked_by_count_then_by_most_recent():
    failures = [_failure("timed out", minutes_ago=50), _failure("timed out", minutes_ago=40),
                _failure(CREDIT, source="llm_call", ref="9 (m)", minutes_ago=1, agent_id=7),
                _failure("KeyError", minutes_ago=5)]
    groups = fc.group_by_cause(failures)
    assert [g.cause for g in groups] == [fc.CAUSE_TIMEOUT, fc.CAUSE_OUT_OF_CREDIT, fc.CAUSE_OTHER]
    assert groups[0].count == 2 and groups[0].last_seen == NOW - timedelta(minutes=40)


def test_a_group_counts_by_source_names_its_agents_once_and_keeps_three_recent_examples():
    failures = [_failure(CREDIT, source="card", ref=f"#{n}", minutes_ago=n, agent_id=7) for n in range(5)]
    failures.append(_failure(CREDIT, source="llm_call", ref="12 (m)", minutes_ago=9, agent_id=8))
    [group] = fc.group_by_cause(failures)
    assert group.count == 6 and group.by_source == {"card": 5, "llm_call": 1}
    assert group.agent_ids == [7, 8]
    assert [e["ref"] for e in group.examples] == ["#0", "#1", "#2"]


def test_the_payload_labels_each_cause_and_cuts_long_messages():
    [row] = fc.as_payload(fc.group_by_cause([_failure("x" * 500 + " timed out")]))
    assert row["cause"] == "timeout" and row["label"] == fc.CAUSE_LABELS["timeout"]
    assert row["last_seen"] == NOW.isoformat()
    assert len(row["examples"][0]["message"]) == fc.MESSAGE_CHARS


def test_failures_with_no_time_still_group():
    groups = fc.group_by_cause([fc.Failure(source="card", ref="#1", message="timed out")])
    assert groups[0].count == 1 and groups[0].last_seen is None


@pytest.mark.parametrize("params, days", [
    ({}, 1), ({"days": 7}, 7), ({"days": "3"}, 3), ({"days": 90}, hd.MAX_DAYS), ({"days": 0}, 1),
    ({"days": -4}, 1), ({"days": "a week"}, 1), ({"days": None}, 1),
])
def test_the_window_is_one_to_fourteen_days(params, days):
    assert hd._window_days(params) == days


def test_every_source_is_read_on_one_naive_utc_clock():
    aware = datetime(2026, 10, 9, 13, 0, tzinfo=timezone(timedelta(hours=1)))
    assert hd._naive_utc(aware) == datetime(2026, 10, 9, 12, 0)
    assert hd._naive_utc(NOW) is NOW and hd._naive_utc(None) is None


def test_the_handler_merges_the_three_sources_and_says_which_were_cut_short(monkeypatch):
    many = [_failure("timed out", source="tool_run", ref=str(n)) for n in range(hd.ROWS_PER_SOURCE)]
    monkeypatch.setattr(hd, "_failed_cards", lambda db, ws, since: [_failure(CREDIT)])
    monkeypatch.setattr(hd, "_failed_llm_calls", lambda db, ws, since: [])
    monkeypatch.setattr(hd, "_failed_tool_runs", lambda db, ws, since: many)
    out = asyncio.run(hd.get_workspace_errors(None, uuid4(), {"days": 2}))
    assert out["success"] is True and out["period_days"] == 2
    assert out["total_failures"] == hd.ROWS_PER_SOURCE + 1
    assert [c["cause"] for c in out["causes"]] == ["timeout", "out_of_credit"]
    assert out["truncated_sources"] == ["tool_run"]


# --- the readers, against the test database: this workspace's failures only ----------


@pytest.fixture
def two_workspaces(db_session, seed_workspace):
    from core.models.composio_cache import ToolExecutionLog
    from core.models.core import BoardTask, LLMUsage

    mine, theirs = UUID(seed_workspace()), UUID(seed_workspace())
    now = datetime.now(timezone.utc)

    def usage(ws, status, message, at):
        return LLMUsage(workspace_id=ws, model_id="google/gemini-2.5-flash", provider="openrouter",
                        tier="aggregator", input_tokens=10, output_tokens=0, total_tokens=10, input_cost=0,
                        output_cost=0, total_cost=0, status=status, error_message=message,
                        created_at=at.replace(tzinfo=None))

    def tool_run(ws, status, message):
        return ToolExecutionLog(workspace_id=ws, app_name="gmail", action_name="GMAIL_SEND_EMAIL",
                                status=status, error_message=message, executed_at=now.replace(tzinfo=None))

    def card(ws, status, message, at, source_type="user"):
        return BoardTask(workspace_id=ws, title="Reconcile the September takings", priority="medium",
                         source_type=source_type, status=status, error_message=message, updated_at=at)

    db_session.add_all([
        card(mine, "failed", CREDIT, now), card(mine, "done", None, now),
        card(mine, "failed", "timed out", now - timedelta(days=3)),                 # outside one day
        card(mine, "failed", "step failed", now, source_type="orchestration_task"),  # a mission step
        card(theirs, "failed", CREDIT, now),
        usage(mine, "rate_limited", None, now), usage(mine, "success", None, now),
        usage(theirs, "error", "boom", now),
        tool_run(mine, "error", "Gmail is not connected"), tool_run(theirs, "error", "nope"),
    ])
    db_session.flush()
    return NS(db=db_session, mine=mine)


def test_the_tool_reads_only_this_workspaces_failures_inside_the_window(two_workspaces):
    out = asyncio.run(hd.get_workspace_errors(two_workspaces.db, two_workspaces.mine, {}))
    assert out["total_failures"] == 3
    by_cause = {c["cause"]: c for c in out["causes"]}
    assert set(by_cause) == {"out_of_credit", "rate_limited", "tool_missing"}
    assert by_cause["out_of_credit"]["by_source"] == {"card": 1}
    assert by_cause["rate_limited"]["examples"][0]["message"] == "rate_limited"   # the status, no message
    assert by_cause["tool_missing"]["examples"][0]["ref"].endswith("(GMAIL_SEND_EMAIL)")


def test_a_wider_window_reaches_older_cards(two_workspaces):
    out = asyncio.run(hd.get_workspace_errors(two_workspaces.db, two_workspaces.mine, {"days": 7}))
    assert out["total_failures"] == 4 and out["period_days"] == 7


# --- wiring, and the skill naming only tools that exist --------------------------------


def test_the_tool_is_registered_as_a_read_and_mapped_to_its_handler():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.platform_actions import register_all_actions
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    registry = ActionRegistry()
    register_all_actions(registry)
    action = registry._actions["platform_get_workspace_errors"]
    assert action.permission_level == "read" and action.category == "analytics"
    assert PLATFORM_HANDLERS["platform_get_workspace_errors"] is hd.get_workspace_errors


def test_every_tool_autos_skill_names_is_registered_and_handled():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.platform_actions import register_all_actions
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    registry = ActionRegistry()
    register_all_actions(registry)
    frontmatter = SEED.read_text(encoding="utf-8").split("---", 2)[1]
    named = re.findall(r"^  - name: (\S+)", frontmatter, re.M)
    assert "platform_get_workspace_errors" in named
    missing = [name for name in named if name not in registry._actions]
    assert missing == [], f"Auto's skill names tools the platform does not register: {missing}"
    unhandled = [name for name in named if name.startswith("platform_") and name not in PLATFORM_HANDLERS]
    assert unhandled == [], f"registered but with no handler: {unhandled}"


@pytest.mark.parametrize("phantom", ["platform_get_my_slow_calls", "platform_get_cost_anomalies",
                                     "platform_get_trace"])
def test_the_skill_no_longer_offers_tools_that_never_existed(phantom):
    assert phantom not in SEED.read_text(encoding="utf-8")
