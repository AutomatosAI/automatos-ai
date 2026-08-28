"""PRD-228 US-001 — fleet read-model against real Postgres (``@integration``).

DB-backed proof of what the pure suite cannot exercise: the actual SQL runs
against the live schema (column names/types, the ``or_`` board filter, the
cost group-by), the 24h window really excludes older usage, and the read-model
is strictly workspace-scoped — a second workspace's agents and cost never leak.

Skips cleanly when no Postgres is reachable (CI runs it per-story push).
PRD-158 lesson: seed ``workspaces`` FIRST for every FK'd table.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.database.database import get_database_url  # noqa: E402
from core.models.approval_grants import ApprovalGrant  # noqa: E402
from core.models.core import Agent, BoardTask, LLMUsage  # noqa: E402
from core.models.orchestration import OrchestrationRun, OrchestrationTask  # noqa: E402
from core.models.watches import Watch  # noqa: E402
from services.fleet_state import get_fleet_state  # noqa: E402

pytestmark = pytest.mark.integration


def _aware(minutes_ago: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)


def _naive_hours(hours_ago: float) -> datetime:
    # llm_usage.created_at is a naive UTC column.
    return datetime.utcnow() - timedelta(hours=hours_ago)


@pytest.fixture(scope="module")
def engine():
    """Real Postgres engine; skip the whole module cleanly when none is up."""
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("agents", "board_tasks", "orchestration_tasks",
                        "watches", "approval_grants", "llm_usage"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"fleet-state suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(engine, new_session):
    """Seed two workspaces of floor state; yield the ids; delete on teardown."""
    ws_main = str(uuid.uuid4())
    ws_other = str(uuid.uuid4())
    s = new_session()
    for ws, name in ((ws_main, "prd228-main"), (ws_other, "prd228-other")):
        s.execute(
            text("INSERT INTO workspaces (id, name) "
                 "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
            {"id": ws, "n": name},
        )
    s.commit()

    def _agent(ws, name):
        a = Agent(name=name, agent_type="custom", workspace_id=ws, status="active")
        s.add(a)
        s.flush()
        return a

    # ws_main roster
    a_builder = _agent(ws_main, "Builder")
    a_research = _agent(ws_main, "Researcher")
    a_stuck = _agent(ws_main, "Stuck")
    a_bench = _agent(ws_main, "Bench")
    z_other = _agent(ws_other, "Intruder")

    # Builder: a leased in-progress board task + one queued (assigned) task.
    b_current = BoardTask(
        workspace_id=ws_main, title="Ship the widget", status="in_progress",
        assigned_agent_id=a_builder.id, started_at=_aware(5),
        lease_until=_aware(-10), updated_at=_aware(1),
    )
    b_queued = BoardTask(
        workspace_id=ws_main, title="Next up", status="assigned",
        assigned_agent_id=a_builder.id,
    )
    s.add_all([b_current, b_queued])
    s.flush()

    # Builder cost: two rows inside 24h (1000 tok / $0.50) + one stale (excluded).
    for tok, cost, hrs in ((600, 0.30, 1.0), (400, 0.20, 3.0)):
        s.add(LLMUsage(
            workspace_id=ws_main, model_id="m", provider="p", tier="direct",
            agent_id=a_builder.id, input_tokens=tok, output_tokens=0,
            total_tokens=tok, input_cost=cost, output_cost=0.0, total_cost=cost,
            created_at=_naive_hours(hrs),
        ))
    s.add(LLMUsage(
        workspace_id=ws_main, model_id="m", provider="p", tier="direct",
        agent_id=a_builder.id, input_tokens=9999, output_tokens=0,
        total_tokens=9999, input_cost=9.9, output_cost=0.0, total_cost=9.9,
        created_at=_naive_hours(30.0),  # outside the rolling 24h window
    ))
    # A live watch owned by Builder, targeting the current board task.
    s.add(Watch(
        workspace_id=ws_main, watch_type="mission", target_type="board_task",
        target_id=str(b_current.id), title="watch: ship", owner_agent_id=a_builder.id,
        status="watching",
    ))

    # Researcher: a running mission task.
    run = OrchestrationRun(workspace_id=ws_main, goal="Do research", created_by="user_x")
    s.add(run)
    s.flush()
    s.add(OrchestrationTask(
        run_id=run.id, title="Draft the brief", sequence_number=1,
        assigned_agent_id=a_research.id, state="running", started_at=_aware(3),
    ))

    # Stuck: a blocked board task with a pending question ask against it.
    b_blocked = BoardTask(
        workspace_id=ws_main, title="Blocked one", status="in_progress",
        assigned_agent_id=a_stuck.id, started_at=_aware(20), blocked_at=_aware(2),
    )
    s.add(b_blocked)
    s.flush()
    ask = ApprovalGrant(
        workspace_id=ws_main, subject_type="board_task", subject_id=str(b_blocked.id),
        status="pending", kind="question", agent_id=a_stuck.id,
        question_md="Which vendor?",
    )
    s.add(ask)

    # ws_other: an in-progress task + cost that must NOT leak into ws_main.
    s.add(BoardTask(
        workspace_id=ws_other, title="Other work", status="in_progress",
        assigned_agent_id=z_other.id, started_at=_aware(1),
    ))
    s.add(LLMUsage(
        workspace_id=ws_other, model_id="m", provider="p", tier="direct",
        agent_id=z_other.id, input_tokens=555, output_tokens=0, total_tokens=555,
        input_cost=5.5, output_cost=0.0, total_cost=5.5, created_at=_naive_hours(1.0),
    ))
    s.flush()
    s.commit()

    ids = {
        "ws_main": ws_main, "ws_other": ws_other,
        "builder": a_builder.id, "research": a_research.id,
        "stuck": a_stuck.id, "bench": a_bench.id, "intruder": z_other.id,
        "ask_id": ask.id,
    }
    s.close()

    yield ids

    s = new_session.sweep()
    for tbl, col in (
        ("approval_grants", "workspace_id"),
        ("watches", "workspace_id"),
        ("llm_usage", "workspace_id"),
        ("board_tasks", "workspace_id"),
        ("orchestration_tasks", None),
        ("orchestration_runs", "workspace_id"),
        ("agents", "workspace_id"),
    ):
        if tbl == "orchestration_tasks":
            s.execute(text(
                "DELETE FROM orchestration_tasks WHERE run_id IN "
                "(SELECT id FROM orchestration_runs WHERE workspace_id "
                "IN (CAST(:a AS uuid), CAST(:b AS uuid)))"),
                {"a": ws_main, "b": ws_other})
            continue
        s.execute(text(
            f"DELETE FROM {tbl} WHERE {col} IN (CAST(:a AS uuid), CAST(:b AS uuid))"),
            {"a": ws_main, "b": ws_other})
    s.execute(text("DELETE FROM workspaces WHERE id IN (CAST(:a AS uuid), CAST(:b AS uuid))"),
              {"a": ws_main, "b": ws_other})
    s.commit()
    s.close()


def _by_id(result):
    return {e["agent_id"]: e for e in result["agents"]}


def test_fleet_shape_and_values(seeded, new_session):
    s = new_session()
    result = get_fleet_state(s, seeded["ws_main"])
    s.close()

    by_id = _by_id(result)
    # Only ws_main's four agents — the intruder never appears.
    assert set(by_id) == {seeded["builder"], seeded["research"],
                          seeded["stuck"], seeded["bench"]}
    assert seeded["intruder"] not in by_id

    builder = by_id[seeded["builder"]]
    assert builder["current"]["kind"] == "board_task"
    assert builder["current"]["title"] == "Ship the widget"
    assert builder["queue_depth"] == 1
    assert builder["watches"] == {"active": 1, "needs_attention": 0}
    # 24h cost = 1000 tokens / $0.50; the 30h-old $9.90 row is excluded.
    assert builder["cost_24h"]["tokens"] == 1000
    assert builder["cost_24h"]["usd"] == pytest.approx(0.50, abs=1e-6)

    research = by_id[seeded["research"]]
    assert research["current"]["kind"] == "mission_task"
    assert research["current"]["title"] == "Draft the brief"

    stuck = by_id[seeded["stuck"]]
    assert stuck["blocked"]["count"] == 1
    assert seeded["ask_id"] in stuck["blocked"]["open_asks"]

    bench = by_id[seeded["bench"]]
    assert bench["current"] is None
    assert bench["queue_depth"] == 0
    assert bench["cost_24h"] == {"tokens": 0, "usd": 0.0}

    assert result["cost_available"] is True
    assert result["cost_source"] == "llm_usage"
    # Deterministic ordering by name.
    assert [e["name"] for e in result["agents"]] == sorted(
        e["name"] for e in result["agents"]
    )


def test_workspace_isolation(seeded, new_session):
    s = new_session()
    other = get_fleet_state(s, seeded["ws_other"])
    s.close()
    by_id = _by_id(other)
    assert set(by_id) == {seeded["intruder"]}
    # ws_other's own cost is intact and not polluted by ws_main.
    assert by_id[seeded["intruder"]]["cost_24h"]["tokens"] == 555
