"""PRD-161 S1 — board dispatch spine: claim / lease / requeue.

DB-backed (real Postgres): ``FOR UPDATE SKIP LOCKED`` and ``LISTEN/NOTIFY`` are
Postgres-only, so this suite follows the test.yml Postgres pattern and skips
cleanly when no DB is reachable. It uses its OWN committed sessions rather than
the rolled-back ``db_session`` fixture, because cross-connection row locking and
NOTIFY delivery only happen across real, committed transactions.
"""
from __future__ import annotations

import concurrent.futures
import select
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from core.database.database import get_database_url
from core.models.core import BoardTask
from services import board_dispatcher


@pytest.fixture(scope="module")
def engine():
    """Real Postgres engine; skip the whole module cleanly when none is up."""
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True, pool_size=8, max_overflow=4)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dispatch suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(engine, new_session):
    """Throwaway workspace + agent. Yields (workspace_id, agent_id); deletes the
    workspace's board_tasks, the agent, and the workspace on teardown."""
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
        {"id": ws_id, "n": "prd161-dispatch"},
    )
    s.commit()
    agent_id = s.execute(
        text(
            "INSERT INTO agents (name, agent_type, workspace_id, status) "
            "VALUES (:n, 'custom', CAST(:w AS uuid), 'active') RETURNING id"
        ),
        {"n": "DISPATCH-TEST", "w": ws_id},
    ).fetchone()[0]
    s.commit()
    s.close()

    yield ws_id, agent_id

    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})
    s.execute(text("DELETE FROM agents WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws_id})
    s.commit()
    s.close()


def _seed_tasks(new_session, ws_id, agent_id, n, *, status="assigned", attempts=0):
    s = new_session()
    for i in range(n):
        s.add(
            BoardTask(
                workspace_id=ws_id,
                title=f"task-{i}",
                status=status,
                priority="medium",
                assigned_agent_id=agent_id,
                source_type="user",
                attempts=attempts,
            )
        )
    s.commit()
    ids = [r[0] for r in s.execute(
        text("SELECT id FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id}
    ).fetchall()]
    s.close()
    return ids


# ── S1 AC: concurrency — 4 workers × 50 tasks → each executed exactly once ──────

def test_four_workers_claim_each_task_exactly_once(seeded, new_session):
    ws_id, agent_id = seeded
    _seed_tasks(new_session, ws_id, agent_id, 50)

    def worker(worker_id: str):
        claimed: list[int] = []
        s = new_session()
        try:
            while True:
                rows = board_dispatcher.claim_tasks(
                    s, worker_id=worker_id, limit=3, lease_seconds=600
                )
                # Only count tasks from THIS test's workspace (suite isolation).
                fresh = [t.id for t in rows if str(t.workspace_id) == ws_id]
                if not rows:
                    break
                claimed.extend(fresh)
        finally:
            s.close()
        return claimed

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(worker, [f"w{i}" for i in range(4)]))

    all_claimed = [tid for r in results for tid in r]
    # Exactly once: no task claimed twice, and all 50 claimed.
    assert len(all_claimed) == len(set(all_claimed)), "a task was double-claimed"
    assert set(all_claimed) == set(
        _ids_for(new_session, ws_id)
    ), "not every task was claimed exactly once"

    s = new_session()
    counts = dict(
        s.execute(
            text(
                "SELECT status, COUNT(*) FROM board_tasks "
                "WHERE workspace_id = CAST(:w AS uuid) GROUP BY status"
            ),
            {"w": ws_id},
        ).fetchall()
    )
    s.close()
    assert counts == {"in_progress": 50}, f"expected 50 in_progress, got {counts}"


def _ids_for(new_session, ws_id):
    s = new_session()
    ids = [r[0] for r in s.execute(
        text("SELECT id FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id}
    ).fetchall()]
    s.close()
    return ids


# ── S1 AC: crash — lease expiry requeues attempts+1, never silently closes done ─

def test_expired_lease_requeues_with_attempt_increment(seeded, new_session):
    ws_id, agent_id = seeded
    [task_id] = _seed_tasks(new_session, ws_id, agent_id, 1)

    # Claim it (attempts → 1, status in_progress, lease set), then simulate a
    # crashed worker by forcing the lease into the past.
    s = new_session()
    claimed = board_dispatcher.claim_tasks(s, worker_id="w-crash", limit=1, lease_seconds=600)
    assert [t.id for t in claimed] == [task_id]
    s.execute(
        text("UPDATE board_tasks SET lease_until = :past WHERE id = :id"),
        {"past": datetime.now(timezone.utc) - timedelta(minutes=5), "id": task_id},
    )
    s.commit()

    out = board_dispatcher.requeue_expired_leases(s, max_attempts=2)
    assert out["requeued"] == [task_id]
    assert out["failed"] == []

    row = s.execute(
        text("SELECT status, attempts, lease_until FROM board_tasks WHERE id = :id"),
        {"id": task_id},
    ).fetchone()
    s.close()
    assert row[0] == "assigned", "crashed task must requeue, not close as done"
    assert row[1] == 1, "attempts must reflect the consumed try"
    assert row[2] is None, "lease must clear on requeue"


# ── S1/Q41: out of attempts → terminal 'failed', not infinite requeue ───────────

def test_expired_lease_fails_after_max_attempts(seeded, new_session):
    ws_id, agent_id = seeded
    [task_id] = _seed_tasks(new_session, ws_id, agent_id, 1, status="in_progress", attempts=2)
    s = new_session()
    s.execute(
        text("UPDATE board_tasks SET lease_until = :past WHERE id = :id"),
        {"past": datetime.now(timezone.utc) - timedelta(minutes=5), "id": task_id},
    )
    s.commit()

    out = board_dispatcher.requeue_expired_leases(s, max_attempts=2)
    assert out["failed"] == [task_id]
    assert out["requeued"] == []

    row = s.execute(
        text("SELECT status, error_message FROM board_tasks WHERE id = :id"),
        {"id": task_id},
    ).fetchone()
    s.close()
    assert row[0] == "failed"
    assert row[1] and "Lease expired" in row[1]


# ── S1 AC: NOTIFY — assign→claim wakeup under 1s in-process ─────────────────────

def test_pg_notify_wakes_a_listener_under_1s(seeded, new_session, engine):
    ws_id, _ = seeded

    raw = engine.raw_connection()
    try:
        raw.connection.autocommit = True
        cur = raw.cursor()
        cur.execute(f"LISTEN {board_dispatcher.NOTIFY_CHANNEL}")

        s = new_session()
        board_dispatcher.notify_task_available(s, workspace_id=ws_id, task_id=4242)
        s.commit()
        s.close()

        deadline = time.monotonic() + 1.0
        received = False
        while time.monotonic() < deadline:
            if select.select([raw.connection], [], [], deadline - time.monotonic())[0]:
                raw.connection.poll()
                while raw.connection.notifies:
                    note = raw.connection.notifies.pop(0)
                    if note.channel == board_dispatcher.NOTIFY_CHANNEL and "4242" in note.payload:
                        received = True
                if received:
                    break
        assert received, "claimant did not receive pg_notify within 1s"
    finally:
        raw.close()


# ── S4 AC: per-agent slots — 5 tasks to one agent → ≤2 concurrent, rest queued ──

def test_per_agent_slots_cap_concurrency(seeded, new_session):
    ws_id, agent_id = seeded
    _seed_tasks(new_session, ws_id, agent_id, 5)  # 5 tasks, ONE agent

    s = new_session()
    claimed = board_dispatcher.claim_tasks(
        s, worker_id="w", limit=10, lease_seconds=600, max_slots_per_agent=2
    )
    # slots already full → a second claim yields nothing until one finishes.
    more = board_dispatcher.claim_tasks(
        s, worker_id="w", limit=10, lease_seconds=600, max_slots_per_agent=2
    )
    counts = dict(
        s.execute(
            text(
                "SELECT status, COUNT(*) FROM board_tasks "
                "WHERE workspace_id = CAST(:w AS uuid) GROUP BY status"
            ),
            {"w": ws_id},
        ).fetchall()
    )
    s.close()

    assert len(claimed) == 2, "at most 2 of one agent's tasks run concurrently"
    assert more == [], "no new claims while the agent's slots are full"
    assert counts.get("in_progress") == 2
    assert counts.get("assigned") == 3, "the rest stay queued (double-texting not dropped)"


def test_slots_free_up_as_tasks_complete(seeded, new_session):
    ws_id, agent_id = seeded
    _seed_tasks(new_session, ws_id, agent_id, 5)

    s = new_session()
    first = board_dispatcher.claim_tasks(
        s, worker_id="w", limit=10, lease_seconds=600, max_slots_per_agent=2
    )
    assert len(first) == 2
    # Finish one → a slot frees → exactly one more becomes claimable.
    s.execute(
        text("UPDATE board_tasks SET status='done' WHERE id = :id"),
        {"id": first[0].id},
    )
    s.commit()
    nxt = board_dispatcher.claim_tasks(
        s, worker_id="w", limit=10, lease_seconds=600, max_slots_per_agent=2
    )
    s.close()
    assert len(nxt) == 1, "one completed task frees exactly one slot"


# ── S5 AC: SLA breach scan — overdue task flagged + (caller) notified, once ──────

def test_sla_breach_flags_overdue_task_once(seeded, new_session):
    ws_id, agent_id = seeded
    s = new_session()
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    t = BoardTask(
        workspace_id=ws_id, title="overdue", status="in_progress",
        priority="medium", assigned_agent_id=agent_id, source_type="user",
        sla_deadline=past,
    )
    s.add(t)
    s.commit()
    tid = t.id
    s.close()

    s = new_session()
    breached = board_dispatcher.scan_sla_breaches(s)
    again = board_dispatcher.scan_sla_breaches(s)  # second tick must not re-flag
    flagged = s.execute(
        text("SELECT sla_breach_notified FROM board_tasks WHERE id = :id"), {"id": tid}
    ).fetchone()[0]
    s.close()

    assert tid in [b["task_id"] for b in breached]
    assert tid not in [b["task_id"] for b in again], "breach is flagged once, not every tick"
    assert flagged is True


def test_sla_breach_ignores_terminal_tasks(seeded, new_session):
    ws_id, agent_id = seeded
    s = new_session()
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    t = BoardTask(
        workspace_id=ws_id, title="done-overdue", status="done",
        priority="medium", assigned_agent_id=agent_id, source_type="user",
        sla_deadline=past,
    )
    s.add(t)
    s.commit()
    tid = t.id
    breached = board_dispatcher.scan_sla_breaches(s)
    s.close()
    assert tid not in [b["task_id"] for b in breached], "terminal tasks are never SLA-breached"


# ── night 3 (#612, #614): the answer sat in sessions/<ticket>; the ticket failed ─

def _expire(new_session, task_id):
    s = new_session()
    s.execute(
        text("UPDATE board_tasks SET lease_until = :past WHERE id = :id"),
        {"past": datetime.now(timezone.utc) - timedelta(minutes=5), "id": task_id},
    )
    s.commit()
    return s


def test_an_expired_ticket_whose_session_left_its_answer_goes_to_review_naming_it(
        seeded, new_session, tmp_path, monkeypatch):
    from config import config

    ws_id, agent_id = seeded
    [task_id] = _seed_tasks(new_session, ws_id, agent_id, 1, status="in_progress", attempts=2)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    folder = tmp_path / ws_id / "sessions" / str(task_id)
    folder.mkdir(parents=True)
    (folder / "cafe-questions-answered.md").write_text("# Seven answers\n")
    s = _expire(new_session, task_id)

    out = board_dispatcher.requeue_expired_leases(s, max_attempts=2)
    assert out["delivered"] == [task_id] and out["failed"] == []
    status, feedback = s.execute(
        text("SELECT status, review_feedback FROM board_tasks WHERE id = :id"), {"id": task_id}).fetchone()
    registered = s.execute(
        text("SELECT file_path FROM deliverables WHERE source_type = 'task' AND source_id = :id"),
        {"id": str(task_id)}).scalar()
    s.close()
    assert status == "review"
    assert f"sessions/{task_id}/cafe-questions-answered.md" in feedback
    assert registered == f"sessions/{task_id}/cafe-questions-answered.md"


def test_an_expired_ticket_whose_session_folder_is_empty_still_fails(seeded, new_session, tmp_path, monkeypatch):
    from config import config

    ws_id, agent_id = seeded
    [task_id] = _seed_tasks(new_session, ws_id, agent_id, 1, status="in_progress", attempts=2)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    folder = tmp_path / ws_id / "sessions" / str(task_id)
    folder.mkdir(parents=True)
    (folder / ".claude-lock").write_text("")                       # nothing the owner would read
    s = _expire(new_session, task_id)

    out = board_dispatcher.requeue_expired_leases(s, max_attempts=2)
    s.close()
    assert out["failed"] == [task_id] and out["delivered"] == []


# ── F094 (night 5): one mission step's card spans its runs ───────────────────

def _registered_before(new_session, ws_id, task_id, path):
    """An EARLIER run of this card registered ``path``."""
    s = new_session()
    s.execute(
        text("INSERT INTO deliverables (workspace_id, source_type, source_id, artifact_type, title, file_path) "
             "VALUES (CAST(:w AS uuid), 'task', :id, 'document', :t, :p)"),
        {"w": ws_id, "id": str(task_id), "t": path.rsplit("/", 1)[-1], "p": path},
    )
    s.commit()
    s.close()


def test_a_re_run_that_never_reported_has_this_runs_files_adopted(seeded, new_session, tmp_path, monkeypatch):
    """The session lane now runs a step's re-run on the same card, so the files an
    earlier run registered no longer mean this run reported. Adoption reads what
    THIS run registered, and records what it adopts where a later step reads it."""
    from config import config

    ws_id, agent_id = seeded
    [task_id] = _seed_tasks(new_session, ws_id, agent_id, 1, status="in_progress", attempts=2)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    folder = tmp_path / ws_id / "sessions" / str(task_id)
    folder.mkdir(parents=True)
    (folder / "offer.md").write_text("# The offer, run 1\n")
    (folder / "offer-revised.md").write_text("# The offer, revised in run 2\n")
    _registered_before(new_session, ws_id, task_id, f"sessions/{task_id}/offer.md")
    s = _expire(new_session, task_id)

    out = board_dispatcher.requeue_expired_leases(s, max_attempts=2)
    paths = sorted(p for (p,) in s.execute(
        text("SELECT file_path FROM deliverables WHERE source_type = 'task' AND source_id = :id "
             "AND deleted_at IS NULL"), {"id": str(task_id)}))
    ref = s.execute(text("SELECT runtime_ref FROM board_tasks WHERE id = :id"), {"id": task_id}).scalar()
    s.execute(text("DELETE FROM deliverables WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.commit()
    s.close()
    assert out["delivered"] == [task_id]
    assert paths == [f"sessions/{task_id}/offer-revised.md", f"sessions/{task_id}/offer.md"]   # run 2's file too
    assert sorted(d["file_path"] for d in ref["deliverables"]) == paths


def test_a_re_run_that_left_nothing_fails_rather_than_passing_on_an_earlier_runs_files(seeded, new_session):
    ws_id, agent_id = seeded
    [task_id] = _seed_tasks(new_session, ws_id, agent_id, 1, status="in_progress", attempts=2)
    _registered_before(new_session, ws_id, task_id, f"projects/cafes/offer-{task_id}.md")
    s = _expire(new_session, task_id)

    out = board_dispatcher.requeue_expired_leases(s, max_attempts=2)
    s.execute(text("DELETE FROM deliverables WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws_id})
    s.commit()
    s.close()
    assert out["failed"] == [task_id] and out["delivered"] == []
