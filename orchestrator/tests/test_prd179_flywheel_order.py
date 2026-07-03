"""PRD-179 S2 (F049) — mission-synthesis flywheel ordering + dedup.

The flywheel's ingest sweep (`CoordinatorService._save_pending_output_documents`)
used to `SELECT ... FROM orchestration_runs WHERE state='completed' LIMIT 3`
with **no ORDER BY** and filtered the already-ingested markers in Python. Once
more than three ingested runs sat at the front of the unordered scan, every
tick re-fetched the same already-done rows and ingested nothing — the flywheel
starved (F049 CONFIRMED, OS review §12.6).

These tests are pure: they drive the sweep against an in-memory fake query
layer that faithfully models the two things the fix must get right —
`ORDER BY created_at DESC` and a SQL-side exclusion of already-ingested /
failed runs — so no Postgres is required. The real SQL is exercised by CI.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern) — nothing here
# touches a DB; the fake query layer stands in for it.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

# config caches POSTGRES_* as class attributes at import — and a sibling test's
# conftest may already have imported config with no env set (local dev). Backfill
# the cached attrs so the lazy engine builder in the CoordinatorService import
# chain doesn't refuse on missing creds. No-op on CI where the vars are real.
from config import config as _config  # noqa: E402

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"):
    if not getattr(_config, _k, None):
        setattr(_config, _k, os.environ[_k])


# ---------------------------------------------------------------------------
# Fake ORM query layer that models ORDER BY / WHERE the way Postgres would.
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_run(idx: int, *, ingested: bool = False, failed: bool = False) -> SimpleNamespace:
    """A fake OrchestrationRun row. created_at ascends with idx (idx 9 newest)."""
    cfg: Dict[str, Any] = {}
    if ingested:
        cfg["output_document_id"] = 1000 + idx
    if failed:
        cfg["output_ingest_failed"] = "2026-01-01T00:00:00+00:00"
    return SimpleNamespace(
        id=idx,
        workspace_id="ws-1",
        state="completed",
        config=cfg,
        created_at=_BASE + timedelta(hours=idx),
    )


class _FakeQuery:
    """Chainable query that honours .filter (marker exclusion), .order_by
    (created_at DESC), and .limit — the three things the fix relies on. Records
    what it was asked to do so tests can assert the real code built the query
    with a SQL-side exclusion and a DESC order (not a Python-side filter)."""

    def __init__(self, rows: List[SimpleNamespace], recorder: dict):
        self._rows = list(rows)
        self._recorder = recorder
        self._desc = False
        self._limit: int | None = None

    def filter(self, *criteria: Any) -> "_FakeQuery":
        # Record the SQL text of each criterion so a test can assert the marker
        # columns are excluded server-side. Then model the intended predicate so
        # the behavioural tests can run without Postgres.
        self._recorder["filter_sql"] = [str(c) for c in criteria]
        kept = [
            r for r in self._rows
            if not (r.config or {}).get("output_document_id")
            and not (r.config or {}).get("output_ingest")
            and not (r.config or {}).get("output_ingest_failed")
        ]
        self._rows = kept
        return self

    def order_by(self, *args: Any) -> "_FakeQuery":
        self._recorder["order_by_sql"] = [str(a) for a in args]
        self._desc = True
        return self

    def limit(self, n: int) -> "_FakeQuery":
        self._recorder["limit"] = n
        self._limit = n
        return self

    def all(self) -> List[SimpleNamespace]:
        rows = self._rows
        if self._desc:
            rows = sorted(rows, key=lambda r: r.created_at, reverse=True)
        if self._limit is not None:
            rows = rows[: self._limit]
        return rows


class _FakeSession:
    def __init__(self, rows: List[SimpleNamespace]):
        self.rows = rows
        self.recorder: dict = {}

    def query(self, *_models: Any) -> _FakeQuery:
        return _FakeQuery(self.rows, self.recorder)

    def flush(self) -> None:  # pragma: no cover - trivial
        pass


@pytest.fixture
def coordinator():
    """A CoordinatorService with the actual sweep, but the per-run ingest call
    stubbed so we assert *which* runs the sweep selected, not the ingest guts."""
    from services.coordinator_service import CoordinatorService

    svc = CoordinatorService()
    svc._save_mission_output_as_document = AsyncMock(return_value=1)
    return svc


@pytest.mark.asyncio
async def test_flywheel_dedup_and_order(coordinator):
    """10 completed missions, none ingested. The sweep must pick the MOST
    RECENT batch (ORDER BY created_at DESC), and a second sweep — after the
    first batch is marked ingested — must pick only *new* (older) runs, never
    re-processing the already-ingested ones (SQL-side exclusion)."""
    runs = [_make_run(i) for i in range(10)]  # idx 9 == newest
    db = _FakeSession(runs)

    # --- First sweep: newest batch by created_at DESC ---
    await coordinator._save_pending_output_documents(db)
    first_batch = [c.args[1].id for c in coordinator._save_mission_output_as_document.call_args_list]

    assert first_batch, "sweep ingested nothing on a fresh backlog (starvation)"
    # Must be the most-recent ids (descending), never an arbitrary/oldest slice.
    assert first_batch == sorted(first_batch, reverse=True), (
        f"batch not ordered newest-first: {first_batch}"
    )
    assert max(first_batch) == 9, f"newest run (id=9) not in first batch: {first_batch}"
    batch_size = len(first_batch)

    # Mark the first batch as ingested (what a successful ingest does).
    for rid in first_batch:
        runs[rid].config = {**(runs[rid].config or {}), "output_document_id": 5000 + rid}

    # --- Second sweep: only NEW runs, none of the already-ingested set ---
    coordinator._save_mission_output_as_document.reset_mock()
    await coordinator._save_pending_output_documents(db)
    second_batch = [c.args[1].id for c in coordinator._save_mission_output_as_document.call_args_list]

    assert not (set(second_batch) & set(first_batch)), (
        f"second sweep re-ingested already-done runs: "
        f"{set(second_batch) & set(first_batch)}"
    )
    # The next newest un-ingested runs surface next (proves progress past 3).
    remaining = sorted((set(range(10)) - set(first_batch)), reverse=True)
    assert second_batch == remaining[:batch_size], (
        f"second batch {second_batch} != expected next-newest {remaining[:batch_size]}"
    )


@pytest.mark.asyncio
async def test_flywheel_query_is_sql_side_ordered_and_excluded(coordinator):
    """The fix must push ordering and exclusion into the query, not do them in
    Python. Assert the real code emitted a DESC order on created_at and a filter
    referencing all three terminal markers — so a regression to an unordered
    ``LIMIT`` with Python-side filtering is caught even without Postgres."""
    db = _FakeSession([_make_run(i) for i in range(4)])
    await coordinator._save_pending_output_documents(db)

    order_sql = " ".join(db.recorder.get("order_by_sql", [])).lower()
    assert "created_at" in order_sql and "desc" in order_sql, (
        f"sweep did not ORDER BY created_at DESC (got {db.recorder.get('order_by_sql')})"
    )

    # The marker keys are bound parameters (SQLAlchemy parameterises ->> keys),
    # so assert the *shape*: three JSONB ->> IS NULL exclusions on config, which
    # is exactly one guard per terminal marker (ingested / opted-out / failed).
    filter_sql = " ".join(db.recorder.get("filter_sql", [])).lower()
    assert filter_sql.count("config ->>") >= 3, (
        f"sweep filter lacks the three JSONB marker exclusions (got {filter_sql})"
    )
    assert filter_sql.count("is null") >= 3, (
        f"sweep filter does not exclude already-handled runs server-side (got {filter_sql})"
    )
    assert db.recorder.get("limit") == 3, "batch size no longer bounded by config"


@pytest.mark.asyncio
async def test_flywheel_excludes_failed_marker(coordinator):
    """A run that previously failed to ingest carries a failure marker and must
    be excluded from the sweep — so a persistently-failing run can never wedge
    the batch and silently starve the newer ones behind it."""
    runs = [
        _make_run(0, failed=True),   # poisoned — must be skipped
        _make_run(1),
        _make_run(2),
    ]
    db = _FakeSession(runs)

    await coordinator._save_pending_output_documents(db)
    picked = [c.args[1].id for c in coordinator._save_mission_output_as_document.call_args_list]

    assert 0 not in picked, "sweep re-processed a run already marked ingest-failed"
    assert set(picked) == {1, 2}, f"expected the two healthy runs, got {picked}"
