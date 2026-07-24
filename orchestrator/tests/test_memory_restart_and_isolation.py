"""PRD-142 Wave 3 · W3-S7 — Memory restart-safe + cross-workspace isolation.

The §H DoD requires every primitive to be (3) Restart-safe — *no user-visible
work lost on process restart* — and (5) Tenant-isolated — *proven by a
cross-workspace test*. For the Memory primitive these collapse to two real-DB
invariants on the L2 ``memory_short_term`` table (the layer the chat path
persists verbatim transcripts to):

1. **Restart-safety**: a row written via ``UnifiedMemoryService.store_transcript``
   survives a process restart, modelled as closing the session that did the
   write and opening a fresh one to read it back.

2. **Cross-workspace isolation**: a transcript written in workspace A is NOT
   readable from workspace B's L2 query path. This is the P0 multi-tenancy
   guard — drop the ``workspace_id`` filter and the assertion trips.

EXTENDS the Wave 2 fixtures (does NOT stand up a parallel harness): same
``_is_ephemeral`` gate as W2-S6 / W2-S5 / J10, same WRITE+COMMIT-then-cleanup
shape, same ``@pytest.mark.integration`` marker. Skips cleanly when no local
Postgres is reachable.
"""
from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

# POSTGRES_* defaults so the engine import doesn't crash before the ephemeral
# gate can skip. Same convention as W2-S6 / golden journeys.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test_db",
}.items():
    os.environ.setdefault(_k, _v)

from core.database.database import engine  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.memory.models import MemoryShortTerm  # noqa: E402

pytestmark = pytest.mark.integration

_EPHEMERAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_EPHEMERAL_DBS = {"test_db", "test"}


def _is_ephemeral(url) -> bool:
    return (
        (url.host or "").lower() in _EPHEMERAL_HOSTS
        and (url.database or "").lower() in _EPHEMERAL_DBS
    )


def _can_handshake_postgres(url, timeout: float = 2.0) -> bool:
    """Fast Postgres handshake probe — fails in seconds when the configured DB
    is not actually reachable as a real Postgres backend.

    A raw TCP probe is not enough: a Docker container that ACCEPTs TCP but
    doesn't speak the Postgres protocol on these credentials would still pass
    the port check, then hang for ~30s on the SQLAlchemy pool timeout. The
    psycopg2 driver supports ``connect_timeout`` so we get a hard ceiling.
    """
    try:
        import psycopg2  # type: ignore
    except Exception:
        return False
    try:
        conn = psycopg2.connect(
            host=url.host or "127.0.0.1",
            port=url.port or 5432,
            user=url.username or "test",
            password=url.password or "test",
            dbname=url.database or "test_db",
            connect_timeout=int(timeout),
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def safe_engine():
    """Refuse to run this WRITE+COMMIT test against anything but an ephemeral DB."""
    if not _is_ephemeral(engine.url):
        pytest.skip(
            "refusing to run memory L2 write tests against a non-ephemeral "
            f"database ({engine.url.host}/{engine.url.database})"
        )
    # Fast pre-flight: when Postgres is not reachable on these credentials,
    # skip in <3s rather than blocking on SQLAlchemy's 30s pool timeout. CI's
    # Postgres service always passes this gate.
    if not _can_handshake_postgres(engine.url):
        pytest.skip(
            f"no reachable Postgres at {engine.url.host}:{engine.url.port}/"
            f"{engine.url.database} for memory integration test"
        )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"no reachable Postgres for memory integration test: {exc}")
    return engine


@pytest.fixture
def two_workspaces_l2(safe_engine):
    """Provision two committed workspaces for L2 isolation testing.

    Cleans up the L2 rows it created plus the workspaces themselves on
    teardown — the test's commits cannot be rolled back by the root
    transactional fixture, so explicit DELETE is the contract.
    """
    Session = sessionmaker(bind=safe_engine)
    db = Session()

    ws_a_id = uuid4()
    ws_b_id = uuid4()
    db.add_all([
        Workspace(id=ws_a_id, name=f"w3s7-mem-iso-a-{ws_a_id.hex[:8]}"),
        Workspace(id=ws_b_id, name=f"w3s7-mem-iso-b-{ws_b_id.hex[:8]}"),
    ])
    db.commit()

    try:
        yield ws_a_id, ws_b_id, Session
    finally:
        # Teardown: delete any L2 rows we wrote, then the workspaces.
        cleanup = Session()
        try:
            cleanup.rollback()  # discard any failed-tx state from the body
            cleanup.query(MemoryShortTerm).filter(
                MemoryShortTerm.workspace_id.in_([ws_a_id, ws_b_id])
            ).delete(synchronize_session=False)
            cleanup.query(Workspace).filter(
                Workspace.id.in_([ws_a_id, ws_b_id])
            ).delete(synchronize_session=False)
            cleanup.commit()
        except Exception:
            cleanup.rollback()
            raise
        finally:
            cleanup.close()
        db.close()


# ---------------------------------------------------------------------------
# 1. L2 transcript survives a process-restart (session close + reopen)
# ---------------------------------------------------------------------------


def test_l2_transcript_survives_session_close_and_reopen(two_workspaces_l2):
    """A transcript written via ``store_transcript`` must persist after the
    session that wrote it closes — modelling a process restart. The next
    session reads the row back.

    This is the §H "restart-safe" check for the Memory primitive: writes that
    completed before restart MUST survive. Fire-and-forget writes that were
    only scheduled (PRD-141 widget latency fix) are a deliberate trade-off
    and out of scope for this assertion — the test only commits a write that
    finished.
    """
    from modules.memory.unified_memory_service import UnifiedMemoryService

    ws_a_id, _ws_b_id, Session = two_workspaces_l2
    ws_a_str = str(ws_a_id)

    service = UnifiedMemoryService()

    # Write the transcript and let it commit through the session pool.
    async def _write():
        row_id = await service.store_transcript(
            workspace_id=ws_a_str,
            turns=[
                {"role": "user", "content": "remember my coffee order"},
                {"role": "assistant", "content": "Got it — black, no sugar."},
            ],
            agent_id=None,
            conversation_id="w3s7-restart-conv",
            metadata={"tier": "global"},
        )
        return row_id

    row_id = asyncio.run(_write())
    assert row_id is not None, "store_transcript must return the new row UUID"

    # Simulate restart: open a brand-new session and re-read.
    fresh = Session()
    try:
        row = (
            fresh.query(MemoryShortTerm)
            .filter(MemoryShortTerm.workspace_id == ws_a_id)
            .filter(MemoryShortTerm.content_type == "transcript")
            .first()
        )
        assert row is not None, (
            "L2 transcript must survive session close+reopen (restart-safety)"
        )
        assert "coffee order" in row.content
        assert row.metadata_.get("conversation_id") == "w3s7-restart-conv"
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# 2. L2 transcript cross-workspace isolation (P0 tenant guard)
# ---------------------------------------------------------------------------


def test_l2_transcript_workspace_isolation(two_workspaces_l2):
    """A transcript written in workspace A must NOT be visible from a
    workspace-scoped read of workspace B. This is the P0 cross-tenant leak
    guard for L2 — drop ``workspace_id`` from the filter and this trips.
    """
    from modules.memory.unified_memory_service import UnifiedMemoryService

    ws_a_id, ws_b_id, Session = two_workspaces_l2
    service = UnifiedMemoryService()

    async def _seed_each():
        a_row = await service.store_transcript(
            workspace_id=str(ws_a_id),
            turns=[{"role": "user", "content": "ws-A secret payload"}],
            metadata={"tier": "global"},
        )
        b_row = await service.store_transcript(
            workspace_id=str(ws_b_id),
            turns=[{"role": "user", "content": "ws-B own data"}],
            metadata={"tier": "global"},
        )
        return a_row, b_row

    a_row, b_row = asyncio.run(_seed_each())
    assert a_row is not None and b_row is not None

    # Re-read from a fresh session, scoped to each workspace.
    fresh = Session()
    try:
        a_visible = (
            fresh.query(MemoryShortTerm)
            .filter(MemoryShortTerm.workspace_id == ws_a_id)
            .all()
        )
        b_visible = (
            fresh.query(MemoryShortTerm)
            .filter(MemoryShortTerm.workspace_id == ws_b_id)
            .all()
        )

        # Each workspace sees exactly its own row.
        a_contents = " ".join(r.content for r in a_visible)
        b_contents = " ".join(r.content for r in b_visible)

        assert "ws-A secret payload" in a_contents
        assert "ws-A secret payload" not in b_contents, (
            "Cross-workspace leak: ws-B can see ws-A's transcript content"
        )
        assert "ws-B own data" in b_contents
        assert "ws-B own data" not in a_contents, (
            "Cross-workspace leak: ws-A can see ws-B's transcript content"
        )
    finally:
        fresh.close()
