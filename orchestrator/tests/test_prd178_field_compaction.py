"""PRD-178 S3 (F063) — workspace-scoped compaction with a resume cursor.

Before this wave the compaction sweep ran unscoped over the whole shared
collection every hour, restarting from offset=None each time (F063: re-scans
everything, no scope, no resume). This corrupts throughput at scale and starves
large workspaces.

The fix:
  * ``compact(workspace_id=...)`` scopes the scroll to that workspace's filter,
    so another workspace's entries are never even scanned (let alone pruned);
  * ``compact`` accepts ``resume_offset`` and returns the next Qdrant scroll
    cursor, so a subsequent run resumes where the last one stopped instead of
    re-scanning compacted entries — persisted across restarts via a cursor
    store backed by the existing ``system_settings`` table (no new table).

Tests mock Qdrant at the boundary and assert the scroll filter/offset the
adapter passes down.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from tests.test_vector_field import (  # noqa: E402
    VectorFieldSharedContext,
    _make_point,
)

pytestmark = pytest.mark.asyncio


def _adapter(client: MagicMock) -> VectorFieldSharedContext:
    adapter = VectorFieldSharedContext.__new__(VectorFieldSharedContext)
    adapter._client = client
    adapter._embedder = MagicMock()
    adapter._decay_rate = 0.1
    adapter._reinforce_bonus = 0.05
    adapter._reinforce_cap = 2.0
    adapter._archival_threshold = 0.05
    adapter._boundary_permeability = 1.0
    adapter._dimension = 2048
    adapter._half_life_access_scale = 0.5
    adapter._bootstrap_done = True
    return adapter


async def test_field_compaction_workspace_scope():
    """Compaction for workspace A scopes its scroll to A's workspace filter, so
    workspace B's entries are never scanned/pruned."""
    client = MagicMock()
    # One prunable point (very old, weak) belonging to workspace A.
    from datetime import datetime, timedelta, timezone
    old = datetime.now(timezone.utc) - timedelta(days=365)
    a_point = _make_point("a1", strength=0.05, access_count=0, last_accessed=old)
    client.scroll = AsyncMock(return_value=([a_point], None))
    client.delete = AsyncMock()

    adapter = _adapter(client)

    with patch(
        "modules.context.adapters.vector_field.config",
        MagicMock(FIELD_PRUNE_THRESHOLD=0.01, FIELD_COMPACTION_MAX_SCAN=10000),
    ):
        result = await adapter.compact(workspace_id="ws-A")

    # The scroll must have been filtered — not a full-collection scan.
    assert client.scroll.await_count >= 1
    _, kwargs = client.scroll.call_args
    scroll_filter = kwargs.get("scroll_filter")
    assert scroll_filter is not None, "compaction must scope by workspace filter"

    # Result exposes a pruned count and a resume cursor.
    assert result.pruned == 1
    assert hasattr(result, "next_offset")


async def test_field_compaction_resume():
    """A second run resumes from the returned cursor: the adapter passes the
    prior run's next_offset as the scroll offset, not None."""
    client = MagicMock()
    # First page returns a cursor "CURSOR-1"; nothing prunable.
    fresh_point = _make_point("keep", strength=1.0, access_count=5)
    client.scroll = AsyncMock(return_value=([fresh_point], "CURSOR-1"))
    client.delete = AsyncMock()
    adapter = _adapter(client)

    with patch(
        "modules.context.adapters.vector_field.config",
        MagicMock(FIELD_PRUNE_THRESHOLD=0.01, FIELD_COMPACTION_MAX_SCAN=1),
    ):
        # First run stops after max_scan=1 with a live cursor.
        first = await adapter.compact(workspace_id="ws-A")
        assert first.next_offset == "CURSOR-1"

        client.scroll.reset_mock()
        # Second run resumes from the cursor.
        await adapter.compact(workspace_id="ws-A", resume_offset=first.next_offset)

    _, kwargs = client.scroll.call_args_list[0]
    assert kwargs.get("offset") == "CURSOR-1", (
        "second run must resume from the persisted cursor, not restart at None"
    )


async def test_full_pass_clears_cursor():
    """When the sweep reaches the end of the collection (Qdrant returns
    offset=None), compact returns next_offset=None so the next run starts a
    fresh full pass."""
    client = MagicMock()
    client.scroll = AsyncMock(return_value=([], None))
    client.delete = AsyncMock()
    adapter = _adapter(client)

    with patch(
        "modules.context.adapters.vector_field.config",
        MagicMock(FIELD_PRUNE_THRESHOLD=0.01, FIELD_COMPACTION_MAX_SCAN=10000),
    ):
        result = await adapter.compact(workspace_id="ws-A")

    assert result.next_offset is None
    assert result.pruned == 0


@pytest.mark.filterwarnings("ignore")
def test_compaction_cursor_store_roundtrip(monkeypatch):
    """The cursor store persists/reads the resume offset via system_settings
    (reused table), keyed per workspace."""
    import modules.context.compaction_cursor as cc
    from modules.context.compaction_cursor import (
        load_compaction_cursor,
        save_compaction_cursor,
    )

    store: dict = {}

    class _FakeSetting:
        # Class-level attrs so `SystemSetting.category == x` (evaluated before
        # the fake .filter() ignores it) resolves without the real ORM columns.
        category = None
        key = None
        value = None

        def __init__(self, category=None, key=None, value=None, **kwargs):
            self.category = category
            self.key = key
            self.value = value
            self.value_type = kwargs.get("value_type", "string")

    # The store constructs SystemSetting(...) — swap in the lightweight fake so
    # no real ORM/table is needed.
    monkeypatch.setattr(cc, "SystemSetting", _FakeSetting)

    class _FakeQuery:
        def __init__(self, rows):
            self._rows = rows

        def filter(self, *a, **k):
            return self

        def first(self):
            return self._rows[0] if self._rows else None

    class _FakeDB:
        def query(self, *a, **k):
            row = store.get("row")
            return _FakeQuery([row] if row else [])

        def add(self, obj):
            store["row"] = obj

        def flush(self):
            pass

        def commit(self):
            pass

    db = _FakeDB()
    # Empty store → None.
    assert load_compaction_cursor(db, "ws-A") is None
    # Save then read back.
    save_compaction_cursor(db, "ws-A", "CURSOR-XYZ")
    assert store["row"].value == "CURSOR-XYZ"
    assert load_compaction_cursor(db, "ws-A") == "CURSOR-XYZ"
    # Clearing (full pass) persists None.
    save_compaction_cursor(db, "ws-A", None)
    assert load_compaction_cursor(db, "ws-A") is None
