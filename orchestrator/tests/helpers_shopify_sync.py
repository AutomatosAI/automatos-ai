"""Shared pure-test harness for the PRD-189 Shopify-integrity stories.

Used by ``test_prd189_s1_catalog_preserve.py``, ``test_prd189_s2_fbt_integrity.py``
and the un-skipped J9 golden journey (``test_golden_journeys.py``). Everything
here mocks at the boundary only:

* **Composio bulk-op + signed-URL download** — a recorded JSONL fixture stands
  in for ``SHOPIFY_BULK_QUERY_OPERATION`` + the GCS download (the
  ``test_prd183_s1_catalog_webhook.py`` posture).
* **Workspace-file persistence** — the REAL :class:`GraphifyService` runs the
  real normalize/merge/cluster pipeline; only its DB-backed export/snapshot
  writers are stubbed on the instance (the ``test_prd164_flywheel.py`` posture),
  so merge semantics under test are the platform's, never a fake's.
* **SQLAlchemy session** — a chainable fake; ``flag_modified`` is a no-op for
  the non-instrumented fake workspace.

No DB, no network, no Composio, no live Shopify store.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402

_ORCHESTRATOR_ROOT = Path(__file__).resolve().parents[1]
SHOPIFY_FIXTURES_DIR = (
    _ORCHESTRATOR_ROOT / "integrations" / "shopify" / "tests" / "fixtures"
)
CATALOG_JSONL_PATH = SHOPIFY_FIXTURES_DIR / "catalog_bulkop.jsonl"
ORDERS_JSONL_PATH = SHOPIFY_FIXTURES_DIR / "orders_bulkop.jsonl"
CATALOG_RESYNC_JSONL_PATH = SHOPIFY_FIXTURES_DIR / "catalog_bulkop_resync.jsonl"


# ---------------------------------------------------------------------------
# Fake session objects (the test_prd183_s1_catalog_webhook.py shape, plus the
# .get() lookup the sync impls use)
# ---------------------------------------------------------------------------


class FakeWorkspace:
    """Workspace stand-in with a plain ``settings`` dict."""

    def __init__(self, wid: str, settings: Optional[Dict[str, Any]] = None):
        self.id = wid
        self.is_active = True
        self.settings: Dict[str, Any] = dict(
            settings or {"shopify_domain": "fixture-lighting.myshopify.com"}
        )


class FakeQuery:
    def __init__(self, workspace: Optional[FakeWorkspace]):
        self._workspace = workspace

    def filter(self, *args, **kwargs) -> "FakeQuery":
        return self

    def first(self) -> Optional[FakeWorkspace]:
        return self._workspace

    def get(self, _pk) -> Optional[FakeWorkspace]:
        return self._workspace


class FakeDb:
    def __init__(self, workspace: Optional[FakeWorkspace]):
        self._workspace = workspace

    def query(self, *args, **kwargs) -> FakeQuery:
        return FakeQuery(self._workspace)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Real GraphifyService with the persistence seam stubbed on the instance
# ---------------------------------------------------------------------------


def make_graph_service(
    workspace_id: Optional[str] = None, existing_graph=None
):
    """A REAL :class:`GraphifyService` whose DB-backed writers are inert.

    The import/merge pipeline (normalize → node_link_graph → merge branch →
    cluster → meta → cache) runs the real code; ``_export_graph`` /
    ``_write_json`` / ``_snapshot_and_diff`` / ``_write_build_report`` /
    ``_prune_history`` are AsyncMocks on the instance so nothing touches
    workspace files or the DB. ``load_graph`` serves from the seeded LRU
    cache, exactly as production does after a build.
    """
    from modules.knowledge.graph_service import GraphifyService

    svc = GraphifyService()
    svc._export_graph = AsyncMock()
    svc._write_json = AsyncMock()
    svc._snapshot_and_diff = AsyncMock(return_value=None)
    svc._write_build_report = AsyncMock()
    svc._prune_history = AsyncMock()
    if workspace_id is not None and existing_graph is not None:
        svc._cache[workspace_id] = existing_graph
    return svc


def silence_graph_primitive(monkeypatch) -> None:
    """No-op the graph primitive heartbeat emit (it opens a DB session)."""
    import modules.knowledge.graph_service as graph_service_mod

    monkeypatch.setattr(
        graph_service_mod, "_emit_graph_primitive", lambda *a, **k: None
    )


# ---------------------------------------------------------------------------
# Boundary mocks for _product_sync_impl / _orders_sync_impl
# ---------------------------------------------------------------------------


def mock_sync_boundaries(
    monkeypatch, *, graph_service, jsonl_texts: List[str]
) -> Dict[str, Any]:
    """Mock the sync impls' externals at the boundary.

    * Composio ``SHOPIFY_BULK_QUERY_OPERATION`` → a successful bulk-op payload
      with a (never-fetched) signed URL.
    * ``EntityManager`` → a fixed Composio entity id.
    * ``httpx.AsyncClient`` → each download returns the next entry of
      ``jsonl_texts`` (one per sync call, in order).
    * ``GraphifyService`` (resolved lazily inside the impls) → the prepared
      real service from :func:`make_graph_service`.
    * ``flag_modified`` → no-op (the fake workspace isn't SQLAlchemy-
      instrumented; the settings dict is asserted directly).

    Returns a ``calls`` dict recording the bulk-op queries executed.
    """
    import core.composio.client as composio_client_mod
    import core.composio.entity_manager as em_mod
    import httpx
    import sqlalchemy.orm.attributes as sa_attrs
    import modules.knowledge.graph_service as graph_service_mod

    calls: Dict[str, Any] = {"bulk_queries": [], "downloads": 0}
    downloads = list(jsonl_texts)

    class _FakeTools:
        @staticmethod
        def execute(tool_name, user_id=None, arguments=None):
            calls["bulk_queries"].append(
                {"tool": tool_name, "user_id": user_id, "arguments": arguments}
            )
            return {
                "successful": True,
                "data": {
                    "url": "https://signed.example/bulkop.jsonl",
                    "bulk_operation_id": f"gid://shopify/BulkOperation/{len(calls['bulk_queries'])}",
                    "object_count": 9,
                    "file_size": 4096,
                },
                "error": None,
                "logId": "log-fixture",
            }

    class _FakeComposioClient:
        class composio:  # noqa: N801 — mirrors the SDK attribute shape
            tools = _FakeTools()

    class _FakeEntityManager:
        def __init__(self, db):
            pass

        def get_or_create_entity(self, workspace_id):
            return {"composio_entity_id": "entity-fixture"}

    class _FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def get(self, url):
            calls["downloads"] += 1
            if not downloads:
                raise AssertionError(
                    "sync harness: more downloads than jsonl_texts provided"
                )
            return _FakeResponse(downloads.pop(0))

    monkeypatch.setattr(
        composio_client_mod, "get_composio_client", lambda: _FakeComposioClient()
    )
    monkeypatch.setattr(em_mod, "EntityManager", _FakeEntityManager)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(sa_attrs, "flag_modified", lambda obj, key: None)
    monkeypatch.setattr(
        graph_service_mod, "GraphifyService", lambda: graph_service
    )
    silence_graph_primitive(monkeypatch)
    return calls


def fbt_edges_of(graph) -> List[tuple]:
    """All ``frequently_bought_with`` edges of a NetworkX graph as
    ``(source, target, attrs)`` triples."""
    return [
        (u, v, attrs)
        for u, v, attrs in graph.edges(data=True)
        if (attrs.get("relation") or "").lower() == "frequently_bought_with"
    ]
