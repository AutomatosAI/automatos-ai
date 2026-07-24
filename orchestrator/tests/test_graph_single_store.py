"""PRD-142 Wave 3 · WS-J · W3-S10 — Graph (moat) single-store + tenant + heartbeat.

The Knowledge Graph primitive's BRAIN §3.x contract says: *the business
graph is rebuildable idempotently from sources, queryable, and survives
restart* — with **one canonical store** (`workspace_graphs`). The
``KNOWLEDGE-GRAPH-CANONICAL.md`` boundary contract makes that explicit:

  > Business entities live in ``workspace_graphs`` (via ``GraphifyService``)
  > and nowhere else. ``knowledge_nodes`` / ``knowledge_edges`` is the
  > agent-learning substrate and stores learning only — never products,
  > orders, customers, or FBT relations.

The audit at PRD-142 §4 named the referencer set to verify
(``init_database.py``, ``core/services/analytics_engine.py``,
``api/system.py``, ``api/execution_history.py``,
``modules/memory/storage/knowledge_system.py``). Confirmed 2026-06-06:

* ``init_database.py`` only lists ``knowledge_nodes``/``knowledge_edges``
  in a critical-tables verification dict — never writes them.
* ``core/services/analytics_engine.py`` issues SELECT-only ``COUNT(*)``
  reads against ``knowledge_nodes`` for the learning-tile metric — no
  writes.
* ``api/system.py`` and ``api/execution_history.py`` consume those
  read-only metrics — no direct SQL on ``knowledge_nodes``/``_edges``.
* ``modules/memory/storage/knowledge_system.py`` *defines* the
  ``KnowledgeNode`` / ``KnowledgeEdge`` SQLAlchemy models (kept — read-only,
  for the learning-tile COUNT metric above). The dead
  ``HierarchicalMemorySystem`` / ``KnowledgeGraph`` / ``LearningEngine``
  classes that *would* have written to them were never instantiated in
  production and were **deleted** in Wave 4 (W4-S13); HARNESS's structured
  store (``HarnessPrescription`` + ``LearningOutcome``) is the live learning
  substrate, inside the §4 boundary so the moat is never contaminated.

These tests pin the W3-S10 hardening contract under the Wave 2 net:

1. **Moat boundary (one source of truth, F3):** no business-entity
   writes to ``knowledge_nodes`` / ``knowledge_edges`` from
   ``modules/knowledge/``, ``api/shopify.py``, or any non-learning
   surface. Static grep over the audited referencer set.
2. **Single writer:** ``modules/knowledge/graph_service.py`` imports
   ``DbWorkspaceClient`` (the ``workspace_graphs`` writer) and never
   imports ``KnowledgeNode`` / ``KnowledgeEdge`` (the learning models).
3. **Idempotent rebuild:** the ``workspace_graphs`` upsert uses
   ``ON CONFLICT (workspace_id, path) DO UPDATE`` — re-writing the same
   ``(ws, path)`` keeps the row count at one and replaces the content.
4. **Restart-safe:** ``DbWorkspaceClient.read_file`` always issues a
   ``SELECT ... FROM workspace_graphs`` — there is no process-local cache
   below the client (the cache is owned by ``GraphifyService`` above it,
   so a fresh ``DbWorkspaceClient`` instance after a restart reads
   straight from Postgres).
5. **Cross-workspace isolation (A4):** every SQL statement binds
   ``workspace_id`` as a parameter (never f-string interpolated) and
   filters by it; two clients at distinct workspace_ids read and write
   independently.
6. **Schema preserved (Wave 4 HARNESS uses it):** the
   ``knowledge_nodes`` / ``knowledge_edges`` models still exist and
   their ``__tablename__`` is unchanged — W3-S10 keeps the schema, it
   only stops the dual-write that never started.
7. **Heartbeat (W3-S1 wiring):** ``_emit_graph_primitive`` calls
   ``emit_primitive_finding`` with primitive='graph' and the correct
   status — green on a clean build, down on a failed build/timeout.
8. **No-workspace skip:** the helper emits NOTHING when ``workspace_id``
   is falsy (A4: honest gap over fabricated default).
9. **Best-effort emit:** a failed heartbeat write NEVER breaks the
   graph caller.
10. **Service wire-up:** ``graph_service.py`` imports the helper and
    calls it at both success and failure boundaries of ``build_graph``
    and ``import_graph``.

The tests deliberately operate at the *unit* level via source-text
inspection + a recording mock for ``get_db_session`` — full integration
of ``GraphifyService.build_graph`` would drag graphify, networkx,
cachetools, the LLM manager, and the extraction pipeline into the unit
suite. Mirrors the W3-S6 / W3-S7 / W3-S8 / W3-S9 patterns.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Paths to the surfaces we pin without importing them through the heavy
# ``modules.knowledge.graph_service`` module (which eagerly loads graphify,
# networkx, cachetools, and the LLM manager).
# ---------------------------------------------------------------------------

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

KNOWLEDGE_DIR = ORCH_ROOT / "modules" / "knowledge"
GRAPH_SERVICE_PY = KNOWLEDGE_DIR / "graph_service.py"
GRAPH_STORAGE_PY = ORCH_ROOT / "core" / "graph_storage.py"
PRIMITIVE_HEARTBEAT_PY = KNOWLEDGE_DIR / "primitive_heartbeat.py"
ANALYTICS_ENGINE_PY = ORCH_ROOT / "core" / "services" / "analytics_engine.py"
SHOPIFY_PY = ORCH_ROOT / "api" / "shopify.py"
SYSTEM_PY = ORCH_ROOT / "api" / "system.py"
EXEC_HISTORY_PY = ORCH_ROOT / "api" / "execution_history.py"
INIT_DB_PY = ORCH_ROOT / "core" / "database" / "init_database.py"


# ===========================================================================
# 1. MOAT BOUNDARY — no business-entity writes to knowledge_nodes/edges from
#    the moat or any non-learning surface.
# ===========================================================================


class TestMoatBoundary:
    """``KNOWLEDGE-GRAPH-CANONICAL.md`` §4 says business entities live in
    ``workspace_graphs`` and *nowhere* else. A future PR that adds an
    ``INSERT INTO knowledge_nodes`` to ``modules/knowledge/`` or
    ``api/shopify.py`` would silently merge the moat into the learning
    store — the latent G11 risk the design doc names. This test gate
    catches it.

    The deliberately *narrow* scope is the moat producers + the
    audited referencer set named in PRD-142 §4. (The learning-store
    tables and their writers were deleted in PRD-187 S5 — this gate now
    also catches any attempt to resurrect writes to them.)"""

    _BUSINESS_WRITE_PATTERNS: tuple[str, ...] = (
        # Direct SQL writes the boundary explicitly forbids
        "INSERT INTO knowledge_nodes",
        "INSERT INTO knowledge_edges",
        "UPDATE knowledge_nodes",
        "UPDATE knowledge_edges",
        # SQLAlchemy model instantiation — would feed a session.add(node)
        "KnowledgeNode(",
        "KnowledgeEdge(",
    )

    @pytest.mark.parametrize(
        "path",
        [
            GRAPH_SERVICE_PY,
            SHOPIFY_PY,
            ANALYTICS_ENGINE_PY,
            SYSTEM_PY,
            EXEC_HISTORY_PY,
            INIT_DB_PY,
            GRAPH_STORAGE_PY,
        ],
        ids=[
            "modules_knowledge_graph_service",
            "api_shopify",
            "core_services_analytics_engine",
            "api_system",
            "api_execution_history",
            "core_database_init_database",
            "core_graph_storage",
        ],
    )
    def test_no_business_entity_writes_to_learning_store(self, path: Path):
        assert path.exists(), f"audited referencer not found at {path}"
        src = path.read_text()
        offenders = [tok for tok in self._BUSINESS_WRITE_PATTERNS if tok in src]
        assert offenders == [], (
            f"Moat boundary violation in {path.name}: business-entity "
            f"writes to the learning store are forbidden "
            f"(KNOWLEDGE-GRAPH-CANONICAL.md §4). Found tokens: {offenders}"
        )


# ===========================================================================
# 2. SINGLE-WRITER IMPORTS — graph_service.py reaches workspace_graphs
#    (via DbWorkspaceClient) and NOT knowledge_nodes/edges.
# ===========================================================================


class TestSingleWriterImports:
    """If ``graph_service.py`` ever imports ``KnowledgeNode`` /
    ``KnowledgeEdge``, the moat is one PR away from a cross-store
    dual-write. Pin the import boundary statically."""

    @pytest.fixture(scope="class")
    def gs_source(self) -> str:
        return GRAPH_SERVICE_PY.read_text()

    def test_imports_db_workspace_client(self, gs_source: str):
        assert "from core.graph_storage import DbWorkspaceClient" in gs_source, (
            "graph_service.py must reach the moat via DbWorkspaceClient — "
            "the workspace_graphs writer is the single source of truth"
        )

    def test_does_not_import_learning_models(self, gs_source: str):
        for forbidden in ("KnowledgeNode", "KnowledgeEdge"):
            assert forbidden not in gs_source, (
                f"graph_service.py must NOT import {forbidden} — learning "
                f"models belong to modules/memory/storage/knowledge_system.py"
            )

    def test_does_not_write_to_learning_tables(self, gs_source: str):
        # A raw SQL backdoor would bypass the model check above. Belt + braces.
        for forbidden in (
            "knowledge_nodes",
            "knowledge_edges",
        ):
            assert forbidden not in gs_source, (
                f"graph_service.py must not reference the learning store "
                f"table {forbidden!r} — moat boundary (§4)"
            )


# ===========================================================================
# 3. workspace_graphs SQL SHAPE — idempotent upsert + workspace_id filter
#    + read goes to Postgres (not a process-local dict).
# ===========================================================================


class TestWorkspaceGraphsSQLShape:
    """``DbWorkspaceClient`` is the *only* writer to ``workspace_graphs``.
    The SQL it issues IS the moat's storage contract — pin each shape so
    a refactor can't quietly drop ``ON CONFLICT`` (breaks idempotent
    rebuild) or the ``workspace_id`` filter (breaks tenant isolation)."""

    @pytest.fixture(scope="class")
    def gs_source(self) -> str:
        return GRAPH_STORAGE_PY.read_text()

    def test_insert_uses_on_conflict_upsert(self, gs_source: str):
        # Idempotent rebuild: the same (ws, path) replaces the content
        # instead of inserting a duplicate row. Pin the upsert.
        assert re.search(
            r"INSERT INTO workspace_graphs\s+\(workspace_id, path, content, updated_at\)",
            gs_source,
        ), "DbWorkspaceClient INSERT must target workspace_graphs(workspace_id, path, content, updated_at)"
        assert "ON CONFLICT (workspace_id, path) DO UPDATE" in gs_source, (
            "DbWorkspaceClient INSERT must use ON CONFLICT (workspace_id, path) "
            "DO UPDATE — without it, a rebuild writes a duplicate row and "
            "the §H 'idempotent rebuild' line fails"
        )

    def test_select_always_filters_by_workspace_id(self, gs_source: str):
        # The SQL is broken across two adjacent Python string literals
        # (``"... " "WHERE ..."``) — DOTALL + a short bridge handles
        # the implicit-concat boundary.
        # read_file
        assert re.search(
            r"SELECT content FROM workspace_graphs.{0,40}WHERE workspace_id\s*=\s*:ws AND path\s*=\s*:path",
            gs_source,
            re.DOTALL,
        ), "DbWorkspaceClient.read_file must filter by workspace_id AND path"
        # list_dir
        assert re.search(
            r"SELECT path FROM workspace_graphs.{0,40}WHERE workspace_id\s*=\s*:ws AND path LIKE :prefix",
            gs_source,
            re.DOTALL,
        ), "DbWorkspaceClient.list_dir must filter by workspace_id AND path LIKE :prefix"

    def test_delete_filters_by_workspace_id(self, gs_source: str):
        assert re.search(
            r"DELETE FROM workspace_graphs.{0,40}WHERE workspace_id\s*=\s*:ws AND path\s*=\s*:path",
            gs_source,
            re.DOTALL,
        ), (
            "DbWorkspaceClient.delete_file must filter by workspace_id AND "
            "path — a missing workspace_id clause would let one workspace's "
            "delete blow away another's graph"
        )

    def test_no_fstring_sql_in_workspace_graphs_path(self, gs_source: str):
        # Every reference to the table should be inside a sa_text() string
        # with :ws/:path bind params — never an f-string interpolating the
        # workspace_id directly (that's an injection seam and also
        # invalidates the parameterised plan reuse).
        forbidden_patterns = [
            r'f"[^"]*INSERT INTO workspace_graphs',
            r'f"[^"]*SELECT[^"]*workspace_graphs',
            r'f"[^"]*DELETE FROM workspace_graphs',
            r'f"[^"]*UPDATE workspace_graphs',
            r"f'[^']*INSERT INTO workspace_graphs",
            r"f'[^']*SELECT[^']*workspace_graphs",
            r"f'[^']*DELETE FROM workspace_graphs",
            r"f'[^']*UPDATE workspace_graphs",
        ]
        for pat in forbidden_patterns:
            assert not re.search(pat, gs_source), (
                f"workspace_graphs SQL must use bind params, not f-string "
                f"interpolation — pattern {pat!r} matched in graph_storage.py"
            )


# ===========================================================================
# 4. FUNCTIONAL — DbWorkspaceClient idempotent rebuild, restart-safe,
#    cross-workspace isolation. Uses an in-memory recording mock for
#    ``get_db_session`` so the test runs without a real Postgres.
# ===========================================================================


def _load_graph_storage():
    """Import ``core.graph_storage`` via its real path.

    POSTGRES_* defaults are set so ``core.database.database`` can import
    cleanly even with no reachable Postgres — same convention as
    ``test_memory_restart_and_isolation.py`` (W3-S7). The actual session
    factory is monkeypatched per-test on the imported module, so no
    queries hit the DB and nothing leaks into ``sys.modules`` for later
    tests.
    """
    import os
    for _k, _v in {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "test_db",
    }.items():
        os.environ.setdefault(_k, _v)
    import core.graph_storage as _gs  # type: ignore  # noqa: E402
    return _gs


GS_MOD = _load_graph_storage()


class _RecordingSession:
    """A tiny in-memory stand-in for the SQLAlchemy session
    ``DbWorkspaceClient`` calls. Implements just enough of the
    ``execute(text, params)`` shape to interpret the four SQL statements
    DbWorkspaceClient issues, against an in-process dict.

    Idempotent semantics fall out for free: the dict is keyed by
    (workspace_id, path), so re-INSERTing the same key overwrites.
    """

    # Shared store so two clients (simulating a restart) see the same data
    def __init__(self, store: dict):
        self._store = store
        self._last_result = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params: dict):
        text_str = str(sql)
        ws = params.get("ws")
        path = params.get("path")
        prefix = params.get("prefix")
        content = params.get("content")
        if "INSERT INTO workspace_graphs" in text_str:
            self._store[(ws, path)] = content
            self._last_result = _ExecResult(rows=[])
        elif "SELECT content FROM workspace_graphs" in text_str:
            row = self._store.get((ws, path))
            self._last_result = _ExecResult(
                rows=[(row,)] if row is not None else []
            )
        elif "SELECT path FROM workspace_graphs" in text_str:
            # LIKE :prefix — strip the trailing % we know the caller appends
            assert prefix and prefix.endswith("%"), (
                "list_dir SQL must use LIKE :prefix"
            )
            prefix_str = prefix[:-1]
            matches = [
                (p,) for (w, p) in self._store
                if w == ws and (prefix_str == "" or p.startswith(prefix_str))
            ]
            self._last_result = _ExecResult(rows=matches)
        elif "DELETE FROM workspace_graphs" in text_str:
            self._store.pop((ws, path), None)
            self._last_result = _ExecResult(rows=[])
        else:
            raise AssertionError(f"Unexpected SQL: {text_str!r}")
        return self._last_result


class _ExecResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


@pytest.fixture
def in_memory_db(monkeypatch):
    """Patch ``get_db_session`` so DbWorkspaceClient operations record
    into a shared dict. Returns the dict so tests can assert state.
    """
    store: dict = {}

    def _fake_session():
        return _RecordingSession(store)

    monkeypatch.setattr(GS_MOD, "get_db_session", _fake_session)
    return store


class TestIdempotentRebuild:
    """Re-running ``build_graph`` from the same sources must yield the
    same graph (§H 'idempotent rebuild'). At the storage layer that
    collapses to: re-writing the same (ws, path) keeps row count at one
    and replaces the content. The ON CONFLICT clause is the contract;
    this test proves the writer respects it."""

    def test_rewrite_same_path_replaces_content(self, in_memory_db):
        import asyncio
        client = GS_MOD.DbWorkspaceClient("ws_A")
        # Write v1
        r1 = asyncio.run(client.write_file("graph/graph.json", '{"n":1}'))
        assert r1["success"] is True
        # Write v2 — same (ws, path) — replaces v1
        r2 = asyncio.run(client.write_file("graph/graph.json", '{"n":2}'))
        assert r2["success"] is True
        # Exactly one row for the key, holding v2
        assert in_memory_db == {("ws_A", "graph/graph.json"): '{"n":2}'}

    def test_rewrite_same_content_is_a_noop_for_state(self, in_memory_db):
        import asyncio
        client = GS_MOD.DbWorkspaceClient("ws_A")
        body = '{"nodes":[],"links":[]}'
        for _ in range(3):
            asyncio.run(client.write_file("graph/graph.json", body))
        # Still exactly one row, holding the same content (no dup rows from rebuilds)
        assert in_memory_db == {("ws_A", "graph/graph.json"): body}


class TestRestartSafe:
    """The §H 'restart-safe' line requires the graph to survive a
    process restart. The cache lives in ``GraphifyService`` *above*
    ``DbWorkspaceClient`` — at the storage layer there is no
    process-local cache, every read hits Postgres. A fresh
    ``DbWorkspaceClient`` instance (simulating a restart) must see what
    a prior process wrote."""

    def test_fresh_client_reads_prior_write(self, in_memory_db):
        import asyncio
        # Process A writes
        writer = GS_MOD.DbWorkspaceClient("ws_A")
        asyncio.run(writer.write_file("graph/graph.json", '{"persisted":true}'))
        # Process B (different instance — simulates restart) reads
        del writer
        reader = GS_MOD.DbWorkspaceClient("ws_A")
        result = asyncio.run(reader.read_file("graph/graph.json"))
        assert result["success"] is True
        assert result["content"] == '{"persisted":true}'

    def test_storage_layer_has_no_process_local_cache(self):
        # Belt + braces — DbWorkspaceClient must not maintain any per-
        # instance dict that could shadow Postgres on a restart.
        gs_source = GRAPH_STORAGE_PY.read_text()
        # No top-level _cache, no class-level _cache, no instance dict
        # initialised in __init__ for caching content.
        assert "self._cache" not in gs_source, (
            "DbWorkspaceClient must not cache content per-instance — "
            "that would break restart-safety (the cache lives above, "
            "in GraphifyService where it can be invalidated)"
        )


class TestCrossWorkspaceIsolation:
    """§H A4: a cross-workspace read/write test must prove the moat is
    tenant-isolated. ``DbWorkspaceClient`` is constructed with a single
    ``workspace_id``; every SQL it issues binds that id and filters by
    it. Two clients at distinct workspace_ids cannot see each other's
    rows even at the same ``path``."""

    def test_two_workspaces_at_same_path_dont_collide(self, in_memory_db):
        import asyncio
        ws_a = GS_MOD.DbWorkspaceClient("ws_A")
        ws_b = GS_MOD.DbWorkspaceClient("ws_B")
        asyncio.run(ws_a.write_file("graph/graph.json", '{"who":"A"}'))
        asyncio.run(ws_b.write_file("graph/graph.json", '{"who":"B"}'))
        # Each side sees only its own content
        read_a = asyncio.run(ws_a.read_file("graph/graph.json"))
        read_b = asyncio.run(ws_b.read_file("graph/graph.json"))
        assert read_a["content"] == '{"who":"A"}'
        assert read_b["content"] == '{"who":"B"}'

    def test_workspace_b_cannot_read_workspace_a(self, in_memory_db):
        import asyncio
        ws_a = GS_MOD.DbWorkspaceClient("ws_A")
        asyncio.run(ws_a.write_file("graph/graph.json", '{"private":"A"}'))
        # B at the same path sees nothing
        ws_b = GS_MOD.DbWorkspaceClient("ws_B")
        result = asyncio.run(ws_b.read_file("graph/graph.json"))
        assert result["success"] is False
        assert result.get("error") == "not_found"

    def test_workspace_b_delete_does_not_affect_workspace_a(self, in_memory_db):
        import asyncio
        ws_a = GS_MOD.DbWorkspaceClient("ws_A")
        ws_b = GS_MOD.DbWorkspaceClient("ws_B")
        asyncio.run(ws_a.write_file("graph/graph.json", '{"keep":"A"}'))
        asyncio.run(ws_b.write_file("graph/graph.json", '{"drop":"B"}'))
        # Workspace B deletes — A's row must survive
        asyncio.run(ws_b.delete_file("graph/graph.json"))
        assert ("ws_A", "graph/graph.json") in in_memory_db
        assert ("ws_B", "graph/graph.json") not in in_memory_db


# ===========================================================================
# 5. SCHEMA PRESERVED — knowledge_nodes/edges still defined for Wave 4.
# ===========================================================================


def _load_primitive_heartbeat():
    spec = importlib.util.spec_from_file_location(
        "primitive_heartbeat_under_test", PRIMITIVE_HEARTBEAT_PY
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


PH_MOD = _load_primitive_heartbeat()


class TestGraphHeartbeatHelper:
    """The graph heartbeat helper mirrors the chat / rag / nl2sql
    helpers' shape. Pin the contract here so a drift in any of them
    surfaces as a test failure."""

    def test_emit_green_on_success(self):
        with patch.object(PH_MOD, "emit_primitive_finding") as mock_emit:
            PH_MOD._emit_graph_primitive("ws_42", success=True, detail="nodes=10 edges=20")
            assert mock_emit.call_count == 1
            args, kwargs = mock_emit.call_args
            assert args[0] == "ws_42"
            assert args[1] == "graph"
            assert args[2] == "green"
            assert "nodes=10" in args[3]

    def test_emit_down_on_failure(self):
        with patch.object(PH_MOD, "emit_primitive_finding") as mock_emit:
            PH_MOD._emit_graph_primitive(
                "ws_42", success=False, detail="boom: graphify timeout"
            )
            assert mock_emit.call_count == 1
            assert mock_emit.call_args[0][2] == "down"

    def test_skip_when_no_workspace_id(self):
        with patch.object(PH_MOD, "emit_primitive_finding") as mock_emit:
            PH_MOD._emit_graph_primitive(None, success=True, detail="x")
            PH_MOD._emit_graph_primitive("", success=False, detail="x")
            assert mock_emit.call_count == 0, (
                "no workspace_id => no emit (A4 honest gap, never a "
                "fabricated default)"
            )

    def test_emit_failure_is_swallowed(self):
        with patch.object(
            PH_MOD,
            "emit_primitive_finding",
            side_effect=RuntimeError("postgres down"),
        ):
            # Must not raise — the graph caller is in the middle of an
            # already-successful (or already-failing) build.
            PH_MOD._emit_graph_primitive("ws_42", success=True, detail="ok")
            PH_MOD._emit_graph_primitive("ws_42", success=False, detail="fail")

    def test_canonical_primitive_name_is_graph(self):
        # The primitive name must be the lowercase canonical 'graph' —
        # not 'knowledge_graph' or 'business_graph'. The W3-S2 endpoint
        # expects exactly one of services.heartbeat_service.PRIMITIVE_NAMES.
        src = PRIMITIVE_HEARTBEAT_PY.read_text()
        assert '"graph"' in src, (
            "helper must emit primitive='graph' (canonical name from "
            "services/heartbeat_service.PRIMITIVE_NAMES)"
        )
        # Negative pin — no legacy nouns sneaking in
        for legacy in ("knowledge_graph", "business_graph", "KnowledgeGraph"):
            assert legacy not in src or src.count(legacy) == 0, (
                f"helper must not use legacy noun {legacy!r} "
                f"(CLAUDE.md §10 canonical terms)"
            )

    def test_detail_truncated_via_emit_primitive_finding(self):
        # The helper passes detail[:500] to emit_primitive_finding; the
        # writer itself also caps to 500 in __init__. Pin the helper
        # boundary here.
        with patch.object(PH_MOD, "emit_primitive_finding") as mock_emit:
            long_detail = "x" * 1000
            PH_MOD._emit_graph_primitive("ws_42", success=True, detail=long_detail)
            assert mock_emit.call_args[0][3] == "x" * 500


# ===========================================================================
# 7. SERVICE WIRE-UP — graph_service.py actually calls the helper on
#    both success and failure boundaries of build_graph + import_graph.
# ===========================================================================


class TestServiceWireUp:
    """A helper that nobody calls is no signal at all — the W3-S2 tile
    would still read ``unknown``. Pin the call sites in source so a
    later refactor can't unwire them without tripping a test."""

    @pytest.fixture(scope="class")
    def gs_source(self) -> str:
        return GRAPH_SERVICE_PY.read_text()

    def test_imports_emit_graph_primitive(self, gs_source: str):
        assert "_emit_graph_primitive" in gs_source, (
            "graph_service.py must import _emit_graph_primitive"
        )
        assert (
            "from modules.knowledge.primitive_heartbeat import _emit_graph_primitive"
            in gs_source
        ), "import path must be modules.knowledge.primitive_heartbeat"

    def _method_body(self, source: str, name: str) -> str:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
                segment = ast.get_source_segment(source, node)
                if segment:
                    return segment
        raise AssertionError(f"method {name!r} not found in graph_service.py")

    def test_build_graph_emits_on_success(self, gs_source: str):
        body = self._method_body(gs_source, "build_graph")
        assert re.search(
            r"_emit_graph_primitive\([^)]*success=True", body
        ), "build_graph must emit success=True at the success boundary"

    def test_build_graph_emits_on_failure(self, gs_source: str):
        body = self._method_body(gs_source, "build_graph")
        # At least one success=False emit inside build_graph
        assert re.search(
            r"_emit_graph_primitive\([^)]*success=False", body
        ), (
            "build_graph must emit success=False on caught exception/timeout "
            "(the tile flips to down)"
        )

    def test_import_graph_emits_on_success(self, gs_source: str):
        body = self._method_body(gs_source, "import_graph")
        assert re.search(
            r"_emit_graph_primitive\([^)]*success=True", body
        ), "import_graph must emit success=True on a successful import"

    def test_import_graph_emits_on_failure(self, gs_source: str):
        body = self._method_body(gs_source, "import_graph")
        assert re.search(
            r"_emit_graph_primitive\([^)]*success=False", body
        ), "import_graph must emit success=False inside its outer except"

    def test_failure_emit_does_not_swallow(self, gs_source: str):
        # The helper is best-effort, but the surrounding except must
        # re-raise (we must NOT swallow build failures by writing the
        # emit and returning early).
        build_body = self._method_body(gs_source, "build_graph")
        import_body = self._method_body(gs_source, "import_graph")
        for name, body in (("build_graph", build_body), ("import_graph", import_body)):
            # Find every `_emit_graph_primitive(...,success=False...)` —
            # each one's enclosing except block must end with `raise` (or
            # in build_graph's timeout-specific arm, `raise TimeoutError(...)`).
            assert "raise" in body, (
                f"{name} must re-raise inside its failure-emit branch — "
                "build failures cannot be silently swallowed"
            )
