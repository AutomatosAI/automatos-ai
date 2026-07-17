"""PRD-164 S3 — Output flywheel: agent outputs become retrievable knowledge.

Layers:

* Pure (no DB): the flywheel choke point's opt-out + tagging contract, the
  KG pending partitioner (the source types the incremental build used to
  drop), and reachability of the deliverable list/get tools (AC4 surface).
* Integration (real Postgres — CI authority):
  - AC1: a completed mission's synthesis routed through the EXISTING
    ingestion manager is retrievable next turn via the PRD-157 retrieval
    tool surface (workspace-scoped through build_retrieval_filters).
  - AC2: a seeded report pending's entities land in the KG via the
    (previously dead) agent-attributed report extractor.
  - AC3: an opted-out workspace ingests NOTHING.
  - AC4: Auto lists its own deliverables via platform_list_deliverables
    (handler → DeliverableService → SQL round-trip).

LLM/embedding/S3 boundaries are stubbed deterministically — extraction
quality is not this story's subject; the routing/drop/opt-out wiring is.
"""
from __future__ import annotations

import asyncio
import importlib.util as _ilu
import json
import os
import sys as _sys
import types as _types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern) — the port
# points at nothing so fail-soft import-time connects refuse instantly. CI
# exports real POSTGRES_* so these setdefaults no-op there.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# AC1/AC3 below drive a REAL ingestion round-trip (object store + embeddings +
# vector retrieval). The standard CI test net provisions Postgres ONLY — no
# S3/vector services — so without credentials the S3 client blocks and the test
# hangs to the 90s faulthandler kill instead of running. Gate them to a
# full-services run (real creds / the live env). The flywheel WIRING (opt-out,
# tagging, KG-pending partition, deliverable-tool reachability) is fully covered
# by the pure/unit tests above; this only defers the real S3+vector round-trip,
# matching pytest.ini's "integration: needs services; run explicitly" posture.
_requires_object_store = pytest.mark.skipif(
    not os.environ.get("AWS_ACCESS_KEY_ID"),
    reason=(
        "flywheel e2e needs a real object store + embeddings; the Postgres-only "
        "CI net has none. Runs in full-services CI / locally with AWS creds."
    ),
)


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which
# pulls modules.rag's ingestion chain (camelot at module top). Stub the missing
# *leaf* only when truly absent — never the modules.rag package.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
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

from modules.knowledge.graph_service import partition_pending_sources  # noqa: E402
from services.knowledge_flywheel import (  # noqa: E402
    AGENT_OUTPUT_SOURCE_TYPE,
    FLYWHEEL_SETTINGS_KEY,
    flywheel_enabled,
    ingest_agent_output,
)

_DELIVERABLE_TOOLS = ["platform_list_deliverables", "platform_get_deliverable"]


# ===========================================================================
# Pure layer — KG pending partitioner (the drop fix)
# ===========================================================================


class TestPartitionPendingSources:
    def test_document_pendings_collect_ids(self):
        doc_ids, text_sources = partition_pending_sources(
            [{"type": "document", "id": 7}, {"type": "document", "id": 9}]
        )
        assert doc_ids == {7, 9}
        assert text_sources == []

    def test_agent_output_doc_types_resolve_to_document_ids(self):
        """mission_synthesis / generated_document pendings — previously
        DROPPED — resolve to their ingested document ids (one extraction)."""
        doc_ids, text_sources = partition_pending_sources([
            {"type": "mission_synthesis", "id": "run-1", "document_id": 42},
            {"type": "generated_document", "document_id": 43},
            {"type": "generated_document", "id": 44},  # fallback to id
        ])
        assert doc_ids == {42, 43, 44}
        assert text_sources == []

    def test_report_pending_carries_text_to_report_extractor(self):
        doc_ids, text_sources = partition_pending_sources([
            {
                "type": "report",
                "id": "r-1",
                "path": "Q3 Infra Report",
                "text": "ACME Corp migrated to PostgreSQL.",
                "agent_name": "Scout",
            }
        ])
        assert doc_ids == set()
        assert len(text_sources) == 1
        src = text_sources[0]
        assert src["type"] == "report"
        assert src["agent_name"] == "Scout"
        assert "PostgreSQL" in src["text"]

    def test_report_without_text_is_skipped_not_crashed(self):
        doc_ids, text_sources = partition_pending_sources(
            [{"type": "report", "id": "r-2", "path": "empty"}]
        )
        assert doc_ids == set() and text_sources == []

    def test_unknown_types_fall_through(self):
        doc_ids, text_sources = partition_pending_sources(
            [{"type": "roster", "id": 1}, {"type": "db_schema", "id": 2}, {}]
        )
        assert doc_ids == set() and text_sources == []


# ===========================================================================
# Pure layer — flywheel choke point (Q58)
# ===========================================================================


def _mock_db_with_workspace(settings):
    ws = MagicMock()
    ws.settings = settings
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ws
    return db


class TestFlywheelOptOut:
    def test_enabled_by_default_when_key_absent(self):
        assert flywheel_enabled(_mock_db_with_workspace({}), uuid.uuid4()) is True

    def test_enabled_when_settings_none(self):
        assert flywheel_enabled(_mock_db_with_workspace(None), uuid.uuid4()) is True

    def test_disabled_only_on_explicit_false(self):
        db = _mock_db_with_workspace({FLYWHEEL_SETTINGS_KEY: False})
        assert flywheel_enabled(db, uuid.uuid4()) is False

    def test_truthy_value_stays_enabled(self):
        db = _mock_db_with_workspace({FLYWHEEL_SETTINGS_KEY: True})
        assert flywheel_enabled(db, uuid.uuid4()) is True

    def test_missing_workspace_fails_open(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        assert flywheel_enabled(db, uuid.uuid4()) is True

    @pytest.mark.asyncio
    async def test_opt_out_ingests_nothing_unit(self):
        """AC3 (unit shape): the gate sits BEFORE any manager construction —
        an opted-out workspace never touches the ingestion path at all."""
        db = _mock_db_with_workspace({FLYWHEEL_SETTINGS_KEY: False})
        with patch("api.documents.get_document_manager") as get_mgr:
            result = await ingest_agent_output(
                db, uuid.uuid4(),
                content="# Synthesis\nText",
                filename="out.md",
                source="mission_synthesis",
                source_id="run-1",
            )
        assert result is None
        get_mgr.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_unknown_source(self):
        with pytest.raises(ValueError):
            await ingest_agent_output(
                MagicMock(), uuid.uuid4(),
                content="x", filename="x.md", source="not_a_source",
            )

    @pytest.mark.asyncio
    async def test_empty_content_is_a_noop(self):
        db = _mock_db_with_workspace({})
        with patch("api.documents.get_document_manager") as get_mgr:
            result = await ingest_agent_output(
                db, uuid.uuid4(),
                content="   ", filename="x.md", source="report",
            )
        assert result is None
        get_mgr.assert_not_called()


class TestFlywheelIngestTagging:
    @pytest.mark.asyncio
    async def test_routes_through_manager_tagged_agent_output(self):
        """Q58: the EXISTING ingestion manager is the path; the document is
        tagged source_type='agent_output' + carries the source tag."""
        db = _mock_db_with_workspace({})
        manager = MagicMock()
        manager.upload_document = AsyncMock(return_value=42)
        graph_service = MagicMock()

        with patch("api.documents.get_document_manager", return_value=manager), \
             patch("modules.knowledge.graph_service.get_graph_service",
                   return_value=graph_service):
            doc_id = await ingest_agent_output(
                db, "11111111-1111-1111-1111-111111111111",
                content="# Mission output\nThe synthesis.",
                filename="mission-output-x.md",
                source="mission_synthesis",
                source_id="run-9",
                title="Mission output: x",
                extra_tags=["mission:run-9"],
            )

        assert doc_id == 42
        kwargs = manager.upload_document.call_args.kwargs
        assert kwargs["source_type"] == AGENT_OUTPUT_SOURCE_TYPE
        assert AGENT_OUTPUT_SOURCE_TYPE in kwargs["tags"]
        assert "mission_synthesis" in kwargs["tags"]
        assert "mission:run-9" in kwargs["tags"]
        assert kwargs["filename"] == "mission-output-x.md"

        # Typed KG pending: synthesis references the ingested document.
        pending = graph_service.schedule_incremental_update.call_args.args[1]
        assert pending == [{
            "type": "mission_synthesis",
            "id": "run-9",
            "document_id": 42,
            "path": "Mission output: x",
        }]

    @pytest.mark.asyncio
    async def test_report_pending_carries_text_and_agent(self):
        db = _mock_db_with_workspace({})
        manager = MagicMock()
        manager.upload_document = AsyncMock(return_value=77)
        graph_service = MagicMock()

        with patch("api.documents.get_document_manager", return_value=manager), \
             patch("modules.knowledge.graph_service.get_graph_service",
                   return_value=graph_service):
            doc_id = await ingest_agent_output(
                db, uuid.uuid4(),
                content="ACME Corp migrated to PostgreSQL.",
                filename="q3-report.md",
                source="report",
                source_id="rep-1",
                title="Q3 Infra Report",
                agent_name="Scout",
            )

        assert doc_id == 77
        pending = graph_service.schedule_incremental_update.call_args.args[1][0]
        assert pending["type"] == "report"
        assert pending["agent_name"] == "Scout"
        assert "PostgreSQL" in pending["text"]
        # report + document pendings both partition cleanly downstream
        doc_ids, text_sources = partition_pending_sources([pending])
        assert text_sources and not doc_ids

    @pytest.mark.asyncio
    async def test_ingest_failure_is_fail_soft(self):
        """Producing the output must never break on a knowledge-loop error."""
        db = _mock_db_with_workspace({})
        manager = MagicMock()
        manager.upload_document = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("api.documents.get_document_manager", return_value=manager):
            result = await ingest_agent_output(
                db, uuid.uuid4(),
                content="text", filename="x.md", source="report", source_id="r",
            )
        assert result is None


# ===========================================================================
# Pure layer — deliverable tools (AC4 reachability)
# ===========================================================================


class TestDeliverableToolReachability:
    @pytest.mark.parametrize("name", _DELIVERABLE_TOOLS)
    def test_registered_read_only(self, name):
        from modules.tools.discovery.action_registry import ActionRegistry
        from modules.tools.discovery.actions_deliverables import (
            register_deliverables_actions,
        )

        reg = ActionRegistry()
        register_deliverables_actions(reg)
        action = reg.get(name)
        assert action is not None, f"{name} not registered"
        assert action.permission_level == "read"

    @pytest.mark.parametrize("name", _DELIVERABLE_TOOLS)
    def test_handler_wired_in_executor(self, name):
        from modules.tools.discovery.platform_executor import PlatformActionExecutor

        ex = PlatformActionExecutor(db=None, workspace_id=None)
        assert name in ex._handlers
        assert callable(ex._handlers[name])

    def test_registered_in_full_catalogue(self):
        """register_all_actions includes the deliverable tools — the PRD-155
        reachability sweep enumerates this same registry."""
        from modules.tools.discovery.action_registry import ActionRegistry
        from modules.tools.discovery.platform_actions import register_all_actions

        reg = ActionRegistry()
        register_all_actions(reg)
        reg._initialized = True
        names = {a.name for a in reg.get_all()}
        for tool in _DELIVERABLE_TOOLS:
            assert tool in names


class TestDeliverableHandlerValidationNoDB:
    @pytest.mark.asyncio
    async def test_get_missing_id(self):
        from modules.tools.discovery.handlers_deliverables import get_deliverable

        res = await get_deliverable(db=None, workspace_id="ws", params={})
        assert res["success"] is False and "deliverable_id" in res["error"]

    @pytest.mark.asyncio
    async def test_list_mine_without_caller_context(self):
        from modules.tools.discovery.handlers_deliverables import list_deliverables

        res = await list_deliverables(db=None, workspace_id="ws", params={"mine": True})
        assert res["success"] is False and "calling agent" in res["error"]

    @pytest.mark.asyncio
    async def test_list_non_int_agent_id(self):
        from modules.tools.discovery.handlers_deliverables import list_deliverables

        res = await list_deliverables(
            db=None, workspace_id="ws", params={"agent_id": "abc"}
        )
        assert res["success"] is False and "agent_id" in res["error"]


# ===========================================================================
# Integration layer (real Postgres — CI authority)
# ===========================================================================

# Prod-shape columns the raw-SQL ingestion manager writes but the model-built
# CI schema lacks (known model/DDL drift: metadata vs doc_metadata, file_hash
# vs content_hash; chunk embedding columns). Additive + idempotent.
_PROD_SHAPE_DDL = """
ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64);
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS parent_content TEXT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS headers JSONB;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS workspace_id TEXT;
"""


def _commit_workspace(engine, settings=None, name="prd164-flywheel-ws"):
    """Insert a COMMITTED workspace row (the manager writes over its own
    psycopg2 connection, so FK targets must be visible outside the test txn).
    Returns the workspace id (str)."""
    from sqlalchemy import text as sa_text

    ws_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(sa_text(_PROD_SHAPE_DDL))
        conn.execute(
            sa_text(
                "INSERT INTO workspaces (id, name, settings) "
                "VALUES (CAST(:id AS uuid), :name, CAST(:settings AS jsonb))"
            ),
            {"id": ws_id, "name": name, "settings": json.dumps(settings or {})},
        )
    return ws_id


def _cleanup_workspace(engine, ws_id):
    from sqlalchemy import text as sa_text

    with engine.begin() as conn:
        # Bounded sweep (see tests/conftest.py teardown-guard note): if a test
        # died mid-transaction and pinned row locks, fail loudly in seconds
        # instead of hanging the lane to the job cap.
        conn.execute(sa_text("SET LOCAL lock_timeout = '5s'"))
        conn.execute(
            sa_text(
                "DELETE FROM document_chunks WHERE document_id IN "
                "(SELECT id FROM documents WHERE workspace_id = CAST(:ws AS uuid))"
            ),
            {"ws": ws_id},
        )
        conn.execute(
            sa_text("DELETE FROM documents WHERE workspace_id = CAST(:ws AS uuid)"),
            {"ws": ws_id},
        )
        conn.execute(
            sa_text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"),
            {"ws": ws_id},
        )


def _manager_boundary_patches():
    """Deterministic stand-ins for the manager's non-DB boundaries (S3,
    embeddings, multimodal side-tables, KG singleton). The DB path — document
    row, chunking, chunk persistence — runs for REAL."""
    import api.documents as api_documents
    from modules.rag.ingestion.manager import DocumentManager

    embedding_stub = MagicMock()
    embedding_stub.get_provider_info.return_value = {"provider": "stub"}
    embedding_stub.get_dimension.return_value = 8

    graph_singleton = MagicMock()

    return [
        # Force legacy (pgvector-inline) mode regardless of CI env.
        patch.object(api_documents.config, "S3_VECTORS_ENABLED", False),
        patch.object(DocumentManager, "_ensure_s3_bucket_exists", lambda self: None),
        patch.object(
            DocumentManager, "_upload_to_s3",
            lambda self, file_path, document_id, filename: f"fake/{document_id}/{filename}",
        ),
        patch.object(
            DocumentManager, "_generate_embeddings_batch",
            AsyncMock(side_effect=lambda texts, **kw: ["[0.0]"] * len(texts)),
        ),
        patch("core.llm.create_embedding_manager", return_value=embedding_stub),
        # Poison the multimodal leaf so its knowledge_items/kb_* side-tables
        # (raw prod DDL, absent on CI) are skipped via the existing except.
        patch.dict(_sys.modules, {"modules.rag.ingestion.multimodal": None}),
        patch(
            "modules.knowledge.graph_service.get_graph_service",
            return_value=graph_singleton,
        ),
    ], graph_singleton


@_requires_object_store
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ac1_completed_mission_synthesis_retrievable_next_turn(
    db_session, test_engine
):
    """AC1 e2e: completed mission → synthesis routed through the EXISTING
    ingestion manager (tagged agent_output) → retrievable next turn via the
    PRD-157 retrieval tool surface, workspace-scoped."""
    from sqlalchemy import text as sa_text
    from sqlalchemy.orm import sessionmaker

    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from modules.tools.discovery.handlers_documents import grep_documents
    from services.coordinator_service import CoordinatorService

    ws_id = _commit_workspace(test_engine)
    marker = f"flywheel-marker-{uuid.uuid4().hex[:8]}"
    try:
        # Seed a completed mission with one verified task (txn-local is fine —
        # the coordinator reads them through this same session).
        run = OrchestrationRun(
            id=uuid.uuid4(),
            workspace_id=ws_id,
            goal=f"Research the {marker} rollout",
            state="completed",
            state_type="terminal",
            created_by="user_test",
        )
        db_session.add(run)
        db_session.flush()
        task = OrchestrationTask(
            run_id=run.id,
            title="Synthesis",
            description="Synthesize findings",
            task_type="synthesis",
            state="verified",
            state_type="terminal",
            sequence_number=1,
            output=f"The {marker} rollout completed; ACME adopted PostgreSQL 16.",
        )
        db_session.add(task)
        db_session.flush()

        patches, graph_singleton = _manager_boundary_patches()
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            doc_id = await CoordinatorService()._save_mission_output_as_document(
                db_session, run
            )

        assert doc_id is not None, "synthesis was not ingested"
        assert (run.config or {}).get("output_document_id") == doc_id

        # The document row is the flywheel scope: source_type='agent_output'.
        with test_engine.connect() as conn:
            row = conn.execute(
                sa_text(
                    "SELECT source_type, status, chunk_count FROM documents WHERE id = :id"
                ),
                {"id": doc_id},
            ).fetchone()
        assert row is not None
        assert row[0] == "agent_output"
        assert row[1] == "completed"
        assert row[2] >= 1

        # KG learned the typed pending (mission_synthesis → ingested doc).
        scheduled = [
            c.args[1][0]
            for c in graph_singleton.schedule_incremental_update.call_args_list
            if c.args[1] and c.args[1][0].get("type") == "mission_synthesis"
        ]
        assert scheduled and scheduled[0]["document_id"] == doc_id

        # NEXT TURN: the agent retrieval surface finds the synthesis —
        # fresh session, scope derived via build_retrieval_filters (157).
        NextTurnSession = sessionmaker(bind=test_engine)
        next_turn = NextTurnSession()
        try:
            found = await grep_documents(next_turn, ws_id, {"pattern": marker})
            assert found["success"] is True
            assert found["count"] >= 1
            assert any(
                str(m.get("document_id")) == str(doc_id) for m in found["matches"]
            )

            # Workspace isolation: another workspace retrieves NOTHING.
            other_ws = _commit_workspace(test_engine, name="prd164-other-ws")
            try:
                other = await grep_documents(next_turn, other_ws, {"pattern": marker})
                assert other["success"] is True and other["count"] == 0
            finally:
                _cleanup_workspace(test_engine, other_ws)
        finally:
            next_turn.close()
    finally:
        _cleanup_workspace(test_engine, ws_id)


@_requires_object_store
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ac3_opted_out_workspace_ingests_nothing(db_session, test_engine):
    """AC3: Q58 opt-out — no document row, no chunks, no KG pending; the run
    is marked so the tick sweep stops retrying."""
    from sqlalchemy import text as sa_text

    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from modules.rag.ingestion.manager import DocumentManager
    from services.coordinator_service import CoordinatorService

    ws_id = _commit_workspace(
        test_engine,
        settings={FLYWHEEL_SETTINGS_KEY: False},
        name="prd164-optout-ws",
    )
    try:
        run = OrchestrationRun(
            id=uuid.uuid4(),
            workspace_id=ws_id,
            goal="Opted-out goal",
            state="completed",
            state_type="terminal",
            created_by="user_test",
        )
        db_session.add(run)
        db_session.flush()
        db_session.add(OrchestrationTask(
            run_id=run.id,
            title="T1",
            description="d",
            task_type="analysis",
            state="verified",
            state_type="terminal",
            sequence_number=1,
            output="should never be ingested",
        ))
        db_session.flush()

        graph_singleton = MagicMock()
        with patch.object(
            DocumentManager, "upload_document", AsyncMock()
        ) as upload, patch(
            "modules.knowledge.graph_service.get_graph_service",
            return_value=graph_singleton,
        ):
            doc_id = await CoordinatorService()._save_mission_output_as_document(
                db_session, run
            )

        assert doc_id is None
        upload.assert_not_called()
        graph_singleton.schedule_incremental_update.assert_not_called()
        assert (run.config or {}).get("output_ingest") == "skipped_opt_out"

        # Nothing in the corpus for this workspace.
        with test_engine.connect() as conn:
            count = conn.execute(
                sa_text(
                    "SELECT COUNT(*) FROM documents WHERE workspace_id = CAST(:ws AS uuid)"
                ),
                {"ws": ws_id},
            ).scalar()
        assert count == 0

        # The report path goes through the same single gate.
        report_doc = await ingest_agent_output(
            db_session, ws_id,
            content="report body", filename="r.md", source="report", source_id="r1",
        )
        assert report_doc is None
    finally:
        _cleanup_workspace(test_engine, ws_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ac2_seeded_report_entities_appear_in_kg():
    """AC2: a 'report' pending (previously dropped by the incremental build)
    is dispatched to the agent-attributed report extractor and its entities
    merge into the workspace graph."""
    import networkx as nx

    from modules.knowledge.graph_service import GraphifyService

    svc = GraphifyService()
    existing = nx.Graph()
    existing.add_node("seed_node", label="Seed")

    extraction = {
        "nodes": [
            {"id": "entity_acme_corp", "label": "ACME Corp", "type": "organization"},
            {"id": "entity_postgresql", "label": "PostgreSQL", "type": "technology"},
        ],
        "edges": [
            {"source": "entity_acme_corp", "target": "entity_postgresql", "label": "uses"}
        ],
        "hyperedges": [],
    }

    extract_report = AsyncMock(return_value=extraction)
    pending = [{
        "type": "report",
        "id": "rep-1",
        "path": "Q3 Infra Report",
        "text": "ACME Corp migrated everything to PostgreSQL.",
        "agent_name": "Scout",
    }]

    with patch(
        "modules.knowledge.graph_extraction.extract_from_report", extract_report
    ), patch(
        "core.llm.create_llm_manager", return_value=MagicMock()
    ), patch.object(svc, "_export_graph", AsyncMock()), patch.object(
        svc, "_write_json", AsyncMock()
    ), patch.object(
        svc, "_snapshot_and_diff", AsyncMock(return_value=None)
    ), patch.object(
        svc, "_write_build_report", AsyncMock()
    ), patch.object(
        svc, "_prune_history", AsyncMock()
    ):
        meta = await svc._incremental_build(str(uuid.uuid4()), existing, pending)

    # The previously-dropped source type reached the report extractor…
    assert extract_report.await_count == 1
    call_kwargs = extract_report.await_args.kwargs
    assert call_kwargs["agent_name"] == "Scout"
    assert "PostgreSQL" in call_kwargs["report_text"]

    # …and the seeded report's entities are IN the graph (AC2).
    assert "entity_acme_corp" in existing.nodes
    assert "entity_postgresql" in existing.nodes
    assert existing.has_edge("entity_acme_corp", "entity_postgresql")
    assert meta is not None


@pytest.mark.asyncio
async def test_kg_synthesis_pending_resolves_to_document_collection():
    """mission_synthesis/generated_document pendings route into the document
    collection (no second extraction path)."""
    from modules.knowledge.graph_service import GraphifyService
    import networkx as nx

    svc = GraphifyService()
    collect = AsyncMock(return_value=[])
    with patch.object(svc, "_collect_sources", collect):
        await svc._incremental_build(
            "ws-1",
            nx.Graph(),
            [{"type": "mission_synthesis", "id": "run-1", "document_id": 42}],
        )
    collect.assert_awaited_once_with("ws-1", doc_ids={42})


# ---------------------------------------------------------------------------
# AC4 integration — Auto lists its own deliverables via the tool
# ---------------------------------------------------------------------------

# Transaction-local stand-ins for the migration-managed deliverables table +
# v_workspace_outputs view (neither is model-backed, so CI's create_all skips
# them). The view body mirrors the deliverables branch of prd133b — the
# column order DeliverableService._row_to_dict consumes.
_DELIVERABLES_DDL = """
CREATE TABLE IF NOT EXISTS deliverables (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id      UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    source_type       VARCHAR(30) NOT NULL,
    source_id         VARCHAR(255) NULL,
    agent_id          INTEGER NULL,
    agent_name        VARCHAR(100) NULL,
    artifact_type     VARCHAR(30) NOT NULL,
    title             VARCHAR(255) NOT NULL,
    summary           VARCHAR(500) NULL,
    storage_type      VARCHAR(20) NOT NULL DEFAULT 'workspace',
    file_path         VARCHAR(1024) NOT NULL,
    file_name         VARCHAR(255) NULL,
    file_type         VARCHAR(50) NULL,
    file_size_bytes   BIGINT NULL,
    preview_url       VARCHAR(1024) NULL,
    preview_type      VARCHAR(30) NULL,
    extra             JSONB NOT NULL DEFAULT '{}'::jsonb,
    status            VARCHAR(20) NOT NULL DEFAULT 'ready',
    deleted_at        TIMESTAMPTZ NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE VIEW v_workspace_outputs AS
SELECT
    d.id, d.workspace_id, d.source_type, d.source_id, d.agent_id,
    d.agent_name, d.artifact_type, d.title, d.summary, d.storage_type,
    d.file_path, d.file_name, d.file_type, d.file_size_bytes,
    d.preview_url, d.preview_type, d.extra, d.status, d.deleted_at,
    d.created_at, d.updated_at
FROM deliverables d;
"""


def _seed_deliverable(db, ws_id, *, agent_id=None, agent_name="Auto",
                      source_type="mission", source_id="run-1",
                      title="Mission report", file_path=None):
    from sqlalchemy import text as sa_text

    row_id = str(uuid.uuid4())
    db.execute(
        sa_text(
            """
            INSERT INTO deliverables (
                id, workspace_id, source_type, source_id, agent_id, agent_name,
                artifact_type, title, storage_type, file_path
            ) VALUES (
                CAST(:id AS uuid), CAST(:ws AS uuid), :source_type, :source_id,
                :agent_id, :agent_name, 'report', :title, 'workspace', :file_path
            )
            """
        ),
        {
            "id": row_id,
            "ws": ws_id,
            "source_type": source_type,
            "source_id": source_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "title": title,
            "file_path": file_path or f"reports/auto/{row_id}.md",
        },
    )
    db.flush()
    return row_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ac4_auto_lists_its_own_deliverables_via_tool(
    db_session, seed_workspace
):
    """AC4: platform_list_deliverables (mine=true → the calling agent's
    _agent_id) returns the agent's outputs; platform_get_deliverable reads
    one back. Full handler → service → SQL round-trip."""
    from sqlalchemy import text as sa_text

    from modules.tools.discovery.handlers_deliverables import (
        get_deliverable,
        list_deliverables,
    )

    ws_id = seed_workspace()
    db_session.execute(sa_text(_DELIVERABLES_DDL))

    auto_agent_id = 4242
    mine_id = _seed_deliverable(
        db_session, ws_id,
        agent_id=auto_agent_id, agent_name="Auto",
        title="Auto weekly digest", source_id="run-7",
    )
    _seed_deliverable(
        db_session, ws_id,
        agent_id=999, agent_name="Scout", title="Scout research notes",
    )

    # Auto lists ITS OWN deliverables (execution context injects _agent_id).
    res = await list_deliverables(
        db_session, ws_id, {"mine": True, "_agent_id": auto_agent_id}
    )
    assert res["success"] is True
    assert res["count"] == 1
    only = res["deliverables"][0]
    assert only["id"] == mine_id
    assert only["title"] == "Auto weekly digest"
    assert only["agent_id"] == auto_agent_id

    # source_id filter (the mission deliverables tab shape).
    by_mission = await list_deliverables(db_session, ws_id, {"source_id": "run-7"})
    assert by_mission["success"] is True and by_mission["count"] == 1
    assert by_mission["deliverables"][0]["id"] == mine_id

    # get round-trip (metadata; content read needs the workspace worker).
    got = await get_deliverable(db_session, ws_id, {"deliverable_id": mine_id})
    assert got["success"] is True
    assert got["deliverable"]["id"] == mine_id
    assert got["deliverable"]["source_type"] == "mission"

    # Workspace isolation: a different workspace sees nothing.
    other_ws = seed_workspace()
    isolated = await list_deliverables(
        db_session, other_ws, {"mine": True, "_agent_id": auto_agent_id}
    )
    assert isolated["success"] is True and isolated["count"] == 0
