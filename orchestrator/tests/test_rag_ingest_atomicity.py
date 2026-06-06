"""PRD-142 Wave 3 · WS-J · W3-S8 — RAG atomicity + tenant + heartbeat.

The RAG primitive's BRAIN §3.x contract says: *ingest→chunk→embed→index is
atomic per document, and delete removes the vector*. The §H DoD adds:
*failure path tested, tenant-isolated, observable via heartbeat finding*.

Today the dual-write to Postgres ``document_chunks`` + the S3 Vectors backend
isn't fully atomic — if the S3 batch insert raises, the in-flight Postgres
chunk inserts rely on garbage-collection rollback rather than an explicit
``conn.rollback()``, and if the chunks commit but a *later* step (status
update, graph hook) fails we'd leak S3 vectors for a doc Postgres
will mark FAILED. The delete path likewise tears down Postgres rows but
never asked the S3 backend to drop the vectors — the §H "delete removes the
vector" line goes one way only.

These tests pin the W3-S8 hardening contract:

* **Atomicity (failure path):** when ``S3Vectors.add_documents`` raises,
  ``_persist_chunks_and_vectors`` rolls the chunk INSERTs back via the
  caller's connection and re-raises — no half-indexed doc.
* **Workspace isolation (A4):** every chunk INSERT carries the document's
  ``workspace_id``; the helper never lets a None / cross-workspace value
  through, so the search-side workspace filter has a real boundary to
  enforce.
* **Delete removes the vector (§H):** the manager's ``delete_document``
  asks the S3 backend to drop the chunk-vectors for the doc when S3
  Vectors mode is on, never just-postgres.
* **Heartbeat (W3-S1 wiring):** the manager emits a ``rag`` /
  ``green`` finding on ingest success and a ``rag`` / ``down`` finding
  on failure via the W3-S1 helper — un-instrumented before this story.

Pure unit shape — heavy infra (boto3, camelot, ``docx``) is stubbed in
``sys.modules`` before importing ``modules.rag.ingestion.manager`` so the
test runs without AWS / DOCX deps, mirroring the
``test_memory_single_write_path`` pattern.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Path / env / heavy-module stubs — must run BEFORE importing the manager.
# ---------------------------------------------------------------------------

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


# boto3 — used at module level in manager.py. Only ``boto3.client`` is
# referenced from the test surface; everything else can be a MagicMock.
if "boto3" not in sys.modules:
    _boto = types.ModuleType("boto3")
    _boto.client = MagicMock(return_value=MagicMock())
    sys.modules["boto3"] = _boto

# botocore.exceptions.ClientError — manager.py imports it at top-level.
if "botocore" not in sys.modules:
    _botocore = types.ModuleType("botocore")
    _botocore_exc = types.ModuleType("botocore.exceptions")

    class _ClientError(Exception):
        pass

    _botocore_exc.ClientError = _ClientError
    sys.modules["botocore"] = _botocore
    sys.modules["botocore.exceptions"] = _botocore_exc

# camelot — optional PDF dep, absent in test env.
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

# magic stubs are typically present, but a couple of CI envs lack it.
if "magic" not in sys.modules:
    _magic = types.ModuleType("magic")
    _magic.from_file = lambda *a, **k: "text/plain"
    sys.modules["magic"] = _magic

# pytesseract / PIL / pandas are pulled by the multimodal processors which
# manager.py reaches only at *call* time — but ``modules.rag.__init__`` does
# top-level eager imports, so stubbing here is the simplest unblock.
sys.modules.setdefault("pytesseract", types.ModuleType("pytesseract"))

if "pandas" not in sys.modules:
    _pandas = types.ModuleType("pandas")
    _pandas.DataFrame = type("DataFrame", (), {})  # type-annotation marker
    sys.modules["pandas"] = _pandas

if "PIL" not in sys.modules:
    _pil = types.ModuleType("PIL")
    _pil_img = types.ModuleType("PIL.Image")
    _pil_img.Image = type("Image", (), {})  # type-annotation marker
    _pil.Image = _pil_img
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _pil_img


# ---------------------------------------------------------------------------
# Parent-package isolation. ``modules.rag.__init__`` eagerly loads the full
# service + chunking + ingestion + multimodal chain (network-touching
# embedding manager construction + heavy tokenizers); the manager module
# itself is testable in isolation, so stub the parent packages BEFORE
# importing it. Mirrors the consumers.chatbot stub pattern in
# ``test_memory_single_write_path``.
# ---------------------------------------------------------------------------

for _pkg in ("modules", "modules.rag", "modules.rag.ingestion"):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = [str(ORCH_ROOT / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub

# semantic_chunker / multimodal are referenced from manager.py inside ``try``
# blocks — they fail to import gracefully. Pre-stub so the try block fails
# fast without dragging in pandas/PIL/Image internals.
sys.modules.setdefault(
    "modules.rag.chunking", types.ModuleType("modules.rag.chunking")
)
_sc_stub = types.ModuleType("modules.rag.chunking.semantic_chunker")
_sc_stub.SemanticChunker = MagicMock()
_sc_stub.ChunkingStrategy = MagicMock()
sys.modules.setdefault("modules.rag.chunking.semantic_chunker", _sc_stub)

# Pre-stub multimodal so import-time references resolve without pytesseract /
# pandas internals being walked.
_mm_stub = types.ModuleType("modules.rag.ingestion.multimodal")
_mm_stub.TableProcessor = MagicMock
_mm_stub.FormulaProcessor = MagicMock
_mm_stub.ImageProcessor = MagicMock
_mm_stub.MultimodalDocumentProcessor = MagicMock
sys.modules.setdefault("modules.rag.ingestion.multimodal", _mm_stub)

import importlib.util  # noqa: E402

_manager_path = ORCH_ROOT / "modules" / "rag" / "ingestion" / "manager.py"
_spec = importlib.util.spec_from_file_location(
    "modules.rag.ingestion.manager", str(_manager_path)
)
_manager_mod = importlib.util.module_from_spec(_spec)
sys.modules["modules.rag.ingestion.manager"] = _manager_mod
_spec.loader.exec_module(_manager_mod)

DocumentChunk = _manager_mod.DocumentChunk
DocumentManager = _manager_mod.DocumentManager
DocumentStatus = _manager_mod.DocumentStatus
DocumentType = _manager_mod.DocumentType


def _bare_manager(*, workspace_id="ws-rag-1", use_s3=True, s3_backend=None):
    """Build a DocumentManager via ``__new__`` so we never run ``__init__``
    (which hits S3, embeddings, and the embedding manager). Attributes
    needed by the methods under test are set explicitly; everything else
    is left undefined so an accidental side-effect blows up loudly.
    """
    mgr = DocumentManager.__new__(DocumentManager)
    mgr.workspace_id = workspace_id
    mgr.use_s3_vectors = use_s3
    mgr._s3_backend = s3_backend if s3_backend is not None else MagicMock()
    mgr._db_initialized = True
    mgr.db_config = {"host": "x"}
    mgr.s3_bucket = "test-bucket"
    return mgr


def _chunk(idx=0, *, document_id=42, content="hello world chunk content", embedding=None):
    c = DocumentChunk(
        document_id=document_id,
        chunk_index=idx,
        content=content,
        embedding=embedding or [0.1] * 8,
        metadata={"chunk_index": idx},
        parent_content=None,
    )
    if not hasattr(c, "headers"):
        c.headers = None
    return c


class _RecordingCursor:
    """Records INSERT statements so we can assert against the dual-write."""

    def __init__(self):
        self.executes: list[tuple[str, tuple]] = []
        self.rowcount = 0

    def execute(self, stmt, params=None):
        self.executes.append((str(stmt), tuple(params) if params else ()))
        # Simulate "1 row deleted/inserted" for DELETE assertions.
        if "DELETE" in str(stmt).upper():
            self.rowcount = 1
        return None

    def close(self):
        pass


class _RecordingConn:
    """Records commit/rollback so atomicity can be asserted."""

    def __init__(self):
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self, *_a, **_k):
        return _RecordingCursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


# ===========================================================================
# 1. ATOMICITY — failure path: S3 raises => conn.rollback() + re-raise.
# ===========================================================================


@pytest.mark.asyncio
async def test_persist_rolls_back_chunks_when_s3_raises():
    """If the S3 batch insert raises, the persist helper MUST call
    ``conn.rollback()`` before re-raising so the in-flight Postgres chunk
    INSERTs don't survive without their vector counterparts. Today's code
    relies on garbage-collection rollback — that's not atomic; this pins
    explicit rollback under the §H DoD's "atomic per document" line.
    """
    backend = MagicMock()
    backend.initialize = AsyncMock(return_value=None)
    backend.add_documents = MagicMock(side_effect=RuntimeError("S3 vector outage"))

    mgr = _bare_manager(s3_backend=backend)
    conn = _RecordingConn()
    cursor = _RecordingCursor()

    chunks = [_chunk(0), _chunk(1)]

    with pytest.raises(RuntimeError, match="S3 vector outage"):
        await mgr._persist_chunks_and_vectors(
            conn=conn,
            cursor=cursor,
            document_id=42,
            workspace_id="ws-rag-1",
            filename="report.pdf",
            file_path="/tmp/report.pdf",
            file_type=DocumentType.PDF,
            filtered_chunks=chunks,
        )

    # The helper queued two chunk INSERTs on the cursor (S3-vectors mode),
    # then attempted the S3 batch — and on failure rolled back BEFORE
    # bubbling the exception up.
    insert_stmts = [
        s for s, _ in cursor.executes if "INSERT INTO document_chunks" in s
    ]
    assert len(insert_stmts) == 2, (
        f"expected 2 chunk INSERTs before S3 call; got {len(insert_stmts)}"
    )
    assert conn.rollbacks == 1, (
        f"expected exactly 1 explicit rollback on S3 failure; got "
        f"{conn.rollbacks}"
    )
    assert conn.commits == 0, "no commit may run on the failure path"


@pytest.mark.asyncio
async def test_persist_returns_vector_ids_on_happy_path():
    """On success the helper returns the list of S3 vector_ids that were
    stored, so the caller can clean them up if a later step (status update,
    graph hook) fails. No rollback fires."""
    backend = MagicMock()
    backend.initialize = AsyncMock(return_value=None)
    backend.add_documents = MagicMock(
        return_value=["doc_42_chunk_0", "doc_42_chunk_1"]
    )

    mgr = _bare_manager(s3_backend=backend)
    conn = _RecordingConn()
    cursor = _RecordingCursor()

    vector_ids = await mgr._persist_chunks_and_vectors(
        conn=conn,
        cursor=cursor,
        document_id=42,
        workspace_id="ws-rag-1",
        filename="report.pdf",
        file_path="/tmp/report.pdf",
        file_type=DocumentType.PDF,
        filtered_chunks=[_chunk(0), _chunk(1)],
    )

    assert vector_ids == ["doc_42_chunk_0", "doc_42_chunk_1"]
    assert conn.rollbacks == 0
    # The helper does NOT commit — the caller controls the outer transaction.
    assert conn.commits == 0


@pytest.mark.asyncio
async def test_persist_no_s3_mode_returns_empty_vector_ids():
    """Legacy pgvector mode (``use_s3_vectors=False``): the chunk INSERT
    carries the embedding column itself, so there's no separate S3 step
    and no vector_ids to track for cleanup."""
    mgr = _bare_manager(use_s3=False, s3_backend=None)
    conn = _RecordingConn()
    cursor = _RecordingCursor()

    vector_ids = await mgr._persist_chunks_and_vectors(
        conn=conn,
        cursor=cursor,
        document_id=99,
        workspace_id="ws-rag-1",
        filename="note.txt",
        file_path="/tmp/note.txt",
        file_type=DocumentType.TEXT,
        filtered_chunks=[_chunk(0)],
    )

    assert vector_ids == []
    # One INSERT with embedding inline; rollback never fires.
    insert_stmts = [
        s for s, _ in cursor.executes if "INSERT INTO document_chunks" in s
    ]
    assert len(insert_stmts) == 1
    assert ", embedding," in insert_stmts[0] or "embedding" in insert_stmts[0]
    assert conn.rollbacks == 0


# ===========================================================================
# 2. WORKSPACE ISOLATION (A4) — chunk INSERTs carry workspace_id; the S3
#    side stores under a workspace-scoped backend.
# ===========================================================================


@pytest.mark.asyncio
async def test_persist_includes_workspace_id_in_chunk_insert():
    """Every chunk INSERT MUST include the document's workspace_id as a
    bound parameter — the read-side filter at ``service.py::_get_candidates``
    relies on it. A None / fabricated workspace_id is the cross-tenant
    leak primitive (A4)."""
    backend = MagicMock()
    backend.initialize = AsyncMock(return_value=None)
    backend.add_documents = MagicMock(return_value=["k1"])

    mgr = _bare_manager(workspace_id="ws-A", s3_backend=backend)
    conn = _RecordingConn()
    cursor = _RecordingCursor()

    await mgr._persist_chunks_and_vectors(
        conn=conn,
        cursor=cursor,
        document_id=7,
        workspace_id="ws-A",  # passed through from the doc row
        filename="x.txt",
        file_path="/tmp/x.txt",
        file_type=DocumentType.TEXT,
        filtered_chunks=[_chunk(0)],
    )

    chunk_inserts = [
        (s, p) for s, p in cursor.executes if "INSERT INTO document_chunks" in s
    ]
    assert chunk_inserts, "expected at least one chunk INSERT"
    for stmt, params in chunk_inserts:
        assert "workspace_id" in stmt, (
            "INSERT INTO document_chunks must name the workspace_id column"
        )
        assert "ws-A" in params, (
            f"workspace_id 'ws-A' must be bound in the chunk INSERT params; "
            f"got params={params!r}"
        )


@pytest.mark.asyncio
async def test_s3_vector_metadata_uses_workspace_scoped_backend():
    """The S3 backend instance carries the workspace_id (it's how the
    per-tenant bucket / key prefix is derived in MockS3VectorsBackend +
    S3VectorsBackend). Ingest passes ``workspace_id`` through to the
    document records given to add_documents so the metadata stays
    workspace-tagged downstream."""
    backend = MagicMock()
    backend.workspace_id = "ws-A"
    backend.initialize = AsyncMock(return_value=None)
    captured: dict = {}

    def _capture(documents, embeddings):
        captured["documents"] = list(documents)
        return [f"k{i}" for i in range(len(documents))]

    backend.add_documents = MagicMock(side_effect=_capture)

    mgr = _bare_manager(workspace_id="ws-A", s3_backend=backend)
    conn = _RecordingConn()
    cursor = _RecordingCursor()

    await mgr._persist_chunks_and_vectors(
        conn=conn,
        cursor=cursor,
        document_id=11,
        workspace_id="ws-A",
        filename="ws-a-doc.txt",
        file_path="/tmp/ws-a-doc.txt",
        file_type=DocumentType.TEXT,
        filtered_chunks=[_chunk(0)],
    )

    assert captured["documents"], "expected the helper to feed documents to S3"
    for doc in captured["documents"]:
        # Each doc dict the helper passes to S3 names its workspace_id so the
        # backend's metadata fan-out can re-assert it on the vector record.
        assert doc.get("workspace_id") == "ws-A", (
            f"S3 add_documents payload must carry workspace_id='ws-A'; got "
            f"{doc!r}"
        )


# ===========================================================================
# 3. DELETE REMOVES THE VECTOR — §H "delete removes the vector".
# ===========================================================================


def test_delete_document_removes_s3_vectors_when_s3_mode():
    """``delete_document`` must drop the chunk-vectors from the S3 backend
    when ``use_s3_vectors=True`` — today the manager tears down Postgres
    rows only, leaving orphan vectors per workspace per deleted doc.
    """
    backend = MagicMock()
    backend.delete_documents = MagicMock(return_value=3)

    mgr = _bare_manager(s3_backend=backend)

    # Replace psycopg2.connect with a recording conn so the postgres delete
    # path runs without a real DB.
    fake_conn = _RecordingConn()
    _manager_mod.psycopg2.connect = MagicMock(return_value=fake_conn)

    ok = mgr.delete_document(42)
    assert ok is True

    backend.delete_documents.assert_called_once_with("42"), (
        "S3 backend must be asked to drop the chunk-vectors for the doc"
    )


def test_delete_document_skips_s3_when_legacy_pgvector_mode():
    """In legacy pgvector mode there are no S3 vectors to drop — the
    delete path must NOT call delete_documents on a backend that does not
    own those vectors."""
    backend = MagicMock()
    mgr = _bare_manager(use_s3=False, s3_backend=backend)

    _manager_mod.psycopg2.connect = MagicMock(return_value=_RecordingConn())

    mgr.delete_document(7)

    backend.delete_documents.assert_not_called()


def test_delete_document_does_not_raise_when_s3_cleanup_fails():
    """Postgres commit has already happened by the time we ask S3 to drop
    its vectors — an S3 outage at that point is a logged warning, not a
    user-visible 500. Otherwise a flaky S3 would block deletes the
    foreground row was already gone for."""
    backend = MagicMock()
    backend.delete_documents = MagicMock(side_effect=RuntimeError("S3 outage"))
    mgr = _bare_manager(s3_backend=backend)

    _manager_mod.psycopg2.connect = MagicMock(return_value=_RecordingConn())

    # Must NOT raise — S3 failure is best-effort after the postgres commit.
    ok = mgr.delete_document(42)
    assert ok is True


# ===========================================================================
# 4. HEARTBEAT (W3-S1 wiring) — rag/green on success, rag/down on failure.
# ===========================================================================


def test_emit_ingest_heartbeat_green_on_success(monkeypatch):
    """The manager exposes a small wrapper that calls
    ``services.heartbeat_service.emit_primitive_finding`` with primitive='rag'
    and the appropriate status. This is the W3-S1 pathfinder wiring for
    the RAG tile."""
    calls: list[tuple] = []

    def _fake_emit(workspace_id, primitive, status, detail=""):
        calls.append((workspace_id, primitive, status, detail))
        return True

    fake_module = types.ModuleType("services.heartbeat_service")
    fake_module.emit_primitive_finding = _fake_emit
    monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_module)

    mgr = _bare_manager(workspace_id="ws-rag-2")
    mgr._emit_ingest_heartbeat(document_id=42, status="green", detail="2 chunks")

    assert calls == [("ws-rag-2", "rag", "green", "2 chunks")]


def test_emit_ingest_heartbeat_down_on_failure(monkeypatch):
    """A failure surfaces as 'down' so the per-primitive tile reflects the
    real outage state instead of staying stale on the last 'green'."""
    calls: list[tuple] = []

    def _fake_emit(workspace_id, primitive, status, detail=""):
        calls.append((workspace_id, primitive, status, detail))
        return True

    fake_module = types.ModuleType("services.heartbeat_service")
    fake_module.emit_primitive_finding = _fake_emit
    monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_module)

    mgr = _bare_manager(workspace_id="ws-rag-2")
    mgr._emit_ingest_heartbeat(
        document_id=99, status="down", detail="S3 vector outage"
    )

    assert calls == [("ws-rag-2", "rag", "down", "S3 vector outage")]


def test_emit_ingest_heartbeat_swallows_emit_failure(monkeypatch):
    """A failed heartbeat write NEVER breaks the ingest path — the §H
    'observable' line is best-effort. (Mirrors the
    ``emit_primitive_finding`` swallow guarantee already in W3-S1.)"""

    def _raising_emit(*_a, **_k):
        raise RuntimeError("heartbeat DB unreachable")

    fake_module = types.ModuleType("services.heartbeat_service")
    fake_module.emit_primitive_finding = _raising_emit
    monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_module)

    mgr = _bare_manager(workspace_id="ws-rag-2")
    # MUST NOT raise.
    mgr._emit_ingest_heartbeat(
        document_id=42, status="green", detail="ok"
    )


def test_emit_ingest_heartbeat_skips_when_no_workspace(monkeypatch):
    """No workspace_id => no per-workspace tile to update — silent skip,
    not a defaulted ws ID (A4: workspace_id is carried, never guessed)."""
    calls: list[tuple] = []

    def _fake_emit(*a, **k):
        calls.append((a, k))
        return True

    fake_module = types.ModuleType("services.heartbeat_service")
    fake_module.emit_primitive_finding = _fake_emit
    monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_module)

    mgr = _bare_manager(workspace_id=None)
    mgr._emit_ingest_heartbeat(document_id=42, status="green")

    assert calls == [], (
        "no workspace_id must mean no emit; never fabricate a default"
    )
