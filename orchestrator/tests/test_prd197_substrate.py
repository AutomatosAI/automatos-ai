"""PRD-197 (reslimmed) — vector-substrate consolidation.

* S1 — the F079 zombie store is DELETED: SearchService / EnhancedVectorStore /
  ContextRetrievalEngine (plus the orphaned SearchConfig) were a parallel
  retrieval stack the live path never imported, whose "cosine" ranking used
  the L2 operator and whose namesake table was dropped in PRD-135. A guard
  pins that no import of them ever comes back.
* S2 — settings-plane truth: every reader queries the canonical
  ``(embeddings, provider|model|cache_dir|dimensions|rerank_model)`` rows that
  PRD-136's migration/seeder/admin-card actually produce. The pre-rename long
  names matched nothing, so the admin embeddings card was a placebo.
* S5 — the open-core/local edition gets a working document read leg:
  ``PgVectorLocalBackend`` over ``document_chunks.embedding`` (which "legacy
  pgvector mode" ingestion already populates when S3 is off), selected by
  config in ``RAGService._get_doc_backend``; S3 Vectors stays the SaaS plane.

All pure — DB/boto3 mocked at the boundary.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

import config as config_mod  # noqa: E402
from modules.search.vector_store import get_vector_store  # noqa: E402
from modules.search.vector_store.backends.pgvector_local_backend import (  # noqa: E402
    PgVectorLocalBackend,
)


# ---------------------------------------------------------------------------
# S1 — the zombie store is gone and stays gone
# ---------------------------------------------------------------------------

_ZOMBIE_FILES = [
    "modules/search/service.py",
    "modules/search/config.py",
    "modules/search/vector_store/store.py",
    "modules/search/retrieval/context_retrieval_engine.py",
    "modules/search/retrieval/__init__.py",
]

_ZOMBIE_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+[\w.]*"
    r"(?:EnhancedVectorStore|ContextRetrievalEngine"
    r"|modules\.search\.service\b|modules\.search\.retrieval\b"
    r"|modules\.search\.config\b)"
    r"|^\s*from\s+modules\.search\s+import\s+.*(?:SearchService|EnhancedVectorStore|ContextRetrievalEngine|SearchConfig)",
    re.M,
)


def test_zombie_store_files_deleted():
    for rel in _ZOMBIE_FILES:
        assert not (_orchestrator_root / rel).exists(), f"{rel} resurrected"


def test_no_zombie_store_importers():
    """No import statement anywhere may reference the deleted layer — comment
    mentions ('do not resurrect') are allowed; imports are not."""
    offenders = []
    for py in _orchestrator_root.rglob("*.py"):
        rel = py.relative_to(_orchestrator_root).as_posix()
        if rel.startswith((".venv", "venv", "node_modules")) or rel == "tests/test_prd197_substrate.py":
            continue
        if _ZOMBIE_IMPORT.search(py.read_text(errors="ignore")):
            offenders.append(rel)
    assert offenders == []


def test_search_package_exports_live_surface_only():
    import modules.search as search_pkg

    assert hasattr(search_pkg, "ContextOptimizer")
    for dead in ("SearchService", "EnhancedVectorStore", "ContextRetrievalEngine", "SearchConfig"):
        assert not hasattr(search_pkg, dead), f"{dead} re-exported"


# ---------------------------------------------------------------------------
# S1/S5 — the facade's pgvector leg is the local backend now
# ---------------------------------------------------------------------------

def test_facade_pgvector_returns_local_backend():
    backend = get_vector_store(backend="pgvector", workspace_id="ws-a")
    assert isinstance(backend, PgVectorLocalBackend)
    assert backend.workspace_id == "ws-a"


def test_facade_pgvector_requires_workspace():
    with pytest.raises(ValueError, match="workspace_id"):
        get_vector_store(backend="pgvector")


# ---------------------------------------------------------------------------
# S5 — pgvector-local read leg
# ---------------------------------------------------------------------------

def test_local_backend_fail_closed_on_mismatched_filter():
    backend = PgVectorLocalBackend(workspace_id="ws-a")
    # Refuses before ever touching the DB — same contract as the S3 backend.
    assert backend.search([0.1] * 4, filters={"workspace_id": "ws-b"}) == []


def test_local_backend_requires_workspace():
    with pytest.raises(ValueError, match="workspace_id"):
        PgVectorLocalBackend(workspace_id="")


def test_local_backend_maps_rows_to_s3_contract(monkeypatch):
    """Rows come back in the exact dict shape RAGService._get_candidates
    consumes from the S3 backend (key/score/content/metadata/external ids)."""
    import core.database.database as db_mod

    rows = [
        SimpleNamespace(
            document_id=7, chunk_index=0, content="chunk text",
            file_name="notes.md", file_path="/docs/notes.md", similarity=0.91,
        )
    ]
    fake_db = MagicMock()
    fake_db.execute.return_value.fetchall.return_value = rows
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: fake_db)

    out = PgVectorLocalBackend(workspace_id="ws-a").search([0.1] * 4, limit=5, min_score=0.5)

    assert len(out) == 1
    hit = out[0]
    assert hit["key"] == "doc_7_chunk_0"
    assert hit["score"] == 0.91
    assert hit["content"] == "chunk text"
    assert hit["file_name"] == "notes.md"
    assert hit["external_file_id"] == "7"
    assert hit["metadata"]["workspace_id"] == "ws-a"
    assert hit["metadata"]["document_id"] == "7"
    fake_db.close.assert_called_once()


# ---------------------------------------------------------------------------
# S5 — RAGService selects the backend by edition config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_edition_rag_constructs_local_backend(monkeypatch):
    import modules.rag.service as svc

    monkeypatch.setattr(config_mod.config, "S3_VECTORS_ENABLED", False, raising=False)
    rag = svc.RAGService.__new__(svc.RAGService)  # bypass __init__ (no DB)
    rag._doc_backends = {}

    backend = await rag._get_doc_backend("ws-local")
    assert isinstance(backend, PgVectorLocalBackend)


@pytest.mark.asyncio
async def test_saas_edition_rag_constructs_s3_backend(monkeypatch):
    import modules.rag.service as svc

    class FakeS3Backend:
        def __init__(self, workspace_id):
            self.workspace_id = workspace_id

        async def initialize(self):
            pass

    monkeypatch.setattr(config_mod.config, "S3_VECTORS_ENABLED", True, raising=False)
    monkeypatch.setattr(
        "modules.search.vector_store.backends.s3_vectors_backend.S3VectorsBackend",
        FakeS3Backend,
    )
    rag = svc.RAGService.__new__(svc.RAGService)
    rag._doc_backends = {}

    backend = await rag._get_doc_backend("ws-saas")
    assert isinstance(backend, FakeS3Backend)


# ---------------------------------------------------------------------------
# S2 — settings-plane truth: readers use the canonical (embeddings, *) keys
# ---------------------------------------------------------------------------

# Only settings READS by the old names are defects (get_system_setting("...")
# or an ORM filter on SystemSetting.key). Bare quoted strings stay legal —
# api/context.py uses "embedding_model" as a JSON response field, which is a
# different namespace entirely.
_STALE_SETTINGS_READS = re.compile(
    r"(?:get_system_setting\(\s*|SystemSetting\.key\s*==\s*)"
    r"[\"'](?:embedding_provider|embedding_model|embedding_cache_dir"
    r"|vector_store_dimensions|rag_rerank_model)[\"']"
)


def test_no_stale_settings_reads():
    """PRD-136 renamed the embedding rows to (embeddings, provider|model|
    cache_dir|dimensions|rerank_model). No prod code may query the old long
    names — those rows no longer exist, so such a read is a silent placebo
    (the reader falls back to config and the admin card changes nothing).
    The PRD-136 migration itself holds the rename map and is exempt."""
    offenders = []
    for py in _orchestrator_root.rglob("*.py"):
        rel = py.relative_to(_orchestrator_root).as_posix()
        if rel.startswith((".venv", "venv", "node_modules", "tests/", "alembic/")):
            continue
        if _STALE_SETTINGS_READS.search(py.read_text(errors="ignore")):
            offenders.append(rel)
    assert offenders == []


def test_embeddings_settings_roundtrip(monkeypatch):
    """The admin embeddings card saves the canonical (embeddings, provider|
    model|cache_dir|dimensions) rows — _load_provider must configure the
    provider from exactly those rows, proving the card is no longer a
    placebo."""
    import core.llm.manager as mgr
    import core.llm.embedding_manager as em

    rows = {
        ("embeddings", "provider"): "openai",
        ("embeddings", "model"): "text-embedding-3-large",
        ("embeddings", "cache_dir"): "/tmp/emb-cache",
        ("embeddings", "dimensions"): "1536",
    }
    monkeypatch.setattr(
        mgr,
        "get_system_setting",
        lambda category, key, default=None: rows.get((category, key), default),
    )
    monkeypatch.setattr(em, "get_credential_field", lambda *a, **k: "sk-test")

    captured = {}

    def _capture(self, cfg):
        captured["cfg"] = cfg
        return MagicMock()

    monkeypatch.setattr(em.EmbeddingManager, "_create_provider", _capture)

    manager = em.EmbeddingManager()
    manager._load_provider()

    cfg = captured["cfg"]
    assert cfg.provider == em.EmbeddingProvider("openai")
    assert cfg.model == "text-embedding-3-large"
    assert cfg.dimension == 1536
    assert cfg.cache_dir == "/tmp/emb-cache"
