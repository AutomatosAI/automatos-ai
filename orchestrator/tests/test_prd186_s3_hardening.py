"""PRD-186 (revised) — S3 Vectors hardening: the live shared-bucket plane
becomes un-silently-breakable.

* S1 — tenant isolation is TOTAL on the shared bucket: search() drops
  unlabeled (missing/None workspace_id) hits, not just mismatched ones; the
  dead unstamped write seam (a method no backend defines) is deleted; the
  disconnect-time delete is file-scoped instead of an index-wide sweep that
  reached every tenant's vectors.
* S2 — a confirmed index-vs-config dimension mismatch raises typed instead of
  logging and serving wrong geometry; the _verify_or_recreate misnomer is gone
  (the code never recreated anything — by design).
* S3 — assert_vector_config_integrity() pins the shared-bucket config rules
  (enabled ⇒ bucket set, dimension positive; a placeholder-less shared bucket
  is the VALID live shape) and validate_security carries it into the hard-fail
  boot phase.
* S4 — the flat-or-better parity gate: fresh live numbers vs the frozen
  aliased baseline, regression flagged beyond ε, exit always 0.
* S5 — stamped_fraction(), the pure half of the coverage probe Gerard runs.

All pure — the s3vectors client is mocked at the boundary; no AWS, no DB.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

import modules.search.vector_store.backends.s3_vectors_backend as s3b_mod  # noqa: E402
from config import config as app_config  # noqa: E402
from evals.retrieval_recall import (  # noqa: E402
    PARITY_EPSILON,
    RetrievalRecallReport,
    VariantTenantResult,
    parity_deltas,
)
from modules.search.vector_store.backends.s3_vectors_backend import (  # noqa: E402
    IndexDimensionMismatchError,
    S3VectorsBackend,
)
from modules.search.vector_store.backends.s3_vectors_mock import (  # noqa: E402
    MockS3VectorsBackend,
)
from scripts.probe_document_vectors import stamped_fraction  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend(monkeypatch):
    """A real S3VectorsBackend bound to ws-a, with boto3 mocked at the boundary."""
    monkeypatch.setattr(s3b_mod.config, "S3_VECTORS_BUCKET", "automatos-vectors", raising=False)
    monkeypatch.setattr(s3b_mod.config, "S3_VECTORS_INDEX_NAME", "documents-index", raising=False)
    monkeypatch.setattr(s3b_mod.config, "S3_VECTORS_DIMENSION", 2048, raising=False)
    monkeypatch.setattr(s3b_mod.config, "S3_VECTORS_METRIC", "cosine", raising=False)
    monkeypatch.setattr(s3b_mod.config, "AWS_REGION", "eu-west-2", raising=False)
    monkeypatch.setattr(s3b_mod.config, "AWS_ACCESS_KEY_ID", "test-key", raising=False)
    monkeypatch.setattr(s3b_mod.config, "AWS_SECRET_ACCESS_KEY", "test-secret", raising=False)

    fake_client = MagicMock()
    monkeypatch.setattr(s3b_mod.boto3, "client", lambda *a, **kw: fake_client)

    b = S3VectorsBackend(workspace_id="ws-a")
    b._setup_complete = True  # keep search/delete tests off the setup path
    return b, fake_client


def _hit(key: str, metadata: dict) -> dict:
    return {"key": key, "distance": 0.1, "metadata": metadata}


# ---------------------------------------------------------------------------
# S1 — isolation completeness on the shared bucket
# ---------------------------------------------------------------------------

def test_search_drops_unlabeled_hit_on_shared_bucket(backend):
    b, client = backend
    client.query_vectors.return_value = {
        "vectors": [
            _hit("k-mine", {"workspace_id": "ws-a", "chunk_text": "mine"}),
            _hit("k-unlabeled", {"chunk_text": "no workspace stamp"}),
            _hit("k-none", {"workspace_id": None, "chunk_text": "explicit None"}),
        ]
    }
    results = b.search(query_embedding=[0.1] * 4, limit=10, min_score=0.5)
    assert [r["key"] for r in results] == ["k-mine"]


def test_search_drops_mismatched_hit(backend):
    b, client = backend
    client.query_vectors.return_value = {
        "vectors": [
            _hit("k-mine", {"workspace_id": "ws-a", "chunk_text": "mine"}),
            _hit("k-theirs", {"workspace_id": "ws-b", "chunk_text": "another tenant"}),
        ]
    }
    results = b.search(query_embedding=[0.1] * 4, limit=10, min_score=0.5)
    assert [r["key"] for r in results] == ["k-mine"]


def test_add_documents_stamps_workspace_id(backend):
    b, client = backend
    b.add_documents(
        documents=[{"external_file_id": "f1", "chunk_index": 0, "chunk_text": "t"}],
        embeddings=[[0.1] * 4],
    )
    put = client.put_vectors.call_args
    stored = put.kwargs["vectors"] if put.kwargs else put.args[0]
    assert all(v["metadata"]["workspace_id"] == "ws-a" for v in stored)


def test_disconnect_delete_is_file_scoped(backend):
    b, client = backend
    client.list_vectors.side_effect = [
        {"vectors": [{"key": "doc_f1_chunk_0"}]},
        {"vectors": [{"key": "doc_f2_chunk_0"}, {"key": "doc_f2_chunk_1"}]},
    ]
    deleted = b.delete_for_files(["f1", "f2"])

    assert deleted == 3
    # Every listing was key-prefix scoped to one of the caller's files —
    # never an index-wide sweep (the shared bucket holds other tenants).
    for call in client.list_vectors.call_args_list:
        assert call.kwargs["keyPrefix"].startswith("doc_f")
    assert not hasattr(b, "delete_all_for_connection")
    assert not hasattr(MockS3VectorsBackend, "delete_all_for_connection")


def test_disconnect_caller_is_file_scoped():
    src = (Path(_orchestrator_root) / "api" / "cloud_documents.py").read_text()
    assert "delete_for_files" in src
    assert "delete_all_for_connection" not in src


def test_ingestion_dead_write_seam_removed():
    """The processor's vector_store leg called a method NO backend defines,
    with metadata that never stamped workspace_id — deleted, not repaired."""
    import inspect

    from modules.rag.ingestion.pipeline import IngestionPipeline
    from modules.rag.ingestion.processor import DocumentProcessor

    assert "vector_store" not in inspect.signature(DocumentProcessor.__init__).parameters
    assert "vector_store" not in inspect.signature(IngestionPipeline.__init__).parameters
    assert not hasattr(DocumentProcessor, "_store_chunks")


# ---------------------------------------------------------------------------
# S2 — dimension mismatch fails loud, misnomer gone
# ---------------------------------------------------------------------------

def test_index_dimension_mismatch_raises(backend):
    b, client = backend
    client.get_index.return_value = {"dimension": 4096}
    with pytest.raises(IndexDimensionMismatchError, match="4096"):
        b._assert_index_dimension()


def test_index_dimension_match_passes(backend):
    b, client = backend
    client.get_index.return_value = {"dimension": 2048}
    b._assert_index_dimension()  # no raise


def test_index_dimension_unverifiable_warns_not_raises(backend):
    b, client = backend
    client.get_index.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied"}}, "GetIndex"
    )
    b._assert_index_dimension()  # reachability problem ≠ confirmed mismatch


def test_verify_or_recreate_misnomer_is_gone():
    assert not hasattr(S3VectorsBackend, "_verify_or_recreate_index")


# ---------------------------------------------------------------------------
# S3 — config-integrity gate (shared-bucket rules)
# ---------------------------------------------------------------------------

def _vector_config(monkeypatch, enabled, bucket, dimension=2048):
    monkeypatch.setattr(app_config, "S3_VECTORS_ENABLED", enabled, raising=False)
    monkeypatch.setattr(app_config, "S3_VECTORS_BUCKET", bucket, raising=False)
    monkeypatch.setattr(app_config, "S3_VECTORS_DIMENSION", dimension, raising=False)


def test_vector_config_integrity_rejects_unset_bucket_when_enabled(monkeypatch):
    _vector_config(monkeypatch, enabled=True, bucket="")
    with pytest.raises(RuntimeError, match="S3_VECTORS_BUCKET"):
        app_config.assert_vector_config_integrity()


def test_vector_config_integrity_rejects_incoherent_dimension(monkeypatch):
    _vector_config(monkeypatch, enabled=True, bucket="automatos-vectors", dimension=0)
    with pytest.raises(RuntimeError, match="S3_VECTORS_DIMENSION"):
        app_config.assert_vector_config_integrity()


def test_vector_config_integrity_accepts_shared_bucket(monkeypatch):
    # The live prod shape: one shared bucket, NO {workspace_id} placeholder.
    _vector_config(monkeypatch, enabled=True, bucket="automatos-vectors")
    app_config.assert_vector_config_integrity()  # no raise


def test_vector_config_integrity_noop_when_disabled(monkeypatch):
    # Open-core local: S3 off, nothing to assert.
    _vector_config(monkeypatch, enabled=False, bucket="")
    app_config.assert_vector_config_integrity()  # no raise


def test_boot_aborts_on_bad_vector_config(monkeypatch):
    """validate_security (the hard-fail boot phase, main.py — OUTSIDE any
    swallowing run_stage) carries the vector-integrity failure."""
    _vector_config(monkeypatch, enabled=True, bucket="")
    with pytest.raises(RuntimeError, match="S3_VECTORS_BUCKET"):
        app_config.validate_security()

    main_src = (Path(_orchestrator_root) / "main.py").read_text()
    assert "config.validate_security()" in main_src


# ---------------------------------------------------------------------------
# S4 — flat-or-better parity gate vs the frozen baseline
# ---------------------------------------------------------------------------

def _row(variant: str, recall_at_5: float) -> VariantTenantResult:
    return VariantTenantResult(
        variant=variant,
        workspace_id="ws-a",
        n_queries=26,
        recall_at_1=recall_at_5,
        recall_at_3=recall_at_5,
        recall_at_5=recall_at_5,
        mrr=0.4,
        recall_at_5_natural=recall_at_5,
        recall_at_5_keyword=recall_at_5,
        n_natural=13,
        n_keyword=13,
    )


_FROZEN = {
    "pilot-a": {
        "variants": {
            "live_baseline": {"mean_recall_at_5": 0.6923},
            "live_rerank": {"mean_recall_at_5": 0.7692},
            "live_hybrid": {"mean_recall_at_5": 0.6923},
        }
    }
}


def test_parity_gate_flat_or_better():
    report = RetrievalRecallReport(results=[
        _row("live_baseline", 0.6923),   # flat → ok
        _row("live_rerank", 0.82),       # better → ok
        _row("live_hybrid", 0.60),       # -9.2 pts → regression
        _row("live_new_lever", 0.90),    # unknown to the baseline → skipped
    ])
    deltas = parity_deltas(report, _FROZEN, tenant_alias="pilot-a")

    assert deltas["live_baseline"]["regression"] is False
    assert deltas["live_rerank"]["regression"] is False
    assert deltas["live_hybrid"]["regression"] is True
    assert "live_new_lever" not in deltas


def test_parity_epsilon_tolerates_run_noise():
    report = RetrievalRecallReport(results=[_row("live_baseline", 0.6923 - PARITY_EPSILON + 0.001)])
    deltas = parity_deltas(report, _FROZEN, tenant_alias="pilot-a")
    assert deltas["live_baseline"]["regression"] is False


def test_parity_baseline_artifact_loads():
    """The frozen artifact ships in-repo (aliased tenants only) so the gate
    has its baseline on main — the same artifact PRD-198/199 consume."""
    path = Path(_orchestrator_root) / "evals" / "baseline" / "kg_retrieval_2026-07.json"
    data = json.loads(path.read_text())
    assert "pilot-a" in data
    assert "live_baseline" in data["pilot-a"]["variants"]
    assert data["pilot-a"]["variants"]["live_rerank"]["mean_recall_at_5"] > \
        data["pilot-a"]["variants"]["live_baseline"]["mean_recall_at_5"]


# ---------------------------------------------------------------------------
# S5 — coverage probe, pure half
# ---------------------------------------------------------------------------

def test_stamped_fraction_empty_sample_is_none():
    assert stamped_fraction([]) is None


def test_stamped_fraction_counts_only_real_stamps():
    sample = [
        {"metadata": {"workspace_id": "ws-a"}},   # stamped
        {"metadata": {"workspace_id": None}},      # explicit None → unstamped
        {"metadata": {}},                          # missing → unstamped
        {"metadata": None},                        # no metadata at all
    ]
    assert stamped_fraction(sample) == 0.25
