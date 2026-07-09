"""RAG document retrieval must fall back to the live pgvector plane when S3
Vectors is enabled but its bucket is mis-templated (dark).

2026-07-09 slow-chat incident: ``_get_candidates`` hard-wired S3VectorsBackend
(PRD-157 S4, anticipating an S3 migration that never landed). With the prod env
``S3_VECTORS_ENABLED=true`` + ``S3_VECTORS_BUCKET=automatos-ai`` (no
``{workspace_id}`` placeholder), the F005 guard fail-closed on every query, so
each Auto turn returned 0 docs after ~80s while the populated pgvector
``document_chunks`` plane (~19k chunks) sat unused. The gate below decides which
backend serves candidates; the actual pgvector query is exercised by the
DB-backed suite.

Pure unit test — the routing decision is a static, config-only function.
"""
import pytest


def _configured():
    try:
        from modules.rag.service import RAGService
    except Exception as e:  # env without the heavy service deps
        pytest.skip(f"rag.service not importable in this env: {e}")
    return RAGService._s3_vectors_backend_configured


def test_prod_misconfig_2026_07_09_routes_to_pgvector():
    # Exact live prod state that caused the outage: enabled, no placeholder.
    assert _configured()(True, "automatos-ai") is False


def test_properly_templated_bucket_uses_s3():
    # Post-PRD-186 relight: a per-workspace bucket → S3 path is used again.
    assert _configured()(True, "automatos-vectors-{workspace_id}") is True


def test_disabled_never_uses_s3():
    fn = _configured()
    assert fn(False, "automatos-vectors-{workspace_id}") is False
    assert fn(False, "automatos-ai") is False


def test_empty_or_none_bucket_routes_to_pgvector():
    fn = _configured()
    assert fn(True, None) is False
    assert fn(True, "") is False
    assert fn(True, "   ") is False  # whitespace, still no placeholder
