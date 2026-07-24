"""PRD-185 S8: the document-vector probe's verdict logic.

The probe itself is read-only I/O against prod (AWS + DB), run by hand. Its
*decision* — is the plane live, dark, or degraded, and what to recommend — is a
pure function so it is pinned here with no DB / network. This guards the gate
that Wave-1 P2-07 (RAG quality) depends on.
"""
import pytest


def _probe():
    try:
        from scripts.probe_document_vectors import (
            classify_plane, recommend,
            VERDICT_LIVE, VERDICT_DARK, VERDICT_DEGRADED, VERDICT_UNKNOWN,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"probe_document_vectors not importable in this env: {e}")
    return classify_plane, recommend, VERDICT_LIVE, VERDICT_DARK, VERDICT_DEGRADED, VERDICT_UNKNOWN


def _p(**kw):
    base = {"index_exists": True, "populated": True, "dimension": 2048, "error": None}
    base.update(kw)
    return base


def test_unconstructable_backend_is_dark():
    classify, _, _LIVE, DARK, _DEG, _UNK = _probe()
    # committed config can't even build the backend (the F005 / placeholder case)
    assert classify("s3_vectors", constructable=False, configured_dimension=2048,
                    probes=[_p()]) == DARK


def test_constructable_but_empty_is_dark():
    classify, _, _LIVE, DARK, _DEG, _UNK = _probe()
    assert classify("s3_vectors", constructable=True, configured_dimension=2048,
                    probes=[_p(populated=False)]) == DARK


def test_populated_matching_dimension_is_live():
    classify, _, LIVE, _DARK, _DEG, _UNK = _probe()
    assert classify("s3_vectors", constructable=True, configured_dimension=2048,
                    probes=[_p(dimension=2048, populated=True)]) == LIVE


def test_dimension_mismatch_is_degraded():
    classify, _, _LIVE, _DARK, DEGRADED, _UNK = _probe()
    # stored at 1536 but config expects 2048 → embeddings unusable → degraded
    assert classify("s3_vectors", constructable=True, configured_dimension=2048,
                    probes=[_p(dimension=1536, populated=True)]) == DEGRADED


def test_all_probes_errored_is_unknown():
    classify, _, _LIVE, _DARK, _DEG, UNKNOWN = _probe()
    assert classify("s3_vectors", constructable=True, configured_dimension=2048,
                    probes=[_p(error="get_index: AccessDenied")]) == UNKNOWN


def test_one_populated_workspace_makes_the_plane_live():
    classify, _, LIVE, _DARK, _DEG, _UNK = _probe()
    probes = [_p(populated=False), _p(populated=True), _p(error="boom")]
    assert classify("s3_vectors", constructable=True, configured_dimension=2048,
                    probes=probes) == LIVE


def test_pgvector_empty_is_dark():
    classify, _, _LIVE, DARK, _DEG, _UNK = _probe()
    assert classify("pgvector", constructable=True, configured_dimension=2048,
                    probes=[_p(populated=False)]) == DARK


def test_recommendation_tracks_verdict():
    classify, recommend, LIVE, DARK, DEGRADED, _UNK = _probe()
    assert "LIVE" in recommend(LIVE, "s3_vectors")
    assert "DARK" in recommend(DARK, "s3_vectors")
    # the s3 dark path names the actual fix + the fold-or-relight decision
    s3_dark = recommend(DARK, "s3_vectors")
    assert "workspace_id" in s3_dark and "P2-16" in s3_dark
    assert "MISMATCH" in recommend(DEGRADED, "s3_vectors")
    # pgvector dark points at ingestion, not the S3 env
    assert "pgvector" in recommend(DARK, "pgvector")
