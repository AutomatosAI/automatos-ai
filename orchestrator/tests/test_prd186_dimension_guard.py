"""PRD-186 S3 — an index/config dimension mismatch fails loud, never mutates.

The sibling hole to F005: ``_verify_or_recreate_index`` only LOGGED when an
existing S3 Vectors index reported a different dimension than
``config.S3_VECTORS_DIMENSION`` (now 2048). 2048-dim vectors written to — or
queried against — a differently-dimensioned index corrupt retrieval silently.

S3 makes a *confirmed* mismatch RAISE at first-use, while keeping the never-delete
invariant (deleting a populated index destroys stored vectors). These are PURE
tests: the s3vectors client is a MagicMock at the boundary — no AWS.
"""
from __future__ import annotations

import os

import pytest

# Dummy POSTGRES_* satisfies the config import chain (blessed pattern); the port
# points at nothing so any fail-soft connect refuses instantly.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). Purge origin-less entries so the
# real backend package imports fresh; conftest's autouse repair re-binds the rest.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from unittest.mock import MagicMock  # noqa: E402

from modules.search.vector_store.backends.s3_vectors_backend import (  # noqa: E402
    S3VectorsBackend,
)


def _backend(monkeypatch, *, configured_dim=2048):
    """Construct a backend bypassing __init__/AWS; only the attrs
    _verify_or_recreate_index touches are set."""
    b = S3VectorsBackend.__new__(S3VectorsBackend)
    b.workspace_id = "ws-1"
    b.bucket_name = "automatos-vectors-ws-1"
    b.index_name = "documents-index"
    b.index_dimension = configured_dim
    b.distance_metric = "cosine"
    b._setup_complete = False
    b.client = MagicMock()
    return b


# ---------------------------------------------------------------------------
# Pure comparison helper — unit-testable without boto3
# ---------------------------------------------------------------------------

class TestDimensionHelpers:
    def test_mismatch_true_when_confirmed_conflict(self):
        from modules.search.vector_store.backends.s3_vectors_backend import (
            _index_dimension_mismatch,
        )
        assert _index_dimension_mismatch(2048, 4096) is True

    def test_mismatch_false_when_equal(self):
        from modules.search.vector_store.backends.s3_vectors_backend import (
            _index_dimension_mismatch,
        )
        assert _index_dimension_mismatch(2048, 2048) is False

    def test_mismatch_false_when_reported_unknown(self):
        # A missing / zero reported dimension is NOT a confirmed mismatch.
        from modules.search.vector_store.backends.s3_vectors_backend import (
            _index_dimension_mismatch,
        )
        assert _index_dimension_mismatch(2048, 0) is False

    def test_reported_dimension_reads_flat_and_nested(self):
        # Prod get_index nests under "index"; tolerate both so the guard is not
        # dead in prod (the old flat-only read would always see 0 → never fire).
        from modules.search.vector_store.backends.s3_vectors_backend import (
            _reported_index_dimension,
        )
        assert _reported_index_dimension({"dimension": 4096}) == 4096
        assert _reported_index_dimension({"index": {"dimension": 4096}}) == 4096
        assert _reported_index_dimension({}) == 0


# ---------------------------------------------------------------------------
# _verify_or_recreate_index — raise on confirmed mismatch, never delete
# ---------------------------------------------------------------------------

class TestVerifyIndexDimension:
    def test_index_dimension_mismatch_raises(self, monkeypatch):
        b = _backend(monkeypatch, configured_dim=2048)
        b.client.get_index.return_value = {"dimension": 4096}
        with pytest.raises(RuntimeError, match="dimension"):
            b._verify_or_recreate_index()

    def test_index_dimension_mismatch_nested_shape_raises(self, monkeypatch):
        b = _backend(monkeypatch, configured_dim=2048)
        b.client.get_index.return_value = {"index": {"dimension": 4096}}
        with pytest.raises(RuntimeError, match="dimension"):
            b._verify_or_recreate_index()

    def test_index_dimension_match_passes(self, monkeypatch):
        b = _backend(monkeypatch, configured_dim=2048)
        b.client.get_index.return_value = {"dimension": 2048}
        b._verify_or_recreate_index()  # no raise

    def test_unknown_reported_dimension_does_not_raise(self, monkeypatch):
        b = _backend(monkeypatch, configured_dim=2048)
        b.client.get_index.return_value = {}  # dimension absent → unconfirmed
        b._verify_or_recreate_index()  # no raise

    def test_client_error_does_not_raise(self, monkeypatch):
        from botocore.exceptions import ClientError
        b = _backend(monkeypatch, configured_dim=2048)
        b.client.get_index.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetIndex"
        )
        b._verify_or_recreate_index()  # unconfirmable → warn+continue, no raise

    def test_mismatch_never_deletes_the_index(self, monkeypatch):
        b = _backend(monkeypatch, configured_dim=2048)
        b.client.get_index.return_value = {"dimension": 4096}
        with pytest.raises(RuntimeError):
            b._verify_or_recreate_index()
        # The :119 invariant: a populated index is NEVER destroyed on mismatch.
        b.client.delete_index.assert_not_called()
        b.client.delete_vectors.assert_not_called()
        b.client.delete_vector_bucket.assert_not_called()
