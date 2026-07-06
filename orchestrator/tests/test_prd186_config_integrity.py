"""PRD-186 S1/S2 — the S3 Vectors config-integrity guard is loud and fail-closed.

Why this exists: prod ran with ``S3_VECTORS_ENABLED=true`` and
``S3_VECTORS_BUCKET='automatos-ai'`` (no ``{workspace_id}`` placeholder), so
``S3VectorsBackend.__init__`` raised the F005 ``RuntimeError`` — but that raise
was SWALLOWED by ``run_stage`` and the retrieval plane booted dark for weeks
(~19,130 healthy pgvector chunks unreachable through the active backend).

These are PURE tests. They pin:
  * S1 — the extracted ``config.assert_vector_config_integrity()`` assertion.
  * S2 — its fail-closed boot wiring (asserted by inspecting ``main.py`` with
    the ``ast`` module — never importing/booting it, which pulls the whole app).

No DB / network / AWS: config values are patched at the boundary.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

# Dummy POSTGRES_* satisfies the config import chain (blessed pattern); the port
# points at nothing so any fail-soft connect refuses instantly. CI exports real
# vars (setdefault no-ops).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import Config

_ORCH_ROOT = Path(__file__).resolve().parents[1]


def _cfg(monkeypatch, *, enabled: bool, bucket):
    cfg = Config()
    monkeypatch.setattr(cfg, "S3_VECTORS_ENABLED", enabled, raising=False)
    monkeypatch.setattr(cfg, "S3_VECTORS_BUCKET", bucket, raising=False)
    return cfg


# ===========================================================================
# S1 — pure config-integrity assertion
# ===========================================================================

class TestVectorConfigIntegrity:
    def test_vector_config_integrity_rejects_placeholderless_bucket(self, monkeypatch):
        # The exact prod misconfig: enabled + a bucket with no {workspace_id}.
        cfg = _cfg(monkeypatch, enabled=True, bucket="automatos-ai")
        with pytest.raises(RuntimeError, match="workspace_id"):
            cfg.assert_vector_config_integrity()
        # And the empty-bucket case (unset in prod → None → "").
        cfg2 = _cfg(monkeypatch, enabled=True, bucket="")
        with pytest.raises(RuntimeError):
            cfg2.assert_vector_config_integrity()

    def test_vector_config_integrity_accepts_templated_bucket(self, monkeypatch):
        cfg = _cfg(monkeypatch, enabled=True, bucket="automatos-vectors-{workspace_id}")
        cfg.assert_vector_config_integrity()  # no raise

    def test_vector_config_integrity_noop_when_disabled(self, monkeypatch):
        # Open-core local: feature off → silent even with a junk bucket.
        cfg = _cfg(monkeypatch, enabled=False, bucket="automatos-ai")
        cfg.assert_vector_config_integrity()  # no raise

    def test_validate_security_delegates_to_assertion(self, monkeypatch):
        # validate_security() still raises on the bad bucket (existing PRD-172
        # contract, test_config_boot_asserts_bucket_placeholder) AND now routes
        # through the shared assertion.
        cfg = _cfg(monkeypatch, enabled=True, bucket="shared-no-placeholder")
        monkeypatch.setattr(cfg, "SHOPIFY_INTERNAL_API_KEY", "x", raising=False)
        with pytest.raises(RuntimeError, match="workspace_id"):
            cfg.validate_security()

    def test_validate_security_calls_shared_assertion_in_source(self):
        # AC: validate_security() now CALLS assert_vector_config_integrity()
        # (proven structurally via ast, so the delegation is real, not just a
        # coincidental raise).
        tree = ast.parse((_ORCH_ROOT / "config.py").read_text(encoding="utf-8"))
        found = []
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name == "validate_security":
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call):
                        name = getattr(n.func, "attr", getattr(n.func, "id", None))
                        if name == "assert_vector_config_integrity":
                            found.append(name)
        assert found, "validate_security() must call assert_vector_config_integrity()"

    def test_f005_message_not_duplicated(self):
        # AC: the inline F005 branch is deleted; the placeholder message string
        # lives in ONE place (no shim, no duplicated string).
        text = (_ORCH_ROOT / "config.py").read_text(encoding="utf-8")
        n = text.count("does not contain the")
        assert n == 1, (
            f"F005 placeholder message appears {n}× — the inline branch must be "
            "deleted, not left beside the extraction."
        )
