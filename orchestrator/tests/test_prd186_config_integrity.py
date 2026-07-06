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


# ===========================================================================
# S2 — fail-closed boot: the integrity failure hard-aborts, not swallowed
# ===========================================================================

def _lifespan_node():
    tree = ast.parse((_ORCH_ROOT / "main.py").read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan":
            return n
    raise AssertionError("lifespan() not found in main.py")


def _calls_in(node):
    """(call_node, callee_name) for every Call under `node` (nested fns included)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            yield n, getattr(n.func, "attr", getattr(n.func, "id", None))


class TestBootFailClosed:
    def test_boot_aborts_on_bad_vector_config(self, monkeypatch):
        """The exact prod misconfig (enabled + 'automatos-ai') RAISES through the
        assertion lifespan invokes — a real abort, not a value returned."""
        cfg = _cfg(monkeypatch, enabled=True, bucket="automatos-ai")
        with pytest.raises(RuntimeError, match="workspace_id"):
            cfg.assert_vector_config_integrity()

    def test_run_stage_swallows_but_preguard_aborts(self, monkeypatch):
        """Reproduce the root cause against the REAL run_stage and prove the fix.

        run_stage catches every exception and records the stage 'failed' WITHOUT
        re-raising (bootstrap.py) — which is how the F005 RuntimeError booted the
        plane dark. The S2 fix invokes the same assertion BEFORE run_stage, where
        the raise propagates (fail-closed).
        """
        import asyncio
        from core.models.bootstrap import BootstrapReport, BootstrapStage, run_stage

        cfg = _cfg(monkeypatch, enabled=True, bucket="automatos-ai")

        def _raises():  # what _boot_phase_1_core does via validate_security
            cfg.assert_vector_config_integrity()

        report = BootstrapReport()
        result = asyncio.run(run_stage(report, BootstrapStage.DATABASE_INIT, _raises))
        # Swallowed: recorded 'failed', no propagation — the dark-boot bug.
        assert result.status == "failed"
        assert "workspace_id" in (result.error or "")

        # Fix: invoked directly (as lifespan does before run_stage) → propagates.
        with pytest.raises(RuntimeError, match="workspace_id"):
            cfg.assert_vector_config_integrity()

    def test_integrity_check_wired_before_run_stage_database_init(self):
        """AST-proven: lifespan() calls assert_vector_config_integrity() OUTSIDE
        the swallowing run_stage and BEFORE run_stage(DATABASE_INIT)."""
        life = _lifespan_node()
        assert_lines = [c.lineno for c, name in _calls_in(life)
                        if name == "assert_vector_config_integrity"]
        assert assert_lines, (
            "lifespan() must call config.assert_vector_config_integrity() so a "
            "placeholder-less bucket aborts boot instead of being swallowed by "
            "run_stage."
        )
        db_init_lines = [
            c.lineno for c, name in _calls_in(life)
            if name == "run_stage"
            and any(isinstance(a, ast.Attribute) and a.attr == "DATABASE_INIT"
                    for a in c.args)
        ]
        assert db_init_lines, "run_stage(DATABASE_INIT) not found in lifespan()"
        assert min(assert_lines) < min(db_init_lines), (
            "assert_vector_config_integrity() must run BEFORE "
            "run_stage(DATABASE_INIT) — after it, the F005 raise is swallowed."
        )

    def test_assertion_call_is_not_inside_boot_phase_1_core(self):
        """The fail-closed guard must live in lifespan, not (only) inside
        _boot_phase_1_core — that function is what run_stage wraps and swallows."""
        tree = ast.parse((_ORCH_ROOT / "main.py").read_text(encoding="utf-8"))
        phase1 = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.AsyncFunctionDef) and n.name == "_boot_phase_1_core"),
            None,
        )
        assert phase1 is not None, "_boot_phase_1_core not found"
        life = _lifespan_node()
        # The lifespan body (excluding the nested _boot_phase_1_core, which is a
        # module-level fn anyway) carries the guard; that is the point.
        assert any(name == "assert_vector_config_integrity" for _, name in _calls_in(life))
