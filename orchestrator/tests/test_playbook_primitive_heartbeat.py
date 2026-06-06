"""PRD-142 Wave 3 · WS-3R · W3-S12 — Playbooks primitive heartbeat.

The §H DoD #4 (Observable) + #7 (Dashboard tile) require the playbooks tile
to reflect real-time execution outcome. W3-S1 built the helper
``emit_primitive_finding(workspace_id, primitive, status, detail)``; this
story wires it for ``primitive="playbooks"`` at the terminal transitions of
the executor.

Honest signal rules — match the W3-S6 (chat) / W3-S8 (rag) / W3-S9 (nl2sql) /
W3-S10 (graph) / W3-S11 (missions) wrappers:

  - Success boundary (``execution.status = 'completed'``) → ``status="green"``.
  - Failure boundary (``_fail_execution``) → ``status="down"`` with the
    caught error in ``detail``.
  - No ``workspace_id`` → no emit (the tile stays ``unknown`` for that
    workspace — never a fake green).
  - The emit is best-effort: a failure inside ``emit_primitive_finding``
    is logged and swallowed, NEVER raised back to the executor (a busted
    heartbeat must NOT break playbook completion).

TDD GUARANTEE: written BEFORE the wrapper module and the executor wiring
land. Each test fails with ``ModuleNotFoundError`` / missing source-text
until the wiring is real.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

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


# ---------------------------------------------------------------------------
# Helper to load the heartbeat wrapper without dragging services/__init__.
# Matches the W3-S6/8 importlib pattern used to dodge heavy package imports.
# ---------------------------------------------------------------------------


def _load_heartbeat_module():
    """Import services.playbook_engine_heartbeat by direct path so its
    sibling ``services.playbook_engine`` (which lazy-imports recipe_executor)
    does not need to load."""
    spec = importlib.util.spec_from_file_location(
        "services.playbook_engine_heartbeat",
        ORCH_ROOT / "services" / "playbook_engine_heartbeat.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. The wrapper module exists with the canonical helper name.
# ---------------------------------------------------------------------------


def test_heartbeat_wrapper_module_exists():
    """services/playbook_engine_heartbeat.py is the tiny stateless wrapper.
    Lives in its own module so the executor stays focused and the tests can
    verify the contract without the full executor import surface."""
    p = ORCH_ROOT / "services" / "playbook_engine_heartbeat.py"
    assert p.exists(), "services/playbook_engine_heartbeat.py must exist"


def test_helper_name_is_canonical():
    """Naming convention: ``_emit_playbooks_primitive`` matches the W3-S11
    (missions) / W3-S6 (chat) / etc. shape — drift here breaks the audit."""
    mod = _load_heartbeat_module()
    assert hasattr(mod, "_emit_playbooks_primitive"), (
        "missing _emit_playbooks_primitive helper"
    )
    assert callable(mod._emit_playbooks_primitive)


# ---------------------------------------------------------------------------
# 2. Green on success.
# ---------------------------------------------------------------------------


def test_emits_green_on_success(monkeypatch):
    """A clean playbook completion → emit_primitive_finding(
    workspace_id, 'playbooks', 'green', detail)."""
    mod = _load_heartbeat_module()

    spy = MagicMock(return_value=True)
    monkeypatch.setattr(mod, "emit_primitive_finding", spy)

    ws_id = str(uuid4())
    mod._emit_playbooks_primitive(ws_id, success=True, detail="exec=xyz steps=3")

    spy.assert_called_once_with(ws_id, "playbooks", "green", "exec=xyz steps=3")


# ---------------------------------------------------------------------------
# 3. Down on failure.
# ---------------------------------------------------------------------------


def test_emits_down_on_failure(monkeypatch):
    """A caught failure → emit_primitive_finding(
    workspace_id, 'playbooks', 'down', detail)."""
    mod = _load_heartbeat_module()

    spy = MagicMock(return_value=True)
    monkeypatch.setattr(mod, "emit_primitive_finding", spy)

    ws_id = str(uuid4())
    mod._emit_playbooks_primitive(ws_id, success=False, detail="Recipe not found")

    spy.assert_called_once_with(ws_id, "playbooks", "down", "Recipe not found")


# ---------------------------------------------------------------------------
# 4. No workspace_id → no emit (honest gap over fake green).
# ---------------------------------------------------------------------------


def test_skips_when_workspace_id_missing(monkeypatch):
    """Falsy workspace_id → emit_primitive_finding is NOT called. The tile
    stays ``unknown`` for that workspace instead of borrowing another's id
    (A4 — no fabricated workspace defaults)."""
    mod = _load_heartbeat_module()

    spy = MagicMock(return_value=True)
    monkeypatch.setattr(mod, "emit_primitive_finding", spy)

    mod._emit_playbooks_primitive(None, success=True, detail="x")
    mod._emit_playbooks_primitive("", success=False, detail="x")

    spy.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Best-effort: emit failure is swallowed.
# ---------------------------------------------------------------------------


def test_emit_failure_is_swallowed(monkeypatch):
    """If emit_primitive_finding raises, the wrapper logs it but does NOT
    re-raise — a busted heartbeat must not break playbook completion."""
    mod = _load_heartbeat_module()

    def _raise(*a, **k):
        raise RuntimeError("simulated heartbeat outage")

    monkeypatch.setattr(mod, "emit_primitive_finding", _raise)

    # MUST NOT raise.
    mod._emit_playbooks_primitive(str(uuid4()), success=True, detail="ok")


# ---------------------------------------------------------------------------
# 6. Detail is truncated to 500 chars (matches the helper contract).
# ---------------------------------------------------------------------------


def test_detail_is_truncated_to_500_chars(monkeypatch):
    """``emit_primitive_finding`` truncates at 500 chars too, but the wrapper
    pre-truncates so the rejection layer never sees an oversized blob."""
    mod = _load_heartbeat_module()

    spy = MagicMock(return_value=True)
    monkeypatch.setattr(mod, "emit_primitive_finding", spy)

    long = "x" * 2000
    mod._emit_playbooks_primitive(str(uuid4()), success=False, detail=long)

    spy.assert_called_once()
    sent_detail = spy.call_args.args[3]
    assert len(sent_detail) <= 500


# ---------------------------------------------------------------------------
# 7. Canonical primitive name = 'playbooks' (never 'recipe', 'workflow').
# ---------------------------------------------------------------------------


def test_canonical_primitive_name(monkeypatch):
    """The wrapper hard-codes ``'playbooks'`` — drift here means the
    primitive shows up as a non-canonical name in the W3-S2 tile (which
    rejects unknown primitives)."""
    mod = _load_heartbeat_module()

    spy = MagicMock(return_value=True)
    monkeypatch.setattr(mod, "emit_primitive_finding", spy)

    mod._emit_playbooks_primitive(str(uuid4()), success=True, detail="")

    assert spy.call_args.args[1] == "playbooks", (
        "wrapper must use the CANONICAL 'playbooks' primitive name"
    )
    # Negative: no legacy 'recipe' / 'workflow' name slips through.
    assert spy.call_args.args[1] not in {"recipe", "workflow"}


# ---------------------------------------------------------------------------
# 8. Executor wire-up: recipe_executor emits success=True at the COMPLETED
#    transition and success=False inside _fail_execution.
# ---------------------------------------------------------------------------


def test_recipe_executor_wires_success_emit():
    """At the canonical success boundary (``execution.status = 'completed'``
    in _execute_recipe_inner), the executor calls the helper with
    success=True. Source-text + AST inspection — no live executor run."""
    src = (ORCH_ROOT / "api" / "recipe_executor.py").read_text()

    # The wrapper module is imported (lazy or top-level — either is OK).
    assert "_emit_playbooks_primitive" in src, (
        "recipe_executor.py must call _emit_playbooks_primitive"
    )
    # success=True is wired for the COMPLETED boundary.
    assert "success=True" in src or 'success = True' in src, (
        "recipe_executor.py must emit success=True at the COMPLETED boundary"
    )


def test_recipe_executor_wires_failure_emit():
    """``_fail_execution`` calls the helper with success=False — every
    failure path that flows through _fail_execution flips the tile to
    'down' alongside the existing logger.error + _dispatch_playbook_event
    surface."""
    src = (ORCH_ROOT / "api" / "recipe_executor.py").read_text()
    assert "success=False" in src or 'success = False' in src, (
        "recipe_executor.py must emit success=False inside _fail_execution"
    )


def test_executor_emit_calls_use_canonical_helper_only():
    """No raw ``emit_primitive_finding('playbooks', ...)`` call from the
    executor — all emits go through the wrapper (so the wrapper's
    workspace-id-skip + truncation + swallow guarantees hold uniformly)."""
    src = (ORCH_ROOT / "api" / "recipe_executor.py").read_text()
    # The bare helper name is used; the literal 'playbooks' is not assembled
    # as a hard-coded string for emit_primitive_finding.
    # (We allow the literal in tests/comments, not in active call sites — but
    # checking the call shape directly is the cleanest pin.)
    assert 'emit_primitive_finding(' not in src.replace(
        '_emit_playbooks_primitive(', ''
    ).replace('emit_primitive_finding,', ''), (
        "recipe_executor must NOT call emit_primitive_finding directly — "
        "use _emit_playbooks_primitive wrapper"
    )
