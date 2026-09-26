"""PRD-184 US-001 — the learning/evaluation theatre packages are deleted, not orphaned.

Two empty-theatre packages signposted away from the real loops:

* ``modules/evaluation/`` — a lone ``__init__`` with ``__all__ = []`` and a
  ``# TODO: Implement`` block. Zero external callers. Advertised a
  ``EvaluationEngine`` that was never built.
* ``modules/learning/feedback/`` and ``modules/learning/patterns/`` — empty
  ``__init__`` files. Zero importers anywhere in the tree.

They are removed rather than kept (honest-empty over silent placebo — the whole
point of the kill-list: the codebase must stop lying to the humans *and agents*
that read it).

RETIRED since (F136, Gerard 2026-09-25): the S10-held ``PlaybookMiner`` and its
only caller, ``api/api_playbooks.py``. The table both read, ``playbooks``, was
dropped by prd135_drop_bucket_6 (their raw-SQL strings hid them from the
dead-code scan), so GET /api/playbooks answered 500 for everyone (night 4, B1).
The whole ``modules/learning/`` package is gone with them.

Pure/static — file reads only, imports no app package.
"""
from __future__ import annotations

import pathlib
import re
import sys

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_SOURCE_DIRS = ("modules", "services", "core", "api", "consumers", "evals")

# Dotted module paths that ONLY ever named the now-deleted packages. Matched on
# word boundaries so the generic words "feedback"/"patterns"/"evaluation" (which
# name unrelated live code — the evals harness, feedback columns, etc.) do NOT
# false-positive; only the specific ``modules.*`` dotted import forms do.
_GONE_TOKENS = (
    "modules.evaluation",
    "modules.learning",
    "api.api_playbooks",
)
_GONE_TOKEN_PATTERNS = tuple(
    (token, re.compile(rf"\b{re.escape(token)}\b")) for token in _GONE_TOKENS
)


def test_learning_evaluation_theatre_dirs_deleted():
    """The three dead package dirs are gone — no ``_legacy`` shim (CLAUDE.md)."""
    for rel in (
        "modules/evaluation",
        "modules/learning",
        "api/api_playbooks.py",
    ):
        assert not (_ORCH / rel).exists(), (
            f"{rel}/ must stay deleted (PRD-184 US-001) — empty theatre, zero callers"
        )


def test_no_learning_evaluation_imports():
    """No live source file imports the deleted packages (no dangling imports)."""
    offenders = []
    for d in _SOURCE_DIRS:
        root = _ORCH / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(errors="ignore")
            for token, pattern in _GONE_TOKEN_PATTERNS:
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(_ORCH)}: {token}")
    for extra in ("main.py", "config.py"):
        text = (_ORCH / extra).read_text(errors="ignore")
        for token, pattern in _GONE_TOKEN_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{extra}: {token}")
    assert not offenders, f"dangling learning/evaluation references: {offenders}"


def test_modules_barrel_drops_evaluation_and_learning():
    """``modules/__init__.py`` advertises neither razed package (F136 retired learning)."""
    src = (_ORCH / "modules" / "__init__.py").read_text()
    assert '"evaluation"' not in src and '"learning"' not in src


def test_the_playbook_miner_and_its_endpoint_are_retired():
    """F136: GET /api/playbooks read a table prd135 dropped and answered 500 for
    everyone; the miner behind POST /api/playbooks/mine wrote to the same table."""
    main = (_ORCH / "main.py").read_text()
    assert "playbooks_router" not in main and "api_playbooks" not in main
    lint = (_ORCH / ".importlinter").read_text()
    assert "modules.learning" not in lint
