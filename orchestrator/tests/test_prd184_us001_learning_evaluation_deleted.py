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

HELD (must SURVIVE — this guard proves the boundary, not just the deletion):
``modules/learning/playbooks/miner.py`` (``PlaybookMiner``) is the S10 retire,
NOT this story. It is a LIVE dependency of ``api/api_playbooks.py`` (mounted at
``main.py``). Its reachability chain — ``modules/learning/__init__.py`` →
``modules/learning/playbooks/__init__.py`` → ``miner.py`` — therefore stays: the
deletion gate forbids breaking a live caller. So ``learning/`` is trimmed to its
real, in-use core, not razed.

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
    "modules.learning.feedback",
    "modules.learning.patterns",
)
_GONE_TOKEN_PATTERNS = tuple(
    (token, re.compile(rf"\b{re.escape(token)}\b")) for token in _GONE_TOKENS
)


def test_learning_evaluation_theatre_dirs_deleted():
    """The three dead package dirs are gone — no ``_legacy`` shim (CLAUDE.md)."""
    for rel in (
        "modules/evaluation",
        "modules/learning/feedback",
        "modules/learning/patterns",
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


def test_modules_barrel_drops_evaluation_keeps_learning():
    """``modules/__init__.py`` no longer advertises the razed ``evaluation``
    package, but STILL lists ``learning`` (the held PlaybookMiner chain lives on)."""
    src = (_ORCH / "modules" / "__init__.py").read_text()
    assert '"evaluation"' not in src, (
        "modules/__init__.py __all__ must drop the deleted 'evaluation' package"
    )
    assert '"learning"' in src, (
        "modules/__init__.py must KEEP 'learning' — PlaybookMiner (held S10) is "
        "still re-exported from it and used by the live api_playbooks router"
    )


def test_held_playbook_miner_chain_survives():
    """Boundary proof: the S10-held PlaybookMiner reachability is intact.

    Deleting this chain would silently break the live ``api/api_playbooks.py``
    router — exactly what the deletion gate forbids. This story trims the dead
    theatre AROUND the miner, it does not touch the miner."""
    assert (_ORCH / "modules" / "learning" / "playbooks" / "miner.py").exists()
    learning_init = (_ORCH / "modules" / "learning" / "__init__.py").read_text()
    assert "from .playbooks import PlaybookMiner" in learning_init
    playbooks_init = (
        _ORCH / "modules" / "learning" / "playbooks" / "__init__.py"
    ).read_text()
    assert "from .miner import PlaybookMiner" in playbooks_init
    # The live held caller still resolves the symbol through the intact chain.
    api_pb = (_ORCH / "api" / "api_playbooks.py").read_text()
    assert "from modules.learning import PlaybookMiner" in api_pb
