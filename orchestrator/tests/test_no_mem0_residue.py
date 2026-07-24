"""PRD-211 US-002 — the dead mem0 residue is gone and the un-split stays locked.

PRD-187 un-split retired the external HTTP mem0 service; the live memory path is
now in-process Qdrant via modules/memory/durable_store.py. That left 7 dead files
still referencing the retired service. This guard pins two things:

1. the 7 residue files do NOT exist (grep-proven to have ZERO live code importers
   before deletion — the only surviving references are archival review snapshots
   and Ralph scaffolding, which are history, not importers); and
2. the un-split stays locked — no HTTP mem0 client (MEM0_API_URL / mem0_client /
   httpx) is reintroduced anywhere under orchestrator/modules/memory/. If the
   external service creeps back in, the in-process guarantee is silently broken;
   this test fails loud instead.

PURE: filesystem + source-grep only. No network, no DB, no git.
"""
from __future__ import annotations

import re
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent

# The 7 dead mem0-residue files retired by the PRD-187 un-split (repo-relative).
_RESIDUE = (
    "orchestrator/mem0_openapi.json",
    "orchestrator/scripts/probe_mem0_endpoints.py",
    "orchestrator/scripts/seed_mem0_user.py",
    "scripts/test_mem0_railway.py",
    "docs/PRDS/39-MEM0-MIGRATION-PRD.md",
    "docs/PRDS/PRD-152-MEM0-INTERNAL-SERVICES-DECOUPLING.md",
    "docs/memory-system/phase1-mem0-async-rollback.md",
)

# An HTTP mem0 client would reintroduce the retired external service. Same tokens
# the acceptance gate greps for (scripts/ralph/acceptance-prd211.sh).
_HTTP_CLIENT = re.compile(r"MEM0_API_URL|mem0_client|httpx")

_MEMORY_DIR = _ORCH / "modules" / "memory"


def test_mem0_residue_files_deleted():
    survivors = [p for p in _RESIDUE if (_REPO / p).exists()]
    assert not survivors, (
        "dead mem0-residue file(s) resurfaced — the PRD-187 un-split retired the "
        f"external HTTP mem0 service; these must stay deleted: {survivors}"
    )


def test_no_http_mem0_client_under_modules_memory():
    """The live memory path is in-process (Qdrant/durable_store). No file under
    modules/memory may carry an HTTP mem0 client — that would un-do the un-split."""
    assert _MEMORY_DIR.is_dir(), f"expected {_MEMORY_DIR} to exist"
    offenders = []
    for path in _MEMORY_DIR.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary / unreadable — not a Python HTTP client
        for lineno, line in enumerate(text.splitlines(), 1):
            if _HTTP_CLIENT.search(line):
                offenders.append(f"{path.relative_to(_REPO)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "HTTP mem0 client residue found under modules/memory/ — the un-split "
        "requires the in-process path only (no MEM0_API_URL / mem0_client / "
        "httpx). Offenders:\n" + "\n".join(offenders)
    )


def test_durable_store_is_the_live_path():
    """Sanity: the in-process replacement the un-split moved to is present."""
    assert (_MEMORY_DIR / "durable_store.py").exists(), (
        "modules/memory/durable_store.py (the in-process memory path) is missing"
    )
