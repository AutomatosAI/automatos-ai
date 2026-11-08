"""Memory-consolidation eval: what folding keeps, and what it loses (#1430 Wave 0).

Consolidation (``UnifiedMemoryService.run_sleep_time_consolidation``, PRD-159 S4, on the
hourly memory job) plans over a workspace's first 200 long-term rows: near-duplicates
(text similarity ≥ 0.82) fold into one, and "contradictions" (the same subject, each
asserting a token the other lacks) lose their older or less confident side. Both are pure
text rules, so this eval runs them exactly, offline, on the real service over
``evals.memory_pipeline``'s in-memory store.

Each of ``evals.memory_recall``'s facts is filed with three restatements of it:
  * exact     the same text (the store folds it at write, by its content hash);
  * light     "As noted, …": the same sentence, a lead-in added;
  * reworded  a handful of words swapped for synonyms.
Optionally with look-alikes (other names and numbers, the same shape: other true facts).

After one consolidation it counts, per fact (an original or a look-alike, with its
restatements), the rows left: two or more is a duplicate kept; none is a fact lost.

Scenarios:
  * per-tenant                  each tenant's facts and restatements in its own workspace;
  * per-tenant + look-alikes     the same with look-alikes beside them;
  * one workspace, over 200      every tenant's, look-alikes included, in one workspace,
                                 past the 200 rows consolidation reads. The offline store
                                 scrolls in the order rows were written (restatements last);
                                 Qdrant scrolls by point id, so a live run reads a random 200.

Published, never a gate (exit 0).

Run (from orchestrator/):
    python -m evals.memory_consolidation
    python -m evals.memory_consolidation --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from collections import Counter
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from evals.memory_pipeline import offline_service, proper_names, with_lookalikes
from evals.memory_recall import MemoryDoc, load_corpus

KINDS = ("exact", "light", "reworded")
LOOKALIKES = 8
ROWS_READ = 200  # what run_sleep_time_consolidation reads per workspace (get_all_memories(limit=200))
_SYNONYMS = {
    "every": "each", "must": "has to", "prefers": "likes", "immediately": "right away", "starts": "begins",
    "all": "every one of the", "wants": "would like", "after": "once", "before": "ahead of",
    "annually": "once a year", "monthly": "each month", "uses": "relies on", "only": "just",
    "sent": "emailed", "handles": "deals with", "requires": "needs", "within": "inside",
}
_WORD = re.compile(r"\b[a-z]+\b")


def restatement(doc: MemoryDoc, kind: str) -> MemoryDoc:
    """The same fact said again, ``kind`` = exact, light or reworded."""
    if kind == "exact":
        text = doc.text
    elif kind == "light":
        text = f"As noted, {doc.text[0].lower()}{doc.text[1:]}"
    else:
        text = _WORD.sub(lambda m: _SYNONYMS.get(m.group(0), m.group(0)), doc.text)
    return replace(doc, memory_id=f"{doc.memory_id}#{kind}", text=text)


def _fact_of(memory_id: str) -> str:
    return memory_id.split("#", 1)[0]


async def _consolidate(service: Any, facts: List[MemoryDoc], restated: List[MemoryDoc]) -> Dict[str, Any]:
    """File the facts, then their restatements, in one workspace; consolidate once; count."""
    workspace = str(uuid.uuid4())
    rows_written = 0
    for d in [*facts, *restated]:
        stored = await service.store_long_term(workspace_id=workspace, content=d.text,
                                               metadata={"memory_id": d.memory_id, "type": d.content_type})
        if not stored.get("success"):
            raise RuntimeError(f"the store refused {d.memory_id}: {stored.get('error')}")
        rows_written += 0 if stored.get("deduped") else 1
    await service.run_sleep_time_consolidation(workspace_id=workspace)
    left = await service.get_all_memories(workspace, limit=rows_written + 1)
    rows_per_fact = Counter(_fact_of(str((m.get("metadata") or {}).get("memory_id"))) for m in left)
    restated_left = Counter(str((m.get("metadata") or {}).get("memory_id")).split("#", 1)[1]
                            for m in left if "#" in str((m.get("metadata") or {}).get("memory_id")))
    return {
        "facts": len(facts),
        "rows_written": rows_written,
        "rows_left": len(left),
        "duplicates_left": sum(n - 1 for n in rows_per_fact.values() if n > 1),
        "restatements_left": {kind: restated_left.get(kind, 0) for kind in KINDS},
        "facts_lost": sorted(d.memory_id for d in facts if rows_per_fact.get(d.memory_id, 0) == 0),
    }


def _scenarios(corpus: List[MemoryDoc]) -> List[Tuple[str, List[List[MemoryDoc]]]]:
    """Each scenario's workspaces, as lists of facts."""
    tenants = sorted({d.workspace_id for d in corpus})
    by_tenant = [[d for d in corpus if d.workspace_id == t] for t in tenants]
    scaled = with_lookalikes(corpus, LOOKALIKES)
    return [
        ("per-tenant", by_tenant),
        (f"per-tenant + {LOOKALIKES} look-alikes", [[d for d in scaled if d.workspace_id == t] for t in tenants]),
        (f"one workspace, over {ROWS_READ} rows", [scaled]),
    ]


def _total(results: List[Dict[str, Any]], label: str, originals: int) -> Dict[str, Any]:
    return {
        "scenario": label,
        "facts": sum(r["facts"] for r in results),
        "rows_written": sum(r["rows_written"] for r in results),
        "rows_left": sum(r["rows_left"] for r in results),
        "restatements_written": originals * (len(KINDS) - 1),   # exact ones never get a row
        "restatements_left": {k: sum(r["restatements_left"][k] for r in results) for k in KINDS},
        "duplicates_left": sum(r["duplicates_left"] for r in results),
        "facts_lost": sorted(m for r in results for m in r["facts_lost"]),
    }


def run_consolidation_eval(service_factory: Callable[[], Any] = offline_service) -> Dict[str, Any]:
    """Every scenario, each workspace on a fresh store."""
    corpus = load_corpus()
    restated_by_fact = {d.memory_id: [restatement(d, k) for k in KINDS] for d in corpus}
    loop = asyncio.new_event_loop()
    try:
        rows = []
        for label, workspaces in _scenarios(corpus):
            results = []
            for facts in workspaces:
                restated = [r for d in facts for r in restated_by_fact.get(d.memory_id, [])]
                results.append(loop.run_until_complete(_consolidate(service_factory(), facts, restated)))
            originals = sum(1 for ws in workspaces for d in ws if d.memory_id in restated_by_fact)
            rows.append(_total(results, label, originals))
    finally:
        loop.close()
    return {"names": len(proper_names(corpus)), "scenarios": rows}


def render(result: Dict[str, Any]) -> str:
    lines = [
        "# Memory-consolidation eval\n",
        "| Scenario | Facts | Rows written | Rows left | Light left | Reworded left | Duplicates left | Facts lost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in result["scenarios"]:
        left = s["restatements_left"]
        lines.append(f"| {s['scenario']} | {s['facts']} | {s['rows_written']} | {s['rows_left']} | "
                     f"{left['light']} | {left['reworded']} | {s['duplicates_left']} | {len(s['facts_lost'])} |")
    lines.append("\nExact restatements never get a row (the store folds them by content hash at write).")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    result = run_consolidation_eval()
    sys.stdout.write((json.dumps(result, indent=2) if args.json else render(result)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
