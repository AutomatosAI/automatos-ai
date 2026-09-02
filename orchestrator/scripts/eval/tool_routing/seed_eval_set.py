"""
Seed the eval set from eval_seed.yaml.

Reads human-curated queries, validates that every `correct_actions` entry
exists in the live ActionRegistry (or is a known top-level tool like
`composio_execute` / `search_knowledge` / `workspace_*`), and emits
`eval_set.jsonl` for the runner.

Run from the orchestrator/ directory so PYTHONPATH resolves:

    cd orchestrator
    python -m scripts.eval.tool_routing.seed_eval_set
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

HERE = Path(__file__).resolve().parent
SEED_YAML = HERE / "eval_seed.yaml"
OUT_JSONL = HERE / "eval_set.jsonl"


# Top-level tools that aren't ActionDefinitions but ARE valid choices the
# LLM can make through `platform_execute`-adjacent paths. Keep in sync
# with the production tool catalog.
KNOWN_TOP_LEVEL_TOOLS: Set[str] = {
    "composio_execute",
    "search_knowledge",
    "semantic_search",
    "generate_document",
    "write_file",
    "smart_query_database",
    "query_database",
    "workspace_read_file",
    "workspace_write_file",
    "workspace_list_dir",
    "workspace_grep",
    "workspace_exec",
    "workspace_git",
}


def _load_registry_action_names() -> Set[str]:
    """Pull all live ActionDefinition names from the orchestrator registry."""
    try:
        from scripts.eval.tool_routing._registry_bootstrap import load_registry
        registry = load_registry()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[seed_eval_set] Could not load ActionRegistry: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    return {a.name for a in registry.get_all()}


def _validate(queries: List[Dict[str, Any]], known: Set[str]) -> List[str]:
    """Return a list of validation errors. Empty list means all good."""
    errors: List[str] = []
    seen_qs: Set[str] = set()

    for i, entry in enumerate(queries):
        prefix = f"[#{i}] "

        q = entry.get("q", "").strip()
        if not q:
            errors.append(prefix + "missing or empty 'q'")
            continue
        if q in seen_qs:
            errors.append(prefix + f"duplicate query: {q!r}")
        seen_qs.add(q)

        # PRD-232 US-012A: abstain rows expect NO tool call (no correct action
        # applies), so they must carry an empty correct_actions list.
        abstain = bool(entry.get("abstain"))
        correct = entry.get("correct_actions") or []
        if abstain:
            if correct:
                errors.append(prefix + "abstain row must have empty 'correct_actions'")
            continue

        if not isinstance(correct, list) or not correct:
            errors.append(prefix + "'correct_actions' must be a non-empty list")
            continue

        for name in correct:
            if name not in known and name not in KNOWN_TOP_LEVEL_TOOLS:
                errors.append(
                    prefix + f"unknown action: {name!r} for query {q!r}"
                )

    return errors


def main() -> int:
    if not SEED_YAML.exists():
        print(f"[seed_eval_set] missing {SEED_YAML}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(SEED_YAML.read_text())
    queries = raw.get("queries") or []
    if not queries:
        print("[seed_eval_set] no queries in eval_seed.yaml", file=sys.stderr)
        return 2

    known = _load_registry_action_names()
    print(f"[seed_eval_set] {len(known)} actions registered in live registry")

    errors = _validate(queries, known)
    if errors:
        print(f"[seed_eval_set] {len(errors)} validation error(s):", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        return 1

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for i, entry in enumerate(queries):
            row = {
                "query_id": f"q{i:03d}",
                "query": entry["q"].strip(),
                "correct_actions": entry.get("correct_actions") or [],
                "category": entry.get("category", "uncategorized"),
                "difficulty": entry.get("difficulty", "easy"),
                # PRD-232 US-012A: abstain rows (no applicable tool) — the runner
                # lets the model NOT call a tool and the scorer counts a no-call
                # as correct for these rows only.
                "abstain": bool(entry.get("abstain", False)),
                "notes": entry.get("notes", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[seed_eval_set] wrote {len(queries)} queries → {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
