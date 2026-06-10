"""PRD-155 S1 — deterministic backend route manifest generator.

Imports the FastAPI app and writes the sorted set of ``{method, path}`` pairs it
serves to ``reports/route-manifest.json``. This manifest is the backend half of
the route contract: the frontend path-extraction suite (S2) asserts every path
the UI calls is a subset of it.

DB-free by construction: SQLAlchemy ``create_engine`` opens no connection until
the first query, and the FastAPI lifespan (which calls ``init_database``) runs
only when the app serves — never on import. So ``from main import app`` reads
the route table without a live database, and this script runs in CI and the
acceptance gate with Postgres unreachable.

Determinism: the route list is sorted by ``(path, method)`` and serialised with
``sort_keys=True``; no timestamps or counters that vary between runs. Two
invocations produce byte-identical output.

Usage (from the ``orchestrator`` directory)::

    python3 -m scripts.dump_routes
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Methods FastAPI auto-adds for every route; they are not part of the contract
# the frontend calls against.
_IGNORED_METHODS = frozenset({"HEAD", "OPTIONS"})

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "reports" / "route-manifest.json"


def collect_pairs(app) -> List[Dict[str, str]]:
    """Flatten ``app.routes`` into deduplicated, sorted ``{method, path}`` pairs.

    HTTP routes expand to one entry per declared method (minus HEAD/OPTIONS).
    Method-less routes (WebSocket endpoints, sub-app mounts, static files) are
    emitted once with an empty method so the manifest is a faithful superset of
    the served surface.
    """
    seen: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path:
            continue
        methods = getattr(route, "methods", None)
        if methods:
            for method in methods:
                if method not in _IGNORED_METHODS:
                    seen.add((method, path))
        else:
            seen.add(("", path))
    return [
        {"method": method, "path": path}
        for path, method in sorted((p, m) for m, p in seen)
    ]


def build_manifest() -> Dict[str, Any]:
    # Lazy import: keeps app construction (and its heavy import chain) out of
    # module import, and makes the no-DB contract explicit — nothing here dials
    # the database.
    from main import app

    pairs = collect_pairs(app)
    return {"route_count": len(pairs), "routes": pairs}


def write_manifest(manifest: Dict[str, Any]) -> Path:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MANIFEST_PATH


def main() -> None:
    manifest = build_manifest()
    out = write_manifest(manifest)
    print(f"Wrote {manifest['route_count']} routes -> {out}")


if __name__ == "__main__":
    main()
