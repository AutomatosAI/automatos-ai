"""
Soft-delete then purge a workspace that ``create_test_workspace.py`` created.

The admin route (``DELETE /api/admin/workspaces/{id}``) needs a signed-in
admin, which an API key is not, so the simulation harness (PRD-247) tears its
throwaway workspaces down through the same two steps the route takes:
mark ``deleted_at``, then ``services.workspace_purge.purge_workspace_sync``.

Guarded on purpose: it refuses any workspace whose ``settings.managed_by`` is
not ``create_test_workspace.py`` or whose ``settings.purpose`` does not start
with ``--expect-purpose`` (default ``sim``). ``--force`` overrides both —
that is an operator decision, never the harness's.

Usage (from the orchestrator root, or piped into the backend container):
    python scripts/purge_test_workspace.py --workspace-id <uuid> [--expect-purpose sim] [--force]
    docker exec -i -w /app automatos_backend python - --workspace-id <uuid> < orchestrator/scripts/purge_test_workspace.py

Prints one JSON line on stdout with the outcome; exit 0 purged, 1 error,
2 refused by the guard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping
from uuid import UUID

MANAGED_BY = "create_test_workspace.py"
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


def is_purgeable(settings: Mapping[str, Any] | None, expect_purpose: str) -> tuple[bool, str]:
    """The guard, kept free of database imports so it can be unit-tested anywhere."""
    values = settings if isinstance(settings, Mapping) else {}
    if values.get("managed_by") != MANAGED_BY:
        return False, f"settings.managed_by is {values.get('managed_by')!r}, not {MANAGED_BY!r}"
    purpose = str(values.get("purpose") or "")
    if not purpose.startswith(expect_purpose):
        return False, f"settings.purpose {purpose!r} does not start with {expect_purpose!r}"
    return True, "ok"


def _bootstrap_path() -> None:
    # Runs as a file (``__file__`` is scripts/…) or from stdin inside the container (cwd is /app).
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
    sys.path.insert(0, os.path.abspath(os.path.join(here, "..")) if here else os.getcwd())


def _result_dict(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    return {k: v for k, v in vars(result).items() if not k.startswith("_")} if hasattr(result, "__dict__") else {"result": str(result)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="soft-delete and purge a test workspace")
    parser.add_argument("--workspace-id", required=True, type=UUID)
    parser.add_argument("--expect-purpose", default="sim", help="settings.purpose must start with this")
    parser.add_argument("--force", action="store_true", help="skip the managed_by/purpose guard (operator decision)")
    args = parser.parse_args(argv)

    _bootstrap_path()
    from sqlalchemy import text  # noqa: E402

    from core.database.database import SessionLocal  # noqa: E402
    from services.workspace_purge import purge_workspace_sync  # noqa: E402

    db = SessionLocal()
    try:
        row = db.execute(
            text("SELECT id, slug, settings, deleted_at FROM workspaces WHERE id = :id"), {"id": str(args.workspace_id)}
        ).fetchone()
        if row is None:
            print(json.dumps({"error": "workspace not found", "workspace_id": str(args.workspace_id)}))
            return EXIT_ERROR
        settings = row.settings if isinstance(row.settings, dict) else json.loads(row.settings or "{}")
        ok, why = is_purgeable(settings, args.expect_purpose)
        if not ok and not args.force:
            print(json.dumps({"error": f"refused: {why}", "workspace_id": str(row.id), "slug": row.slug}))
            return EXIT_REFUSED
        if row.deleted_at is None:
            db.execute(text("UPDATE workspaces SET deleted_at = NOW(), is_active = false WHERE id = :id"),
                       {"id": str(row.id)})
            db.commit()
    except Exception as exc:  # noqa: BLE001 — report, never hide
        db.rollback()
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "workspace_id": str(args.workspace_id)}))
        return EXIT_ERROR
    finally:
        db.close()

    result = purge_workspace_sync(args.workspace_id)
    payload = {"workspace_id": str(args.workspace_id), "slug": row.slug, "purged": True, **_result_dict(result)}
    print(json.dumps(payload, default=str))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
