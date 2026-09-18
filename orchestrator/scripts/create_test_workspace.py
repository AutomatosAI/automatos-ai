"""
Create a dedicated test workspace for the nightly API test suite.

The nightly test suite (`tests/run_nightly.py`) runs ~143 mutating HTTP
calls (POST/PUT/PATCH/DELETE) against `WORKSPACE_ID` from `tests/.env`.
Until this script is run, that target is the user's real workspace —
which means every night agents, personas, BYOK keys, channels, routing
rules, missions are created and (mostly) cleaned up in their live env.

This script provisions an isolated workspace named "TEST - Nightly Suite"
plus a service-type SDK API key, and prints both so they can be dropped
into `tests/.env` and the nightly recipe.

Idempotent: re-running finds the existing workspace by slug instead of
creating a duplicate. Re-running ALWAYS mints a fresh API key (the old
plaintext is unrecoverable, so this is the only way to recover access
if the key is lost).

Usage:
    DATABASE_URL=postgres://... python orchestrator/scripts/create_test_workspace.py
    # a differently named workspace (the simulation harness, PRD-247, names its
    # throwaway workspaces sim-<pack>-<date> and purges them afterwards):
    python orchestrator/scripts/create_test_workspace.py --slug sim-smoke-20260918 \
        --name "SIM smoke 20260918" --key-name "sim key" --purpose sim:smoke
    # or piped into the backend container, no DATABASE_URL needed on the host:
    docker exec -i -w /app automatos_backend python - --slug ... < orchestrator/scripts/create_test_workspace.py

Output (stdout):
    WORKSPACE_ID=<uuid>
    API_KEY=ak_srv_xxxxx...

Pipe straight into tests/.env if you want:
    python orchestrator/scripts/create_test_workspace.py >> tests/.env
"""

from __future__ import annotations

import argparse
import os
import sys
from uuid import uuid4

# Allow running as `python orchestrator/scripts/create_test_workspace.py`, or
# piped into `python -` from the orchestrator root (no __file__ then).
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")) if _HERE else os.getcwd())

from core.database.database import SessionLocal  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.services.api_key_service import ApiKeyService  # noqa: E402


WORKSPACE_NAME = "TEST - Nightly Suite"
WORKSPACE_SLUG = "test-nightly-suite"
KEY_NAME = "Nightly Test Suite Key"
PURPOSE = "nightly-test-suite"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="create (or find by slug) an isolated test workspace + service key")
    parser.add_argument("--slug", default=WORKSPACE_SLUG)
    parser.add_argument("--name", default=WORKSPACE_NAME)
    parser.add_argument("--key-name", default=KEY_NAME)
    parser.add_argument("--purpose", default=PURPOSE, help="stored in settings.purpose; the purge script checks it")
    parser.add_argument("--no-key", action="store_true",
                        help="create/find the workspace only; mint no API key (the simulation harness runs as the "
                             "anonymous local operator and scopes by X-Workspace-ID)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    db = SessionLocal()
    try:
        existing = (
            db.query(Workspace)
            .filter(Workspace.slug == args.slug)
            .first()
        )

        if existing:
            workspace = existing
            print(f"# Found existing workspace: {workspace.id}", file=sys.stderr)
        else:
            # owner_id is NOT NULL in DB schema (model is out of date).
            # Reuse the owner of the most recently created active workspace
            # so the test workspace is owned by a real user.
            template = (
                db.query(Workspace)
                .filter(
                    Workspace.is_active.is_(True),
                    Workspace.owner_id.isnot(None),
                )
                .order_by(Workspace.created_at.desc())
                .first()
            )
            if template is None:
                raise RuntimeError(
                    "No active workspace with owner_id found — cannot infer owner."
                )

            workspace = Workspace(
                id=uuid4(),
                name=args.name,
                slug=args.slug,
                owner_id=template.owner_id,
                plan="basic",  # PRD-222 W2·S1: entry tier (renamed from 'starter')
                is_personal=False,
                is_active=True,
                webhook_key=uuid4().hex,
                settings={
                    "purpose": args.purpose,
                    "managed_by": "create_test_workspace.py",
                    "warning": (
                        "This workspace is mutated nightly by the API test "
                        "suite. Do not store production data here."
                    ),
                },
            )
            db.add(workspace)
            db.flush()
            print(
                f"# Created new workspace: {workspace.id} "
                f"(owner_id={template.owner_id})",
                file=sys.stderr,
            )

        if args.no_key:
            db.commit()
            print(f"WORKSPACE_ID={workspace.id}")
            print("# no API key minted (--no-key)", file=sys.stderr)
            return 0

        # key_type="server" (private/admin). PRD-195 S1: empty/None permissions
        # now grant NOTHING (least privilege, one semantic on every plane) — a
        # test key that should reach everything must carry the explicit "*"
        # full grant (honoured by modules.policy.roles.has_permission).
        key_result = ApiKeyService.create_api_key(
            db=db,
            workspace_id=workspace.id,
            name=args.key_name,
            key_type="server",
            permissions=["*"],
        )
        db.commit()

        print(f"WORKSPACE_ID={workspace.id}")
        print(f"API_KEY={key_result['key']}")
        print(
            f"# key_prefix={key_result['key_prefix']} "
            f"key_id={key_result['id']}",
            file=sys.stderr,
        )
        return 0

    except Exception as exc:
        db.rollback()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
