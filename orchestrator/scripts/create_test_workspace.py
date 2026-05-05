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

Output (stdout):
    WORKSPACE_ID=<uuid>
    API_KEY=ak_srv_xxxxx...

Pipe straight into tests/.env if you want:
    python orchestrator/scripts/create_test_workspace.py >> tests/.env
"""

from __future__ import annotations

import os
import sys
from uuid import uuid4

# Allow running as `python orchestrator/scripts/create_test_workspace.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.database.database import SessionLocal  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.services.api_key_service import ApiKeyService  # noqa: E402


WORKSPACE_NAME = "TEST - Nightly Suite"
WORKSPACE_SLUG = "test-nightly-suite"
KEY_NAME = "Nightly Test Suite Key"


def main() -> int:
    db = SessionLocal()
    try:
        existing = (
            db.query(Workspace)
            .filter(Workspace.slug == WORKSPACE_SLUG)
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
                name=WORKSPACE_NAME,
                slug=WORKSPACE_SLUG,
                owner_id=template.owner_id,
                plan="starter",
                is_personal=False,
                is_active=True,
                webhook_key=uuid4().hex,
                settings={
                    "purpose": "nightly-test-suite",
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

        # key_type="server" (private/admin); permissions=None grants all
        # — see ApiKeyService.check_permissions: empty perms = unrestricted
        key_result = ApiKeyService.create_api_key(
            db=db,
            workspace_id=workspace.id,
            name=KEY_NAME,
            key_type="server",
            permissions=None,
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
