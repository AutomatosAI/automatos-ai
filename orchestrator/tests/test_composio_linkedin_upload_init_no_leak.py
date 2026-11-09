"""GET /api/composio/linkedin/test-upload-init never echoes its own
exception text (CodeQL py/stack-trace-exposure #206).

A workspace-admin-gated smoke test, but still a diagnostic endpoint that
answers over HTTP — a failure's text (an internal URL, a library version,
a credential-store detail) stayed server-side after this fix.
"""
from __future__ import annotations

import asyncio
import os
import uuid

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from types import SimpleNamespace  # noqa: E402

import api.composio as composio_api  # noqa: E402
import core.composio.deny_list as deny_list  # noqa: E402
import core.composio.linkedin_image_workaround as linkedin  # noqa: E402

SECRET = "internal-host db-primary.vpc.internal:5432 refused connection"


def test_a_credential_load_failure_does_not_leak_its_text(monkeypatch):
    async def _not_denied(slug):
        return None

    def _boom(workspace_id):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(deny_list, "composio_action_denial_async", _not_denied)
    monkeypatch.setattr(linkedin, "_load_linkedin_credentials", _boom)

    ctx = SimpleNamespace(workspace_id=uuid.uuid4())
    result = asyncio.run(composio_api.test_linkedin_upload_init(ctx=ctx))

    assert result["ok"] is False
    assert SECRET not in result["error"]
    assert result["error"] == "LinkedIn upload-init failed"
