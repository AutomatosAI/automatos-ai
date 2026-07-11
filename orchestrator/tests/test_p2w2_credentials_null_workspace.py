"""PRD-195 S7 (P2-14 / auth-identity J4, Sec §3.2.b) — credential NULL-workspace deny.

The BOLA gap: ``_check_credential_workspace`` used to treat a credential row
with ``workspace_id IS NULL`` as belonging to EVERY tenant, and by-name
resolution ignored workspaces entirely ("For MVP: just find the credential by
name"). Closed here:

- NULL workspace ⇒ deny for non-admin callers (404 — existence-hiding); the
  system-admin hierarchy and the env-API-key service lane may still read
  globally-seeded rows (G3);
- ``get_credential_by_name`` filters on the caller's workspace (global rows
  only via the explicit ``include_global`` admin lane, own-workspace row
  preferred on a name collision);
- the sound by-id mismatch check keeps behaving.

Pure: MagicMock store/session, no DB.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api.credentials import _check_credential_workspace  # noqa: E402


def _ctx(system_role="user", auth_type="clerk", workspace_id=None):
    return SimpleNamespace(
        user=SimpleNamespace(
            id="u", email=None, role="user", system_role=system_role,
            clerk_user_id="clerk_x",
        ),
        workspace_id=workspace_id or uuid.uuid4(),
        auth_type=auth_type,
    )


def _cred(workspace_id):
    return SimpleNamespace(workspace_id=workspace_id)


# ---------------------------------------------------------------------------
# The NULL hole is closed
# ---------------------------------------------------------------------------

def test_null_workspace_credential_denied_cross_tenant():
    """Yesterday: a NULL-workspace row passed for every tenant. Today: 404 for
    a plain member; admins / the env-key service lane still read it (G3)."""
    with pytest.raises(HTTPException) as ei:
        _check_credential_workspace(_cred(None), _ctx())
    assert ei.value.status_code == 404

    # G3: globally-seeded rows stay readable for the admin hierarchy...
    _check_credential_workspace(_cred(None), _ctx(system_role="admin"))
    _check_credential_workspace(_cred(None), _ctx(system_role="super_admin"))
    # ...and the env-API-key service lane.
    _check_credential_workspace(_cred(None), _ctx(auth_type="api_key"))


def test_by_id_paths_unchanged():
    ws = uuid.uuid4()
    # own workspace row passes
    _check_credential_workspace(_cred(ws), _ctx(workspace_id=ws))
    # foreign non-null row 404s (existence-hiding) — even for system admins;
    # cross-workspace reads of TENANT rows were never allowed here.
    with pytest.raises(HTTPException) as ei:
        _check_credential_workspace(_cred(uuid.uuid4()), _ctx(workspace_id=ws))
    assert ei.value.status_code == 404


# ---------------------------------------------------------------------------
# By-name lookup is workspace-scoped
# ---------------------------------------------------------------------------

class _RecordingQuery:
    def __init__(self):
        self.filters = []
        self.orders = []

    def filter(self, *exprs):
        self.filters.extend(exprs)
        return self

    def order_by(self, *exprs):
        self.orders.extend(exprs)
        return self

    def first(self):
        return None


def _run_lookup(**kwargs):
    from core.credentials.service import CredentialStore

    store = CredentialStore.__new__(CredentialStore)
    q = _RecordingQuery()
    store.db = SimpleNamespace(query=lambda model: q)
    store.get_credential_by_name("shared-name", **kwargs)
    return " || ".join(str(e) for e in q.filters), [str(e) for e in q.orders]


def test_resolve_by_name_scoped():
    ws = uuid.uuid4()

    # Tenant lane: the workspace filter is on the wire.
    sql, _ = _run_lookup(workspace_id=ws)
    assert "workspace_id" in sql, f"no workspace scoping in: {sql}"
    assert "IS NULL" not in sql.upper(), "tenant lane must not see global rows"

    # Admin lane: own-workspace OR global, own row preferred.
    sql, orders = _run_lookup(workspace_id=ws, include_global=True)
    assert "workspace_id" in sql
    assert "IS NULL" in sql.upper()
    assert orders and "IS NULL" in orders[0].upper(), (
        "collision preference (own workspace first) missing"
    )

    # In-process platform lane (core/llm/manager): unscoped by design.
    sql, _ = _run_lookup()
    assert "workspace_id" not in sql


def test_resolve_endpoint_passes_caller_workspace():
    """Source pin: the resolve name-branch threads ctx.workspace_id into the
    scoped lookup (repoint if api/credentials.py refactors)."""
    src = (_ORCH / "api" / "credentials.py").read_text(encoding="utf-8")
    assert "workspace_id=ctx.workspace_id" in src, (
        "resolve_credential name-branch no longer scopes by caller workspace"
    )


def test_resolve_endpoint_foreign_row_404s():
    """End-to-end through the endpoint: a resolved row belonging to another
    workspace 404s even for the admin caller the in-handler gate admits."""
    from api.credentials import resolve_credential
    from core.models.credentials import CredentialResolveRequest

    ws = uuid.uuid4()
    foreign = SimpleNamespace(
        id=7, name="shared-name", credential_type_id=1,
        environment="production", workspace_id=uuid.uuid4(),
    )
    store = MagicMock()
    store.get_credential_by_name.return_value = foreign

    req = SimpleNamespace(headers={}, client=SimpleNamespace(host="1.2.3.4"))
    req.headers = MagicMock()
    req.headers.get.return_value = None

    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            resolve_credential(
                resolve_request=CredentialResolveRequest(
                    credential_name="shared-name", service_name="test-suite"
                ),
                request=req,
                ctx=_ctx(system_role="admin", workspace_id=ws),
                store=store,
            )
        )
    assert ei.value.status_code == 404
    # and the lookup itself was workspace-scoped
    kwargs = store.get_credential_by_name.call_args.kwargs
    assert kwargs.get("workspace_id") == ws


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
