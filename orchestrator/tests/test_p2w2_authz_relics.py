"""PRD-195 S8 (P2-14) — the 5→1 collapse is FINISHED: relics stay dead.

Deletion guards (the PRD-185 S5 import-regression shape — repoint, don't
delete, if a refactor renames targets):

- the agent-tool RBAC fossil (``api/permissions.py``, vocabulary #5) is
  unimportable and unmounted; its models are gone from ``core.models``;
- no dangling reference to the fossil's symbols survives anywhere in the
  backend tree (alembic's DROP migration and these tests excepted);
- the frontend ghost role ``customer_manager`` (vocabulary #3's invention)
  is gone; the frontend SystemRole set matches the backend's;
- SDK ``VALID_PERMISSIONS`` speak the canonical G1 vocabulary — no legacy
  ``workflows:*`` strings anywhere a checker or picker reads them.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

RELIC_SYMBOLS = (
    "AgentToolPermission",
    "PermissionAuditLog",
    "AGENT_TOOL_PERMISSIONS",
)
# Deleted ORM models whose NAMES also exist legitimately elsewhere (the
# modules/tools registry Enum is called ToolCategory) — swept only as
# core.models attributes, not as tree-wide strings.
RELIC_MODEL_ATTRS = RELIC_SYMBOLS[:2] + ("ToolConfiguration", "ToolCategory")

_SKIP_DIRS = {"__pycache__", "alembic", "node_modules", ".git"}


def _py_files():
    for root, dirs, files in os.walk(_ORCH):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def test_removed_authz_relics_unimportable():
    assert importlib.util.find_spec("api.permissions") is None, (
        "api/permissions.py (the fossil router) is importable again"
    )

    import core.models as models

    for attr in RELIC_MODEL_ATTRS:
        assert not hasattr(models, attr), f"core.models.{attr} survived the S8 deletion"

    # The live tier reader keeps its model — deletion was surgical.
    assert hasattr(models, "Tool"), "Tool (live PRD-123 tier reader) must survive"


def test_fossil_router_unmounted():
    main_src = (_ORCH / "main.py").read_text(encoding="utf-8")
    assert "api.permissions" not in main_src
    assert "permissions_router" not in main_src
    manifest = (_ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8")
    assert '"/permissions' not in manifest, "fossil routes survived in the committed manifest"


def test_no_dangling_relic_imports():
    this_file = Path(__file__).resolve()
    offenders = []
    for path in _py_files():
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for sym in RELIC_SYMBOLS:
            if sym in text:
                offenders.append(f"{path.relative_to(_ORCH)}: {sym}")
    assert not offenders, "dangling fossil references:\n  " + "\n  ".join(offenders)


def test_frontend_role_type_matches_backend():
    ghost_hits = []
    fe = _REPO / "frontend"
    for root, dirs, files in os.walk(fe):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".next", ".git")]
        for f in files:
            if f.endswith((".ts", ".tsx")):
                p = Path(root) / f
                if "customer_manager" in p.read_text(encoding="utf-8", errors="ignore"):
                    ghost_hits.append(str(p.relative_to(_REPO)))
    assert not ghost_hits, f"customer_manager ghost survived in: {ghost_hits}"

    role_ctx = (fe / "contexts" / "role-context.tsx").read_text(encoding="utf-8")
    assert "'super_admin' | 'admin' | 'user'" in role_ctx, (
        "frontend SystemRole no longer mirrors the backend system-role set"
    )


def test_user_context_role_twin_collapsed():
    """G1: UserContext carries ONE role field — ``system_role``. The ``role``
    init parameter is still ACCEPTED (core/auth/hybrid.py — the do-not-modify
    shared dependency — passes it at four mint sites) but is discarded; there
    is no ``.role`` attribute to read anymore."""
    from core.auth.dependencies import UserContext

    u = UserContext(id="u", email=None, role="admin", system_role="user")
    assert u.system_role == "user"
    assert not hasattr(u, "role"), (
        ".role survived the twin collapse — nothing may read it"
    )
    # default construction (the anonymous lane) still works
    assert UserContext().system_role == "user"


def test_current_workspace_returns_real_workspace_role():
    """C.1: /api/workspaces/current used to return the SYSTEM-role twin while
    the frontend typed it as a workspace role. Source pin: it now resolves the
    per-tenant role (repoint if api/workspaces.py refactors)."""
    src = (_ORCH / "api" / "workspaces.py").read_text(encoding="utf-8")
    assert "resolve_workspace_role" in src
    assert '"role": ctx.user.role' not in src

    # G5 companion: the frontend provider derives the viewer read-only flag
    # from that role and the marquee write affordances consume it.
    provider = (_REPO / "frontend" / "components" / "workspace-provider.tsx").read_text(
        encoding="utf-8"
    )
    assert "canEdit" in provider
    for component in (
        Path("frontend") / "components" / "agents" / "agent-management.tsx",
        Path("frontend") / "components" / "missions" / "mission-list.tsx",
        Path("frontend") / "components" / "documents" / "document-management.tsx",
        Path("frontend") / "components" / "deliverables" / "deliverables-blogs.tsx",
    ):
        text = (_REPO / component).read_text(encoding="utf-8")
        assert "canEdit" in text, f"{component} lost its viewer read-only affordance"


def test_valid_permissions_speak_canonical_vocabulary():
    from api.api_keys import VALID_PERMISSIONS

    legacy = [p for p in VALID_PERMISSIONS if p.startswith("workflows:")]
    assert not legacy, f"legacy workflows:* scopes survived: {legacy}"
    for expected in ("missions:read", "missions:execute", "playbooks:read", "playbooks:execute"):
        assert expected in VALID_PERMISSIONS

    for rel in (
        Path("frontend") / "components" / "settings" / "ApiKeyManager.tsx",
        Path("frontend") / "app" / "marketplace" / "publish" / "page.tsx",
    ):
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert "workflows:read" not in text and "workflows:execute" not in text, (
            f"{rel} still offers legacy workflows:* scopes"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
