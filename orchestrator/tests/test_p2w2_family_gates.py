"""PRD-195 S3–S6 (P2-14) — per-family representative viewer/editor pairs.

Two layers per representative route, both pure:

1. **Wiring** — the route decorator in source carries the exact
   ``require_workspace_permission("<perm>")`` dependency (the boundary sweep
   already proves a marker exists on every mutating route; this pins the
   *string* on the family's marquee routes so a refactor can't silently swap
   ``agents:create`` for something a viewer holds).
2. **Semantics** — the gate itself, driven directly with a fake ctx +
   MagicMock session (the ``test_p2w0_cockpit_reach.py`` idiom): viewer 403,
   editor (or admin, where the matrix says owner/admin) passes.

Representative, not exhaustive — exhaustive coverage is the boundary sweep's
job. Rows are grouped by family story.
"""
from __future__ import annotations

import asyncio
import os
import re
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

from core.auth.workspace_permission import (  # noqa: E402
    require_workspace_permission,
    workspace_permission_granted,
)

# (family, module, route-path-regex-fragment, permission, allowed_role)
# allowed_role: the LOWEST role the matrix grants the permission to — proven
# to pass while viewer is proven to 403.
REPRESENTATIVES = [
    # ---- S3 · agents, skills & tools plane --------------------------------
    ("S3", "api/agents.py", r'"/"', "agents:create", "editor"),
    ("S3", "api/agents.py", r'"/\{agent_id\}/execute"', "agents:execute", "editor"),
    ("S3", "api/agents.py", r'"/\{agent_id\}"', "agents:delete", "admin"),
    ("S3", "api/workspace_skills.py", r'"/\{skill_id\}"', "agents:update", "editor"),
    ("S3", "api/tools.py", r'"/add-to-workspace"', "workspace:manage", "admin"),
    ("S3", "api/chat.py", r'""', "agents:execute", "editor"),
]


def _ctx(clerk_user_id="clerk_x"):
    return SimpleNamespace(
        user=SimpleNamespace(
            id="u", email=None, role="user", system_role="user",
            clerk_user_id=clerk_user_id,
        ),
        workspace_id=uuid.uuid4(),
        auth_type="clerk",
    )


def _db(role):
    db = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = (role,)
    db.execute.return_value = result
    return db


@pytest.mark.parametrize(
    "story,module,path_re,perm,allowed_role",
    REPRESENTATIVES,
    ids=[f"{s}:{m.split('/')[-1]}:{p}" for s, m, _, p, _ in REPRESENTATIVES],
)
def test_representative_route_wiring_and_semantics(story, module, path_re, perm, allowed_role):
    # 1) Wiring: the decorator carries the exact permission string.
    src = (_ORCH / module).read_text(encoding="utf-8")
    pattern = re.compile(
        r"@\w+\.(?:get|post|put|patch|delete)\(\s*\n?\s*" + path_re
        + r"[^@]*?require_workspace_permission\(\""
        + re.escape(perm) + r"\"\)",
        re.S,
    )
    assert pattern.search(src), (
        f"{module}: no route matching {path_re} gated with {perm!r}"
    )

    # 2) Semantics: viewer 403s, the lowest granted role passes.
    assert workspace_permission_granted(_db("viewer"), _ctx(), perm) is False
    assert workspace_permission_granted(_db(allowed_role), _ctx(), perm) is True

    dep = require_workspace_permission(perm)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(dep(SimpleNamespace(path_params={}), ctx=_ctx(), db=_db("viewer")))
    assert ei.value.status_code == 403
    ok_ctx = _ctx()
    assert asyncio.run(
        dep(SimpleNamespace(path_params={}), ctx=ok_ctx, db=_db(allowed_role))
    ) is ok_ctx


def test_read_shaped_posts_stay_viewer_reachable():
    """Deliberate exceptions, documented: a handful of POST routes are pure
    queries/telemetry (search, retrieval, cost estimation) — they carry
    ``<resource>:read`` so a viewer's read-only surface keeps working. The
    sweep still requires the marker; this pins the deliberate strings."""
    for module, path_re, perm in [
        ("api/models_endpoints.py", r'"/estimate-cost"', "agents:read"),
        ("api/query.py", r'"/platform-help"', "agents:read"),
    ]:
        src = (_ORCH / module).read_text(encoding="utf-8")
        pattern = re.compile(
            r"@\w+\.post\(\s*\n?\s*" + path_re
            + r"[^@]*?require_workspace_permission\(\""
            + re.escape(perm) + r"\"\)",
            re.S,
        )
        assert pattern.search(src), f"{module} {path_re} should carry {perm}"
    # and a viewer indeed passes a :read gate
    assert workspace_permission_granted(_db("viewer"), _ctx(), "agents:read") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
