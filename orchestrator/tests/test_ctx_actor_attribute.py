"""The request principal lives on ``ctx.user`` — never on ``ctx`` itself.

2026-08-06: ``POST /api/documents/upload`` 500'd for the Inbuild UK client
with ``'RequestContext' object has no attribute 'clerk_user_id'``. PRD-168 S4
(2026-06-12) threaded "real actor identity" through write paths as
``ctx.clerk_user_id`` in three routes — but ``RequestContext`` has no such
attribute; the principal is ``ctx.user.clerk_user_id`` (``UserContext``).
Every request through those routes raised AttributeError from the day the
commit landed.

These tests pin both sides so the class of bug cannot return:

1. A static sweep of ``api/`` for ``ctx.clerk_user_id`` — the exact broken
   spelling — must find nothing.
2. The dataclass contracts: ``RequestContext`` must NOT gain the attribute
   (which would re-legalise the broken spelling and split the actor identity
   across two homes), and ``UserContext`` must keep ``clerk_user_id``.

Pure static/dataclass tests — no app boot, no DB.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from core.auth.dependencies import RequestContext, UserContext  # noqa: E402

# ``ctx.clerk_user_id`` / ``context.clerk_user_id`` — reading the principal
# off the request context object instead of its ``.user``.
_BROKEN_SPELLING = re.compile(r"\bctx\.clerk_user_id\b|\bcontext\.clerk_user_id\b")


def test_no_route_reads_clerk_user_id_off_the_context():
    offenders: list[str] = []
    for py in sorted((_orchestrator_root / "api").rglob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if _BROKEN_SPELLING.search(line):
                offenders.append(f"{py.relative_to(_orchestrator_root)}:{i}: {line.strip()}")
    assert not offenders, (
        "The principal lives on ctx.user, not ctx — these lines raise "
        "AttributeError on every request:\n" + "\n".join(offenders)
    )


def test_request_context_has_no_clerk_user_id_field():
    names = {f.name for f in dataclasses.fields(RequestContext)}
    assert "clerk_user_id" not in names, (
        "Adding clerk_user_id to RequestContext splits the actor identity "
        "across two homes — it lives on UserContext"
    )


def test_user_context_keeps_clerk_user_id():
    names = {f.name for f in dataclasses.fields(UserContext)}
    assert "clerk_user_id" in names
