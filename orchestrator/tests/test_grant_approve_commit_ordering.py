"""The approval endpoint must commit the yes BEFORE resuming the call.

2026-08-06 incident (grant 77, Inbuild UK): a human clicked Approve on a
gated ``platform_delete_agent``; the resume re-dispatched the stored call;
the confirmation gate re-asked; nothing executed. Root cause: SessionLocal
is ``autoflush=False``. ``grant_grant`` mutates ``status`` only in the
identity map, and the gate's ``consume_tool_grant`` → ``find_active_grant``
runs REAL SQL — the database still said ``pending``, so the gate could not
see the approval it was resuming under. Evidence: grant 77 sat GRANTED and
unexpired with ``revoked_by`` empty — destructive consumes stamp
``system:consumed``, so the gate provably never matched it.

The existing S4 suite (test_p2w2_grant_resume.py) mocks the executor with a
fake session, so the re-dispatch contract is pinned but the gate's SQL
visibility is out of frame — which is exactly where this broke. This test
pins the fix at the seam that failed: the endpoint's ordering. Same AST
pattern as test_p2w2_cors_boot_guard's lifespan placement pin.
"""

from __future__ import annotations

import ast
from pathlib import Path

_API = Path(__file__).resolve().parent.parent / "api" / "approval_grants.py"


def _grant_approval_call_order() -> list[str]:
    """The ordered relevant calls inside grant_approval, by name."""
    tree = ast.parse(_API.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == "grant_approval"
    )
    interesting = {"grant_grant", "_requeue_subject", "commit"}
    order: list[str] = []
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        name = None
        if isinstance(call.func, ast.Name):
            name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            name = call.func.attr
        if name in interesting:
            order.append((call.lineno, name))  # type: ignore[arg-type]
    return [n for _, n in sorted(order)]  # type: ignore[misc]


def test_approval_is_committed_before_the_resume():
    order = _grant_approval_call_order()
    assert "grant_grant" in order and "_requeue_subject" in order, order

    grant_i = order.index("grant_grant")
    resume_i = order.index("_requeue_subject")
    assert grant_i < resume_i, f"approve must precede resume: {order}"

    between = order[grant_i + 1 : resume_i]
    assert "commit" in between, (
        "grant_approval must db.commit() between grant_grant and "
        "_requeue_subject — with autoflush=False the gate's SQL cannot see "
        f"an uncommitted GRANTED row and re-asks (grant 77). Order: {order}"
    )


def test_executed_result_still_persisted_after_resume():
    """The follow-up commit (executed_result + audit path) must survive."""
    order = _grant_approval_call_order()
    resume_i = order.index("_requeue_subject")
    assert "commit" in order[resume_i + 1 :], (
        f"executed_result must be committed after the resume. Order: {order}"
    )
