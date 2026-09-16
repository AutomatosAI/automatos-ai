"""PRD-242 S1 — one principal→PK resolver, and the document routes use it.

``UserContext.id`` is the Clerk subject string (SaaS) or the operator EMAIL
(local, PRD-233 S6) — never ``users.id``. ``GET /api/documents/variables`` (and
the preview / generate routes) handed it straight to ``User.id == …``, so the
Template Studio 500'd in both editions — surfaced in the browser as
"Failed to fetch" because the 500 carried no CORS header.

Pure: the db is a MagicMock; the request context is a stub.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE, resolve_user_pk

_API = Path(__file__).resolve().parent.parent / "api"


def _ctx(**user):
    return SimpleNamespace(user=SimpleNamespace(**user))


def test_int_id_fast_path_needs_no_query():
    db = MagicMock()
    assert resolve_user_pk(db, _ctx(id=42, email="x@y")) == 42
    db.execute.assert_not_called()


def test_local_operator_lane_uses_raw_claims_pk_without_a_query():
    # PRD-233 S6 binds id=email and stashes the integer PK on raw_claims.
    db = MagicMock()
    ctx = _ctx(
        id="local@automatos.local",
        email="local@automatos.local",
        raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": 1},
    )
    assert resolve_user_pk(db, ctx) == 1
    db.execute.assert_not_called()


def test_a_clerk_jwt_claim_named_user_id_is_never_trusted_as_the_pk():
    # The Clerk lane's raw_claims is the raw JWT payload. A custom claim that
    # merely happens to be called user_id must go through the DB lookup, never
    # be adopted as our PK (security review, PRD-242).
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (17,)
    ctx = _ctx(id="user_38Zabc", email="jane@acme.com", clerk_user_id="user_38Zabc", raw_claims={"user_id": 1, "sub": "user_38Zabc"})
    assert resolve_user_pk(db, ctx) == 17
    db.execute.assert_called_once()


def test_hybrid_local_operator_context_emits_the_claim_source_contract():
    # The resolver trusts raw_claims["user_id"] only when hybrid.py stamps the source.
    src = (Path(__file__).resolve().parent.parent / "core" / "auth" / "hybrid.py").read_text(encoding="utf-8")
    assert f'"source": "{LOCAL_OPERATOR_CLAIM_SOURCE}"' in src


def test_clerk_string_resolves_through_clerk_user_id_then_email():
    db = MagicMock()
    db.execute.return_value.fetchone.side_effect = [None, (17,)]
    ctx = _ctx(id="user_38Zabc", email="jane@acme.com", clerk_user_id="user_38Zabc", raw_claims=None)
    assert resolve_user_pk(db, ctx) == 17
    assert db.execute.call_count == 2
    first_params = db.execute.call_args_list[0].args[1]
    second_params = db.execute.call_args_list[1].args[1]
    assert first_params == {"value": "user_38Zabc"}
    assert second_params == {"value": "jane@acme.com"}


def test_unresolvable_or_principal_less_is_none_not_user_one():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    assert resolve_user_pk(db, None) is None
    assert resolve_user_pk(db, SimpleNamespace(user=None)) is None
    assert resolve_user_pk(db, _ctx(id="user_missing", email=None, clerk_user_id=None, raw_claims=None)) is None


def test_bool_is_not_an_int_pk():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    assert resolve_user_pk(db, _ctx(id=True, email=None, clerk_user_id=None, raw_claims={"user_id": False})) is None


def _user_id_kwargs_in(module: str) -> list[str]:
    """Every ``user_id=<expr>`` keyword passed in the module, as source text."""
    tree = ast.parse((_API / module).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "user_id":
                    found.append(ast.unparse(kw.value))
    return found


def test_document_routes_never_pass_ctx_user_id_as_user_id():
    """The regression guard: no route hands the principal string to an integer column."""
    exprs = _user_id_kwargs_in("document_generation.py")
    assert exprs, "expected user_id keywords in document_generation.py"
    assert all("ctx.user.id" not in e for e in exprs), exprs
    assert any("resolve_user_pk" in e for e in exprs)


def test_positional_resolver_calls_in_document_routes_use_resolve_user_pk():
    src = (_API / "document_generation.py").read_text(encoding="utf-8")
    assert "ctx.user.id if ctx.user else None" not in src
