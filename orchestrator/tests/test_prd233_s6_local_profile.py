"""PRD-233 S6 — local profile: Auto knows who it is talking to.

The local edition has exactly one operator and no login. PRD-209 seeded the
``users`` row and made the anonymous lane carry its email; this story binds the
row (id / name / email) into the session, adds the profile API that edits it,
and threads the name into Auto's greeting through the path the platform already
had for names. Pins:

* hybrid.py local lane — the anonymous context carries the operator row (email
  as ``id`` — the tree's own Clerk-less binding — integer PK + name in
  ``raw_claims``); a missing row or a DB error keeps PRD-209's email-only lane
  (never a 500); the saas lane is untouched (``UserContext()``).
* the operator-row cache — one read per TTL; ``invalidate_local_operator_cache``
  (called by PUT /api/profile) and TTL expiry both force a re-read.
* GET/PUT /api/profile — local round-trip (PUT name → GET reflects, cache
  invalidated, next session carries the new name); email is read-only (422);
  no markup / no non-http avatar / no blank username (422); username clash
  (409); saas PUT is 403 "managed by your identity provider".
* greeting — ``resolve_known_user_name`` reads ``users.name`` by the INTEGER id
  (blank → None); ``atom_identity_clause`` mirrors the personality wording and
  is empty when unknown; IdentitySection renders "talking to <name>" from the
  ``user_name`` kwarg it already had, "ready to help" without; source guards
  pin the seam in consumers/chatbot/service.py.

DB-free: an in-memory SQLite ``users`` table stands in for Postgres. No
personal data — placeholder names only (public repo).
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain; the
# closed port refuses instantly and nothing here touches a real database.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.profile as profile_mod  # noqa: E402
import core.auth.hybrid as hybrid  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.core import User  # noqa: E402
from uuid import UUID  # noqa: E402

OPERATOR_EMAIL = "operator@example.test"
OPERATOR_ID = 1
SEEDED_NAME = "Local Operator"
LOCAL_WS = UUID("00000000-0000-0000-0000-0000000000aa")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def users_db():
    """An in-memory ``users`` table seeded like docker-entrypoint.sh does."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    User.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    session.add(
        User(id=OPERATOR_ID, username="local", email=OPERATOR_EMAIL, name=SEEDED_NAME, is_active=True)
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def local_lane(users_db, monkeypatch):
    """Drive hybrid.py into the local posture against the SQLite users table."""
    hybrid.invalidate_local_operator_cache()
    monkeypatch.setattr(hybrid.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(hybrid.config, "REQUIRE_AUTH", False)
    monkeypatch.setattr(hybrid.config, "DEFAULT_WORKSPACE_ID", str(LOCAL_WS))
    monkeypatch.setattr(hybrid.config, "WORKSPACE_ID", None)
    monkeypatch.setattr(hybrid.config, "AUTH_DEBUG", False)
    monkeypatch.setattr(hybrid.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    monkeypatch.setattr(hybrid, "SessionLocal", lambda: users_db)
    monkeypatch.setattr(hybrid, "_workspace_exists", lambda db, ws: True)
    monkeypatch.setattr(hybrid, "_assert_workspace_usable", lambda db, ws, *, is_admin: None)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    yield users_db
    hybrid.invalidate_local_operator_cache()


def _anonymous_request():
    request = MagicMock()
    request.method = "GET"
    request.headers = {}
    request.query_params = {}
    request.state = MagicMock(spec=[])
    return request


def _local_ctx():
    return RequestContext(
        workspace_id=LOCAL_WS,
        user=UserContext(id=OPERATOR_EMAIL, email=OPERATOR_EMAIL, system_role="super_admin"),
        auth_type="anonymous",
    )


def _client(db, ctx):
    app = FastAPI()
    app.include_router(profile_mod.router)
    app.dependency_overrides[hybrid.get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[profile_mod._PROFILE_WRITE_GATE] = lambda: ctx
    return TestClient(app)


def _chat_service_module():
    """consumers.chatbot.service, re-imported if a sibling left a stub."""
    mod = sys.modules.get("consumers.chatbot.service")
    if mod is None or not getattr(mod, "__file__", None):
        mod = importlib.import_module("consumers.chatbot.service")
    return mod


# ---------------------------------------------------------------------------
# hybrid.py — the local lane binds the operator row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_anonymous_session_carries_the_operator_row(local_lane):
    ctx = await hybrid.get_request_context_hybrid(_anonymous_request())

    assert ctx.auth_type == "anonymous"
    assert ctx.workspace_id == LOCAL_WS
    assert ctx.user.system_role == "super_admin"
    assert ctx.user.email == OPERATOR_EMAIL
    # The Clerk-less binding the tree already uses (clerk lane: id = clerk id
    # OR email): every resolver falls back to users.email == ctx.user.email.
    assert ctx.user.id == OPERATOR_EMAIL
    assert ctx.user.clerk_user_id is None
    assert ctx.user.raw_claims["source"] == "local_operator"
    assert ctx.user.raw_claims["user_id"] == OPERATOR_ID
    assert ctx.user.raw_claims["name"] == SEEDED_NAME
    assert ctx.user.raw_claims["username"] == "local"


@pytest.mark.asyncio
async def test_ctx_user_id_is_never_the_integer_pk_and_resolves_by_email(local_lane):
    """The trap: ``ctx.user.id`` is compared against ``users.clerk_user_id``
    (varchar) across api/*; an integer there is a Postgres type error. The
    platform's resolution order (clerk id, then email) must land on the row."""
    ctx = await hybrid.get_request_context_hybrid(_anonymous_request())
    assert not isinstance(ctx.user.id, int)

    by_clerk = local_lane.query(User).filter(User.clerk_user_id == ctx.user.id).first()
    by_email = local_lane.query(User).filter(User.email == ctx.user.email).first()
    assert by_clerk is None
    assert by_email is not None and by_email.id == OPERATOR_ID


@pytest.mark.asyncio
async def test_missing_operator_row_keeps_prd209_email_only_lane(local_lane):
    local_lane.query(User).delete()
    local_lane.commit()

    ctx = await hybrid.get_request_context_hybrid(_anonymous_request())

    assert ctx.user.email == OPERATOR_EMAIL
    assert ctx.user.system_role == "super_admin"
    assert ctx.user.id is None
    assert ctx.user.raw_claims is None


@pytest.mark.asyncio
async def test_operator_lookup_db_error_never_500s(local_lane, monkeypatch):
    def _boom(db, email):
        raise RuntimeError("users table unreachable")

    monkeypatch.setattr(hybrid, "_load_local_operator_row", _boom)
    ctx = await hybrid.get_request_context_hybrid(_anonymous_request())
    assert ctx.user.email == OPERATOR_EMAIL
    assert ctx.user.raw_claims is None


@pytest.mark.asyncio
async def test_saas_anonymous_lane_is_byte_identical(local_lane, monkeypatch):
    monkeypatch.setattr(hybrid.config, "AUTH_EDITION", "saas")
    loads = []
    monkeypatch.setattr(hybrid, "_load_local_operator_row", lambda db, email: loads.append(email))

    ctx = await hybrid.get_request_context_hybrid(_anonymous_request())

    assert ctx.user == UserContext()
    assert loads == [], "saas must never touch the operator-row lookup"


# ---------------------------------------------------------------------------
# The operator-row cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_reads_once_until_invalidated_or_expired(local_lane, monkeypatch):
    real_load = hybrid._load_local_operator_row
    calls = []

    def _counting(db, email):
        calls.append(email)
        return real_load(db, email)

    monkeypatch.setattr(hybrid, "_load_local_operator_row", _counting)
    clock = {"now": 1000.0}
    monkeypatch.setattr(hybrid.time, "monotonic", lambda: clock["now"])

    await hybrid.get_request_context_hybrid(_anonymous_request())
    await hybrid.get_request_context_hybrid(_anonymous_request())
    assert len(calls) == 1, "second request inside the TTL must hit the cache"

    hybrid.invalidate_local_operator_cache()
    await hybrid.get_request_context_hybrid(_anonymous_request())
    assert len(calls) == 2, "explicit invalidation forces a re-read"

    clock["now"] += hybrid._LOCAL_OPERATOR_CACHE_TTL_SECONDS + 1
    await hybrid.get_request_context_hybrid(_anonymous_request())
    assert len(calls) == 3, "TTL expiry is the backstop for out-of-band writers"


@pytest.mark.asyncio
async def test_missing_row_is_not_cached(local_lane, monkeypatch):
    """A seed that lands after the first request must be picked up at once."""
    local_lane.query(User).delete()
    local_lane.commit()
    await hybrid.get_request_context_hybrid(_anonymous_request())

    local_lane.add(User(id=OPERATOR_ID, username="local", email=OPERATOR_EMAIL, name="Late Seed"))
    local_lane.commit()
    ctx = await hybrid.get_request_context_hybrid(_anonymous_request())
    assert ctx.user.raw_claims["name"] == "Late Seed"


# ---------------------------------------------------------------------------
# GET / PUT /api/profile
# ---------------------------------------------------------------------------

def test_get_profile_local_reflects_the_operator_row(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    client = _client(users_db, _local_ctx())

    resp = client.get("/api/profile")

    assert resp.status_code == 200
    body = resp.json()
    assert body["edition"] == "local" and body["editable"] is True
    assert body["id"] == OPERATOR_ID
    assert body["email"] == OPERATOR_EMAIL
    assert body["name"] == SEEDED_NAME
    assert body["username"] == "local"
    assert body["system_role"] == "super_admin"
    assert "LOCAL_OPERATOR_EMAIL" in body["email_note"]


def test_put_then_get_round_trip_and_cache_invalidation(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    invalidations = []
    monkeypatch.setattr(profile_mod, "invalidate_local_operator_cache", lambda: invalidations.append(1))
    client = _client(users_db, _local_ctx())

    resp = client.put(
        "/api/profile",
        json={"name": "Test Operator", "username": "tester", "avatar_url": "https://example.test/a.png"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "Test Operator"
    assert invalidations == [1]

    got = client.get("/api/profile").json()
    assert got["name"] == "Test Operator"
    assert got["username"] == "tester"
    assert got["avatar_url"] == "https://example.test/a.png"
    assert got["email"] == OPERATOR_EMAIL, "email is the lookup key — untouched"


@pytest.mark.asyncio
async def test_put_name_reaches_the_next_session(local_lane, monkeypatch):
    """End to end inside the process: the hybrid lane cached the seeded name;
    PUT invalidates; the very next anonymous request carries the new name."""
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    first = await hybrid.get_request_context_hybrid(_anonymous_request())
    assert first.user.raw_claims["name"] == SEEDED_NAME

    client = _client(local_lane, _local_ctx())
    assert client.put("/api/profile", json={"name": "Renamed Operator"}).status_code == 200

    second = await hybrid.get_request_context_hybrid(_anonymous_request())
    assert second.user.raw_claims["name"] == "Renamed Operator"


def test_put_partial_update_leaves_other_fields_alone(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    client = _client(users_db, _local_ctx())

    assert client.put("/api/profile", json={"username": "solo"}).status_code == 200
    got = client.get("/api/profile").json()
    assert got["username"] == "solo"
    assert got["name"] == SEEDED_NAME


def test_put_blank_name_clears_it_so_the_greeting_falls_back(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    client = _client(users_db, _local_ctx())

    resp = client.put("/api/profile", json={"name": "   "})
    assert resp.status_code == 200
    assert resp.json()["name"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "someone@else.test"},              # read-only lookup key
        {"name": "<script>alert(1)</script>"},        # no HTML
        {"avatar_url": "javascript:alert(1)"},        # http(s) only
        {"avatar_url": "not a url"},
        {"username": ""},                             # NOT NULL + unique column
        {"username": "bad name!"},
        {"system_role": "user"},                      # unknown field
        {"name": "x" * 256},                          # length
    ],
)
def test_put_rejects_invalid_input(users_db, monkeypatch, payload):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    client = _client(users_db, _local_ctx())

    assert client.put("/api/profile", json=payload).status_code == 422
    assert client.get("/api/profile").json()["name"] == SEEDED_NAME


def test_put_username_clash_is_409(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    users_db.add(User(id=2, username="taken", email="second@example.test"))
    users_db.commit()
    client = _client(users_db, _local_ctx())

    resp = client.put("/api/profile", json={"username": "taken"})
    assert resp.status_code == 409
    assert client.get("/api/profile").json()["username"] == "local"


def test_put_without_operator_row_is_404_not_500(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", "nobody@example.test")
    client = _client(users_db, _local_ctx())
    assert client.put("/api/profile", json={"name": "x"}).status_code == 404


def test_put_in_saas_is_403_managed_by_identity_provider(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(profile_mod.config, "LOCAL_OPERATOR_EMAIL", OPERATOR_EMAIL)
    client = _client(users_db, _local_ctx())

    resp = client.put("/api/profile", json={"name": "Nope"})

    assert resp.status_code == 403
    assert resp.json()["detail"] == profile_mod.MANAGED_BY_IDENTITY_PROVIDER
    assert users_db.query(User).filter(User.email == OPERATOR_EMAIL).one().name == SEEDED_NAME


def test_get_profile_saas_resolves_by_clerk_id_and_is_read_only(users_db, monkeypatch):
    monkeypatch.setattr(profile_mod.config, "AUTH_EDITION", "saas")
    users_db.add(User(id=7, username="clerk_user", email="member@example.test", clerk_user_id="user_abc", name="Member"))
    users_db.commit()
    ctx = RequestContext(
        workspace_id=LOCAL_WS,
        user=UserContext(id="user_abc", email="member@example.test", clerk_user_id="user_abc"),
        auth_type="clerk",
    )
    client = _client(users_db, ctx)

    body = client.get("/api/profile").json()

    assert body["edition"] == "saas" and body["editable"] is False
    assert body["id"] == 7 and body["name"] == "Member"
    assert body["email_note"] == profile_mod.EMAIL_NOTE_SAAS


def test_put_is_gated_by_the_workspace_permission_dependency():
    put_routes = [r for r in profile_mod.router.routes if "PUT" in getattr(r, "methods", set())]
    assert len(put_routes) == 1
    gate_calls = [d.call for d in put_routes[0].dependant.dependencies]
    assert profile_mod._PROFILE_WRITE_GATE in gate_calls


def test_router_is_declared_in_the_mount_manifest():
    from router_manifest import MANIFEST_ROUTERS

    assert any(spec.module == "api.profile" and not spec.optional for spec in MANIFEST_ROUTERS)


# ---------------------------------------------------------------------------
# Greeting / context — one source for the name, both prompt lanes
# ---------------------------------------------------------------------------

def test_resolve_known_user_name_reads_users_name_by_integer_id(users_db):
    svc = _chat_service_module()
    assert svc.resolve_known_user_name(users_db, OPERATOR_ID) == SEEDED_NAME
    assert svc.resolve_known_user_name(users_db, 999) is None
    assert svc.resolve_known_user_name(users_db, None) is None


def test_resolve_known_user_name_blank_name_is_none(users_db):
    svc = _chat_service_module()
    users_db.query(User).filter(User.id == OPERATOR_ID).update({"name": "   "})
    users_db.commit()
    assert svc.resolve_known_user_name(users_db, OPERATOR_ID) is None


def test_resolve_known_user_name_db_error_is_none():
    svc = _chat_service_module()
    db = MagicMock()
    db.query.side_effect = RuntimeError("down")
    assert svc.resolve_known_user_name(db, OPERATOR_ID) is None


def test_atom_identity_clause_mirrors_personality_wording():
    svc = _chat_service_module()
    assert svc.atom_identity_clause("Test Operator") == " You're talking to Test Operator."
    assert svc.atom_identity_clause(None) == ""
    assert svc.atom_identity_clause("   ") == ""
    assert svc.atom_identity_clause(MagicMock()) == ""


@pytest.mark.asyncio
async def test_identity_section_greets_by_name_from_the_user_name_kwarg(monkeypatch):
    """The path the platform already had: IdentitySection → personality
    prompt renders ``user_name``. Present → "talking to <name>"; blank → the
    "ready to help" fallback. No second greeting path."""
    import consumers.chatbot.personality as personality
    from modules.context.sections.base import SectionContext
    from modules.context.sections.identity import IdentitySection

    monkeypatch.setattr(personality, "load_orchestrator_settings", lambda ws: {})
    agent = SimpleNamespace(
        id=1, name="Auto", agent_type="assistant", description=None,
        use_custom_persona=False, custom_persona_prompt=None, persona=None,
    )

    def _ctx(**kwargs):
        return SectionContext(agent=agent, workspace_id="ws_1", workspace_name="WS", kwargs=kwargs)

    named = await IdentitySection().render(_ctx(personality=True, user_name="Test Operator"))
    assert "talking to Test Operator" in named

    blank = await IdentitySection().render(_ctx(personality=True, user_name=None))
    assert "ready to help" in blank
    assert "talking to" not in blank


def test_chat_service_seeds_the_greeting_state_from_the_users_row():
    """Source guard: the seam lives in ONE module — the driving human's integer
    id is remembered next to the viewer subject, the orchestrator state is
    seeded from users.name, and the ATOM prompt reads that same state."""
    src = (_ORCH / "consumers" / "chatbot" / "service.py").read_text(encoding="utf-8")

    assert re.search(
        r'self\._viewer_subject_id = f"user:\{user_id\}" if user_id else None\s*\n'
        r"(?:.*\n){0,3}\s*self\._driving_user_id = user_id",
        src,
    ), "stream_response_with_agent must remember the driving users.id"
    assert re.search(
        r"smart_chat\.orchestrator\.state\.user_name = \(\s*\n\s*smart_chat\.orchestrator\.state\.user_name\s*\n"
        r"\s*or resolve_known_user_name\(self\.db, getattr\(self, \"_driving_user_id\", None\)\)",
        src,
    ), "_prepare_llm_messages must seed ConversationState.user_name from the users row"
    assert 'atom_identity_clause(smart_chat.get_user_name())' in src, (
        "the ATOM prompt must read the same ConversationState.user_name"
    )
    # The full path forwards the state as the `user_name` kwarg IdentitySection reads.
    orch = (_ORCH / "consumers" / "chatbot" / "smart_orchestrator.py").read_text(encoding="utf-8")
    assert "user_name=self.state.user_name," in orch
    identity = (_ORCH / "modules" / "context" / "sections" / "identity.py").read_text(encoding="utf-8")
    assert 'ctx.kwargs.get("_user_name") or ctx.kwargs.get("user_name")' in identity
