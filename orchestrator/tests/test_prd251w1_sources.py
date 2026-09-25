"""PRD-251 S1.4 (US-109) — facts carry sources: resolution and search (D7).

``modules/socials/sources.py`` over in-memory SQLite: the social tables, SQLite
copies of ``workspaces`` / ``document_templates`` / ``documents``, and stand-ins
for the two relations the resolver reads with raw SQL: ``agent_reports`` (no
ORM model) and the ``v_workspace_outputs`` view (a table here, with the view's
columns). The real socials router runs on a mini FastAPI app with the real gate;
only the master switch and the member role are stubbed. Pins:

* each source kind resolves in its own workspace, and another workspace's
  source never does: it reads as not found, and a save that binds it is 422;
* a claim bound to a deleted Deliverable fails validation: a save binding it is
  422 naming the claim, and an approval counts it as unsourced (422 without the
  override; with it, the approval record says why);
* a save checks only the sources it adds or changes;
* a metric is read at its timestamp: the latest report at or before ``as_of``
  whose metrics carry it as a figure; ``as_of`` is required, never in the future;
* a malformed ref is refused before any query reaches the database;
* ``GET /api/socials/sources`` offers candidates per kind from the caller's
  workspace only, newest first, matching ``q`` case-insensitively with LIKE
  wildcards taken literally, never a deleted item; it is in the committed route
  manifest.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.socials as socials_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.core import Document, DocumentTemplate  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import service, sources  # noqa: E402
from modules.socials.sources import SourceNotResolved  # noqa: E402

WS_A = uuid.uuid4()
WS_B = uuid.uuid4()
NOW = datetime(2026, 9, 23, 12, 0)
MANIFEST = _ORCH / "reports" / "route-manifest.json"
SOURCES_URL = "/api/socials/sources"
DELETED = "the Deliverable was deleted"


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table: sa.Table, metadata: sa.MetaData) -> sa.Table:
    """A column-for-column copy SQLite can build (JSONB / ARRAY → JSON, UUID → CHAR(32))."""
    columns = [sa.Column(col.name, _portable(col.type), primary_key=col.primary_key) for col in table.columns]
    return sa.Table(table.name, metadata, *columns)


_TABLES = sa.MetaData()
_sqlite_copy(Workspace.__table__, _TABLES)
_sqlite_copy(DocumentTemplate.__table__, _TABLES)
DOCUMENTS = _sqlite_copy(Document.__table__, _TABLES)
# agent_reports has no ORM model (alembic prd76_agent_reports + prd133b's
# deleted_at): the columns the resolver reads.
AGENT_REPORTS = sa.Table(
    "agent_reports",
    _TABLES,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("workspace_id", sa.String(36), nullable=False),
    sa.Column("report_type", sa.String(30)),
    sa.Column("title", sa.String(255), nullable=False),
    sa.Column("summary", sa.String(500)),
    sa.Column("metrics", sa.JSON),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("deleted_at", sa.DateTime(timezone=True)),
)
# v_workspace_outputs is a view over blog_posts, agent_reports and deliverables
# (alembic prd133b_outputs_view); a table with its columns stands in.
OUTPUTS = sa.Table(
    "v_workspace_outputs",
    _TABLES,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("workspace_id", sa.String(36), nullable=False),
    sa.Column("artifact_type", sa.String(50)),
    sa.Column("title", sa.String(255)),
    sa.Column("summary", sa.Text),
    sa.Column("preview_url", sa.Text),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("deleted_at", sa.DateTime(timezone=True)),
)
_DOCUMENT_IDS = itertools.count(1)


def _ctx(workspace_id, user_id="member-1"):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id=user_id, clerk_user_id=f"clerk-{user_id}", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def env(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _TABLES.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])

    session = sessionmaker(bind=engine)()
    for ws_id in (WS_A, WS_B):
        session.add(
            Workspace(
                id=ws_id, name=f"ws-{ws_id.hex[:6]}", plan="basic", plan_limits={},
                settings={"socials": {"enabled": True}}, onboarding={}, created_at=NOW, updated_at=NOW,
            )
        )
    session.commit()

    state = SimpleNamespace(session=session, ctx=_ctx(WS_A), role="owner", master="true")
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: state.master)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)

    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _at(days_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def _deliverable(env, ws, title="Q3 revenue chart", *, summary=None, days_ago=1.0, deleted=False) -> str:
    ident = str(uuid.uuid4())
    env.session.execute(
        OUTPUTS.insert().values(
            id=ident, workspace_id=str(ws), artifact_type="image", title=title, summary=summary,
            preview_url=f"/api/workspaces/{ws}/files/raw?path=outputs/{ident}.png",
            created_at=_at(days_ago), deleted_at=_at(0) if deleted else None,
        )
    )
    env.session.commit()
    return ident


def _report(env, ws, title="Weekly sales", *, metrics=None, summary=None, days_ago=1.0, deleted=False) -> str:
    ident = str(uuid.uuid4())
    env.session.execute(
        AGENT_REPORTS.insert().values(
            id=ident, workspace_id=str(ws), report_type="summary", title=title, summary=summary,
            metrics=metrics or {}, created_at=_at(days_ago), deleted_at=_at(0) if deleted else None,
        )
    )
    env.session.commit()
    return ident


def _document(env, ws, name="pricing.pdf", *, description=None, days_ago=1.0) -> str:
    ident = next(_DOCUMENT_IDS)
    env.session.execute(
        DOCUMENTS.insert().values(
            id=ident, filename=f"{ident}-{name}", original_filename=name, description=description,
            workspace_id=ws, status="processed", upload_date=_at(days_ago).replace(tzinfo=None),
        )
    )
    env.session.commit()
    return str(ident)


def _soft_delete(env, table: sa.Table, ident: str) -> None:
    env.session.execute(table.update().where(table.c.id == ident).values(deleted_at=_at(0)))
    env.session.commit()


def _make_source(env, kind: str, ws):
    """A source of ``kind`` that exists in ``ws``, and the title it resolves to."""
    if kind == "deliverable":
        return {"kind": kind, "ref": _deliverable(env, ws, "Q3 revenue chart")}, "Q3 revenue chart"
    if kind == "report":
        return {"kind": kind, "ref": _report(env, ws, "Weekly sales")}, "Weekly sales"
    if kind == "document":
        return {"kind": kind, "ref": _document(env, ws, "pricing.pdf")}, "pricing.pdf"
    if kind == "metric":
        _report(env, ws, "Sales, week 38", metrics={"orders": 321}, days_ago=2)
        return {"kind": kind, "ref": "orders", "as_of": _at(1).isoformat()}, "orders"
    if kind == "url":
        return {"kind": kind, "ref": "https://websummit.com/tickets"}, "websummit.com"
    raise AssertionError(f"add a {kind!r} source to _make_source: every source kind must be covered")


def _create(env, **body):
    resp = env.client.post("/api/socials/posts", json={"title": "Countdown", **body})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _submit(env, post):
    resp = env.client.post(f"/api/socials/posts/{post['id']}/submit")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _approve(env, post, **extra):
    return env.client.post(
        f"/api/socials/posts/{post['id']}/approve", json={"content_hash": post["content_hash"], **extra}
    )


def _row(env, post):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(post["id"])).to_dict()


def _search(env, **params):
    resp = env.client.get(SOURCES_URL, params={k: v for k, v in params.items() if v is not None})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _refs(env, **params):
    return [c["ref"] for c in _search(env, **params)["candidates"]]


def _claim(name, source, value="$1.2M"):
    return {"variables": {name: {"value": value, "claim": True}}, "sources": {name: source}}


class _NoDatabase:
    """A session that fails the test if anything queries it."""

    def execute(self, *args, **kwargs):
        raise AssertionError("a malformed source reached the database")

    def query(self, *args, **kwargs):
        raise AssertionError("a malformed source reached the database")


# ---------------------------------------------------------------------------
# Resolution: each kind, in its own workspace only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", service.SOURCE_KINDS)
def test_each_source_kind_resolves_in_its_own_workspace(env, kind):
    source, title = _make_source(env, kind, WS_A)
    resolved = sources.resolve(env.session, WS_A, source)
    assert (resolved.kind, resolved.title) == (kind, title)
    assert sources.unresolved(env.session, WS_A, {"claim": source}) == {}
    if kind == "metric":
        assert resolved.value == 321 and resolved.detail == "Sales, week 38"


@pytest.mark.parametrize("kind", [k for k in service.SOURCE_KINDS if k != "url"])
def test_a_source_from_another_workspace_is_refused(env, kind):
    source, _ = _make_source(env, kind, WS_B)
    assert sources.resolve(env.session, WS_B, source).kind == kind  # it is there, in its own workspace

    with pytest.raises(SourceNotResolved, match="in this workspace"):
        sources.resolve(env.session, WS_A, source)
    resp = env.client.post("/api/socials/posts", json={"title": "Borrowed", **_claim("figure", source)})
    assert resp.status_code == 422, resp.text
    assert list(resp.json()["detail"]["unresolved"]) == ["figure"]
    assert env.client.get("/api/socials/posts").json()["total"] == 0


def test_every_kind_is_resolved_and_searched():
    assert set(sources._RESOLVERS) == set(sources._SEARCHERS) == set(service.SOURCE_KINDS)


# ---------------------------------------------------------------------------
# A deleted source
# ---------------------------------------------------------------------------


def test_a_claim_bound_to_a_deleted_deliverable_fails_validation(env):
    chart = _deliverable(env, WS_A, "Q3 revenue chart")
    claim = _claim("revenue", {"kind": "deliverable", "ref": chart})
    post = _create(env, **claim)
    _submit(env, post)
    _soft_delete(env, OUTPUTS, chart)

    assert sources.unresolved(env.session, WS_A, post["sources"]) == {"revenue": DELETED}

    # A save that binds it is refused, naming the claim and why.
    resp = env.client.post("/api/socials/posts", json={"title": "Again", **claim})
    assert resp.status_code == 422
    assert resp.json()["detail"]["unresolved"] == {"revenue": DELETED}

    # The approval counts the claim as unsourced.
    resp = _approve(env, post)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["claims"] == ["revenue"] and detail["unresolved"] == {"revenue": DELETED}
    assert "could not be found" in detail["message"]
    assert _row(env, post)["status"] == "needs_approval"

    # An explicit override approves it, and the record says why.
    resp = _approve(env, post, override_unsourced=True)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved" and body["override_unsourced"] is True
    entry = body["review_log"][-1]
    assert entry["overridden_claims"] == ["revenue"]
    assert entry["unresolved_sources"] == {"revenue": DELETED}


def test_a_deleted_report_and_a_removed_document_do_not_resolve(env):
    report = _report(env, WS_A, "Weekly sales", deleted=True)
    with pytest.raises(SourceNotResolved, match="the report was deleted"):
        sources.resolve(env.session, WS_A, {"kind": "report", "ref": report})

    document = _document(env, WS_A, "pricing.pdf")
    env.session.execute(DOCUMENTS.delete().where(DOCUMENTS.c.id == int(document)))
    env.session.commit()
    with pytest.raises(SourceNotResolved, match="no document"):
        sources.resolve(env.session, WS_A, {"kind": "document", "ref": document})


def test_only_a_claim_counts_as_unsourced_when_its_source_is_gone():
    post = SimpleNamespace(
        variables={"users": {"value": 1200, "claim": True}, "city": {"value": "Lisbon", "claim": False}},
        sources={"users": {"kind": "report", "ref": str(uuid.uuid4())}, "city": {"kind": "url", "ref": "https://lisbon.pt"}},
    )
    assert service.unsourced_claims(post) == []
    assert service.unsourced_claims(post, {"city": "gone"}) == []
    assert service.unsourced_claims(post, {"users": "gone", "city": "gone"}) == ["users"]
    exc = service.UnsourcedClaims(["users"], {"users": "the report was deleted", "city": "gone"})
    assert exc.unresolved == {"users": "the report was deleted"}
    assert "users (the report was deleted)" in str(exc) and "no source" not in str(exc)


def test_a_claim_whose_source_resolves_is_approved_without_the_override(env):
    report = _report(env, WS_A, "Weekly sales")
    post = _create(env, **_claim("orders", {"kind": "report", "ref": report}, value=321))
    _submit(env, post)
    resp = _approve(env, post)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved" and body["override_unsourced"] is False
    assert "overridden_claims" not in body["review_log"][-1]


def test_a_save_checks_only_the_sources_it_adds_or_changes(env):
    chart = _deliverable(env, WS_A, "Q3 revenue chart")
    kept = {"kind": "deliverable", "ref": chart}
    post = _create(env, **_claim("revenue", kept))
    url = f"/api/socials/posts/{post['id']}"
    _soft_delete(env, OUTPUTS, chart)

    # Edits that keep the source as it was are saved; the approval checks it again.
    assert env.client.patch(url, json={"title": "Renamed"}).status_code == 200
    both = {"revenue": kept, "site": {"kind": "url", "ref": "https://websummit.com"}}
    resp = env.client.patch(url, json={"sources": both})
    assert resp.status_code == 200, resp.text

    # Re-binding a claim to another workspace's report is refused, and nothing is written.
    theirs = _report(env, WS_B, "Their sales")
    resp = env.client.patch(url, json={"sources": {**both, "revenue": {"kind": "report", "ref": theirs}}})
    assert resp.status_code == 422
    assert list(resp.json()["detail"]["unresolved"]) == ["revenue"]
    assert _row(env, post)["sources"] == both


# ---------------------------------------------------------------------------
# A metric at a timestamp
# ---------------------------------------------------------------------------


def test_a_metric_is_read_at_its_timestamp(env):
    older = _report(env, WS_A, "Sales, week 36", metrics={"orders": 300}, days_ago=10)
    newer = _report(env, WS_A, "Sales, week 38", metrics={"orders": 321}, days_ago=3)
    _report(env, WS_A, "Sales, week 39", metrics={"orders": None, "refunds": 2}, days_ago=2)  # no figure for orders
    _report(env, WS_A, "Heartbeat", metrics={"llm_calls": 4}, days_ago=1)
    _report(env, WS_B, "Their sales", metrics={"orders": 999}, days_ago=1)

    def read(days_ago):
        return sources.resolve(env.session, WS_A, {"kind": "metric", "ref": "orders", "as_of": _at(days_ago).isoformat()})

    week36 = read(5)
    assert (week36.value, week36.detail, week36.report_id) == (300, "Sales, week 36", older)
    latest = read(0)
    assert (latest.value, latest.report_id) == (321, newer)
    assert datetime.fromisoformat(latest.as_of) < _at(2)
    with pytest.raises(SourceNotResolved, match="carries the metric 'orders'"):
        read(20)

    _soft_delete(env, AGENT_REPORTS, newer)
    assert read(0).value == 300


@pytest.mark.parametrize(
    "as_of, message",
    [
        (None, "needs as_of"),
        ("next tuesday", "not an ISO date-time"),
        ((datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), "in the future"),
    ],
)
def test_a_metric_needs_a_timestamp_that_is_not_in_the_future(as_of, message):
    with pytest.raises(SourceNotResolved, match=message):
        sources.resolve(_NoDatabase(), WS_A, {"kind": "metric", "ref": "orders", "as_of": as_of})


# ---------------------------------------------------------------------------
# Malformed refs never reach the database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, ref",
    [
        ("deliverable", "report-42"),
        ("report", "r-1"),
        ("document", "abc"),
        ("document", "0"),
        ("document", "-3"),
        ("document", "²"),
        ("document", str(2**31)),
        ("url", "websummit.com"),
        ("url", "ftp://example.com/deck.pdf"),
        ("url", "javascript:alert(1)"),
        ("url", "https://user:secret@example.com/"),
        ("url", "https://exa mple.com/"),
        ("url", "https://[::1/"),
        ("url", "https://example.com:port/"),
        ("rumour", "x"),
    ],
)
def test_a_malformed_ref_is_refused_before_any_query(kind, ref):
    with pytest.raises(SourceNotResolved):
        sources.resolve(_NoDatabase(), WS_A, {"kind": kind, "ref": ref})


def test_a_url_resolves_by_its_shape_alone():
    resolved = sources.resolve(_NoDatabase(), WS_A, {"kind": "url", "ref": "https://WebSummit.com/tickets?day=1"})
    assert (resolved.title, resolved.ref) == ("websummit.com", "https://WebSummit.com/tickets?day=1")


# ---------------------------------------------------------------------------
# GET /api/socials/sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["deliverable", "report", "document"])
def test_the_source_search_offers_only_the_callers_workspace(env, kind):
    mine, _ = _make_source(env, kind, WS_A)
    theirs, _ = _make_source(env, kind, WS_B)
    body = _search(env, kind=kind)
    refs = [c["ref"] for c in body["candidates"]]
    assert refs == [mine["ref"]] and body["total"] == 1
    assert theirs["ref"] not in refs
    assert {c["kind"] for c in body["candidates"]} == {kind}


def test_the_source_search_matches_q_case_insensitively_with_wildcards_taken_literally(env):
    chart = _deliverable(env, WS_A, "Q3 Revenue chart", days_ago=4)
    plan = _deliverable(env, WS_A, "Hiring plan", summary="Revenue per head", days_ago=3)
    organic = _deliverable(env, WS_A, "100% organic", days_ago=2)
    book = _deliverable(env, WS_A, "Brand_book", days_ago=1)

    assert _refs(env, kind="deliverable", q="REVENUE") == [plan, chart]
    assert _refs(env, kind="deliverable", q="%") == [organic]
    assert _refs(env, kind="deliverable", q="_") == [book]
    assert _refs(env, kind="deliverable", q="nothing like it") == []


def test_the_source_search_skips_deleted_items_and_lists_newest_first_up_to_the_limit(env):
    old = _deliverable(env, WS_A, "Chart one", days_ago=3)
    new = _deliverable(env, WS_A, "Chart two", days_ago=1)
    _deliverable(env, WS_A, "Chart three", days_ago=2, deleted=True)
    _report(env, WS_A, "Chart report", deleted=True)

    assert _refs(env, kind="deliverable", q="chart") == [new, old]
    assert _refs(env, kind="deliverable", q="chart", limit=1) == [new]
    assert _refs(env, kind="report", q="chart") == []


def test_the_source_search_offers_metrics_with_their_latest_figure(env):
    _report(env, WS_A, "Sales, week 36", metrics={"orders": 300, "order_value": 41.5}, days_ago=10)
    newer = _report(env, WS_A, "Sales, week 38", metrics={"orders": 321, "orders_by_region": {"eu": 1}}, days_ago=3)
    _report(env, WS_A, "Heartbeat", metrics={"llm_calls": 4}, days_ago=1)
    _report(env, WS_B, "Their sales", metrics={"orders": 999}, days_ago=1)

    candidates = {c["ref"]: c for c in _search(env, kind="metric", q="ORDER")["candidates"]}
    assert set(candidates) == {"orders", "order_value"}
    orders = candidates["orders"]
    assert (orders["value"], orders["report_id"], orders["detail"]) == (321, newer, "Sales, week 38")
    assert candidates["order_value"]["value"] == 41.5

    # Stored as a source, the candidate resolves to the same figure.
    stored = {"kind": orders["kind"], "ref": orders["ref"], "as_of": orders["as_of"]}
    resolved = sources.resolve(env.session, WS_A, stored)
    assert (resolved.value, resolved.report_id) == (321, newer)


def test_the_source_search_offers_a_url_only_when_q_is_one(env):
    body = _search(env, kind="url", q="https://websummit.com/tickets")
    assert [(c["kind"], c["ref"], c["title"]) for c in body["candidates"]] == [
        ("url", "https://websummit.com/tickets", "websummit.com")
    ]
    assert _search(env, kind="url", q="websummit")["candidates"] == []
    assert _search(env, kind="url")["candidates"] == []


def test_the_source_search_without_a_kind_covers_every_kind(env):
    for kind in service.SOURCE_KINDS:
        if kind != "url":
            _make_source(env, kind, WS_A)
    kinds = [c["kind"] for c in _search(env)["candidates"]]
    assert set(kinds) == {"deliverable", "report", "document", "metric"}
    # Grouped in the order of SOURCE_KINDS.
    assert kinds == sorted(kinds, key=service.SOURCE_KINDS.index)


@pytest.mark.parametrize(
    "params",
    [{"kind": "rumour"}, {"limit": 0}, {"limit": sources.SEARCH_MAX_LIMIT + 1}, {"q": "x" * (sources.SEARCH_QUERY_MAX_CHARS + 1)}],
)
def test_the_source_search_refuses_bad_parameters(env, params):
    assert env.client.get(SOURCES_URL, params=params).status_code == 422


def test_the_source_search_is_behind_the_gate_and_open_to_every_member(env):
    _make_source(env, "deliverable", WS_A)
    env.role = "viewer"
    assert env.client.get(SOURCES_URL).status_code == 200
    env.master = "false"
    assert env.client.get(SOURCES_URL).status_code == 404


def test_the_source_search_is_in_the_committed_route_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "GET", "path": SOURCES_URL} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
