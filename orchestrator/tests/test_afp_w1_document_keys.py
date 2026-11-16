"""PRD-274 Wave 1 (§5.2): a product's server key manages its workspace's knowledge documents.

``POST /api/documents/upload`` and ``DELETE /api/documents/{id}`` take a member with their
permission as before (``documents:create``, ``documents:delete``) or a server key listing
``documents:write``. ``GET /api/documents`` (both paths) and ``GET /api/documents/{id}`` take any
signed-in member as before, and ``POST /api/documents/search`` a member with ``documents:read``,
or a key listing ``documents:read``. A publishable key is refused everywhere.

A key is bound to its own workspace and reaches only documents no team and no person limits
(``services/document_key_access.py``): anything else answers 404, as a missing document does,
and nothing is removed. Its search maps a hit only by the hit's own id, never by file name. Its
upload names the key, never answers with a hidden document, and is refused (409) where it would
write over one. It deletes through the removal members use. Real keys and rows on the real
database; the ingestion manager, the retrieval funnel and the workspace worker are fakes that
record what they are given.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from starlette.requests import Request

from api import documents
from core.auth import server_key_gate as gate
from core.auth.scopes import DOCUMENTS_READ, DOCUMENTS_WRITE
from services.document_key_access import UPLOAD_CLASH

UPLOAD = ("POST", "/api/documents/upload")
LIST = ("GET", "/api/documents")
LIST_SLASH = ("GET", "/api/documents/")
GET = ("GET", "/api/documents/{document_id:int}")
SEARCH = ("POST", "/api/documents/search")
DELETE = ("DELETE", "/api/documents/{document_id}")
# Each route a key reaches: (the member permission, as before; the key's scope).
ROUTES = {
    UPLOAD: ("documents:create", DOCUMENTS_WRITE),
    LIST: (None, DOCUMENTS_READ),
    LIST_SLASH: (None, DOCUMENTS_READ),
    GET: (None, DOCUMENTS_READ),
    SEARCH: ("documents:read", DOCUMENTS_READ),
    DELETE: ("documents:delete", DOCUMENTS_WRITE),
}
# Document routes no key reaches: they keep the member-only sign-in.
NOT_FOR_KEYS = [
    ("GET", "/api/documents/{document_id}/content"),
    ("GET", "/api/documents/{document_id}/download"),
    ("GET", "/api/documents/{document_id}/delete-impact"),
    ("POST", "/api/documents/{document_id}/reprocess"),
]
PRIVATE_TEXT = "Priya's notes: the Harbour contract is not renewing."
TEAM_TEXT = "Finance only: the margin on the Christmas box is 61%."
THEIR_TEXT = "Another workspace's price list."


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def routes():
    from fastapi.routing import APIRoute, iter_route_contexts

    from main import app

    return {(m, r.path): r for r in iter_route_contexts(app.routes)
            if isinstance(r.original_route, APIRoute) for m in r.methods}


@pytest.fixture
def workspaces(new_session):
    ws, other = str(uuid.uuid4()), str(uuid.uuid4())
    s = new_session()
    for workspace_id, name in ((ws, "w1-document-keys"), (other, "w1-document-keys-other")):
        s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :name)"),
                  {"id": workspace_id, "name": name})
    s.commit()
    s.close()
    yield ws, other
    s = new_session.sweep()
    s.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents "
                   "WHERE workspace_id IN (CAST(:a AS uuid), CAST(:b AS uuid)))"), {"a": ws, "b": other})
    for table in ("documents", "deliverables", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id IN (CAST(:a AS uuid), CAST(:b AS uuid))"),  # noqa: S608
                  {"a": ws, "b": other})
    s.execute(text("DELETE FROM workspaces WHERE id IN (CAST(:a AS uuid), CAST(:b AS uuid))"), {"a": ws, "b": other})
    s.commit()
    s.close()


def _document(workspace_id, filename, **columns):
    from core.models import Document

    return Document(workspace_id=uuid.UUID(workspace_id), filename=filename, original_filename=filename,
                    status="completed", **columns)


def _add(new_session, *rows):
    s = new_session()
    try:
        s.add_all(rows)
        s.commit()
        return [row.id for row in rows]
    finally:
        s.close()


@pytest.fixture
def library(new_session, workspaces):
    """The key's workspace holds a shared document, one stored before the visibility columns had
    values, one a team limits, one a person keeps, and the knowledge copy of a Deliverable written
    through ``POST /api/documents/generate`` (the row ``_registered_response`` registers: source
    ``document``, storage ``generated``); that Deliverable; and another workspace's document."""
    ws, other = workspaces
    s = new_session()
    deliverable = str(s.execute(text(
        "INSERT INTO deliverables (workspace_id, source_type, artifact_type, title, storage_type, file_path, file_name) "
        "VALUES (CAST(:ws AS uuid), 'document', 'document', 'Invoice INV-0042', 'generated', "
        "'generated/invoice-inv-0042.pdf', 'invoice-inv-0042.pdf') RETURNING id"), {"ws": ws}).scalar())
    s.commit()
    s.close()
    rows = {
        "shared": _document(ws, "pricing.md", team_access=[], is_shared=True),
        "unset": _document(ws, "faq.md", team_access=None, is_shared=None),
        "team": _document(ws, "margins.md", team_access=["finance"], is_shared=True),
        "private": _document(ws, "notes.md", team_access=[], is_shared=False),
        "copy": _document(ws, "invoice-inv-0042.pdf", team_access=[], is_shared=True,
                          tags=["added-by-owner", f"deliverable-added:{deliverable}"]),
        "theirs": _document(other, "pricing.md", team_access=[], is_shared=True),
    }
    ids = dict(zip(rows, _add(new_session, *rows.values())))
    return SimpleNamespace(ws=ws, other=other, ids=ids, deliverable=deliverable)


@pytest.fixture
def unlimited(monkeypatch):
    class _Open:
        def check(self, *args, **kwargs):
            return (True, 1, 0, 60)

    monkeypatch.setattr("api.widgets.rate_limit.RateLimitStore", _Open)


@pytest.fixture
def manager(new_session, monkeypatch):
    """The ingestion manager: it deletes by bare id on committed rows, as the real one does, so
    only the route's own scoping keeps other rows safe; and it records what it is asked."""
    asked = SimpleNamespace(deleted=[], processed=[])

    class _Manager:
        def __init__(self, workspace_id):
            self.workspace_id = workspace_id

        def delete_document(self, document_id):
            asked.deleted.append((self.workspace_id, document_id))
            s = new_session()
            try:
                s.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
                s.commit()
            finally:
                s.close()
            return True

        async def _process_document(self, document_id, path, file_type, *args, **kwargs):
            asked.processed.append((self.workspace_id, document_id))

    monkeypatch.setattr(documents, "get_document_manager", _Manager)
    return asked


@pytest.fixture
def worker(monkeypatch):
    """The workspace worker: the files documents/ holds, and what is written there."""
    import core.workspace_client as wc

    state = SimpleNamespace(files={}, written={})

    async def write_file(self, path, content):
        state.written[path] = content
        state.files[path] = content
        return {"success": True, "path": path}

    async def list_dir(self, path="."):
        return {"path": path, "entries": [{"name": name, "type": "file"} for name in state.files]}

    monkeypatch.setattr(wc.WorkspaceClient, "write_file", write_file)
    monkeypatch.setattr(wc.WorkspaceClient, "list_dir", list_dir)
    return state


@pytest.fixture
def uploads(monkeypatch, tmp_path, manager, worker):
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    return SimpleNamespace(folder=tmp_path, manager=manager, worker=worker)


def _key(new_session, ws, permissions, key_type="server"):
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        return ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme backend", key_type=key_type,
                                            permissions=list(permissions))["key"]
    finally:
        s.close()


def _request(token=None, **headers):
    pairs = [(k.lower().replace("_", "-").encode(), v.encode()) for k, v in headers.items()]
    if token:
        pairs.append((b"authorization", f"Bearer {token}".encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": pairs, "query_string": b"",
                    "path_params": {}})


def _gate(route):
    [found] = [d.call for d in route.dependant.dependencies if getattr(d.call, gate.KEY_SCOPE_MARKER_ATTR, None)]
    return found


def _through(dep, request):
    """A route's gate, run as FastAPI runs it: the member-or-key gate takes a session, the read gate none."""
    if "db" in inspect.signature(dep).parameters:
        return asyncio.run(dep(request, db=None))
    return asyncio.run(dep(request))


def _key_ctx(new_session, routes, route, ws, permissions):
    """The context a key listing ``permissions`` gets from ``route``'s own gate."""
    return _through(_gate(routes[route]), _request(_key(new_session, ws, permissions)))


def _member(ws):
    return SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="clerk",
                           user=SimpleNamespace(id="user_1", clerk_user_id="user_1", email=None))


def _call(new_session, route_function, **kwargs):
    s = new_session()
    try:
        answer = route_function(db=s, **kwargs)
        return asyncio.run(answer) if inspect.iscoroutine(answer) else answer
    finally:
        s.close()


def _listed(new_session, ctx, team=None):
    listed = _call(new_session, documents.list_documents, ctx=ctx, skip=0, limit=1000, status=None, file_type=None,
                   search=None, team=team, source_type=None, exclude_source_type=None, content_hash=None)
    return {doc.id for doc in listed}


def _named(library, *names):
    return {library.ids[name] for name in names}


def _remaining(new_session, library):
    """Which of the library's documents still exist, and whether its Deliverable is still live."""
    s = new_session()
    try:
        ids = {row.id for row in s.execute(text(
            "SELECT id FROM documents WHERE workspace_id IN (CAST(:a AS uuid), CAST(:b AS uuid))"),
            {"a": library.ws, "b": library.other})}
        live = s.execute(text("SELECT deleted_at IS NULL FROM deliverables WHERE id = CAST(:id AS uuid)"),
                         {"id": library.deliverable}).scalar()
    finally:
        s.close()
    return {name for name, document_id in library.ids.items() if document_id in ids}, live


# ── the gates ────────────────────────────────────────────────────────────────

def test_each_route_takes_a_key_listing_its_scope_bound_to_its_own_workspace(new_session, workspaces, unlimited,
                                                                            routes):
    ws, other = workspaces
    assert ROUTES
    for route, (_permission, scope) in ROUTES.items():
        token = _key(new_session, ws, [scope])
        ctx = _through(_gate(routes[route]), _request(token))
        assert (str(ctx.workspace_id), gate.is_server_key(ctx)) == (ws, True), route
        with pytest.raises(HTTPException) as elsewhere:
            _through(_gate(routes[route]), _request(token, x_workspace_id=other))
        assert elsewhere.value.status_code == 403, route


def test_each_route_refuses_a_key_without_its_scope_and_a_publishable_key(new_session, workspaces, unlimited, routes):
    """Delete rides on documents:write: a key that only reads can't delete, nor upload."""
    ws, _ = workspaces
    assert ROUTES
    for route, (_permission, scope) in ROUTES.items():
        other_scope = DOCUMENTS_READ if scope == DOCUMENTS_WRITE else DOCUMENTS_WRITE
        for permissions, key_type in (([other_scope], "server"), ([], "server"), ([scope], "public")):
            with pytest.raises(HTTPException) as refused:
                _through(_gate(routes[route]), _request(_key(new_session, ws, permissions, key_type=key_type)))
            assert refused.value.status_code == 403, (route, permissions, key_type)


def test_a_member_meets_each_routes_gate_as_before(monkeypatch, routes):
    member = SimpleNamespace(workspace_id=uuid.uuid4(), auth_type="clerk",
                             user=SimpleNamespace(id="u", clerk_user_id="c", system_role="user"))

    async def signed_in(request):
        return member

    asked = []
    monkeypatch.setattr(gate, "get_request_context_hybrid", signed_in)
    monkeypatch.setattr("core.auth.workspace_permission.workspace_permission_granted",
                        lambda db, ctx, permission: asked.append(permission) or False)
    assert ROUTES
    for route, (permission, _scope) in ROUTES.items():
        if permission is None:  # list and get: any signed-in member, no permission asked
            assert _through(_gate(routes[route]), _request()) is member, route
            continue
        with pytest.raises(HTTPException) as refused:
            _through(_gate(routes[route]), _request())
        assert refused.value.status_code == 403, route
    assert asked == ["documents:create", "documents:read", "documents:delete"]


def test_the_routes_carry_their_member_permission_and_key_scope_and_no_other_gate(routes):
    from api.api_keys import VALID_PERMISSIONS
    from core.auth.hybrid import get_request_context_hybrid
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import limiter

    def calls(dependant):
        for sub in dependant.dependencies:
            yield sub.call
            yield from calls(sub)

    def gates(route):
        return [(getattr(c, PERMISSION_MARKER_ATTR, None), getattr(c, gate.KEY_SCOPE_MARKER_ATTR, None))
                for c in calls(routes[route].dependant)
                if getattr(c, PERMISSION_MARKER_ATTR, None) or getattr(c, gate.KEY_SCOPE_MARKER_ATTR, None)]

    assert ROUTES and NOT_FOR_KEYS
    for route, pair in ROUTES.items():
        assert gates(route) == [pair], route
        assert get_request_context_hybrid not in set(calls(routes[route].dependant)), route
        endpoint = routes[route].endpoint
        assert f"{endpoint.__module__}.{endpoint.__name__}" in limiter._exempt_routes, route
    for route in NOT_FOR_KEYS:
        assert all(scope is None for _permission, scope in gates(route)), route
    assert {DOCUMENTS_READ, DOCUMENTS_WRITE} <= set(VALID_PERMISSIONS)


# ── list and get ─────────────────────────────────────────────────────────────

def test_a_key_lists_only_what_no_team_or_person_limits_in_its_own_workspace(new_session, library, unlimited, routes):
    ctx = _key_ctx(new_session, routes, LIST, library.ws, [DOCUMENTS_READ])
    reachable = _named(library, "shared", "unset", "copy")
    assert _listed(new_session, ctx) == reachable
    narrowed = _listed(new_session, ctx, "finance")                  # asking for a team widens nothing
    assert library.ids["shared"] in narrowed and narrowed <= reachable


def test_a_member_lists_every_document_of_the_workspace_as_before(new_session, library):
    assert _listed(new_session, _member(library.ws)) == _named(library, "shared", "unset", "team", "private", "copy")


def test_a_key_gets_a_document_it_may_read_and_404_for_any_other(new_session, library, unlimited, routes):
    ctx = _key_ctx(new_session, routes, GET, library.ws, [DOCUMENTS_READ])
    for name in ("shared", "unset", "copy"):
        assert _call(new_session, documents.get_document, document_id=library.ids[name], ctx=ctx).id == library.ids[name]
    for name in ("team", "private", "theirs"):
        with pytest.raises(HTTPException) as missing:
            _call(new_session, documents.get_document, document_id=library.ids[name], ctx=ctx)
        assert missing.value.status_code == 404, name
    member = _member(library.ws)
    assert _call(new_session, documents.get_document, document_id=library.ids["team"], ctx=member).id == library.ids["team"]


# ── search ───────────────────────────────────────────────────────────────────

def _hit(document_id, file_name, content, score):
    return {"score": score, "file_name": file_name, "content": content, "chunk_index": 0,
            "metadata": {"document_id": document_id}}


def _search(new_session, ctx):
    return _call(new_session, documents.semantic_search, body=None, query="opening hours", limit=None,
                 min_similarity=None, document_ids=None, team=None, ctx=ctx)


def test_a_keys_search_hears_only_of_documents_it_may_read_even_under_a_shared_name(new_session, library, unlimited,
                                                                                    routes, monkeypatch):
    """A reachable document shares its name with a private one. The private one's passage, named
    by file name only or by its id, never reaches the key: not its id, not its text, not a preview."""
    [namesake] = _add(new_session, _document(library.ws, "notes.md", team_access=[], is_shared=True))
    s = new_session()
    s.execute(text("INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (:id, 0, :content)"),
              {"id": library.ids["private"], "content": PRIVATE_TEXT})
    s.commit()
    s.close()
    hits = [
        _hit(None, "notes.md", PRIVATE_TEXT, 0.99),               # the private passage, by file name only
        _hit(library.ids["private"], "notes.md", PRIVATE_TEXT, 0.98),
        _hit(library.ids["team"], "margins.md", TEAM_TEXT, 0.97),
        _hit(library.ids["theirs"], "pricing.md", THEIR_TEXT, 0.96),
        _hit(library.ids["shared"], "pricing.md", "Harvest Club: £21 a month.", 0.5),
        _hit(namesake, "notes.md", "Opening hours: 8 to 4.", 0.4),
    ]
    searched = []

    async def funnel(query, max_chunks, workspace_id, team, *, floor=None):
        searched.append(workspace_id)
        return [dict(hit) for hit in hits]

    monkeypatch.setattr(documents, "_retrieval_hits", funnel)
    answer = _search(new_session, _key_ctx(new_session, routes, SEARCH, library.ws, [DOCUMENTS_READ]))

    assert searched == [library.ws]
    assert [r["document_id"] for r in answer["results"]] == [library.ids["shared"], namesake]
    assert answer["total_results"] == 2
    assert [r["excerpt"] for r in answer["results"]] == ["Harvest Club: £21 a month.", "Opening hours: 8 to 4."]
    for secret in (PRIVATE_TEXT, TEAM_TEXT, THEIR_TEXT):
        assert secret not in repr(answer)

    member = {r["document_id"] for r in _search(new_session, _member(library.ws))["results"]}
    assert _named(library, "private", "team") <= member                    # a member's search is as before


# ── delete ───────────────────────────────────────────────────────────────────

ALL = {"shared", "unset", "team", "private", "copy", "theirs"}


def _delete(new_session, ctx, document_id):
    return _call(new_session, documents.delete_document, document_id=document_id, ctx=ctx)


def test_a_key_deletes_one_document_and_leaves_every_other_and_its_deliverable(new_session, library, unlimited,
                                                                               routes, manager):
    ctx = _key_ctx(new_session, routes, DELETE, library.ws, [DOCUMENTS_WRITE])
    assert _delete(new_session, ctx, library.ids["shared"]) == {"message": "Document deleted successfully"}
    assert _remaining(new_session, library) == (ALL - {"shared"}, True)
    assert manager.deleted == [(library.ws, library.ids["shared"])]    # the removal members use, in its workspace


def test_a_key_cannot_delete_another_workspaces_or_a_limited_document(new_session, library, unlimited, routes,
                                                                      manager):
    ctx = _key_ctx(new_session, routes, DELETE, library.ws, [DOCUMENTS_WRITE])
    for name in ("theirs", "team", "private"):
        with pytest.raises(HTTPException) as missing:
            _delete(new_session, ctx, library.ids[name])
        assert missing.value.status_code == 404, name
    assert _remaining(new_session, library) == (ALL, True)
    assert manager.deleted == []


def test_a_key_deletes_its_deliverables_knowledge_copy_when_it_names_it_and_the_deliverable_stays(
        new_session, library, unlimited, routes, manager):
    ctx = _key_ctx(new_session, routes, DELETE, library.ws, [DOCUMENTS_WRITE])
    _delete(new_session, ctx, library.ids["copy"])
    assert _remaining(new_session, library) == (ALL - {"copy"}, True)
    assert manager.deleted == [(library.ws, library.ids["copy"])]


def test_a_member_deletes_as_before(new_session, library, manager):
    member = _member(library.ws)
    _delete(new_session, member, library.ids["team"])
    with pytest.raises(HTTPException) as missing:
        _delete(new_session, member, library.ids["theirs"])
    assert missing.value.status_code == 404
    assert _remaining(new_session, library) == (ALL - {"team"}, True)


# ── upload ───────────────────────────────────────────────────────────────────

class _Upload:
    def __init__(self, filename, data):
        self.filename, self._data = filename, data

    async def read(self):
        return self._data

    async def seek(self, _pos):
        pass


def _upload(new_session, ctx, filename, data):
    return _call(new_session, documents.handle_request, ctx=ctx, file=_Upload(filename, data), description=None,
                 tags=None, team_access=None)


def _row(new_session, document_id):
    s = new_session()
    try:
        return s.execute(text("SELECT workspace_id, filename, created_by, created_by_user_id, content_hash "
                              "FROM documents WHERE id = :id"), {"id": document_id}).fetchone()
    finally:
        s.close()


def _names_in(new_session, ws):
    s = new_session()
    try:
        return sorted(row.filename for row in s.execute(
            text("SELECT filename FROM documents WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws}))
    finally:
        s.close()


def test_a_keys_upload_lands_in_its_workspace_named_for_the_key(new_session, library, unlimited, routes, uploads):
    ctx = _key_ctx(new_session, routes, UPLOAD, library.ws, [DOCUMENTS_WRITE])
    answer = _upload(new_session, ctx, "opening-hours.md", b"Opening hours: 8 to 4, Monday to Saturday.\n")
    row = _row(new_session, answer.document_id)
    assert (str(row.workspace_id), row.filename) == (library.ws, "opening-hours.md")
    assert row.created_by == ctx.user.id and ctx.user.id.startswith("sdk:") and row.created_by_user_id is None
    assert uploads.manager.processed == [(library.ws, answer.document_id)]


def test_a_keys_upload_never_answers_with_or_writes_over_a_hidden_document(new_session, library, unlimited, routes,
                                                                            uploads):
    data = b"Board minutes, 3 November: the Harbour contract is not renewing.\n"
    [kept] = _add(new_session, _document(library.ws, "board-minutes.md", team_access=[], is_shared=False,
                                         content_hash=hashlib.sha256(data).hexdigest()))
    member = _upload(new_session, _member(library.ws), "minutes-again.md", data)
    assert (member.status, member.document_id) == ("duplicate", kept)   # a member's upload is as before
    key = _key_ctx(new_session, routes, UPLOAD, library.ws, [DOCUMENTS_WRITE])

    answer = _upload(new_session, key, "minutes-copy.md", data)          # the same bytes as a private document
    assert answer.status != "duplicate" and answer.document_id != kept
    for name in ("notes.md", "margins.md"):                               # the name of a private, a team document
        with pytest.raises(HTTPException) as clash:
            _upload(new_session, key, name, b"A new text under that name.\n")
        assert (clash.value.status_code, clash.value.detail) == (409, UPLOAD_CLASH), name
    assert _names_in(new_session, library.ws).count("notes.md") == 1
    assert _names_in(new_session, library.ws).count("margins.md") == 1
    assert _row(new_session, kept).content_hash == hashlib.sha256(data).hexdigest()


def test_a_keys_spreadsheet_never_writes_over_a_hidden_spreadsheets_copy(new_session, library, unlimited, routes,
                                                                          uploads):
    original = "name,role\nPriya,lead\n"
    _add(new_session, _document(library.ws, "roster list.csv", file_type="csv", team_access=[], is_shared=False))
    uploads.worker.files["documents/roster_list.csv"] = original
    key = _key_ctx(new_session, routes, UPLOAD, library.ws, [DOCUMENTS_WRITE])

    for name in ("roster list.csv", "roster_list.csv"):                   # its name, and a name with its copy's path
        with pytest.raises(HTTPException) as clash:
            _upload(new_session, key, name, b"name,role\nSam,stand-in\n")
        assert (clash.value.status_code, clash.value.detail) == (409, UPLOAD_CLASH), name
    assert uploads.worker.files["documents/roster_list.csv"] == original and uploads.worker.written == {}
    assert list(uploads.folder.iterdir()) == []                           # nothing was saved either
    assert "roster_list.csv" not in _names_in(new_session, library.ws)

    _upload(new_session, key, "rota.csv", b"day,shift\nMonday,early\n")   # a name of its own goes through
    assert set(uploads.worker.written) == {"documents/rota.csv"}
    member = _upload(new_session, _member(library.ws), "roster_list.csv", b"name,role\nSam,stand-in\n")
    assert member.document_id and "documents/roster_list.csv" in uploads.worker.written   # a member as before
