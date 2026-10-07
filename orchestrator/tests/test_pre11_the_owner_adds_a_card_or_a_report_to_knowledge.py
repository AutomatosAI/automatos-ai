"""PRE-11 (Gerard, 7 Oct): "Add to Knowledge" on a board ticket card and on a report,
the way F354 built it for Deliverables.

- Who may: a workspace owner or admin, and the platform super-admin; in the local
  edition the operator (its session is the super-admin). An editor, viewer or member
  is refused. The report's route no longer sits behind the super-admin lock.
- Adding again answers with the copy already filed (``already_added``).
- ``DELETE`` removes the owner's copy; the card or the report stays.
- The board's and the Reports page's answers carry ``knowledge_document_id``.
- Another workspace's card or report is 404.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

ANSWER = "Cafés pay on 30-day terms."
REPORT = "# Supplier terms\n## Result\nCafés pay in 30 days.\n## Execution Metrics\n- Model: x\n- Cost: $0.01"

# agent_reports is raw DDL (alembic prd76/wave1c) the test schema never builds: a temp
# table with the columns services/report_knowledge.report_row reads (the F235 precedent).
_REPORTS_DDL = (
    "CREATE TEMP TABLE agent_reports (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), workspace_id uuid NOT NULL, "
    "agent_name varchar(255), report_type varchar(50) DEFAULT 'task', title text DEFAULT 'Supplier terms', "
    "file_path text DEFAULT 'reports/writer/r.md', linked_task_ids jsonb NOT NULL DEFAULT '[]'::jsonb, "
    "grade integer, acknowledged_at timestamptz, created_at timestamptz NOT NULL DEFAULT now())")


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """A workspace with cards and reports, an ingestion manager that files what it is
    given as a Document row (and deletes it), and reports that read as ``REPORT``."""
    from core.models.core import BoardTask, Document

    db = db_session
    ws = UUID(seed_workspace())
    uploads = []

    async def _upload(**kwargs):
        with open(kwargs["file_path"], encoding="utf-8") as fh:
            uploads.append({**kwargs, "content": fh.read()})
        doc = Document(filename=kwargs["filename"], workspace_id=ws, status="completed",
                       source_type=kwargs["source_type"], tags=kwargs["tags"])
        db.add(doc)
        db.flush()
        return doc.id

    def _delete(document_id):
        db.query(Document).filter(Document.id == document_id).delete()
        db.flush()
        return True

    async def _read_report(self, report_id, include_content=True):
        return {"success": True, "report": {"id": str(report_id), "content": REPORT}}

    monkeypatch.setattr("api.documents.get_document_manager",
                        lambda workspace_id: NS(upload_document=_upload, delete_document=_delete))
    monkeypatch.setattr("services.report_service.ReportService.get_report", _read_report)
    db.execute(text("DROP TABLE IF EXISTS pg_temp.agent_reports"))
    db.execute(text(_REPORTS_DDL))

    def card(status="done", result=ANSWER, in_ws=ws):
        made = BoardTask(workspace_id=in_ws, title="Payment terms for cafés", status=status, priority="low",
                         result=result)
        db.add(made)
        db.flush()
        return made

    def report(in_ws=ws, tasks=()):
        return str(db.execute(text(
            "INSERT INTO agent_reports (workspace_id, linked_task_ids) VALUES (CAST(:ws AS uuid), CAST(:t AS jsonb)) "
            "RETURNING id"), {"ws": str(in_ws), "t": f"[{', '.join(str(t) for t in tasks)}]"}).scalar())

    return NS(db=db, ws=ws, uploads=uploads, card=card, report=report)


def _ctx(ws):
    return NS(workspace_id=ws, user=NS(clerk_user_id="user_owner", id=1))


def _add_card(cafe, task_id, ws=None):
    from api.add_to_knowledge import add_card_to_knowledge

    return asyncio.run(add_card_to_knowledge(task_id, ctx=_ctx(ws or cafe.ws), db=cafe.db))


def _remove_card(cafe, task_id, ws=None):
    from api.add_to_knowledge import remove_card_from_knowledge

    return remove_card_from_knowledge(task_id, ctx=_ctx(ws or cafe.ws), db=cafe.db)


def _add_report(cafe, report_id, ws=None):
    from api.add_report_to_knowledge import add_report_to_knowledge

    return asyncio.run(add_report_to_knowledge(report_id, ctx=_ctx(ws or cafe.ws), db=cafe.db))


def _remove_report(cafe, report_id, ws=None):
    from api.add_report_to_knowledge import remove_report_from_knowledge

    return remove_report_from_knowledge(report_id, ctx=_ctx(ws or cafe.ws), db=cafe.db)


def _card_state(cafe, task):
    """The card's knowledge_document_id, as the board's list and its ticket view serve it."""
    from api.board_tasks import list_tasks
    from services.board_task_view import enrich_with_agents

    listed = list_tasks(ctx=_ctx(cafe.ws), db=cafe.db, status=None, agent_id=None, priority=None, search=None,
                        parent_task_id=None, limit=100, offset=0, finished_limit=None)
    [on_board] = [t for t in listed["tasks"] if t["id"] == task.id]
    [viewed] = enrich_with_agents([task], cafe.db, cafe.ws)
    assert on_board["knowledge_document_id"] == viewed["knowledge_document_id"]
    return viewed["knowledge_document_id"]


def _report_state(cafe, report_id, monkeypatch):
    """The report's knowledge_document_id, as the Reports list and the report's view serve it."""
    import api.reports as reports_api

    async def _listed(self, **_filters):
        return {"success": True, "reports": [{"id": report_id}], "total": 1}

    monkeypatch.setattr("services.report_service.ReportService.list_reports", _listed)
    listed = asyncio.run(reports_api.list_reports(agent_id=None, report_type=None, status=None, graded=None,
                                                  period="30d", limit=20, offset=0, ctx=_ctx(cafe.ws), db=cafe.db))
    viewed = asyncio.run(reports_api.get_report(report_id, ctx=_ctx(cafe.ws), db=cafe.db))
    assert listed["reports"][0]["knowledge_document_id"] == viewed["report"]["knowledge_document_id"]
    return viewed["report"]["knowledge_document_id"]


# ── a card ──────────────────────────────────────────────────────────────────

def test_an_approved_cards_answer_is_added_once_and_the_board_says_so(cafe):
    task = cafe.card()
    assert _card_state(cafe, task) is None

    got = _add_card(cafe, task.id)
    again = _add_card(cafe, task.id)

    assert got["success"] is True and got["already_added"] is False
    [filed] = cafe.uploads
    assert ANSWER in filed["content"] and filed["source_type"] is None           # the owner's, not agent_output
    assert {"added-by-owner", f"card:{task.id}"} <= set(filed["tags"])
    assert again == {**got, "already_added": True} and len(cafe.uploads) == 1      # filed once
    assert _card_state(cafe, task) == got["document_id"]


def test_removing_a_cards_copy_deletes_it_and_keeps_the_card(cafe):
    from core.models.core import BoardTask, Document

    task = cafe.card()
    doc_id = _add_card(cafe, task.id)["document_id"]

    got = _remove_card(cafe, task.id)

    assert got == {"success": True, "task_id": task.id, "removed": 1}
    assert cafe.db.query(Document).filter(Document.id == doc_id).first() is None
    assert cafe.db.query(BoardTask).filter(BoardTask.id == task.id).first() is not None
    assert _card_state(cafe, task) is None
    assert _remove_card(cafe, task.id)["removed"] == 0                  # nothing left to remove is not an error
    assert _add_card(cafe, task.id)["already_added"] is False           # and it can be added again


def test_a_document_the_owner_did_not_add_is_neither_shown_nor_removed(cafe):
    """Only the owner's copy counts: a document carrying the card's tag but not
    ``added-by-owner`` is not the card's copy, and Remove never deletes it."""
    from core.models.core import Document

    task = cafe.card()
    other = Document(filename="card.md", workspace_id=cafe.ws, status="completed", source_type="agent_output",
                     tags=["agent_output", f"card:{task.id}"])
    cafe.db.add(other)
    cafe.db.flush()

    assert _card_state(cafe, task) is None
    assert _remove_card(cafe, task.id)["removed"] == 0
    assert cafe.db.query(Document).filter(Document.id == other.id).first() is not None


# ── a report ────────────────────────────────────────────────────────────────

def test_a_report_is_added_once_without_its_metrics_and_its_answers_say_so(cafe, monkeypatch):
    report_id = cafe.report()
    assert _report_state(cafe, report_id, monkeypatch) is None

    got = _add_report(cafe, report_id)
    again = _add_report(cafe, report_id)

    assert got["success"] is True and got["already_added"] is False and got["report_id"] == report_id
    [filed] = cafe.uploads
    assert "Cafés pay in 30 days." in filed["content"] and "Execution Metrics" not in filed["content"]
    assert {"added-by-owner", f"report-added:{report_id}"} <= set(filed["tags"])
    assert again == {**got, "already_added": True} and len(cafe.uploads) == 1
    assert _report_state(cafe, report_id, monkeypatch) == got["document_id"]


def test_removing_a_reports_copy_deletes_it_and_keeps_the_report(cafe, monkeypatch):
    from core.models.core import Document

    report_id = cafe.report()
    doc_id = _add_report(cafe, report_id)["document_id"]

    got = _remove_report(cafe, report_id)

    assert got == {"success": True, "report_id": report_id, "removed": 1}
    assert cafe.db.query(Document).filter(Document.id == doc_id).first() is None
    assert cafe.db.execute(text("SELECT 1 FROM agent_reports WHERE id = CAST(:id AS uuid)"),
                           {"id": report_id}).fetchone() is not None
    assert _report_state(cafe, report_id, monkeypatch) is None


def test_a_report_waiting_for_its_ticket_is_refused(cafe):
    report_id = cafe.report(tasks=[cafe.card(status="review").id])

    with pytest.raises(HTTPException) as refused:
        _add_report(cafe, report_id)

    assert refused.value.status_code == 409 and "isn't approved yet" in refused.value.detail
    assert cafe.uploads == []


def test_another_workspaces_card_or_report_is_not_found(cafe, seed_workspace):
    other = UUID(seed_workspace())
    task, report_id = cafe.card(), cafe.report()

    for call, source in ((_add_card, task.id), (_remove_card, task.id),
                         (_add_report, report_id), (_remove_report, report_id), (_add_report, "not-a-uuid")):
        with pytest.raises(HTTPException) as missing:
            call(cafe, source, ws=other if source != "not-a-uuid" else None)
        assert missing.value.status_code == 404, call.__name__
    assert cafe.uploads == []


# ── who may ─────────────────────────────────────────────────────────────────

def _knowledge_routes():
    from api.add_report_to_knowledge import router as report_router
    from api.add_to_knowledge import router as card_router

    return [(method, route) for r in (card_router, report_router) for route in r.routes
            for method in route.methods if route.path.endswith("/add-to-knowledge")]


def _dependant_calls(dependant) -> set:
    calls = {getattr(dependant, "call", None)}
    for sub in getattr(dependant, "dependencies", []) or []:
        calls |= _dependant_calls(sub)
    return calls


def test_every_knowledge_route_is_behind_the_owner_or_admin_gate_only():
    from core.auth.super_admin import require_super_admin
    from core.auth.workspace_admin import require_workspace_admin
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    seen = set()
    for method, route in _knowledge_routes():
        calls = _dependant_calls(route.dependant)
        assert require_workspace_admin in calls, f"{method} {route.path}"
        assert require_super_admin not in calls, f"{method} {route.path} is still super-admin only"
        assert not any(getattr(c, PERMISSION_MARKER_ATTR, None) for c in calls), f"{method} {route.path}"
        seen.add((method, route.path))
    assert seen == {(m, p) for m in ("POST", "DELETE")
                    for p in ("/api/v1/tasks/{task_id}/add-to-knowledge", "/api/reports/{report_id}/add-to-knowledge")}


@pytest.fixture
def team(db_session, seed_workspace):
    """A workspace with one member of each role, each with a Clerk identity."""
    db = db_session
    ws = seed_workspace()

    def person(role=None, *, in_ws=ws, active=True):
        clerk = f"user_{uuid4().hex[:10]}"
        user_id = db.execute(text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) "
                                  "RETURNING id"), {"e": f"{clerk}@cafe.test", "u": clerk, "c": clerk}).scalar()
        if role:
            db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :user, :role, :active)"),
                       {"ws": str(in_ws), "user": user_id, "role": role, "active": active})
        db.flush()
        return NS(clerk=clerk, id=user_id)

    return NS(db=db, ws=UUID(ws), person=person)


def _signed_in(ws, clerk, system_role="user"):
    from core.auth.dependencies import RequestContext, UserContext

    return RequestContext(workspace_id=ws, user=UserContext(id=clerk, clerk_user_id=clerk, system_role=system_role),
                          auth_type="clerk")


def _allowed(db, ctx) -> bool:
    from core.auth.workspace_admin import require_workspace_admin

    try:
        return asyncio.run(require_workspace_admin(ctx=ctx, db=db)) is ctx
    except HTTPException as refused:
        assert refused.status_code == 403
        return False


@pytest.mark.parametrize("role, allowed", [("owner", True), ("admin", True), ("editor", False),
                                           ("viewer", False), ("member", False)])
def test_in_saas_an_owner_or_admin_may_and_nobody_else(team, role, allowed):
    assert _allowed(team.db, _signed_in(team.ws, team.person(role).clerk)) is allowed


def test_the_workspaces_owner_and_the_super_admin_may_and_strangers_may_not(team, seed_workspace):
    from core.auth.dependencies import RequestContext, UserContext

    owner = team.person()
    team.db.execute(text("UPDATE workspaces SET owner_id = :u WHERE id = CAST(:ws AS uuid)"),
                    {"u": owner.id, "ws": str(team.ws)})
    assert _allowed(team.db, _signed_in(team.ws, owner.clerk)) is True              # owns it, no member row
    assert _allowed(team.db, _signed_in(team.ws, "user_platform", "super_admin")) is True
    assert _allowed(team.db, _signed_in(team.ws, team.person("admin", active=False).clerk)) is False
    assert _allowed(team.db, _signed_in(team.ws, team.person("admin", in_ws=seed_workspace()).clerk)) is False
    api_key = RequestContext(workspace_id=team.ws, user=UserContext(id="api_key", system_role="admin"),
                             auth_type="api_key")
    assert _allowed(team.db, api_key) is False
    saas_anonymous = RequestContext(workspace_id=team.ws, user=UserContext(), auth_type="anonymous")
    assert _allowed(team.db, saas_anonymous) is False


def test_in_the_local_edition_the_operator_may(team, monkeypatch):
    """The local session is the operator's, and hybrid.py makes it the super-admin."""
    from core.auth import hybrid
    from core.auth.dependencies import RequestContext

    monkeypatch.setattr(hybrid, "_resolve_local_operator", lambda db, email: {
        "email": email, "id": 1, "name": "Operator", "username": "operator", "avatar_url": None})
    operator = hybrid._local_operator_user_context(team.db)
    ctx = RequestContext(workspace_id=team.ws, user=operator, auth_type="anonymous")

    assert _allowed(team.db, ctx) is True


def test_over_http_a_member_is_refused_and_an_admin_gets_through(team, cafe):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.add_report_to_knowledge import router as report_router
    from api.add_to_knowledge import router as card_router
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    who = {}
    app = FastAPI()
    app.include_router(card_router)
    app.include_router(report_router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: who["ctx"]
    app.dependency_overrides[get_db] = lambda: team.db
    client = TestClient(app, raise_server_exceptions=False)
    task, report_id = cafe.card(in_ws=team.ws), cafe.report(in_ws=team.ws)
    paths = (f"/api/v1/tasks/{task.id}/add-to-knowledge", f"/api/reports/{report_id}/add-to-knowledge")

    who["ctx"] = _signed_in(team.ws, team.person("member").clerk)
    for path in paths:
        for verb in (client.post, client.delete):
            assert verb(path).status_code == 403, path
    assert cafe.uploads == []

    who["ctx"] = _signed_in(team.ws, team.person("admin").clerk)
    for path in paths:
        assert client.delete(path).json()["removed"] == 0
