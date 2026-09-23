"""F091 (night 3) — a confirmation card names what it acts on, and is never staged
for something that is not there.

Night 3's delete card named no document, and #503 was an id Auto had made up.
Before a gated call stages its card, the ids in its parameters are looked up in
the workspace: a missing one fails the call back to the model with no card, and
one that exists is named on the card.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from modules.tools.execution.subject_targets import (
    Target, missing_targets_error, named_subject, resolve_targets,
)

_TABLES = ("documents", "agents", "board_tasks")


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        for t in _TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{t}"))
            session.execute(text(f"CREATE TEMP TABLE {t} (LIKE public.{t} INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        for t in _TABLES:
            session.execute(text(f"DROP TABLE IF EXISTS pg_temp.{t}"))
        session.commit()
        session.close()


def _seed(db, ws):
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, status) "
                    "VALUES (716, CAST(:ws AS uuid), 'christmas-box-2026.csv', 'completed')"), {"ws": ws})
    db.execute(text("INSERT INTO agents (id, name, agent_type, workspace_id, status) "
                    "VALUES (294, 'Auto', 'custom', CAST(:ws AS uuid), 'active')"), {"ws": ws})
    db.execute(text("INSERT INTO board_tasks (id, workspace_id, title, status, priority, source_type, attempts) "
                    "VALUES (484, CAST(:ws AS uuid), 'Brand voice summary', 'done', 'medium', 'user', 0)"),
               {"ws": ws})


def test_an_id_that_is_there_is_named_and_one_that_is_not_is_missing(db):
    ws = str(uuid4())
    _seed(db, ws)
    found, missing = resolve_targets(db, ws, {"document_id": 716, "agent_id": "294", "task_id": 484})
    assert [(t.noun, t.ident, t.name) for t in found] == [
        ("document", 716, "christmas-box-2026.csv"), ("agent", 294, "Auto"), ("ticket", 484, "Brand voice summary")]
    assert missing == []
    found, missing = resolve_targets(db, ws, {"document_id": 503})
    assert found == [] and [(t.noun, t.ident) for t in missing] == [("document", 503)]


def test_another_workspaces_row_or_a_made_up_id_is_missing(db):
    ws = str(uuid4())
    _seed(db, ws)
    _, missing = resolve_targets(db, str(uuid4()), {"document_id": 716})
    assert [t.ident for t in missing] == [716]
    _, missing = resolve_targets(db, ws, {"document_id": "the-christmas-sheet"})
    assert [t.ident for t in missing] == ["the-christmas-sheet"]


def test_a_scheduled_task_id_is_looked_up_as_a_scheduled_task(db):
    """platform_cancel_scheduled_task's task_id is a schedule, not a ticket."""
    ws = str(uuid4())
    db.execute(text("DROP TABLE IF EXISTS pg_temp.agent_scheduled_tasks"))
    db.execute(text("CREATE TEMP TABLE agent_scheduled_tasks (id int, workspace_id uuid, description text)"))
    db.execute(text("INSERT INTO agent_scheduled_tasks VALUES (41, CAST(:ws AS uuid), 'Weekly roast-day reminder')"),
               {"ws": ws})
    found, missing = resolve_targets(db, ws, {"task_id": 41}, "platform_cancel_scheduled_task")
    assert [(t.noun, t.name) for t in found] == [("scheduled task", "Weekly roast-day reminder")] and not missing
    _, missing = resolve_targets(db, ws, {"task_id": 41})             # any other action: a ticket, and there is none
    assert [t.noun for t in missing] == ["ticket"]


def test_what_the_owner_and_the_model_are_told():
    assert named_subject([Target("document_id", 716, "document", "christmas-box-2026.csv")]) == (
        " on 'christmas-box-2026.csv' (document #716)")
    err = missing_targets_error("platform_delete_document", [Target("document_id", 503, "document")])
    assert err["success"] is False
    assert err["error"].startswith("No document #503 in this workspace — nothing was asked or done.")
    assert "call platform_delete_document with a real id" in err["error"]


def test_the_card_carries_the_named_subject(monkeypatch):
    from core.services import approval_grants as svc
    from modules.tools.execution import tool_grants

    made = {}
    monkeypatch.setattr(svc, "find_pending_grant", lambda *a, **k: None)
    monkeypatch.setattr(svc, "create_grant", lambda db, ws, **kw: made.update(kw) or NS(id=1, details=None))
    tool_grants.issue_tool_grant(object(), str(uuid4()), action="platform_delete_document",
                                 params={"document_id": 716}, description="Delete a document",
                                 subject=" on 'christmas-box-2026.csv' (document #716)")
    assert made["reason"] == ("Confirmation required before running platform_delete_document on "
                              "'christmas-box-2026.csv' (document #716): Delete a document")


def test_the_gate_looks_before_it_asks():
    from modules.tools.discovery import platform_executor

    source = inspect.getsource(platform_executor.PlatformActionExecutor)
    assert source.index("resolve_targets(self.db, self.workspace_id, params, action_name)") < source.index(
        "return tool_grants.attach_ask_grant(")
    assert "return missing_targets_error(action_name, missing)" in source
