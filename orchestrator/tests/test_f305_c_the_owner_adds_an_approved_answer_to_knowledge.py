"""F305 (night 9): "Add to knowledge" files an approved card's answer as the OWNER's
document (not agent_output), which the default search then finds; a card that is not
approved is refused with a message saying so.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import HTTPException

ANSWER = "Cafés pay on 30-day terms."


@pytest.fixture
def owner(db_session, seed_workspace, monkeypatch):
    from core.models.core import BoardTask, Document

    ws = UUID(seed_workspace())
    filed = {}

    async def _upload(**kwargs):
        with open(kwargs["file_path"], encoding="utf-8") as fh:
            filed.update(kwargs, content=fh.read())
        doc = Document(filename=kwargs["filename"], workspace_id=ws, status="completed",
                       source_type=kwargs["source_type"], tags=kwargs["tags"])
        db_session.add(doc)
        db_session.flush()
        return doc.id

    monkeypatch.setattr("api.documents.get_document_manager", lambda workspace_id: NS(upload_document=_upload))

    def card(status, result=ANSWER):
        made = BoardTask(workspace_id=ws, title="Payment terms for cafés", status=status, priority="low",
                         result=result)
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, filed=filed, card=card)


def _ctx(ws):
    return NS(workspace_id=ws, user=NS(clerk_user_id="user_owner", id=1))


def test_an_approved_card_becomes_the_owners_document_and_search_finds_it(owner):
    from api.add_to_knowledge import add_card_to_knowledge
    from services.agents_writing import owners_passages_only

    task = owner.card("done")
    got = asyncio.run(add_card_to_knowledge(task.id, ctx=_ctx(owner.ws), db=owner.db))

    assert got["success"] is True
    assert owner.filed["source_type"] is None and "added-by-owner" in owner.filed["tags"]
    assert ANSWER in owner.filed["content"] and owner.filed["created_by"] == "user_owner"

    search = AsyncMock(return_value={"success": True, "results": [{"document_id": got["document_id"],
                                                                  "content": ANSWER}]})
    found = asyncio.run(owners_passages_only("results")(search)(owner.db, owner.ws, {"query": "café terms"}))
    assert [r["document_id"] for r in found["results"]] == [got["document_id"]]   # the default search finds it


@pytest.mark.parametrize("status", ["review", "in_progress", "blocked", "inbox"])
def test_a_card_that_is_not_approved_is_refused(owner, status):
    from api.add_to_knowledge import add_card_to_knowledge

    task = owner.card(status)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(add_card_to_knowledge(task.id, ctx=_ctx(owner.ws), db=owner.db))

    assert refused.value.status_code == 409 and "isn't approved yet" in refused.value.detail
    assert owner.filed == {}


def test_another_workspaces_card_is_not_found(owner, seed_workspace):
    from api.add_to_knowledge import add_card_to_knowledge

    task = owner.card("done")
    with pytest.raises(HTTPException) as missing:
        asyncio.run(add_card_to_knowledge(task.id, ctx=_ctx(UUID(seed_workspace())), db=owner.db))

    assert missing.value.status_code == 404 and owner.filed == {}


def test_a_reports_metrics_stay_out_of_the_document():
    from services.owner_knowledge import _METRICS

    report = "# Report\n## Result\nCafés pay in 30 days.\n## Execution Metrics\n- Model: x\n- Cost: $0.01"
    assert _METRICS.sub("", report).strip() == "# Report\n## Result\nCafés pay in 30 days."
