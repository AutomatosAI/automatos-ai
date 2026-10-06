"""Deliverables carry tags (Gerard, 7 Oct).

Deliverables had no tags. They now live in ``extra["tags"]`` (no migration): trimmed,
lowercase, each once, at most 10 of at most 40 characters, refused in plain words where
they come in (the generate route's body, generate_document's arguments). The list and
detail answers return ``tags``; the list takes a ``tag`` filter; a Deliverable a board
card produced (a session's files, a document generated while working the card) carries
the card's tags after its own.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from services.deliverable_filters import TAG_CONDITION, list_filter
from services.deliverable_service import DeliverableService
from services.deliverable_tags import MAX_TAGS, TagsRefused, tags_of, validated_tags

WS = uuid4()
CARD = 612


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


def test_tags_are_trimmed_lowercase_and_each_once():
    assert validated_tags(["  Invoice ", "invoice", "Q3   Report", ""]) == ["invoice", "q3 report"]
    assert validated_tags("Invoice, Harbourline ,") == ["invoice", "harbourline"]
    assert validated_tags(None) == []


@pytest.mark.parametrize("raw, says", [
    ({"tag": "invoice"}, "must be a list"),
    (["invoice", 7], "must be text"),
    (["x" * 41], "at most 40 characters"),
    ([f"tag {n}" for n in range(MAX_TAGS + 1)], "at most 10 tags"),
])
def test_tags_that_break_the_rules_are_refused_saying_which_rule(raw, says):
    with pytest.raises(TagsRefused, match=says):
        validated_tags(raw)


def test_stored_or_card_tags_are_read_leniently():
    many = [f"Tag {n}" for n in range(12)]
    assert tags_of(["Harbourline", 3, "x" * 41, "harbourline", "Session"]) == ["harbourline", "session"]
    assert tags_of(many) == [f"tag {n}" for n in range(10)]
    assert tags_of("invoice") == [] and tags_of(None) == []


# ---------------------------------------------------------------------------
# The list's tag filter, and the answers' tags
# ---------------------------------------------------------------------------


def test_the_tag_filter_matches_the_cleaned_tag_as_a_bound_parameter():
    where, params = list_filter(WS, {"tag": "  Q3 ", "search": "box"})
    assert TAG_CONDITION in where and params["tag"] == "q3"
    assert params["search"] == "%box%" and params["workspace_id"] == str(WS)

    where, params = list_filter(WS, {"tag": "   "})
    assert TAG_CONDITION not in where and "tag" not in params


def _row(**overrides):
    values = {
        "id": uuid4(), "workspace_id": WS, "source_type": "task", "source_id": str(CARD), "agent_id": 7,
        "agent_name": "Ops", "artifact_type": "document", "title": "Invoice", "summary": None,
        "storage_type": "generated", "file_path": "generated/invoice.pdf", "file_name": "invoice.pdf",
        "file_type": "pdf", "file_size_bytes": 10, "preview_url": "/api/documents/generated/invoice.pdf",
        "preview_type": None, "extra": {"tags": ["invoice", "harbourline"]}, "status": "ready",
        "created_at": None, "updated_at": None,
    }
    return NS(**{**values, **overrides})


def test_the_list_filters_by_tag_and_answers_with_each_deliverables_tags():
    db = MagicMock()
    db.execute.side_effect = [MagicMock(scalar=lambda: 1), MagicMock(fetchall=lambda: [_row()])]

    out = DeliverableService(db, WS).list_deliverables(tag="Invoice")

    assert out["deliverables"][0]["tags"] == ["invoice", "harbourline"]
    count_sql, count_params = db.execute.call_args_list[0][0]
    assert "jsonb_array_elements_text" in str(count_sql) and count_params["tag"] == "invoice"


def test_a_deliverable_without_tags_answers_an_empty_list():
    assert DeliverableService._row_to_dict(_row(extra=None))["tags"] == []
    assert DeliverableService._row_to_dict(_row(extra={"tags": "not a list"}))["tags"] == []


def test_the_detail_answer_carries_the_tags():
    db = MagicMock()
    db.execute.return_value = MagicMock(fetchone=lambda: _row())

    out = asyncio.run(DeliverableService(db, WS).get_deliverable("d-1"))

    assert out["deliverable"]["tags"] == ["invoice", "harbourline"]


# ---------------------------------------------------------------------------
# A board card's Deliverable carries the card's tags
# ---------------------------------------------------------------------------


def _db(card_tags):
    db = MagicMock()
    db.execute.return_value = MagicMock(fetchone=lambda: (uuid4(), True))
    db.query.return_value.filter.return_value.scalar.return_value = card_tags
    return db


def _registered_extra(db) -> dict:
    return json.loads(db.execute.call_args[0][1]["extra"])


def test_a_cards_deliverable_carries_the_cards_tags_after_its_own():
    db = _db(["Harbourline", "Session", "invoice"])

    out = DeliverableService(db, WS).register(file_path="sessions/612/invoice.png", source_type="task",
                                              source_id=str(CARD), extra={"tags": ["invoice"], "task_id": CARD})

    assert out["success"] is True
    assert _registered_extra(db) == {"tags": ["invoice", "harbourline", "session"], "task_id": CARD}


@pytest.mark.parametrize("source_type, source_id", [("chat", None), ("mission", "m-1"), ("task", "task-1")])
def test_a_deliverable_from_no_card_reads_no_card(source_type, source_id):
    db = _db(["harbourline"])

    DeliverableService(db, WS).register(file_path="outputs/chart.png", source_type=source_type, source_id=source_id)

    db.query.assert_not_called()
    assert "tags" not in _registered_extra(db)


def test_a_card_whose_tags_cannot_be_read_still_gets_its_deliverable():
    db = _db([])
    db.query.return_value.filter.return_value.scalar.side_effect = OperationalError("SELECT", {}, Exception("gone"))

    out = DeliverableService(db, WS).register(file_path="sessions/612/chart.png", source_type="task",
                                              source_id=str(CARD))

    assert out["success"] is True and "tags" not in _registered_extra(db)
    db.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Where tags come in: generate_document, the generate route
# ---------------------------------------------------------------------------

PARAMS = {"title": "Invoice HL-2026-0145", "format": "pdf",
          "data": {"sections": [{"title": "Invoice", "content": "Two bags of Guji, 2 x 12.50."}]}}


def test_generate_document_reads_its_tags():
    from modules.tools.execution import generate_document_tool as gdt

    assert gdt.document_request({**PARAMS, "tags": [" Invoice", "Harbourline"]}).tags == ("invoice", "harbourline")
    assert gdt.document_request(PARAMS).tags == ()
    with pytest.raises(gdt.DocumentArgsRefused, match="The document was not made: a tag has at most 40"):
        gdt.document_request({**PARAMS, "tags": ["x" * 41]})


def test_generate_document_registers_its_tags(monkeypatch):
    import modules.documents.generation_service as generation_service
    from modules.tools.execution import generate_document_tool as gdt

    asked = []

    class _Service:
        def __init__(self, db, workspace_id):
            pass

        async def generate(self, **kwargs):
            return NS(filename="invoice.pdf", format="pdf", download_url="/api/documents/generated/invoice.pdf",
                      size=2048, content="# Invoice", template_id=None, template_name=None)

        def register_as_deliverable(self, result, **kwargs):
            asked.append(kwargs)
            return {"success": True, "deliverable_id": "d-1"}

        def share_link(self, result):
            return None

    async def _no_ingest(*args, **kwargs):
        return None

    monkeypatch.setattr(generation_service, "DocumentGenerationService", _Service)
    monkeypatch.setattr(gdt, "_ingest", _no_ingest)
    request = gdt.document_request({**PARAMS, "tags": ["invoice"]})

    asyncio.run(gdt.make_document(None, request, NS(id=7, name="Ops"), WS, card_id=CARD))

    assert asked[0]["tags"] == ("invoice",) and asked[0]["source_id"] == str(CARD)


def test_the_generate_route_validates_its_tags():
    from api.document_generation import GenerateDocumentRequest

    body = GenerateDocumentRequest(title="R", format="pdf", data={}, tags=[" Invoice ", "invoice", "Q3"])
    assert body.tags == ["invoice", "q3"]
    assert GenerateDocumentRequest(title="R", format="pdf", data={}).tags == []
    with pytest.raises(ValidationError, match="at most 10 tags"):
        GenerateDocumentRequest(title="R", format="pdf", data={}, tags=[f"t{n}" for n in range(11)])


def test_a_generated_documents_deliverable_records_its_tags(monkeypatch):
    import modules.documents.generation_service as gs
    import services.deliverable_service as ds
    from modules.documents.deliverable_extra import deliverable_extra

    result = gs.GeneratedDocument(path="/x/invoice.pdf", format="pdf", filename="invoice.pdf", size=9)
    assert deliverable_extra(result, tags=["Invoice", "q3"])["tags"] == ["invoice", "q3"]
    assert "tags" not in deliverable_extra(result)

    registered = []

    class _Deliverables:
        def __init__(self, db, workspace_id):
            pass

        def register(self, **kwargs):
            registered.append(kwargs)
            return {"success": True, "deliverable_id": "d-1"}

    monkeypatch.setattr(ds, "DeliverableService", _Deliverables)
    monkeypatch.setattr(gs, "copy_when_registered", lambda *args: False)

    gs.DocumentGenerationService(object(), WS).register_as_deliverable(result, title="Invoice", tags=["invoice"])

    assert registered[0]["extra"]["tags"] == ["invoice"]


# ---------------------------------------------------------------------------
# Both schemas an agent reads declare the argument
# ---------------------------------------------------------------------------


def test_both_generate_document_schemas_declare_tags():
    from modules.agents.services.agent_platform_tools import AgentPlatformTools
    from modules.tools.registry.tool_registry import ToolRegistry
    from services.deliverable_tag_schemas import TAGS_DESCRIPTION

    spec = ToolRegistry().get_tool("generate_document")
    param = next(p for p in spec.parameters if p.name == "tags")
    assert (param.type, param.required, param.description) == ("array", False, TAGS_DESCRIPTION)
    assert spec.to_openai_format()["parameters"]["properties"]["tags"]["items"] == {"type": "string"}

    inline = next(t for t in AgentPlatformTools.get_available_tools(object()) if t["name"] == "generate_document")
    assert inline["parameters"]["properties"]["tags"]["description"] == TAGS_DESCRIPTION
    assert "tags" not in inline["parameters"]["required"]
