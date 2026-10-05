"""F346 (night 10b): Auto and the agents see every template, what each needs, and why a call failed.

* ``list_templates`` stopped at "Executive Summary": every template came back as an
  object with its description, the 18 social starters included, and a chat turn keeps
  about 2,000 tokens of a tool's answer, so the owner's own templates were cut off.
  Here the seeded workspace plus the owner's templates is listed through the same
  formatting and the same token cut the chat applies, and "Zebra" is still there.
* ``get_template_schema`` learned a table's columns only from sample data, so a
  template made by POST (no sample) gave guessed columns. Its answer now carries
  each table's columns from the blocks, and it takes a template's name too.
* ``data`` sent as JSON text with ``\\'`` for an apostrophe (4 of 55 calls) failed
  as "not a complete object"; now it generates, and text that truly is not JSON is
  refused naming the problem.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from types import SimpleNamespace
from typing import Any, List

import pytest

from core.context_guard import truncate_to_token_budget
from modules.documents.presets import PRESETS
from modules.documents.seed_templates import STARTER_TEMPLATES
from modules.documents.social_starters import social_starters
from modules.tools.discovery.handlers_documents import get_template_schema, list_templates
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from services import session_tools as st
from tests.f298_fixtures import call_tool, recorded

WS = uuid.UUID("00000000-0000-0000-0000-0000000346a1")
# What a chat turn keeps of one tool's answer (consumers/chatbot/service.py runs the loop with it).
CHAT_TOOL_TOKENS = inspect.signature(ToolLoopExecutor.__init__).parameters["content_truncate_tokens"].default
OWNERS_OWN = ("Harbourline Quote", "Lantern Kitchen Statement", "Monthly Wholesale Report", "Price List", "Zebra")


FILTERED_COLUMNS = ("id", "name", "format", "category")


def _keeps(row: Any, criterion: Any) -> bool:
    """Whether ``row`` passes one ``Column == value`` filter (the workspace and is_active ones pass)."""
    column = getattr(getattr(criterion, "left", None), "key", None)
    if column not in FILTERED_COLUMNS:
        return True
    return getattr(row, column) == getattr(getattr(criterion, "right", None), "value", None)


class _Rows:
    """A session holding template rows, sorted by name as the service orders them; filters apply."""

    def __init__(self, rows: List[Any]):
        self.rows = sorted(rows, key=lambda row: row.name)

    def query(self, *args: Any) -> "_Rows":
        return self

    def filter(self, *criteria: Any) -> "_Rows":
        return _Rows([row for row in self.rows if all(_keeps(row, c) for c in criteria)])

    def order_by(self, *args: Any) -> "_Rows":
        return self

    def all(self) -> List[Any]:
        return list(self.rows)

    def first(self) -> Any:
        return self.rows[0] if self.rows else None


def _row(name: str, fmt: str = "pdf", category: str = "general", description: str = "", **columns: Any) -> Any:
    fields = dict(id=uuid.uuid4(), name=name, format=fmt, category=category, description=description,
                  blocks=None, data_schema={}, sample_data={}, template_content=None)
    return SimpleNamespace(**{**fields, **columns})


def _seeded_workspace() -> List[Any]:
    """Every starter a workspace with Socials on is seeded with, and the owner's own templates."""
    starters = [_row(t["name"], t["format"], t["category"], t["description"], data_schema=t["data_schema"])
                for t in STARTER_TEMPLATES]
    presets = [_row(p["name"], p["format"], p["category"], p["description"], blocks=p["blocks"]) for p in PRESETS]
    socials = [_row(s["name"], s["format"], s["category"], s["description"], blocks=s["blocks"])
               for s in social_starters()]
    owners = [_row(name, "pdf", "invoice", "The owner's own template, made in the Studio.") for name in OWNERS_OWN]
    return starters + presets + socials + owners


def _what_chat_sees(answer: dict) -> str:
    """The answer as Auto reads it: formatted for the model, then cut to the chat's token budget."""
    return truncate_to_token_budget(ToolResultFormatter.format_for_llm(answer, "platform_list_templates"),
                                    CHAT_TOOL_TOKENS)


# --------------------------------------------------------------------------- #
# list_templates: every template, the owner's own included
# --------------------------------------------------------------------------- #


def test_auto_sees_the_owners_template_named_zebra():
    answer = asyncio.run(list_templates(_Rows(_seeded_workspace()), WS, {}))

    seen = _what_chat_sees(answer)

    assert "(truncated)" not in seen
    for name in OWNERS_OWN:
        assert name in seen, name
    assert answer["count"] == len(STARTER_TEMPLATES) + len(PRESETS) + len(OWNERS_OWN)


def test_social_templates_are_listed_with_their_format_and_a_name_search_narrows():
    rows = _Rows(_seeded_workspace())
    socials = asyncio.run(list_templates(rows, WS, {"format": "social_image"}))
    found = asyncio.run(list_templates(rows, WS, {"name": "zeb"}))

    assert socials["count"] and all(" | social_image | " in line for line in socials["templates"])
    assert found["count"] == 1 and found["templates"][0].startswith("Zebra | pdf | invoice | ")


def test_the_session_tool_forwards_the_name_search():
    ctx = st.SessionContext(task_id=3460, agent_id=268, agent_name="Business Analyst", workspace_id=str(WS))

    assert st.resolve_parameters(st.get_tool("list_templates"), {"name": "Zebra"}, ctx) == {"name": "Zebra"}


# --------------------------------------------------------------------------- #
# get_template_schema: a table's columns from the blocks, by id or by name
# --------------------------------------------------------------------------- #

PRICE_LIST_BLOCKS = {"blocks": [
    {"type": "heading", "id": "h", "level": 1, "content": [{"type": "variable", "path": "data.title"}]},
    {"type": "data_table", "id": "prices", "path": "data.prices", "columns": [
        {"key": "blend", "label": "Blend"}, {"key": "price_per_kg", "label": "Per kg", "align": "right"},
        {"key": "notes", "label": "Notes", "optional": True},
    ]},
    {"type": "text", "id": "terms", "content": [{"type": "variable", "path": "data.terms", "fallback": "Prices exclude VAT."}]},
]}


def _posted_price_list() -> Any:
    """A template made by POST /templates: blocks and no sample data."""
    return _row("Price List", blocks=PRICE_LIST_BLOCKS, sample_data={})


@pytest.mark.parametrize("params", [
    pytest.param({"template_name": "Price List"}, id="by-name"),
    pytest.param({"template_id": "Price List"}, id="a-name-in-template-id"),
])
def test_a_template_without_sample_data_still_lists_its_table_columns(params):
    template = _posted_price_list()

    schema = asyncio.run(get_template_schema(_Rows([template]), WS, params))

    assert schema["success"] is True and schema["id"] == str(template.id) and schema["sample_data"] == {}
    assert schema["tables"] == [{"field": "prices", "columns": ["blend", "price_per_kg", "notes"],
                                 "optional_columns": ["notes"], "required": True}]
    assert schema["required_fields"] == ["data.prices", "data.title"]
    assert schema["fallback_fields"] == {"data.terms": "Prices exclude VAT."}


def test_a_template_name_that_is_not_there_is_said_plainly():
    schema = asyncio.run(get_template_schema(_Rows([]), WS, {"template_name": "Zebra"}))

    assert schema["success"] is False and "No template 'Zebra'" in schema["error"]


def test_the_session_tool_takes_a_template_name():
    ctx = st.SessionContext(task_id=3461, agent_id=268, agent_name="Business Analyst", workspace_id=str(WS))
    tool = st.get_tool("get_template_schema")

    assert st.resolve_parameters(tool, {"template_name": "Price List"}, ctx) == {"template_name": "Price List"}


# --------------------------------------------------------------------------- #
# data sent as JSON text: an apostrophe written as \' generates
# --------------------------------------------------------------------------- #

APOSTROPHE = "\\'"  # a backslash, then an apostrophe: what the model wrote
DATA_TEXT = ('{"recipient_name": "Tom O' + APOSTROPHE + 'Brien", '
             '"subject": "Lantern Kitchen' + APOSTROPHE + 's October order", "body": "It ships on Friday."}')


def test_data_with_an_escaped_apostrophe_generates_the_document(monkeypatch):
    with pytest.raises(json.JSONDecodeError):
        json.loads(DATA_TEXT)  # strict JSON refuses it: this is the call that failed
    service = recorded(monkeypatch)

    answer = call_tool({"title": "Letter to Tom", "format": "pdf", "template_name": "Branded Letter", "data": DATA_TEXT})

    assert answer["success"] is True
    (asked,) = service.asked
    assert asked["data"]["recipient_name"] == "Tom O'Brien"
    assert asked["data"]["subject"] == "Lantern Kitchen's October order"


def test_data_that_is_not_json_is_refused_naming_the_problem(monkeypatch):
    service = recorded(monkeypatch)

    answer = call_tool({"title": "Letter", "format": "pdf", "template_name": "Branded Letter",
                        "data": '{"recipient_name": "Tom", "body": "unterminated}'})

    assert answer["success"] is False
    assert "line 1, column" in answer["error"] and "near" in answer["error"]
    assert service.asked == []
