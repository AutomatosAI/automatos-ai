"""Gerard, 7 Oct: Auto sees a ticket by its board number only.

The ticket tools answered with a ticket's number (#0950) and its database id (892). A model that
copied the id into its next call named #0892 instead, because 892 was also a board number. What
Auto, an API agent or a CLI session reads of a ticket tool's answer now names each ticket by its
number alone; the answer itself (the chat's cards, the frontend) keeps the ids.
"""
from __future__ import annotations

import asyncio
import json
import operator
import re
from types import SimpleNamespace as NS
from uuid import UUID

from core.llm.usage_context import LANE_CHAT, usage_scope
from modules.tools.formatting.result_formatter import ToolResultFormatter
from services.session_tools_rpc import render_result
from services.ticket_refs import by_ticket_number

WS = UUID("8e2f3a4b-5c6d-4e7f-9a8b-0c1d2e3f4a5b")
OTHER = NS(id=2147, workspace_seq=892, workspace_id=WS, title="Design Brand Kit", source_type="user")
NEW = NS(id=892, workspace_seq=950, workspace_id=WS, title="Price list for the café", source_type="user")


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *clauses):
        rows = self.rows
        for clause in clauses:
            key = getattr(getattr(clause, "left", None), "key", None)
            if key and getattr(clause, "operator", None) is operator.eq:
                rows = [r for r in rows if getattr(r, key, None) == clause.right.value]
        return _Query(rows)

    def all(self):
        return list(self.rows)

    def first(self):
        return self.rows[0] if self.rows else None


class _Db:
    """#0892 is id 2147; the new ticket, #0950, is id 892."""

    def query(self, *columns):
        return _Query([OTHER, NEW])


def _as_auto_reads(answer, tool="platform_get_task"):
    return ToolResultFormatter.format_for_llm(answer, tool)


def test_a_ticket_auto_reads_has_its_number_and_no_id():
    answer = {"success": True,
              "task": {"id": 2147, "number": "#0892", "title": "Design Brand Kit", "status": "review"},
              "frontend_data": {"task_card": {"id": 2147, "number": "#0892", "status": "review"}}}

    text = _as_auto_reads(answer)

    assert "#0892" in text and "2147" not in text
    assert answer["task"]["id"] == 2147 and answer["frontend_data"]["task_card"]["id"] == 2147   # the UI keeps it


def test_a_list_auto_reads_names_each_ticket_by_number_and_an_unnumbered_one_by_its_id():
    answer = {"success": True, "total": 2,
              "tasks": [{"id": 2147, "number": "#0892", "title": "Design Brand Kit"},
                        {"id": 61, "number": None, "title": "From before numbering"}]}

    tasks = json.loads(_as_auto_reads(answer, "platform_list_tasks").split("\n\n", 1)[1])["tasks"]

    assert tasks == [{"number": "#0892", "title": "Design Brand Kit"},
                     {"id": 61, "number": None, "title": "From before numbering"}]


def test_a_bulk_move_auto_reads_lists_the_numbers():
    answer = {"success": True, "status": "cancelled", "updated": [2147], "updated_numbers": ["#0892"],
              "failed": [{"task_id": 999999, "error": "not found", "number": None}]}

    seen = json.loads(_as_auto_reads(answer, "platform_update_task_status").split("\n\n", 1)[1])

    assert seen["updated"] == ["#0892"] and "updated_numbers" not in seen
    assert seen["failed"] == [{"task_id": 999999, "error": "not found", "number": None}]   # what was sent


def test_a_session_reads_a_ticket_by_its_number_only():
    rendered = render_result({"success": True, "tasks": [{"id": 2147, "number": "#0892", "title": "Design"}]})

    body = rendered["content"][0]["text"]
    assert "#0892" in body and "2147" not in body


@by_ticket_number
async def _create_task(db, workspace_id, params):
    return {"success": True, "task_id": NEW.id, "status": "inbox", "title": NEW.title}


@by_ticket_number
async def _update_task(db, workspace_id, params):
    return {"success": True, "task_id": params["task_id"], "updated": {"title": params["title"]}}


def test_create_then_update_by_the_number_in_the_answer_changes_the_new_ticket():
    """The new ticket's id (892) is another ticket's number (#0892): the id from the answer
    would have changed #0892. The answer Auto reads gives #0950, and #0950 is the new ticket."""
    with usage_scope(request_type=LANE_CHAT, execution_id="chat:f369e"):
        created = asyncio.run(_create_task(_Db(), WS, {"title": NEW.title}))
        read = _as_auto_reads(created, "platform_create_task")
        (number,) = re.findall(r"#\d{4}", read)
        edited = asyncio.run(_update_task(_Db(), WS, {"task_id": number, "title": "Price list, autumn"}))

    assert created["task_id"] == NEW.id and created["number"] == "#0950"   # the answer keeps the id for the UI
    assert number == "#0950" and '"task_id"' not in read
    assert edited["task_id"] == NEW.id
