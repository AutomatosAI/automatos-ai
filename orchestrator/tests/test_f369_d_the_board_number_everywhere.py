"""Gerard, 7 Oct: "same number everywhere, it has to be easy… a user can say a board number to
Auto and it knows."

A ticket reference a person or an agent gives ("#0892", "0892", "892", 892) is the ticket with
that board number in the workspace: in Auto's chat, in the platform tools on a board run, in a
CLI session's tools, on a confirmation card, for a playbook run's card and in the board's search.
Only the platform's own id in a structured call (a ``TicketId``) is read as the id.

The board: #0892 "Design Brand Kit" is id 2147; #0708, a playbook run's card, is id 892.
"""
from __future__ import annotations

import asyncio
import contextlib
import operator
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql

from core.llm.usage_context import LANE_BOARD_TASK, LANE_SESSION, usage_scope
from services.ticket_numbers import TicketId
from services.ticket_refs import by_ticket_number, ticket_id_named

WS = UUID("7d1e2f3a-4b5c-4d6e-8f7a-9b0c1d2e3f4a")
RUN = UUID("3c4d5e6f-7a8b-4c9d-8e0f-1a2b3c4d5e6f")
BRAND_KIT = NS(id=2147, workspace_seq=892, workspace_id=WS, title="Design Brand Kit", source_type="orchestration",
               orchestration_run_id=RUN, parent_task_id=None, source_id=None, status="in_progress")
RECIPE_RUN = NS(id=892, workspace_seq=708, workspace_id=WS, title="Weekly roast recipe", source_type="recipe",
                orchestration_run_id=None, parent_task_id=None, source_id="exec-45d8ac862a79", status="done")
SAID = ("#0892", "0892", "892", 892)


class _Query:
    """Rows of the board, narrowed by each ``column == value`` the call filters on."""

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
    def __init__(self, *rows):
        self.rows = rows or (BRAND_KIT, RECIPE_RUN)

    def query(self, *columns):
        return _Query(list(self.rows))

    def execute(self, statement, params):
        row = next((r for r in self.rows if r.id == params["id"]), None)
        return NS(first=lambda: (row.title,) if row else None)

    def begin_nested(self):
        return contextlib.nullcontext()


@by_ticket_number
async def _update_task(db, workspace_id, params):
    return {"success": True, "edited": params["task_id"]}


@pytest.mark.parametrize("said", SAID)
def test_a_platform_tool_on_a_board_run_takes_the_board_number(said):
    with usage_scope(request_type=LANE_BOARD_TASK, execution_id="board_task:2150"):
        answer = asyncio.run(_update_task(_Db(), WS, {"task_id": said}))

    assert answer["edited"] == BRAND_KIT.id, said


@pytest.mark.parametrize("said", SAID[:3])
def test_a_sessions_get_mission_takes_the_board_number(said):
    """A CLI session's get_mission is platform_get_mission, which reads a card's number (F241 7b)."""
    from modules.tools.discovery.mission_refs import READS, takes_card_numbers
    from services.session_tools import SessionContext, get_tool, resolve_parameters

    asked = {}

    @takes_card_numbers(READS)
    async def get_mission(db, workspace_id, params):
        asked.update(params)
        return {"success": True}

    ctx = SessionContext(task_id=2150, agent_id=7, agent_name="Analyst", workspace_id=WS)
    params = resolve_parameters(get_tool("get_mission"), {"mission_id": said}, ctx)
    with usage_scope(request_type=LANE_SESSION, execution_id="session:2150"):
        asyncio.run(get_mission(_Db(), WS, params))

    assert asked == {"mission_id": str(RUN)}, said


def test_a_sessions_task_list_shows_each_tickets_number():
    from services.session_tools import _project_list_tasks

    listed = _project_list_tasks({"success": True, "tasks": [{"id": 2147, "number": "#0892", "title": "Design"}]})

    assert listed["tasks"] == [{"number": "#0892", "id": 2147, "title": "Design"}]


def test_the_platforms_own_id_in_a_structured_call_is_the_id():
    with usage_scope(request_type=LANE_BOARD_TASK, execution_id="board_task:2150"):
        answer = asyncio.run(_update_task(_Db(), WS, {"task_id": TicketId(892)}))
        found, error = ticket_id_named(_Db(), WS, TicketId(892))

    assert answer["edited"] == RECIPE_RUN.id and type(answer["edited"]) is int
    assert (found, error) == (RECIPE_RUN.id, None)


def test_a_bare_number_no_ticket_has_is_an_id_and_none_names_no_ticket():
    assert ticket_id_named(_Db(RECIPE_RUN), WS, 892) == (RECIPE_RUN.id, None)     # an id a tool's answer gave
    found, error = ticket_id_named(_Db(), WS, "4321")
    assert found is None and "No ticket #4321" in error


@pytest.mark.parametrize("said", SAID)
def test_the_hierarchy_gate_checks_the_owner_of_the_ticket_the_call_will_change(monkeypatch, said):
    """An agent's update to #0892 was checked against #0708's owner (id 892), or, as "#0892", against no row."""
    import core.security.hierarchy_permissions as hp

    checked = []
    analyst = NS(name="Analyst", is_system_agent=False, reports_to_id=None, workspace_id=WS, status="active")
    monkeypatch.setattr(hp, "_agent_row", lambda db, agent_id: analyst)
    monkeypatch.setattr(hp, "_owner_id", lambda db, sql, target_id: checked.append(target_id))

    hp.can_actor_modify(_Db(), actor_agent_id=9, target_type=hp.TARGET_TASK, workspace_id=WS, target_id=said,
                        source="platform_tool")

    assert checked == [BRAND_KIT.id], said


def test_the_gates_other_callers_and_a_ticket_named_by_nothing_are_checked_as_given(monkeypatch):
    import core.security.hierarchy_permissions as hp

    checked = []
    analyst = NS(name="Analyst", is_system_agent=False, reports_to_id=None, workspace_id=WS, status="active")
    monkeypatch.setattr(hp, "_agent_row", lambda db, agent_id: analyst)
    monkeypatch.setattr(hp, "_owner_id", lambda db, sql, target_id: checked.append(target_id))

    hp.can_actor_modify(_Db(), actor_agent_id=9, target_type=hp.TARGET_TASK, workspace_id=WS, target_id=892)
    hp.can_actor_modify(_Db(), actor_agent_id=9, target_type=hp.TARGET_TASK, workspace_id=WS, target_id=4321,
                        source="platform_tool")

    assert checked == [892, 4321]


def test_a_confirmation_card_names_the_ticket_the_call_will_change():
    from modules.tools.execution.subject_targets import resolve_targets

    found, missing = resolve_targets(_Db(), WS, {"task_id": "892"}, "platform_update_task")

    assert [(t.ident, t.called) for t in found] == [(BRAND_KIT.id, "ticket #0892")] and not missing


def test_a_playbook_runs_card_is_found_by_its_bare_number():
    from modules.tools.discovery.playbook_run_results import _run_id_for_card

    assert _run_id_for_card(_Db(), WS, "708") == RECIPE_RUN.source_id
    assert _run_id_for_card(_Db(), WS, "exec-45d8ac862a79") is None        # an execution id names no card


def test_the_owner_naming_card_892_in_words_means_0892():
    from modules.tools.discovery.card_words_said import card_said

    assert card_said(_Db(), WS, "892") == (BRAND_KIT.id, False)
    assert card_said(_Db(RECIPE_RUN), WS, "892") == (RECIPE_RUN.id, True)          # no #0892: its id


@pytest.mark.parametrize("said", ("#0892", "0892", "892"))
def test_the_boards_search_finds_a_ticket_by_its_number(said):
    from services.ticket_numbers import title_or_number

    sql = str(title_or_number(said).compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

    assert "board_tasks.workspace_seq = 892" in sql and "ILIKE" in sql.upper(), said


@pytest.mark.parametrize("said", ("Guji", "#0051.3", "1234567890"))
def test_the_boards_search_by_words_is_by_title_only(said):
    from services.ticket_numbers import title_or_number

    assert "workspace_seq" not in str(title_or_number(said).compile(dialect=postgresql.dialect()))
