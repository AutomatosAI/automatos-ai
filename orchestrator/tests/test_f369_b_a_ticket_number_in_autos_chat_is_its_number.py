"""F369 (night 10c, chat 16bb619c 18:10): Auto tags the ticket it just called #0892.

Auto filed "Design Brand Kit" and told the owner it was #0892 (id 2147). "Please tag that
ticket" became platform_update_task {task_id: 892}: 892 is #0892's number and the id of #0708, a
finished run. The call was refused naming both (F241's rule for a tie), and Auto asked the owner
"confirm the exact task ID". In Auto's chat a ticket is named by its number (PRD-252 R4), so
there the number is the one meant; elsewhere a tie is still refused.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

from core.llm.usage_context import LANE_BOARD_TASK, LANE_CHAT, usage_scope
from services.ticket_numbers import read_bare_refs
from services.ticket_refs import by_ticket_number

WS = UUID("5a4b3c2d-1e0f-4a9b-8c7d-6e5f4a3b2c1d")
RECIPE_RUN = NS(id=892, workspace_seq=708, title="Weekly roast recipe")            # #0708, id 892
DESIGN_BRAND_KIT = NS(id=2147, workspace_seq=892, title="Design Brand Kit")        # #0892, id 2147


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *clauses):
        return self

    def all(self):
        return list(self.rows)

    def first(self):
        return None


class _Db:
    """The workspace's board: the two tickets 892 can name."""

    def query(self, *columns):
        return _Query([RECIPE_RUN, DESIGN_BRAND_KIT])


@by_ticket_number
async def _update_task(db, workspace_id, params):
    return {"success": True, "edited": params["task_id"]}


def test_in_autos_chat_892_is_ticket_0892():
    with usage_scope(request_type=LANE_CHAT, execution_id="chat:16bb619c"):
        found, refused = read_bare_refs(_Db(), WS, [892])
        answer = asyncio.run(_update_task(_Db(), WS, {"task_id": 892, "tags": ["sim-night-2026-10-06"]}))

    assert refused is None and found == {892: DESIGN_BRAND_KIT.id}
    assert answer == {"success": True, "edited": DESIGN_BRAND_KIT.id}


def test_outside_autos_chat_the_tie_is_still_refused_naming_both():
    with usage_scope(request_type=LANE_BOARD_TASK, execution_id="board_task:2150"):
        found, refused = read_bare_refs(_Db(), WS, [892])

    assert found == {} and "892 is both the id of" in refused


def test_a_ref_only_one_reading_names_is_unchanged_in_chat():
    class _OnlyAnId(_Db):
        def query(self, *columns):
            return _Query([RECIPE_RUN])

    with usage_scope(request_type=LANE_CHAT, execution_id="chat:16bb619c"):
        found, refused = read_bare_refs(_OnlyAnId(), WS, [892])

    assert refused is None and found == {892: RECIPE_RUN.id}
