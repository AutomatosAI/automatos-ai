"""PRD-255 US-014 (FR-11): the Brand designer's proposal card, and the save of what the owner approved.

``platform_propose_brand_kit`` (session: ``propose_brand_kit``) checks a proposal with the
kit's own validation, draws the Brand Board from it into the ticket's folder WITHOUT
saving it, and files a question card on the caller's own ticket that carries the
proposal and the board (options Approve / Revise). ``platform_save_approved_brand_kit``
(session: ``save_approved_brand_kit``) saves exactly that proposal, through
``platform_update_brand_kit``'s handler, and only when the owner answered Approve.

Boundaries faked: the session (a query evaluator over rows), the workspace worker,
the child-process board render, and the question filing (``raise_session_ask`` /
``stage_question``). The kit's validation and writer are the real ones.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import operator
import os
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, List
from uuid import UUID

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.models.approval_grants import KIND_QUESTION, ApprovalGrant  # noqa: E402
from modules.documents import brand_board_render, brand_fonts  # noqa: E402
from modules.tools.discovery import brand_proposal_card as card  # noqa: E402
from modules.tools.discovery import handlers_asks  # noqa: E402
from modules.tools.discovery import handlers_brand_proposals as bp  # noqa: E402
from modules.tools.discovery.session_ticket import (  # noqa: E402
    BOARD_CARD_PARAM,
    SESSION_TICKET_PARAM,
    session_ticket_params,
)
from modules.tools.execution import session_document_folder  # noqa: E402
from services import cli_host_service  # noqa: E402

TICKET = 2201
OTHER_TICKET = 2202
AGENT_ID = 311
WS = UUID("5e0c4b2a-7d1f-4a3b-8c2d-1e2f3a4b5c61")
OTHER_WS = UUID("6f1d5c3b-8e2a-4b4c-9d3e-2f3a4b5c6d72")
BOARD_PNG = b"\x89PNG\r\n\x1a\n the proposed board"
STORED = {"brand_kit": {"name": "Hollow Leaf", "primary_color": "#1F3A2E"}}
# accent_use "sparing" is the default already: the card must not list it as a change.
PROPOSAL = {"accent_color": "#C8553D", "accent_use": "sparing"}


# ── the fakes ──────────────────────────────────────────────────────────────

class _Query:
    """A query over rows: each ``Model.column == value`` clause is applied for real; a lock is recorded."""

    def __init__(self, rows: List[Any], model: str, locked: List[str]) -> None:
        self.rows, self.model, self.locked = rows, model, locked

    def filter(self, *clauses: Any) -> "_Query":
        rows = self.rows
        for clause in clauses:
            assert clause.operator is operator.eq, clause
            key, value = clause.left.key, clause.right.value
            rows = [r for r in rows if getattr(r, key, None) == value]
        return _Query(rows, self.model, self.locked)

    def with_for_update(self) -> "_Query":
        self.locked.append(self.model)
        return self

    def first(self) -> Any:
        return self.rows[0] if self.rows else None

    def all(self) -> List[Any]:
        return list(self.rows)


class _Db:
    def __init__(self, grants: List[Any] = (), task_status: str = "in_progress") -> None:
        self.workspace = NS(id=WS, settings=copy.deepcopy(STORED))
        self.rows = {
            "Workspace": [self.workspace, NS(id=OTHER_WS, settings={})],
            "Agent": [NS(id=AGENT_ID, workspace_id=WS, name="Brand Designer")],
            "BoardTask": [NS(id=TICKET, workspace_id=WS, status=task_status, runtime_ref={}),
                          NS(id=OTHER_TICKET, workspace_id=OTHER_WS, status="in_progress", runtime_ref={})],
            "ApprovalGrant": list(grants),
        }
        self.commits = 0
        self.rollbacks = 0
        self.locked: List[str] = []

    def query(self, model: Any) -> _Query:
        return _Query(self.rows.get(model.__name__, []), model.__name__, self.locked)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _Worker:
    written: Dict[Any, bytes] = {}

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id

    async def write_binary(self, path: str, pieces: Any) -> Dict[str, Any]:
        self.written[(self.workspace_id, path)] = b"".join([piece async for piece in pieces])
        return {"success": True, "path": path}


@pytest.fixture
def filed(monkeypatch):
    """The board render, the worker and both ways of filing a card faked; what each was given."""
    seen: Dict[str, List[Any]] = {"board": [], "session": [], "board_card": []}
    _Worker.written = {}
    monkeypatch.setattr(session_document_folder, "WorkspaceClient", _Worker)
    monkeypatch.setattr(brand_fonts, "brand_kit_for_media_render", lambda kit: {**kit, "inlined": True})

    def render(kit, fmt, timeout_s=None):
        seen["board"].append({"kit": kit, "fmt": fmt})
        return BOARD_PNG

    async def raise_session_ask(db, **kwargs):
        seen["session"].append(kwargs)
        return {"success": True, "result": {"ask_id": 901, "message": "asked"}}

    async def stage_question(db, workspace_id, **kwargs):
        seen["board_card"].append({"workspace_id": workspace_id, **kwargs})
        return {"success": True, "ask_id": 902, "parked": True}

    monkeypatch.setattr(brand_board_render, "render_board_isolated", render)
    monkeypatch.setattr(cli_host_service, "raise_session_ask", raise_session_ask)
    monkeypatch.setattr(handlers_asks, "stage_question", stage_question)
    return seen


def _session(**params: Any) -> Dict[str, Any]:
    return session_ticket_params("platform_propose_brand_kit", {**params, "_agent_id": AGENT_ID},
                                 {"session_task_id": TICKET})


def _propose(params: Dict[str, Any], db: _Db = None) -> Dict[str, Any]:
    return asyncio.run(bp.propose_brand_kit(db or _Db(), WS, params))


def _grant(grant_id: int, status: str = "granted", answer: str = "Approve", ticket: int = TICKET,
           workspace_id: UUID = WS, proposal: Dict[str, Any] = None, base: str = None,
           answered_by: str = "user:7") -> ApprovalGrant:
    marker = {"proposal": dict(proposal or PROPOSAL), "board_path": f"sessions/{ticket}/board.png",
              "base": base if base is not None else card.fingerprint(STORED["brand_kit"])}
    return ApprovalGrant(id=grant_id, workspace_id=workspace_id, subject_type="board_task", subject_id=str(ticket),
                         kind=KIND_QUESTION, status=status, answer_text=answer if status == "granted" else None,
                         answered_by=answered_by if status == "granted" else None,
                         details={"cli_ask": {"task_id": ticket}, card.PROPOSAL_MARKER: marker})


def _save(db: _Db, ticket_param: str = SESSION_TICKET_PARAM, ticket: int = TICKET) -> Dict[str, Any]:
    return asyncio.run(bp.save_approved_brand_kit(db, WS, {ticket_param: ticket}))


# ── the card carries the proposal and the board ─────────────────────────────

def test_a_sessions_proposal_card_carries_the_proposal_and_the_board_and_saves_nothing(filed):
    db = _Db()
    answer = _propose(_session(brand_kit=PROPOSAL, why="The logo's terracotta, used for highlights only."), db)

    assert answer["success"] is True and answer["saved"] is False and answer["ask_id"] == 901
    path = answer["board_path"]
    assert path == f"sessions/{TICKET}/{card.board_name(PROPOSAL)}"
    assert _Worker.written == {(str(WS), path): BOARD_PNG}
    assert filed["board"][0]["fmt"] == "png" and filed["board"][0]["kit"]["accent_color"] == "#C8553D"
    asked = filed["session"][0]
    assert asked["task_id"] == TICKET and asked["workspace_id"] == WS and asked["agent_id"] == AGENT_ID
    assert asked["options"] == ["Approve", "Revise"]
    assert asked["details"][card.PROPOSAL_MARKER] == {
        "proposal": PROPOSAL, "board_path": path, "base": card.fingerprint(STORED["brand_kit"])}
    assert "Brand Designer" in asked["question"] and "terracotta" in asked["question"]
    assert "`accent_color`" in asked["question"] and "`#C8553D`" in asked["question"]
    assert "accent_use" not in asked["question"]           # unchanged: not listed
    assert path in asked["question"] and "**Approve** saves it" in asked["question"]
    assert answer["changes"] == [{"field": "accent_color", "old": answer["changes"][0]["old"], "new": "`#C8553D`"}]
    assert db.workspace.settings == STORED and db.commits == 0


def test_a_board_runs_card_is_parked_behind_the_question(filed):
    params = session_ticket_params("platform_propose_brand_kit", {"brand_kit": PROPOSAL, "_agent_id": AGENT_ID},
                                   {"board_task_id": TICKET})
    db = _Db()
    answer = _propose(params, db)

    assert answer["success"] is True and answer["ask_id"] == 902 and filed["session"] == []
    asked = filed["board_card"][0]
    assert asked["subject_type"] == "board_task" and asked["subject_id"] == str(TICKET)
    assert asked["park"] is db.rows["BoardTask"][0] and asked["options"] == ["Approve", "Revise"]
    assert asked["details"][card.PROPOSAL_MARKER]["proposal"] == PROPOSAL
    assert db.workspace.settings == STORED


def test_another_workspaces_card_and_a_finished_card_are_refused(filed):
    other = session_ticket_params("platform_propose_brand_kit", {"brand_kit": PROPOSAL},
                                  {"board_task_id": OTHER_TICKET})
    assert _propose(other)["error"] == f"ticket {OTHER_TICKET} is not in this workspace"
    done = session_ticket_params("platform_propose_brand_kit", {"brand_kit": PROPOSAL}, {"board_task_id": TICKET})
    assert "already done" in _propose(done, _Db(task_status="done"))["error"]
    assert filed["board_card"] == [] and filed["board"] == [] and _Worker.written == {}


@pytest.mark.parametrize("proposal, words", [
    ({"accent_color": "orange"}, "must be a hex color"),
    ({"logo_path": "brand/evil.png"}, "The kit has no field logo_path"),
    ({"name": "Hollow Leaf"}, "changes nothing"),
    ("less orange", "brand_kit is an object of kit fields"),
    ({}, "brand_kit is an object of kit fields"),
])
def test_a_proposal_the_kit_refuses_files_no_card_and_draws_nothing(filed, proposal, words):
    db = _Db()
    answer = _propose(_session(brand_kit=proposal), db)

    assert answer["success"] is False and words in answer["error"]
    assert filed["board"] == [] and filed["session"] == [] and _Worker.written == {}
    assert db.workspace.settings == STORED


def test_without_a_ticket_nothing_is_drawn_or_asked(filed):
    spoofed = session_ticket_params("platform_propose_brand_kit",
                                    {"brand_kit": PROPOSAL, SESSION_TICKET_PARAM: TICKET, BOARD_CARD_PARAM: TICKET},
                                    {"user_id": "owner"})
    assert SESSION_TICKET_PARAM not in spoofed and BOARD_CARD_PARAM not in spoofed
    answer = _propose(spoofed)
    assert answer["success"] is False and "has none" in answer["error"]
    assert filed["board"] == [] and filed["session"] == [] and filed["board_card"] == []


def test_the_server_gives_the_ticket_only_to_the_actions_that_work_it():
    board = {"board_task_id": 77}
    assert session_ticket_params("platform_save_approved_brand_kit", {}, board) == {BOARD_CARD_PARAM: 77}
    assert session_ticket_params("platform_render_preview", {}, board) == {}       # still a session's only
    assert session_ticket_params("platform_list_templates", {BOARD_CARD_PARAM: 5}, board) == {}
    both = {"board_task_id": 77, "session_task_id": TICKET}
    assert session_ticket_params("platform_propose_brand_kit", {}, both) == {SESSION_TICKET_PARAM: TICKET}


def test_a_failed_board_draw_files_no_card(filed, monkeypatch):
    from modules.documents.thumbnails.render import ThumbnailError

    def broken(kit, fmt, timeout_s=None):
        raise ThumbnailError("timed out")

    monkeypatch.setattr(brand_board_render, "render_board_isolated", broken)
    answer = _propose(_session(brand_kit=PROPOSAL))
    assert answer["success"] is False and "could not be drawn" in answer["error"] and filed["session"] == []


def test_every_change_is_shown_whole_and_a_card_too_long_to_show_whole_is_refused(filed):
    tagline = "Small-batch coffee roasted by the harbour, for the cafés and kitchens that care where it came from"
    answer = _propose(_session(brand_kit={"tagline": tagline}))
    assert answer["success"] is True and f"`{tagline}`" in filed["session"][0]["question"]

    filed["session"].clear()
    phrases = [f"never say this phrase, number {i:02d}, in any of our copy or on any of our pages" for i in range(50)]
    long_voice = {"voice": {"banned_phrases": phrases}}      # valid (50 at most), too long to show whole
    refused = _propose(_session(brand_kit=long_voice))
    assert refused["success"] is False and "propose fewer fields per card" in refused["error"]
    assert filed["session"] == [] and len(filed["board"]) == 1      # only the first card's board was drawn


def test_a_card_still_waiting_for_the_owner_blocks_a_second_one(filed):
    db = _Db([_grant(30, status="pending")])
    answer = _propose(_session(brand_kit=PROPOSAL), db)

    assert answer["success"] is False and "still waiting" in answer["error"] and answer["ask_id"] == 30
    assert filed["board"] == [] and filed["session"] == []


def test_the_session_ask_keeps_its_own_marker_beside_the_cards(monkeypatch):
    seen = {}

    async def stage_question(db, workspace_id, **kwargs):
        seen.update(kwargs)
        return {"success": True, "ask_id": 903}

    monkeypatch.setattr(handlers_asks, "stage_question", stage_question)
    db = _Db()
    filed = asyncio.run(cli_host_service.raise_session_ask(
        db, task_id=TICKET, workspace_id=WS, agent_id=AGENT_ID, agent_name="Brand Designer",
        question="q", options=["Approve", "Revise"], details={card.PROPOSAL_MARKER: {"proposal": PROPOSAL}}))

    assert filed["success"] is True and seen["park"] is None
    assert seen["details"] == {card.PROPOSAL_MARKER: {"proposal": PROPOSAL},
                               cli_host_service.SESSION_ASK_MARKER: {"task_id": TICKET}}


# ── the kit changes only after the owner's Approve ──────────────────────────

@pytest.mark.parametrize("ticket_param", [SESSION_TICKET_PARAM, BOARD_CARD_PARAM])
def test_an_approved_proposal_is_saved_exactly_once(ticket_param):
    grant = _grant(41)
    db = _Db([grant])
    saved = _save(db, ticket_param)

    assert saved["success"] is True and saved["ask_id"] == 41 and saved["changed"] == sorted(PROPOSAL)
    kit = db.workspace.settings["brand_kit"]
    assert kit["accent_color"] == "#C8553D" and kit["name"] == "Hollow Leaf" and db.commits == 1
    assert grant.details[card.PROPOSAL_MARKER]["saved_at"]
    assert grant.details["cli_ask"] == {"task_id": TICKET}
    assert "ApprovalGrant" in db.locked and "Workspace" in db.locked     # held from the check to the commit

    again = _save(db, ticket_param)
    assert again["success"] is False and "already saved" in again["error"] and db.commits == 1


@pytest.mark.parametrize("grant, words", [
    (_grant(42, status="pending"), "has not answered"),
    (_grant(43, answer="Less orange, please"), "'Less orange, please'"),
    (_grant(44, answer="Approve, but warmer"), "sent your proposal back"),
    (_grant(45, answer="Revise"), "sent your proposal back"),
    (_grant(46, status="denied"), "was denied, not approved"),
    (_grant(47, base="a kit drawn over before"), "changed after you proposed"),
    (_grant(48, answered_by=None), "was granted, not approved"),
])
def test_without_a_plain_approve_on_the_unchanged_kit_nothing_is_saved(grant, words):
    db = _Db([grant])
    saved = _save(db)

    assert saved["success"] is False and words in saved["error"]
    assert db.workspace.settings == STORED and db.commits == 0
    assert "saved_at" not in grant.details[card.PROPOSAL_MARKER]


@pytest.mark.parametrize("answer", ["Approve", "approve.", " APPROVE! "])
def test_approve_is_read_in_any_case(answer):
    assert card.is_approve(answer)


def test_the_newest_card_on_the_ticket_decides():
    db = _Db([_grant(50, answer="Approve"), _grant(51, answer="warmer please")])
    saved = _save(db)
    assert saved["success"] is False and "warmer please" in saved["error"] and db.commits == 0


def test_another_workspaces_or_tickets_approval_saves_nothing():
    db = _Db([_grant(60, workspace_id=OTHER_WS), _grant(61, ticket=OTHER_TICKET)])
    saved = _save(db)
    assert saved["success"] is False and "no proposal card" in saved["error"] and db.workspace.settings == STORED


def test_a_question_that_is_not_a_proposal_saves_nothing():
    plain = ApprovalGrant(id=70, workspace_id=WS, subject_type="board_task", subject_id=str(TICKET),
                          kind=KIND_QUESTION, status="granted", answer_text="Approve", answered_by="user:7",
                          details={"cli_ask": {"task_id": TICKET}})
    assert _save(_Db([plain]))["error"].startswith("Your ticket has no proposal card")


def test_a_save_outside_a_ticket_is_refused():
    db = _Db([_grant(80)])
    saved = asyncio.run(bp.save_approved_brand_kit(db, WS, {}))
    assert saved["success"] is False and "has none" in saved["error"] and db.workspace.settings == STORED


def test_a_stored_proposal_the_kit_now_refuses_is_rolled_back():
    db = _Db([_grant(90, proposal={"accent_color": "terracotta"})])
    saved = _save(db)
    assert saved["success"] is False and "must be a hex color" in saved["error"]
    assert db.rollbacks == 1 and db.commits == 0 and db.workspace.settings == STORED


# ── wiring: the 3-file pattern, the gate, the session tools ────────────────

def _gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_hierarchy_gate.py"
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def test_the_two_actions_are_registered_writes_routed_and_on_the_gates_allow_list():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    gate = _gate()
    for name, handler in (("platform_propose_brand_kit", bp.propose_brand_kit),
                          ("platform_save_approved_brand_kit", bp.save_approved_brand_kit)):
        action = get_action_registry().get(name)
        assert action is not None and action.permission_level == "write" and not action.admin_only, name
        assert PLATFORM_HANDLERS[name] is handler
        assert name in gate.ALLOW_LIST and gate.collect_registrations()[name].permission_level == "write"
    assert get_action_registry().get("platform_save_approved_brand_kit").parameters["properties"] == {}


def test_the_session_tools_work_their_own_ticket_and_the_save_takes_no_fields():
    from services import session_tool_groups as groups
    from services import session_tools as st

    documents = next(g for g in groups.SESSION_TOOL_GROUPS if g.id == "documents")
    ctx = st.SessionContext(task_id=TICKET, agent_id=AGENT_ID, agent_name="Brand Designer", workspace_id=str(WS))
    for name, action, reads in (("get_brand_kit", "platform_get_brand_kit", True),
                                ("propose_brand_kit", "platform_propose_brand_kit", False),
                                ("save_approved_brand_kit", "platform_save_approved_brand_kit", False)):
        tool = st.get_tool(name)
        assert tool.action == action and tool.reads_only is reads and name in documents.tools
    for name in ("propose_brand_kit", "save_approved_brand_kit"):
        assert name in st.CARD_ATTRIBUTED_TOOLS
    sent = {"brand_kit": PROPOSAL, "why": "w", SESSION_TICKET_PARAM: 9, "accent_color": "#000000"}
    assert st.resolve_parameters(st.get_tool("propose_brand_kit"), sent, ctx) == {"brand_kit": PROPOSAL, "why": "w"}
    assert st.resolve_parameters(st.get_tool("save_approved_brand_kit"), {"accent_color": "#000000"}, ctx) == {}
    with pytest.raises(st.SessionToolRefused):
        st.resolve_parameters(st.get_tool("propose_brand_kit"), {"why": "w"}, ctx)
