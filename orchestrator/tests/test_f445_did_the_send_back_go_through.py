"""P256-FIX-N17-6 (night 17, F445): "did that go through?" reads the card's send-back, not its status.

Night 17, chat cc4e3e29: "Send #1745 back…" raised the card, the owner clicked it, the redo ran and
returned to Review. "Did that one go through?" read #1745 with platform_get_task, saw Review with an
answer and said "No, it hasn't gone through … no send-back", offering to send it back again. The read
named nothing of the send-back (it lives in ``planning_data``, and the redo had blanked its times).

On the owner's own chat a read of the card now says it (``services.ticket_redo.send_back_seen``): the
send-backs since the brief was agreed (when, by whom, the notes), the newest send-back asked for on it
and its approval, and one line, ``said``. An agent's run, a session, a mission tool's read of a card and
a widget turn get nothing. The right answer ("It was sent back … with your note") survives the claim
check after a read that shows the send-back went through, and only then.

Each call runs through the executor's whole stack (the send-back test's desk: Auto, the dial on).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from tests import test_prd256_fix_send_back_waits_for_the_click as t3

shop = t3.shop
desk = t3.desk
GET, MOVE = "platform_get_task", t3.MOVE
NOTE = t3.NOTE
REDO = "Hello Larder & Loaf, the Guji first at £21 a kilo…"
RIGHT = ("It was sent back to Content Creator with your note, and the redo is back in Review.",
         "It's been sent back with your note.")
OWNER_ID, OWNER_CLERK = 7, "user_owner"


@pytest.fixture
def owner(desk):
    """users.id 7, whose Clerk id is the one the owner's chat carries (t3._owners_chat)."""
    from sqlalchemy import text

    desk.db.execute(text("INSERT INTO users (id, username, email, clerk_user_id) VALUES (:id, 'owner7', "
                         "'owner7@cafe.test', :clerk) ON CONFLICT (id) DO UPDATE SET clerk_user_id = :clerk"),
                    {"id": OWNER_ID, "clerk": OWNER_CLERK})
    desk.db.flush()
    return desk


def _ask(desk, params):
    asked = t3._call(desk, params, t3._owners_chat())
    assert t3._asked(asked), asked
    return asked["grant_id"]


def _grant(desk, grant_id):
    from core.models.approval_grants import ApprovalGrant

    return desk.db.get(ApprovalGrant, grant_id)


def _sent_back_by_the_click(desk):
    """A Review card sent back from chat, the owner's click (as 'user:7'), the send-back run: Assigned."""
    from core.services.approval_grants import grant_grant

    task = t3._card(desk, "review")
    params = {"task_id": t3._number(task), "status": "assigned", "note": NOTE}
    grant_grant(_grant(desk, _ask(desk, params)), granted_by=t3.CLICKER)
    desk.db.flush()
    ran = t3._call(desk, params, t3._owners_chat())
    assert ran["success"] is True and ran["sent_back"] is True, ran
    return task, params


def _redone(desk, task):
    """The redo answers and the card is back in Review, finished after the send-back."""
    task.status, task.result = "review", REDO
    task.completed_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    desk.db.flush()


def _read(desk, task, caller_context="owner"):
    context = t3._owners_chat() if caller_context == "owner" else caller_context
    answer = t3._call(desk, {"task_id": t3._number(task)}, context, action=GET)
    assert answer["success"] is True, answer
    return answer


# ── (a) the redo is back in Review ──────────────────────────────────────────────────

def test_a_redone_card_back_in_review_says_it_went_through(owner):
    task, _params = _sent_back_by_the_click(owner)
    _redone(owner, task)

    card = _read(owner, task)["task"]

    assert card["status"] == "review" and card["times_sent_back"] == 1
    said = card["sent_back"]
    assert said["notes"][-1]["note"] == NOTE
    assert said["by"] == "you"                            # signed 'user:user_owner', the owner's Clerk ref
    assert said["went_through"] is True and card["send_back_card"]["decision"] == "approved"
    assert "redo's" in said["said"] and " at " in said["said"] and "waiting" not in said["said"]


# ── (b) a second send-back still waits ──────────────────────────────────────────────

def test_a_new_send_back_still_waiting_never_reads_as_gone_through(owner):
    task, params = _sent_back_by_the_click(owner)
    _redone(owner, task)
    _ask(owner, params)                                    # the second send-back: its card waits

    card = _read(owner, task)["task"]

    assert card["send_back_card"]["decision"] == "waiting"
    said = card["sent_back"]
    assert said["went_through"] is False and "still waiting for your click" in said["said"]
    assert "redo's" not in said["said"] and "went through" not in said["said"]


# ── (c) declined and lapsed ─────────────────────────────────────────────────────────

def test_a_declined_and_a_lapsed_send_back_say_so(owner):
    from core.services.approval_grants import deny_grant

    declined = t3._card(owner, "review")
    grant = _grant(owner, _ask(owner, {"task_id": t3._number(declined), "status": "assigned", "note": NOTE}))
    deny_grant(grant, revoked_by=t3.CLICKER)
    lapsed = t3._card(owner, "review")
    grant = _grant(owner, _ask(owner, {"task_id": t3._number(lapsed), "status": "assigned", "note": NOTE}))
    grant.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    owner.db.flush()

    no, late = _read(owner, declined)["task"], _read(owner, lapsed)["task"]

    assert (no["send_back_card"]["decision"], late["send_back_card"]["decision"]) == ("declined", "lapsed")
    assert no["times_sent_back"] == late["times_sent_back"] == 0
    assert no["sent_back"]["went_through"] is late["sent_back"]["went_through"] is False
    assert "was declined; nothing changed" in no["sent_back"]["said"]
    assert "lapsed with no click; nothing changed" in late["sent_back"]["said"]


# ── (d) the redo waiting, running, failed ───────────────────────────────────────────

def test_a_running_a_waiting_and_a_failed_redo(owner):
    task, _params = _sent_back_by_the_click(owner)

    waiting = _read(owner, task)["task"]["sent_back"]
    task.status = "in_progress"
    owner.db.flush()
    running = _read(owner, task)["task"]["sent_back"]
    task.status, task.result = "failed", None
    owner.db.flush()
    failed = _read(owner, task)["task"]["sent_back"]

    assert waiting["went_through"] is running["went_through"] is failed["went_through"] is True
    assert "Its redo waits for Content Creator." in waiting["said"]
    assert "Its redo is running." in running["said"]
    assert "The redo failed and left no answer." in failed["said"]
    assert "The last draft is the one that was sent back." in failed["said"]           # F243's kept draft
    assert not any("redo's" in said["said"] for said in (waiting, running, failed))


# ── (e) only the owner's own turn ───────────────────────────────────────────────────

def test_only_the_owners_turn_sees_the_note(owner):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery.handlers_board_tasks import get_board_task
    from modules.tools.discovery.handlers_missions import get_mission
    from tests.test_f155_a_widget_turn_sees_tasks_and_playbooks_as_their_status import VISITOR_TASK

    task, _params = _sent_back_by_the_click(owner)
    _redone(owner, task)
    spoofed = {"task_id": t3._number(task), "_driving_user_id": OWNER_ID}

    assert _read(owner, task)["task"]["sent_back"]["notes"][-1]["note"] == NOTE      # the owner's chat
    for context in (None, {"board_task_id": 85}, {"field_context": {"field_id": 3}}):   # an agent, a session
        assert "sent_back" not in _read(owner, task, context)["task"], context
        spoof = t3._call(owner, spoofed, context, action=GET)                          # stripped, never trusted
        assert "sent_back" not in spoof["task"] and "times_sent_back" not in spoof["task"], context
    on_mission = asyncio.run(get_mission(owner.db, owner.ws, {"mission_id": t3._number(task),
                                                              "_driving_user_id": OWNER_ID}))
    assert "not a mission" in on_mission["note"] and "sent_back" not in on_mission["task"]
    with turn_surface(WIDGET, ("chat", "tasks:read"), None):
        visitor = asyncio.run(get_board_task(owner.db, owner.ws, {"task_id": task.id, "_driving_user_id": OWNER_ID}))
    assert set(visitor["task"]) == VISITOR_TASK


# ── (f) the right answer survives the claim check ───────────────────────────────────

def test_the_right_answer_survives_the_claim_check(owner):
    from consumers.chatbot.receipts import unbacked_sentences

    task, params = _sent_back_by_the_click(owner)
    _redone(owner, task)
    went = (GET, {"task_id": t3._number(task)}, _read(owner, task))
    _ask(owner, params)
    waits = (GET, {"task_id": t3._number(task)}, _read(owner, task))
    memory = ("platform_store_memory", {"content": "Lead with the price."}, {"success": True, "memory_id": "m-1"})
    other_card = {**went[2], "task": {**went[2]["task"], "number": "#9998"}}

    for sentence in RIGHT:
        assert unbacked_sentences(sentence, [went]) == [], sentence                     # what the read found
        assert unbacked_sentences(sentence, [waits]) != [], sentence                    # its newest ask waits
        assert unbacked_sentences(sentence, [went, memory]) != [], sentence             # the turn wrote
        assert unbacked_sentences(sentence, [went, (GET, {"task_id": "#9998"}, other_card)]) != [], sentence
    assert unbacked_sentences("I've sent it back with your note.", [went]) != []      # the writer's own work


# ── (g) the wiring ──────────────────────────────────────────────────────────────────

def test_the_tool_and_the_executor_wire_it():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.platform_executor import _DRIVER_AWARE_ACTIONS

    described = get_action_registry().get(GET).description
    assert GET in _DRIVER_AWARE_ACTIONS
    assert described.startswith("Get full details of a specific task by ID.")
    assert "sent back" in described and "send_back_card" in described and "platform_my_actions" in described
    assert "is showing its redo" not in described


# ── the line, DB-free ───────────────────────────────────────────────────────────────

SENT_AT = datetime(2026, 10, 15, 20, 40, tzinfo=timezone.utc)


def _iso(moment):
    return moment.isoformat()


def _record(at=SENT_AT, by="you"):
    return {"times": 1, "last_at": _iso(at), "by": by, "notes": [{"note": NOTE, "at": _iso(at)}]}


def _asked(decision, at):
    return {"asked_at": _iso(at), "decision": decision, "card": "Send #1745 back"}


def _task(status, completed_at=None, result=None, runs=()):
    return NS(status=status, completed_at=completed_at, result=result,
              planning_data={"previous_runs": list(runs)})


def test_send_back_said_when_the_redo_is_back():
    from services.ticket_redo import send_back_said

    finished = SENT_AT + timedelta(minutes=3)
    said = send_back_said(_task("review", finished), _record(), _asked("approved", SENT_AT - timedelta(minutes=1)))
    assert said == ("The send-back by you at 20:40 on 15 Oct went through. "
                    "The result shown is the redo's, finished at 20:43 on 15 Oct.")
    assert send_back_said(_task("done", finished), _record(), None).endswith("the redo's, finished at 20:43 on 15 Oct.")
    stale = send_back_said(_task("review", SENT_AT - timedelta(minutes=9)), _record(), None)
    assert "redo's" not in stale and stale.endswith("It is now in Review.")


@pytest.mark.parametrize("status, line", [
    ("assigned", "Its redo waits for Content Creator."), ("in_progress", "Its redo is running."),
    ("failed", "The redo failed and left no answer."), ("blocked", "It is now blocked."),
])
def test_send_back_said_by_the_cards_status(status, line):
    from services.ticket_redo import send_back_said

    said = send_back_said(_task(status), _record(), None, agent="Content Creator")
    assert said == f"The send-back by you at 20:40 on 15 Oct went through. {line}"


def test_a_failed_redo_names_the_draft_it_kept():
    from services.ticket_redo import SENT_BACK, send_back_said

    kept = _task("failed", runs=[{"why": SENT_BACK, "result": "the draft", "at": _iso(SENT_AT)}])
    assert send_back_said(kept, _record(), None).endswith(
        "The redo failed and left no answer. The last draft is the one that was sent back.")
    assert send_back_said(_task("assigned"), _record(), None).endswith("Its redo waits for its agent.")


def test_an_ask_after_the_last_send_back_decides_the_line():
    from services.ticket_redo import send_back_said, went_through

    later, earlier = SENT_AT + timedelta(minutes=10), SENT_AT - timedelta(minutes=1)
    task = _task("review", SENT_AT + timedelta(minutes=3))
    waits = send_back_said(task, _record(), _asked("waiting", later))
    assert waits == ("A send-back asked for at 20:50 on 15 Oct is still waiting for your click. "
                     "The card was last sent back at 20:40 on 15 Oct.")
    assert send_back_said(task, None, _asked("waiting", later)) == (
        "A send-back asked for at 20:50 on 15 Oct is still waiting for your click.")
    assert send_back_said(task, _record(), _asked("approved", later)) == (
        "You approved the send-back asked at 20:50 on 15 Oct, but no send-back is on the card after it.")
    assert send_back_said(task, None, _asked("declined", later)) == (
        "The send-back asked at 20:50 on 15 Oct was declined; nothing changed.")
    assert send_back_said(task, None, _asked("lapsed", later)) == (
        "The send-back asked at 20:50 on 15 Oct lapsed with no click; nothing changed.")
    assert send_back_said(task, None, None) == "No send-back is on the card."
    assert "went through" in send_back_said(task, _record(), _asked("declined", earlier))   # an older ask
    assert (went_through(_record(), None), went_through(_record(), _asked("approved", earlier)),
            went_through(_record(), _asked("waiting", later)), went_through(None, None)) == (True, True, False, False)
    for said in (waits, send_back_said(task, _record(), _asked("approved", later))):
        assert "went through" not in said and "redo's" not in said


@pytest.mark.parametrize("by, who", [
    ("user:7", "you"), ("user:user_owner", "you"), ("user:12", "a member of your team"),
    ("user:unknown", "a member of your team"), ("platform_tool", "an agent"), ("agent:4", "an agent"), (None, "an agent"),
])
def test_send_back_record_says_who_sent_it_back(by, who):
    from services.ticket_redo import SENT_BACK, send_back_record

    task = NS(planning_data={
        "previous_runs": [{"why": SENT_BACK, "result": "d", "by": by, "at": _iso(SENT_AT)}],
        "owner_corrections": [{"note": f"note {n}", "by": by, "at": _iso(SENT_AT)} for n in range(5)]})

    record = send_back_record(task, ["user:7", "user:user_owner"])

    assert record["by"] == who and record["times"] == 1 and record["last_at"] == _iso(SENT_AT)
    assert [n["note"] for n in record["notes"]] == ["note 2", "note 3", "note 4"]       # the newest three


def test_send_back_record_counts_only_since_the_agreed_brief():
    from services.ticket_redo import RUN_NOW, SENT_BACK, send_back_record

    agreed = SENT_AT + timedelta(hours=1)
    before = {"why": SENT_BACK, "by": "user:7", "at": _iso(SENT_AT)}
    task = NS(planning_data={"previous_runs": [before, {"why": RUN_NOW, "at": _iso(agreed + timedelta(minutes=5))}],
                             "previous_briefs": [{"description": "old", "at": _iso(agreed)}]})
    assert send_back_record(task, ["user:7"]) is None
    after = {"why": SENT_BACK, "by": "platform_tool", "at": _iso(agreed + timedelta(minutes=9))}
    task.planning_data = {**task.planning_data, "previous_runs": [before, after]}
    record = send_back_record(task, ["user:7"])
    assert (record["times"], record["by"], record["last_at"]) == (1, "an agent", after["at"])
