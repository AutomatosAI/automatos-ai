"""F274 (night 7b, the Needs-you check, friction 13) — every Needs-you row says what it is.

At 20:09 the owner gave #0192 (Ruth's split bag) to the Support Agent, and the
board's gate asked for approval. Needs you listed it as "board task requires
approval under 'always_ask' policy", with no number: an approval named its
ticket's number as ticket_number where every other row says number, and its title
was the gate's policy sentence. #0188's plan approval had no number either. At
19:44 Needs you said 6, and five of the six were one failed mission: #0176 and the
four steps it left open (#0176.9-.12), each with mission_id null, so they read as
five decisions. Now every row names its ticket as number, an approval of a ticket
is titled with the ticket, and a step whose mission ended carries that mission
(its id, and its card's number and title): the row opens the mission, where the
step is resumed or let go, and the widget lists the mission's steps as one. Each
step still counts, as F246's reconciliation counts it.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from tests import test_f246_needs_you_counts_what_waits as f246

# F246's Postgres and night (#0176 failed with steps .2 and .3 left open; #0031's
# plan waiting; an approval on #0170), as fixtures of this module too.
engine = f246.engine
night = f246.night

RUTH = "Reply to Ruth at Quayside Pantry - split bag in Thursday's delivery"


@pytest.fixture
def gate_asked(night, new_session, monkeypatch):
    """#0192 given to the Support Agent, and the board's gate asking for approval
    (the workspace's policy is always_ask), as at 20:09."""
    import services.board_approval as board_approval

    async def _no_bell(*args, **kwargs):
        return None

    monkeypatch.setattr(board_approval, "_dispatch_approval_pending", _no_bell)
    monkeypatch.setattr(board_approval, "_audit_governance", lambda *args, **kwargs: None)
    s = new_session()
    agent = f246._agent(s, night.ws, "Support Agent")
    card = f246._card(s, night.ws, 192, RUTH, "blocked", agent=agent)
    outcome = board_approval.evaluate_board_task_approval(
        s, workspace_id=night.ws, task_id=card, agent_id=agent, _policy_override="always_ask")
    s.commit()
    return NS(card=card, grant=outcome.grant.id)


def test_an_approval_of_a_ticket_is_named_by_its_number_and_title(night, gate_asked, new_session):
    from services.needs_you import needs_you

    rows = needs_you(new_session(), UUID(night.ws))["rows"]["approval"]
    row = next(r for r in rows if r["id"] == str(gate_asked.grant))

    # night: no number, and titled "board task requires approval under 'always_ask' policy"
    assert (row["ticket_id"], row["number"], row["title"], row["agent_name"]) == (
        gate_asked.card, "#0192", RUTH, "Support Agent")


def test_every_question_and_approval_names_its_ticket_as_every_row_does(night, gate_asked, new_session):
    from services.needs_you import needs_you

    rows = needs_you(new_session(), UUID(night.ws))["rows"]

    assert {(r["source"], r["number"]) for r in rows["approval"]} == {
        ("grant", "#0192"), ("grant", "#0170"), ("mission", "#0031")}         # #0188's plan read as none
    assert {r["number"] for r in rows["question"]} == {"#0176.1", "#0161"}
    assert not any("ticket_number" in r for r in rows["approval"] + rows["question"])


def test_the_steps_a_failed_mission_left_open_read_as_that_mission(night, new_session):
    from services.needs_you import needs_you

    out = needs_you(new_session(), UUID(night.ws))
    card = next(r for r in out["rows"]["failed"] if r["ticket_id"] == night.missions.card)
    steps = [r for r in out["rows"]["stuck"] if r["why"] == "mission_ended"]
    others = [r for r in out["rows"]["stuck"] if r["why"] != "mission_ended"]

    assert card["mission_id"] == night.missions.ended                       # the failed card opens its mission
    assert {r["number"] for r in steps} == {"#0176.2", "#0176.3"}
    assert {(r["mission_id"], r["mission_number"], r["mission_title"]) for r in steps} == {
        (night.missions.ended, "#0176", "Mission: prepare the cafés")}       # night: mission_id null on each
    assert all(r["mission_id"] is None and "mission_number" not in r for r in others)
    assert out["counts"]["stuck"] == 6                                      # each step still counts (F246)
