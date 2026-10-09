"""PRD-256 FX-014 (E5): a named agent gets its ticket without the classifier, and a shared name is asked about.

Night 12: ``handoffs.the_lane`` turned a named agent into a ticket only on a DELEGATE verdict, which the
rubric no longer offers, so "Get OPS to…" and "Ask RESEARCHER…" stayed with Auto; "Give #1057 to
CHRISTMAS BOX" was no hand-on at all (only a card numbered #0xxx was read as one); and with two
agents called OPS, "Get OPS to…" could only fail. Here, through the dispatch (``lane_for``):

- a name shared by several active agents is an ASSIGN turn that asks which, listing each as FX-012's
  refusal does, and never files a copy; the dispatch keeps that directive over the ticket's;
- the answer, "267, the operations one", is OPS 267's ticket, filed by its agent_id;
- a bare number is an answer to any numbered question: it hands nothing over;
- P256-FIX-RVW-33: the answer after "Give #1057 to OPS" hands #1057 on to the agent picked, never a copy.

No model is called and no database is opened: the roster is the test's.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from api.chat_dispatch import lane_for, said_before
from consumers.chatbot.addressed_agents import addressed_by_name, id_reply
from consumers.chatbot.auto import ASSIGN_TOOL_HINTS, Action, AutoBrain, Complexity, ComplexityAssessment
from consumers.chatbot.handoffs import the_lane

AUTO = 1
OPS, SHOP_OPS = 267, 284
ROSTER = [
    NS(id=AUTO, name="Auto", job_title="Workspace orchestrator", is_system_agent=True),
    NS(id=2, name="NEWSROOM", job_title="Newsletter editor"),
    NS(id=30, name="TRACKER", job_title="Order tracker"),
    NS(id=57, name="RESEARCHER", job_title="Market researcher"),
    NS(id=OPS, name="OPS", job_title="Operations Manager"),
    NS(id=SHOP_OPS, name="OPS", job_title="Shop floor assistant"),
    NS(id=412, name="CHRISTMAS BOX", job_title="Seasonal buyer"),
]
CANDIDATES = "267 · OPS · Operations Manager; 284 · OPS · Shop floor assistant"


class _Brain:
    _match_roster_agent = AutoBrain._match_roster_agent

    def _active_agents(self):
        return ROSTER


def _turn(said, action=Action.RESPOND):
    """The lane the dispatch runs for ``said`` when the tiers said ``action`` (night 12: RESPOND)."""
    tiers = ComplexityAssessment(complexity=Complexity.MOLECULE, action=action, reasoning="tiers")
    return lane_for(AUTO, the_lane(_Brain(), said, tiers), said)


# ── a shared name: Auto asks which, and files nothing ─────────────────────────

@pytest.mark.parametrize("action", [Action.RESPOND, Action.ASSIGN, Action.MISSION])
def test_get_ops_to_with_two_ops_asks_which_listing_both(action):
    lane = _turn("Get OPS to reorder the green stock under 50 kg.", action)
    verdict = lane.assessment

    assert lane.agent_id == AUTO and verdict.action == Action.ASSIGN and not lane.suggest_mission
    assert (verdict.target_agent_id, verdict.target_agent_name) == (None, "OPS")
    assert set(ASSIGN_TOOL_HINTS) <= set(verdict.tool_hints)
    directive = verdict.context_directive
    assert "ask which agent" in directive and CANDIDATES in directive
    assert "do NOT create a new agent" in directive and "do NOT file anything yet" in directive
    assert "platform_create_task and that agent's agent_id" in directive
    assert "confirm the agent first" not in directive       # the ticket's own directive did not replace it


def test_a_card_given_to_a_shared_name_asks_which_and_never_makes_a_card():
    directive = _turn("Give #1057 to OPS").assessment.context_directive

    assert CANDIDATES in directive
    assert 'platform_assign_task with task_id "#1057"' in directive and "platform_create_task" not in directive


# ── the answer: the id picks the agent ────────────────────────────────────────

@pytest.mark.parametrize("said, agent_id", [
    ("267, the operations one", OPS),
    ("284 — the shop floor one, please", SHOP_OPS),
    ("267 (ops)", OPS),
    ("agent 284", SHOP_OPS),
])
def test_the_id_given_after_the_clash_is_that_agents_ticket_by_its_id(said, agent_id):
    verdict = _turn(said).assessment

    assert (verdict.action, verdict.target_agent_id, verdict.target_agent_name) == (Action.ASSIGN, agent_id, "OPS")
    assert f'assigned_agent_name="OPS" and agent_id={agent_id}' in verdict.context_directive


@pytest.mark.parametrize("said", [
    "2",                                    # an answer to "option 1 or 2?"
    "30, the big bags",                     # agent 30 is the TRACKER, not bags
    "267, the operations one, and also please send the Kerbside invoice tonight",   # too long for an answer
    "999, the operations one",              # nobody has that id
    "1, the workspace one",                 # Auto never takes a ticket
])
def test_a_number_that_does_not_pick_an_agent_hands_nothing_over(said):
    assert _turn(said).assessment.action == Action.RESPOND
    assert id_reply(said, [agent for agent in ROSTER if agent.id != AUTO]) is None


# ── P256-FIX-RVW-33: the card clashed on rides to the answer ───────────────────

def _two_turns(first, answer):
    """Night 12's two turns through the dispatch: ``first`` (which asks which), then ``answer``, with the
    history the chat route reads (the answer is saved before the lane is picked)."""
    asked = _turn(first).assessment
    history = [
        {"role": "user", "parts": [{"type": "text", "text": first}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "Which OPS? 267 · OPS · Operations Manager …"}]},
        {"role": "user", "parts": [{"type": "text", "text": answer}, {"type": "text", "text": "[page: board]"}]},
    ]
    tiers = ComplexityAssessment(complexity=Complexity.MOLECULE, action=Action.RESPOND, reasoning="tiers")
    return asked, lane_for(AUTO, the_lane(_Brain(), answer, tiers), answer, said_before(history)).assessment


@pytest.mark.parametrize("answer, agent_id", [("267, the operations one", OPS), ("agent 284", SHOP_OPS)])
def test_the_id_given_after_a_cards_clash_hands_that_card_on(answer, agent_id):
    asked, verdict = _two_turns("Give #1057 to OPS", answer)

    assert 'platform_assign_task with task_id "#1057"' in asked.context_directive
    assert (verdict.action, verdict.target_agent_id, verdict.target_agent_name) == (Action.ASSIGN, agent_id, "OPS")
    directive = verdict.context_directive
    assert f'platform_assign_task with task_id "#1057" and agent_name "OPS" and agent_id {agent_id}' in directive
    assert "platform_create_task" not in directive and "do NOT create a new card" in directive


def test_a_deferred_card_stays_deferred_after_the_answer():
    _asked, verdict = _two_turns("Give #1057 to OPS, no rush", "267, the operations one")

    assert 'task_id "#1057"' in verdict.context_directive and "the user asked to defer it" in verdict.context_directive


@pytest.mark.parametrize("first", [
    "Get OPS to reorder the green stock under 50 kg.",     # a ticket clash: no card to carry
    "Give #1057 to CHRISTMAS BOX",                          # the card went to another name
    "Approve #1057",                                         # a card acted on, handed to nobody
])
def test_the_id_given_after_no_cards_clash_still_files_the_ticket(first):
    _asked, verdict = _two_turns(first, "267, the operations one")

    assert (verdict.action, verdict.target_agent_id) == (Action.ASSIGN, OPS)
    assert 'platform_create_task' in verdict.context_directive and "#1057" not in verdict.context_directive
    assert 'assigned_agent_name="OPS" and agent_id=267' in verdict.context_directive


@pytest.mark.parametrize("history, said", [
    ([], ""),
    ([{"role": "user", "parts": [{"type": "text", "text": "267, the operations one"}]}], ""),
    ([{"role": "user", "parts": [{"type": "file", "url": "x"}]}, {"role": "user", "parts": []}], ""),
    ([{"role": "user", "parts": [{"type": "text", "text": "Give #1057 to OPS"}]},
      {"role": "assistant", "parts": [{"type": "text", "text": "Which one?"}]},
      {"role": "user", "parts": [{"type": "text", "text": "267"}]}], "Give #1057 to OPS"),
])
def test_said_before_is_the_owners_message_before_the_latest(history, said):
    assert said_before(history) == said


# ── names and cards ───────────────────────────────────────────────────────────

def test_a_card_past_0999_is_handed_on_by_its_agent_id():
    verdict = _turn("Give #1057 to CHRISTMAS BOX").assessment

    assert (verdict.action, verdict.target_agent_id) == (Action.ASSIGN, 412)
    assert 'task_id "#1057" and agent_name "CHRISTMAS BOX" and agent_id 412' in verdict.context_directive


def test_a_named_agent_is_its_ticket_by_its_agent_id_whatever_the_tiers_said():
    verdict = _turn("Ask RESEARCHER to find three cafés in Leith.", Action.MISSION).assessment

    assert (verdict.action, verdict.target_agent_id) == (Action.ASSIGN, 57)
    assert 'assigned_agent_name="RESEARCHER" and agent_id=57' in verdict.context_directive


def test_an_orders_number_is_no_card():
    verdict = _turn("Ask RESEARCHER to chase order #1043").assessment

    assert (verdict.action, verdict.target_agent_id) == (Action.ASSIGN, 57)
    assert "hand the card on" not in verdict.context_directive


@pytest.mark.parametrize("said", [
    "Approve #1057, tell RESEARCHER thanks",          # the owner's act on a card stays Auto's
    "What did RESEARCHER say about the Leith cafés?",
    "Delete MARKET-MANAGER and OPS",
])
def test_a_name_only_mentioned_or_a_card_acted_on_stays_autos(said):
    lane = _turn(said)

    assert (lane.agent_id, lane.assessment.action, lane.assessment.target_agent_id) == (AUTO, Action.RESPOND, None)


def test_work_for_several_agents_stays_the_tiers_lane_whatever_they_said():
    said = "Have RESEARCHER and CHRISTMAS BOX plan the Christmas launch together"
    lane = _turn(said, Action.MISSION)

    assert lane.assessment.action == Action.MISSION and lane.suggest_mission is True
    # P256-FIX-RVW-12: the tiers said respond, and nobody gets the whole job (CHRISTMAS BOX was dropped)
    lane = _turn(said)
    assert (lane.agent_id, lane.assessment.action, lane.assessment.target_agent_id) == (AUTO, Action.RESPOND, None)


@pytest.mark.parametrize("said, name", [
    ("Have support tickets been answered today?", "Support"),   # the name modifies a noun with a verb of its own
    ("Get sales figures for Q3", "Sales"),                      # a fetch, not "get Sales to …"
    ("Get OPS's stock report", "OPS"),                          # a possessive
    ("Have Support answered the club emails?", "Support"),      # a past participle: a question about it
])
def test_a_name_with_no_task_after_it_hands_nothing_over(said, name):
    roster = [*ROSTER, NS(id=610, name="Support", job_title="Customer support"),
              NS(id=611, name="Sales", job_title="Wholesale sales lead")]
    brain = _Brain()
    brain._active_agents = lambda: roster
    tiers = ComplexityAssessment(complexity=Complexity.MOLECULE, action=Action.RESPOND, reasoning="tiers")

    assert addressed_by_name(said, name) is False
    assert the_lane(brain, said, tiers) is tiers


@pytest.mark.parametrize("said, name", [
    ("Have Support check the refund queue.", "Support"),
    ("Ask Sales: how many club boxes sold this week?", "Sales"),
    ("Ask RESEARCHER what the going rate is.", "RESEARCHER"),
    ("Get OPS to reorder the green stock.", "OPS"),
])
def test_a_name_with_a_task_after_it_is_handed_the_work(said, name):
    assert addressed_by_name(said, name) is True


# ── P256-FIX-RVW-16: a hand-off the owner forbids or asks about hands nothing over ──

@pytest.mark.parametrize("said, name", [
    ("Don't ask OPS to check anything", "OPS"),
    ("Never have WRITER touch the About page", "WRITER"),
    ("I told you not to let OPS post", "OPS"),
    ("No need to ask RESEARCHER to price it.", "RESEARCHER"),
    ("Did you ask OPS to check the stock?", "OPS"),
    ("Why didn't you get OPS to check?", "OPS"),
    ("Should I ask OPS to check?", "OPS"),
    ("What did you ask RESEARCHER to find?", "RESEARCHER"),
    ("Have sales risen this week?", "Sales"),
    ("Have Support caught up?", "Support"),
    ("Have sales come in?", "Sales"),
    ("Have OPS finished?", "OPS"),                                     # P256-FIX-RVW-29: a participle after the name
    ("Ask OPS to check the stock, or just do it yourself.", "OPS"),     # keeps_it_with_auto
])
def test_a_hand_off_forbidden_or_asked_about_hands_nothing_over(said, name):
    assert addressed_by_name(said, name) is False


# ── P256-FIX-RVW-29: "Have <name> <bare verb> …?" is a request, not a present perfect ──

ONE_OPS = [*(agent for agent in ROSTER if agent.id != SHOP_OPS), NS(id=58, name="WRITER", job_title="Copywriter")]


class _OneOpsBrain(_Brain):
    def _active_agents(self):
        return ONE_OPS


@pytest.mark.parametrize("said, name, agent_id", [
    ("Have OPS check the stock?", "OPS", OPS),
    ("Have WRITER draft the About page?", "WRITER", 58),
    ("Have RESEARCHER find the Leith cafés?", "RESEARCHER", 57),
])
@pytest.mark.parametrize("action", [Action.RESPOND, Action.ASSIGN, Action.MISSION])
def test_have_a_name_and_a_bare_verb_with_a_question_mark_hands_the_work_over(said, name, agent_id, action):
    tiers = ComplexityAssessment(complexity=Complexity.MOLECULE, action=action, reasoning="tiers")
    verdict = the_lane(_OneOpsBrain(), said, tiers)

    assert addressed_by_name(said, name) is True
    assert addressed_by_name(said.rstrip("?"), name) is True        # as the same words without "?"
    assert (verdict.action, verdict.target_agent_id, verdict.target_agent_name) == (Action.ASSIGN, agent_id, name)


@pytest.mark.parametrize("said", ["Have Support caught up?", "Have sales risen this week?", "Have OPS finished?"])
def test_have_a_name_and_a_past_participle_with_a_question_mark_hands_nothing_over(said):
    roster = [*ONE_OPS, NS(id=610, name="Support", job_title="Customer support"),
              NS(id=611, name="Sales", job_title="Wholesale sales lead")]
    brain = _OneOpsBrain()
    brain._active_agents = lambda: roster
    tiers = ComplexityAssessment(complexity=Complexity.MOLECULE, action=Action.RESPOND, reasoning="tiers")

    assert the_lane(brain, said, tiers) is tiers


@pytest.mark.parametrize("action", [Action.RESPOND, Action.ASSIGN, Action.MISSION])
def test_dont_ask_researcher_files_no_ticket_whatever_the_tiers_said(action):
    tiers = ComplexityAssessment(complexity=Complexity.MOLECULE, action=action, reasoning="tiers")

    assert the_lane(_Brain(), "Don't ask RESEARCHER to price the Kerbside offer", tiers) is tiers


@pytest.mark.parametrize("said, name", [
    ("Don't forget to ask OPS to check the stock.", "OPS"),           # the negation is of forgetting
    ("Could you get OPS to reorder the green stock?", "OPS"),         # a request, as "Can you get …?" is
    ("Stop and ask OPS to check the stock.", "OPS"),                  # "and" opens a new clause
    ("We're not ready, but ask OPS to check the stock.", "OPS"),
    ("Should I order 1.5 kg? Ask OPS to check.", "OPS"),              # the question is another sentence
    ("Have Sales to pull the Q3 figures?", "Sales"),                  # "to <verb>" after the name
])
def test_a_hand_off_beside_a_negation_or_a_question_is_still_handed_over(said, name):
    assert addressed_by_name(said, name) is True
