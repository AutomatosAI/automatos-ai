"""PRD-256 US-011: one hand-off table instead of four lanes.

Brand work went to the Brand designer (PRD-255 US-014, F362: brand_assign_lane and the routing half
of brand_to_the_designer), customer paperwork with no template named to the team (F337(c):
paperwork_to_the_team), social media work to the Social Media Director (F379: socials_assign_lane),
each through its own module and the routing half of named_template_note. They are one table now
(``consumers.chatbot.handoffs``), read once per turn by the classifier. The deleted modules' tests
are here, each against the table:

- the golden: the routing asks (the eval's wrong-routing and wrong-tool rows, listed below) route the
  same way after the change as before it (``fixtures/prd256_routing_golden.json``, generated from the
  four lanes before they were deleted): the lane AutoBrain pins and the note the turn reads last;
- the table: one row per kind, the lane modules gone, the classifier's reading is the note's;
- brand work (test_prd255w2_brand_routing, test_f362_a, test_f362_b), customer paperwork
  (test_f337_c), each as it was asserted before;
- night 12 (FX-014): the 33 hand-off shapes the classifier's verdict lost ("Get OPS to…", "Ask
  RESEARCHER…", "Give #1057 to CHRISTMAS BOX", "267, the operations one", "Delete MARKET-MANAGER")
  route to their lane whatever the tiers said (the golden file's ``night12`` rows). Social media work's row keeps its own file
  (test_f379_d), against the table;
- P256-FIX-RVW-12: a name is handed work only when a task follows it, so "Have support tickets been
  answered today?", "Get sales figures for Q3" and "Get OPS's stock report" stay the tiers', and work
  handed to two teammates together stays the tiers' whatever they said (``night12.task_after_the_name``).
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot import brand_to_the_designer, handoffs
from consumers.chatbot import named_template_note as note_module
from consumers.chatbot.auto import Action, AutoBrain, Complexity, ComplexityAssessment, apply_assign_bias
from consumers.chatbot.handoffs import (
    BRAND, HANDOFFS, PAPERWORK, REMOVED_NOTE, SOCIALS, UNAVAILABLE_NOTE, Turn, asks_for_brand_work,
    asks_for_paperwork, auto_always_answers, hands_off, read_the_table, the_turn, turn_note,
)
from consumers.chatbot.named_template_note import fills_the_named_template, read_note
from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT
from modules.documents.presets import INVOICE

WS = UUID("4d9b3a19-6c0e-4f2a-9b1c-0d1e2f3a4b50")
DESIGNER = NS(id=347, name="Brand Designer", status="active")
DIRECTOR = NS(id=348, name="Social Media Director")
PASSAGES = "passages from brand-notes.md"
TIERS = "the tiers' assessment"
PRICE_LIST = ("Can you make me a wholesale price list for cafés, as a spreadsheet? Every coffee we sell, price per "
              "kg, minimum order, carriage and payment terms.")
NIGHT_10C = ["Make the orange an accent only", "More space between sections",
             "Not you — my brand. The documents feel a bit cold. Warmer, please."]
GOLDEN = Path(__file__).parent / "fixtures" / "prd256_routing_golden.json"

# The eval's routing asks: (category, the owner's words, whether they open the conversation).
ROUTING_ASKS = [
    ("wrong-routing", "Help me design my brand.", False),
    ("wrong-routing", "Can you improve the kit? It looks dated.", False),
    ("wrong-routing", "Make our templates match the new logo.", False),
    ("wrong-routing", "Less orange, please.", False),
    ("wrong-routing", "Make the orange an accent only", False),
    ("wrong-routing", "More space between sections", False),
    ("wrong-routing", "Warmer, please", True),
    ("wrong-routing", "Warmer, please", False),
    ("wrong-routing", "Make our invoice template look like us.", False),
    ("wrong-routing", "Auto: my social media person should draft the October Harvest Club post.", False),
    ("wrong-routing", "Can you get my social media director to do a 15-second video of our top three cafés?", False),
    ("wrong-routing", "Draft an Instagram post for the October Harvest Club box.", False),
    ("wrong-routing", "Make a carousel for the Harvest Club: two coffees, £32, on sale Monday.", False),
    ("wrong-routing", "Have our social media team put together a LinkedIn post about the wholesale offer.", False),
    ("wrong-routing", "Get Jim to make a carousel for the Harvest Club.", False),
    ("wrong-routing", "Ask NEWSROOM to draft a LinkedIn post about the wholesale offer.", False),
    ("wrong-routing", PRICE_LIST, True),
    ("wrong-routing", "Write the Quay letter: to Maya Osei, moving to 30-day terms from 1 November.", False),
    ("wrong-routing", "Please draft a quote for Rosa, 10 kg vs 12 kg of Harbour Blend.", False),
    ("wrong-routing", "Do me a flyer for the October Harvest Club box.", False),
    ("wrong-routing", "Put together an invoice for Salt Kitchen: 8 kg Harbour Blend at £22.", False),
    ("wrong-routing", "Design a flyer in our brand colours for the Harvest Club.", False),
    ("wrong-routing", "Give #0192 to the Support Agent", False),
    ("wrong-routing", "Ask Jim to chase the Salt Kitchen invoice.", False),
    ("wrong-routing", "Plan next week's roasting schedule", False),
    ("wrong-tool", "Make the Lantern Kitchen invoice on my Harbourline Invoice template.", False),
    ("wrong-tool", "Make an invoice for Salt Kitchen on my Seaside Invoice template.", False),
    ("wrong-tool", "Can you write the Quay letter yourself? I want it now.", False),
    ("wrong-tool", "Make the carousel yourself, don't bother the team.", False),
    ("wrong-tool", "Improve the kit, but don't bother the team.", False),
    ("wrong-tool", "What colour is our accent?", False),
    ("wrong-tool", "Did Rosa pay the invoice from September?", False),
    ("wrong-tool", "Send the invoice to Rosa.", False),
    ("wrong-tool", "Which Instagram posts are waiting for approval?", False),
    ("wrong-tool", "Approve the Harvest Club post.", False),
    ("wrong-tool", "Make a flyer for our Instagram.", False),
    ("wrong-tool", "How do I give my agent two photos?", False),
    ("wrong-tool", "Make a quote card post on my Quote card template.", False),
    ("wrong-tool", "Fix the font size in the table.", False),
]
GOLDEN_ROWS = [
    NS(id=UUID("00000000-0000-0000-0000-000000000a01"), name="Harbourline Invoice", format="pdf", category="invoice",
       blocks=None, version=1),
    NS(id=UUID("00000000-0000-0000-0000-000000000a02"), name="Branded Invoice", format="pdf", category="invoice",
       blocks=None, version=1),
    NS(id=UUID("00000000-0000-0000-0000-000000000a03"), name="Quote card", format="social_image", category="social",
       version=1, blocks={"variables_schema": {"quote": {"type": "text", "label": "Quote"}}}),
]
NOTE_LABELS = [
    ("The owner asked for brand work: ", "brand-designer-ticket"),
    ("The ticket is a Socials post.", "socials-ticket"),
    ("Socials, as it works today.", "how-socials-works"),
    ("The owner named their document template ", "named-template"),
    ("The owner named a document template ", "template-not-found"),
    ("The owner asked for a", "paperwork-team-ticket"),
]


# ── shared fakes ───────────────────────────────────────────────────────────

class _ChatDb:
    """The chat's session, faked: a savepoint only."""

    def begin_nested(self):
        return contextlib.nullcontext()


def _brain(onboarding=False):
    """What ``hands_off`` reads of AutoBrain."""
    return NS(_db=object(), _workspace_id=str(WS), _onboarding_active=lambda: onboarding)


async def _tiers(brain, message, conversation_length=0):
    return TIERS


_assess = hands_off(_tiers)


@pytest.fixture
def designer(monkeypatch):
    from core.seeds import seed_brand_designer

    found = {"agent": DESIGNER, "looked": []}

    def find(db, workspace_id):
        found["looked"].append(workspace_id)
        return found["agent"]

    monkeypatch.setattr(seed_brand_designer, "find_brand_designer", find)
    return found


@pytest.fixture
def templates(monkeypatch):
    from modules.documents import template_service

    read = []

    class _Templates:
        def __init__(self, db):
            self.db = db

        def list_templates(self, workspace_id, format=None, category=None):
            read.append(workspace_id)
            return []

    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)
    return read


def _note(said, *earlier):
    """The hand-off note the table gives a turn no classifier read (``said`` the latest)."""
    texts = [said, *earlier]
    return turn_note(object(), WS, the_turn(texts), no_template_named=False)


def _team_note(said):
    return turn_note(object(), WS, Turn(read_the_table(said)), no_template_named=True)


def _wrapped_turn(chat, said, messages):
    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": PASSAGES})
        yield "searched"

    async def run():
        return [f async for f in fills_the_named_template(retrieval_first)(chat, said, messages, None, "c", [])]

    assert asyncio.run(run()) == ["searched"]
    return messages


def _turn(said, widget_mode=False, db=None, workspace_id=WS):
    chat = NS(db=db or _ChatDb(), workspace_id=str(workspace_id), widget_mode=widget_mode)
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": said}]
    return _wrapped_turn(chat, said, messages)


# ── the golden: the same asks route the same way ───────────────────────────

@pytest.fixture
def golden_workspace(monkeypatch, designer):
    from modules.documents import template_service

    class _Templates:
        def __init__(self, db):
            pass

        def list_templates(self, workspace_id, format=None, category=None):
            return list(GOLDEN_ROWS)

    golden = json.loads(GOLDEN.read_text())
    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)
    monkeypatch.setattr(handoffs, "find_social_media_director", lambda db, workspace_id: DIRECTOR)
    monkeypatch.setattr(handoffs, "agent_names", lambda db, workspace_id: golden["roster"])
    monkeypatch.setattr(note_module, "_schema", lambda db, workspace_id, row: {"data_fields": []})
    return golden


def _label(note):
    if note is None:
        return None
    return next(label for start, label in NOTE_LABELS if note.startswith(start))


def _routed(said, opening):
    """The lane AutoBrain pins and the note the turn reads last, in one turn's context."""
    texts = [said] if opening else [said, "Hello."]

    async def turn():
        verdict = await _assess(_brain(), said, 1 if opening else 3)
        return verdict, read_note(_ChatDb(), WS, texts)

    verdict, note = asyncio.run(turn())
    lane = f"ASSIGN {verdict.target_agent_name}" if verdict != TIERS else "tiers"
    return lane, _label(note)


def test_the_golden_lists_the_routing_asks_of_this_test(golden_workspace):
    listed = [(r["category"], r["said"], r["opening"]) for r in golden_workspace["routes"]]
    assert listed == [tuple(ask) for ask in ROUTING_ASKS]
    assert {category for category, _, _ in ROUTING_ASKS} == {"wrong-routing", "wrong-tool"}


@pytest.mark.parametrize("index", range(len(ROUTING_ASKS)))
def test_the_routing_asks_route_as_they_did_before_the_table(golden_workspace, index):
    before = golden_workspace["routes"][index]
    assert _routed(before["said"], before["opening"]) == (before["lane"], before["note"]), before["said"]


# ── night 12 (FX-014): a named agent gets its ticket whatever the tiers said ──

NIGHT_12 = json.loads(GOLDEN.read_text())["night12"]
TIERS_SAID = [Action.RESPOND, Action.DELEGATE, Action.MISSION, Action.ASSIGN]


class _Night12Brain:
    """What the table and the lane read of AutoBrain: c1's roster, AutoBrain's own roster match."""

    _db, _workspace_id = object(), str(WS)
    _match_roster_agent = AutoBrain._match_roster_agent

    def _onboarding_active(self):
        return False

    def _active_agents(self):
        return [NS(**{"is_system_agent": False, "status": "active", **agent}) for agent in NIGHT_12["roster"]]


@pytest.fixture
def night12_workspace(monkeypatch, designer):
    monkeypatch.setattr(handoffs, "find_social_media_director", lambda db, workspace_id: DIRECTOR)
    monkeypatch.setattr(handoffs, "agent_names", lambda db, workspace_id: [a["name"] for a in NIGHT_12["roster"]])


def _night12_lane(said, action):
    """The lane AutoBrain's two wrappers give a turn whose tiers said ``action`` (night 12: RESPOND)."""
    tiers = ComplexityAssessment(complexity=Complexity.MOLECULE, action=action, reasoning=TIERS)

    async def classifier(brain, message, conversation_length=0):
        return tiers

    verdict = asyncio.run(hands_off(auto_always_answers(classifier))(_Night12Brain(), said, 3))
    if verdict is tiers or verdict.action != Action.ASSIGN:
        return "tiers"
    if verdict.target_agent_id is None:
        ids = sorted(a["id"] for a in NIGHT_12["roster"] if a["name"] == verdict.target_agent_name)
        return f"ASK {verdict.target_agent_name} ({', '.join(map(str, ids))})"
    return f"ASSIGN {verdict.target_agent_id} {verdict.target_agent_name}"


def test_the_golden_holds_night_12s_33_hand_off_shapes():
    said = " | ".join(route["said"] for route in NIGHT_12["routes"]).lower()
    lanes = [route["lane"] for route in NIGHT_12["routes"]]

    assert len(lanes) == 33
    for shape in ("get ops to", "ask researcher", "have market-manager", "give #1057 to christmas box",
                  "267, the operations one", "delete market-manager", "set up something every wednesday"):
        assert shape in said
    assert "ASK OPS (267, 284)" in lanes and "ASSIGN 267 OPS" in lanes and "tiers" in lanes


@pytest.mark.parametrize("action", TIERS_SAID)
@pytest.mark.parametrize("index", range(len(NIGHT_12["routes"])))
def test_night_12s_hand_offs_take_their_lane_whatever_the_tiers_said(night12_workspace, index, action):
    route = NIGHT_12["routes"][index]
    assert _night12_lane(route["said"], action) == route["lane"], route["said"]


# ── P256-FIX-RVW-12: a name is handed work only when a task follows it ──────

TASK_AFTER_THE_NAME = NIGHT_12["task_after_the_name"]


def test_the_golden_holds_the_names_that_hand_nothing_over_and_the_joined_rows():
    by_said = {route["said"]: route["lane"] for route in TASK_AFTER_THE_NAME}
    names = {agent["name"] for agent in NIGHT_12["roster"]}

    assert {"Support", "Sales"} <= names
    for said in ("Have support tickets been answered today?", "Get sales figures for Q3", "Get OPS's stock report",
                 "Have Support answered the club emails?", "Have RESEARCHER and WRITER plan the launch."):
        assert by_said[said] == "tiers", said
    assert by_said["Have Support check the refund queue."] == "ASSIGN 610 Support"
    assert by_said["Ask Sales: how many club boxes sold this week?"] == "ASSIGN 611 Sales"


@pytest.mark.parametrize("action", TIERS_SAID)
@pytest.mark.parametrize("index", range(len(TASK_AFTER_THE_NAME)))
def test_a_name_is_handed_work_only_when_a_task_follows_it_whatever_the_tiers_said(night12_workspace, index, action):
    route = TASK_AFTER_THE_NAME[index]
    assert _night12_lane(route["said"], action) == route["lane"], route["said"]


# ── the table ──────────────────────────────────────────────────────────────

def test_one_table_holds_each_kind_its_role_and_its_note():
    assert [row.kind for row in HANDOFFS] == [BRAND, SOCIALS, PAPERWORK]
    assert [row.role for row in HANDOFFS][:2] == ["the Brand designer", "the Social Media Director"]
    paperwork = HANDOFFS[2]
    assert paperwork.owner is None and paperwork.unless_a_template_is_named
    assert HANDOFFS[0].no_setting_or_mission and not HANDOFFS[1].no_setting_or_mission


@pytest.mark.parametrize("module", ["brand_assign_lane", "paperwork_to_the_team", "socials_assign_lane"])
def test_the_lane_modules_are_deleted(module):
    assert importlib.util.find_spec(f"consumers.chatbot.{module}") is None


def test_the_routing_halves_are_gone_and_the_field_note_stays():
    for gone in ("asks_for_brand_work", "designer_note", "DESIGNER_NOTE"):
        assert not hasattr(brand_to_the_designer, gone)
    for gone in ("designer_note", "team_note"):
        assert not hasattr(note_module, gone)
    assert callable(note_module.found_note) and callable(note_module.not_found_note)


def test_autobrain_assess_wears_the_table_once():
    import inspect

    from consumers.chatbot import auto

    source = inspect.getsource(auto.AutoBrain)
    assert source.count("@hands_off") == 1
    assert "brand_work_goes_to_the_designer" not in source and "social_work_goes_to_the_director" not in source


def test_the_classifier_reads_the_table_once_and_the_note_reads_its_reading(designer, templates, monkeypatch):
    reads = []
    real = handoffs.read_the_table

    def counted(message, opening=False):
        reads.append(message)
        return real(message, opening)

    monkeypatch.setattr(handoffs, "read_the_table", counted)

    async def turn():
        verdict = await _assess(_brain(), "Help me design my brand.", 3)
        return verdict, read_note(_ChatDb(), WS, ["Help me design my brand."])

    verdict, note = asyncio.run(turn())
    assert reads == ["Help me design my brand."]
    assert verdict.target_agent_name == "Brand Designer" and 'assigned_agent_name "Brand Designer"' in note
    assert designer["looked"] == [WS]       # the pin found the designer; the note named it from the pin


def test_the_note_keeps_the_classifiers_reading_of_an_opening(designer, templates):
    async def turn():
        await _assess(_brain(), "Warmer, please", 4)       # not the opening message, to the classifier
        return read_note(_ChatDb(), WS, ["Warmer, please"])

    assert asyncio.run(turn()) is None
    assert read_note(_ChatDb(), WS, ["Warmer, please"]).startswith("The owner asked for brand work")


def test_paperwork_pins_nothing_and_its_note_files_the_ticket(designer, templates):
    async def turn():
        verdict = await _assess(_brain(), PRICE_LIST, 1)
        return verdict, read_note(_ChatDb(), WS, [PRICE_LIST])

    verdict, note = asyncio.run(turn())
    assert verdict == TIERS and note.startswith("The owner asked for a price list for their customers")


# ── brand work (PRD-255 US-014, F362) ──────────────────────────────────────

@pytest.mark.parametrize("said", [
    "Help me design my brand.",
    "Can you improve the kit? It looks dated.",
    "Make our templates match the new logo.",
    "Redesign our brand kit from the logo.",
    "Less orange, please.",
    "Could you make the brand warmer?",
    "We need more space on the pages.",
    "Make our invoice template look like us.",
    "Sort out our colours on everything.",
    "More orange in the header, less in the body.",
])
def test_a_brand_ask_is_read(said):
    assert asks_for_brand_work(said) is True


@pytest.mark.parametrize("said", [
    "What colour is our accent?",                                     # a question about the kit
    "Did the designer finish the brand kit?",
    "How would you improve our brand?",
    "Design my brand yourself, I want it now.",                       # kept with Auto
    "Improve the kit, but don't bother the team.",
    "Make me an invoice for Salt Kitchen: 8 kg Harbour Blend.",       # customer paperwork (F337(c))
    "Design a flyer in our brand colours for the Harvest Club.",
    PRICE_LIST,
    "Make the Lantern Kitchen invoice on my Harbourline Invoice template.",   # a named template (F351)
    "We need more space in the calendar next week.",                  # not the look
    "I need more space in my documents folder.",
    "It's warmer today.",
    "Less red tape for our suppliers, please.",                       # not a colour
    "More gold stock for the shop.",
    "Fix the font size in the table.",                                # not the owner's brand
    "Build a colour palette for my garden.",
    "Make something with my template.",                               # fills one (F351)
    "",
])
def test_anything_else_is_not_brand_work(said):
    assert asks_for_brand_work(said) is False


@pytest.mark.parametrize("said", NIGHT_10C)
def test_the_nights_brand_asks_are_read_as_brand_work(said):
    assert asks_for_brand_work(said) is True


def test_a_style_word_on_its_own_is_brand_work_only_when_it_opens_the_conversation():
    assert asks_for_brand_work("Warmer, please", opening=True) is True
    assert asks_for_brand_work("A bit more space, please.", opening=True) is True
    assert asks_for_brand_work("Warmer, please") is False          # after a draft: its tone, perhaps
    assert asks_for_brand_work("It's warmer today.", opening=True) is False


def _never_seed(workspace_id):
    raise AssertionError("the workspace already has its designer: nothing is seeded")


def _seeding_with(monkeypatch, seed):
    real = brand_to_the_designer.designer_name
    monkeypatch.setattr(brand_to_the_designer, "designer_name", lambda db, workspace_id: real(db, workspace_id, seed))


def test_the_note_files_the_ticket_for_the_designer_by_name_and_keeps_the_kit_from_auto(designer, monkeypatch):
    _seeding_with(monkeypatch, _never_seed)
    note = _note("Help me design my brand.")

    assert designer["looked"] == [WS]
    assert note.startswith("The owner asked for brand work")
    assert "don't design it yourself" in note
    assert "don't call platform_update_brand_kit, platform_propose_brand_kit or the template tools" in note
    assert 'platform_create_task with assigned_agent_name "Brand Designer"' in note
    assert "platform_update_task_status to 'in_progress'" in note
    assert "carries the owner's words exactly" in note and "invents none" in note
    for step in ("read the logo", "approves, with the Brand Board drawn from the proposal and not yet saved",
                 "save it only after the owner approves", "an invoice, a letter, a proposal and three social cards",
                 "report back with the board and the set"):
        assert step in note
    assert DISPATCH_CONTRACT_FRAGMENT in note


def test_a_workspace_without_a_designer_gets_one_seeded_and_named(designer, monkeypatch):
    designer["agent"] = None
    seeded = []

    def seed(workspace_id):
        seeded.append(workspace_id)
        return "Brand Designer"

    _seeding_with(monkeypatch, seed)
    assert seeded == [] and 'assigned_agent_name "Brand Designer"' in _note("Improve the kit.")
    assert seeded == [WS]


def test_a_designer_the_owner_removed_is_not_brought_back(designer, monkeypatch):
    designer["agent"] = None
    _seeding_with(monkeypatch, lambda workspace_id: None)
    note = _note("Less orange.")

    assert note == REMOVED_NOTE
    assert "platform_create_task" not in note and "Don't change the brand kit" in note


def test_a_seed_that_fails_is_logged_and_auto_still_keeps_its_hands_off_the_kit(designer, monkeypatch, caplog):
    designer["agent"] = None

    def broken(workspace_id):
        raise RuntimeError("database gone")

    _seeding_with(monkeypatch, broken)
    with caplog.at_level(logging.ERROR):
        note = _note("Design my brand.")
    assert note == UNAVAILABLE_NOTE and "could not be found or seeded" in caplog.text


def test_no_note_and_no_lookup_for_anything_else(designer):
    assert _note("What colour is our accent?") is None
    assert designer["looked"] == []


def test_the_note_says_a_style_ask_is_one_ticket_never_a_mission_or_a_setting(designer):
    note = _note("More space between sections")

    assert "never a mission (platform_create_mission)" in note
    assert "never a platform setting (platform_update_system_setting)" in note


def test_a_brand_ask_gets_the_designer_note_before_any_template_is_read(designer, templates):
    note = read_note(_ChatDb(), WS, ["Make our templates match the new logo."])

    assert 'assigned_agent_name "Brand Designer"' in note and templates == []


def test_paperwork_still_goes_to_the_team_beside_a_designer(designer, templates):
    assert read_note(_ChatDb(), WS, [PRICE_LIST]) == _team_note(PRICE_LIST)


def test_an_opening_style_word_gets_the_designer_note_and_a_later_one_does_not(designer, templates):
    opening = read_note(_ChatDb(), WS, ["Warmer, please"])
    later = read_note(_ChatDb(), WS, ["Warmer, please", "Draft a thank-you email to Rosa."])

    assert opening is not None and 'assigned_agent_name "Brand Designer"' in opening
    assert later is None


def test_the_owners_brand_ask_reaches_the_model_as_the_designer_note(designer, templates):
    messages = _turn("Help me design my brand.")

    assert messages[-2]["content"] == PASSAGES
    assert messages[-1]["role"] == "system" and messages[-1]["content"].startswith("The owner asked for brand work")


def test_a_widget_visitors_brand_ask_gets_no_note(designer, templates):
    messages = _turn("Help me design my brand.", widget_mode=True)

    assert messages[-1]["content"] == PASSAGES and designer["looked"] == []


def _autobrain(monkeypatch, onboarding=False):
    brain = AutoBrain(object(), str(WS))
    brain._redis = None
    monkeypatch.setattr(brain, "_onboarding_active", lambda: onboarding)

    def no_tier(*_a, **_k):
        raise AssertionError("a brand ask is decided before the cache and the tiers")

    for tier in ("_cache_lookup", "_run_fast_heuristics", "_decision_classify", "_llm_classify"):
        monkeypatch.setattr(brain, tier, no_tier)
    return brain


@pytest.mark.parametrize("said", NIGHT_10C)
def test_autobrain_pins_a_brand_ask_to_the_designers_ticket(designer, monkeypatch, said):
    verdict = asyncio.run(_autobrain(monkeypatch).assess(said, 3))

    assert verdict.action == Action.ASSIGN and verdict.complexity == Complexity.MOLECULE
    assert (verdict.target_agent_id, verdict.target_agent_name) == (347, "Brand Designer")
    apply_assign_bias(verdict, said)
    assert "assigned_agent_name=\"Brand Designer\"" in verdict.context_directive
    assert "platform_create_task" in verdict.tool_hints


def test_an_opening_warmer_please_is_pinned_and_a_later_one_is_left_to_the_tiers(designer, monkeypatch):
    opening = asyncio.run(_autobrain(monkeypatch).assess("Warmer, please", 1))
    assert opening.target_agent_name == "Brand Designer"

    with pytest.raises(AssertionError, match="decided before the cache"):
        asyncio.run(_autobrain(monkeypatch).assess("Warmer, please", 4))


@pytest.mark.parametrize("found", [None, NS(id=347, name="Brand Designer", status="paused")])
def test_without_an_active_designer_the_tiers_decide(designer, monkeypatch, found):
    designer["agent"] = found

    with pytest.raises(AssertionError, match="decided before the cache"):
        asyncio.run(_autobrain(monkeypatch).assess(NIGHT_10C[0], 3))


def test_mid_onboarding_the_onboarding_pin_still_comes_first(designer, monkeypatch):
    verdict = asyncio.run(_autobrain(monkeypatch, onboarding=True).assess(NIGHT_10C[0], 3))

    assert verdict.action == Action.RESPOND and "Onboarding active" in verdict.reasoning


def test_a_proposal_made_with_no_ticket_tells_auto_to_file_it_for_the_designer(designer):
    from modules.tools.discovery import handlers_brand_proposals as bp

    answer = asyncio.run(bp.propose_brand_kit(object(), WS, {"brand_kit": {"accent_use": "sparing"}}))

    assert answer["success"] is False and "has none" in answer["error"]
    assert 'platform_create_task (assigned_agent_name "Brand Designer"' in answer["error"]
    assert "ask a human" not in answer["error"]


# ── a brand turn changes no setting and starts no mission (Gerard, 7 Oct) ──

REFUSED = ("This is a brand change: it goes to the Brand Designer on a ticket "
           "(platform_create_task assigned to Brand Designer).")


def _execute(action_name, params):
    """A call outside a chat a person drives (no conversation), so only the brand turn's check applies."""
    from modules.tools.discovery.follows_the_owner import follows_the_owner

    @follows_the_owner          # the executor's check before any gate or handler
    async def ran(self, action_name, params, caller_context=None):
        return {"success": True, "ran": action_name}

    return ran(NS(db=None, workspace_id=WS), action_name, params)


def _brand_turn(said, *calls, onboarding=False):
    """One chat turn: AutoBrain assesses ``said``, then the turn's tool calls run in its context."""
    async def turn():
        verdict = await _assess(_brain(onboarding), said, 3)
        return verdict, [await call() for call in calls]
    return asyncio.run(turn())


def test_a_brand_turn_refuses_the_setting_and_the_mission_with_where_the_work_goes(designer):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = object()     # the refusal comes before any gate, so no executor state is read
    verdict, (setting, mission, blog) = _brand_turn(
        "More space between sections",
        lambda: PlatformActionExecutor.execute(executor, "platform_update_system_setting", {"key": "spacing"}),
        lambda: PlatformActionExecutor.execute(executor, "platform_create_mission", {"goal": "Warmer documents"}),
        lambda: PlatformActionExecutor.execute(executor, "platform_create_blog_post", {"topic": "Our new look"}),
    )

    assert verdict.target_agent_name == "Brand Designer"
    for refused in (setting, mission, blog):
        assert refused == {"success": False, "error": REFUSED}


def test_a_brand_turn_still_files_the_designers_ticket(designer):
    _, (filed, started) = _brand_turn(
        "Make the orange an accent only",
        lambda: _execute("platform_create_task", {"assigned_agent_name": "Brand Designer"}),
        lambda: _execute("platform_update_task_status", {"task_id": "#0901", "status": "in_progress"}),
    )

    assert filed == {"success": True, "ran": "platform_create_task"}
    assert started == {"success": True, "ran": "platform_update_task_status"}


def test_a_turn_that_is_not_a_brand_ask_runs_the_setting_and_the_mission(designer):
    verdict, (setting, mission) = _brand_turn(
        "Plan next week's roasting schedule",
        lambda: _execute("platform_update_system_setting", {"key": "spacing"}),
        lambda: _execute("platform_create_mission", {"goal": "Roasting schedule"}),
    )

    assert verdict == TIERS
    assert setting["success"] is True and mission["success"] is True


def test_the_next_assessment_clears_a_brand_turn(designer):
    async def two_turns():
        await _assess(_brain(), "Warmer, please", 1)
        await _assess(_brain(), "Plan next week's roasting schedule", 3)
        return await _execute("platform_create_mission", {"goal": "Roasting schedule"})

    assert asyncio.run(two_turns())["success"] is True


def test_mid_onboarding_a_brand_ask_is_no_brand_turn(designer):
    verdict, (mission,) = _brand_turn(
        "Make the orange an accent only",
        lambda: _execute("platform_create_mission", {"goal": "Warmer documents"}),
        onboarding=True,
    )

    assert verdict == TIERS and mission["success"] is True


# ── customer paperwork (F337(c)) ───────────────────────────────────────────

D2564ECF = PRICE_LIST
PAPER_PASSAGES = "passages from wholesale-terms.md"


@pytest.mark.parametrize("said, kind", [
    (D2564ECF, "price list"),                                                            # chat d2564ecf
    ("Write the Quay letter: to Maya Osei, moving to 30-day terms from 1 November.", "letter"),
    ("Please draft a quote for Rosa, 10 kg vs 12 kg of Harbour Blend.", "quote"),
    ("Do me a flyer for the October Harvest Club box.", "flyer"),
    ("Put together an invoice for Salt Kitchen: 8 kg Harbour Blend at £22.", "invoice"),
])
def test_paperwork_asked_for_in_plain_english_is_read(said, kind):
    assert asks_for_paperwork(said) == kind


@pytest.mark.parametrize("said", [
    "Did Rosa pay the invoice from September?",                                          # a question about one
    "Send the invoice to Rosa.",                                                         # sending, not making
    "How many bags of Kayon Mountain did Lantern Kitchen order in September?",
    "Can you write the Quay letter yourself? I want it now.",                            # kept with Auto
    "Make the flyer, but don't bother the team with it.",
    "",
])
def test_anything_else_is_not_paperwork(said):
    assert asks_for_paperwork(said) is None
    assert _team_note(said) is None


def test_the_note_hands_it_to_the_right_agent_with_the_owners_facts():
    note = _team_note(D2564ECF)

    assert note.startswith("The owner asked for a price list for their customers and named none of their templates.")
    assert "don't write it yourself in this reply" in note
    assert "platform_recommend_agent" in note and "platform_create_task with assigned_agent_name" in note
    assert "exactly as the owner gave them and invents none" in note
    assert "who has it and its card number" in note and "on one of their templates" in note
    assert DISPATCH_CONTRACT_FRAGMENT in note


@pytest.fixture
def shop(db_session, seed_workspace):
    from core.models.core import DocumentTemplate

    ws = UUID(seed_workspace())
    db_session.add(DocumentTemplate(workspace_id=ws, name="Harbourline Invoice", format="pdf",
                                    category=INVOICE["category"], blocks=INVOICE["blocks"], data_schema={},
                                    sample_data={}, created_by="owner"))
    db_session.flush()
    return NS(db=db_session, ws=ws)


def _shop_turn(shop, said, widget_mode=False):
    chat = NS(db=shop.db, workspace_id=str(shop.ws), widget_mode=widget_mode)
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": said}]

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": PAPER_PASSAGES})
        yield "searched"

    async def run():
        return [f async for f in fills_the_named_template(retrieval_first)(chat, said, messages, None, "c", [])]

    assert asyncio.run(run()) == ["searched"]
    return messages


def test_a_turn_asking_for_paperwork_with_no_template_gets_the_team_note(shop):
    messages = _shop_turn(shop, D2564ECF)

    assert messages[-2]["content"] == PAPER_PASSAGES
    assert messages[-1]["role"] == "system" and messages[-1]["content"] == _team_note(D2564ECF)


def test_a_named_template_is_still_filled_by_auto(shop):
    messages = _shop_turn(shop, "Make the Lantern Kitchen invoice on my Harbourline Invoice template.")

    assert "Use THIS template" in messages[-1]["content"]
    assert "platform_create_task" not in messages[-1]["content"]


def test_a_widget_visitors_turn_gets_no_team_note(shop):
    messages = _shop_turn(shop, D2564ECF, widget_mode=True)

    assert messages[-1]["content"] == PAPER_PASSAGES


def test_the_note_reads_as_english():
    assert _team_note("Make an invoice for Salt Kitchen.").startswith("The owner asked for an invoice for their customers")
