"""F337 (b), night 10: Auto's own chat keeps to the brand kit, knows today in the owner's zone,
and writes only the owner's facts.

Night 10 (Monday 5 October), from Auto's chat: a counter flyer (PDF 1fc4eef7) that said
"delightful", which the kit bans, posted "October 15th" and called a red apple green; a
wholesale flyer from "Harbour Coffee Roasters" (e9d50f2e), not Harbourline; a quote and a letter
signed "Sincerely, [Your Company Name]". #953 gave the brand kit to card runs, mission steps and
playbook steps and left Auto's chat unwired. Now every owner's turn in a workspace with a kit
ends Auto's system prompt with the kit's rules (read off the event loop), the saved reply has its
placeholder company and signer filled, and a banned word the reply uses is said after it.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from core.models.workspaces import Workspace

WS = "5e0c1a2b-3d4e-4f60-8a7b-9c0d1e2f3a37"
NAME = "Harbourline Coffee Roasters"
SIGNED = "Gerard, Harbourline Coffee Roasters"
KIT = {"name": NAME,   # 3 to 5 tone words, or the read keeps the default voice (no sign-off, no banned words)
       "voice": {"tone": ["warm", "plain", "local"], "banned_phrases": ["delightful", "exquisite"],
                 "sign_off": SIGNED}}
SYSTEM = "You are Auto."
ASKED = "I want a one-page flyer for the October Harvest Club box to leave on the counter."
# Night 10's flyer copy (PDF 1fc4eef7) and Auto's reply about it (chat 494e8101).
FLYER_COPY = "Each selection offers a unique and delightful tasting experience, perfect for the autumn season."
NIGHT_10_REPLY = ('Please note, the flyer currently uses the word "delightful," which your brand kit has flagged '
                  "as a banned word. You might want to revise this before it's distributed.")


@pytest.fixture(autouse=True)
def fresh_kits():
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    yield
    forget_cached_kits()


class _Session:
    """A session holding one workspace's settings; it records the thread of every read."""

    def __init__(self, settings):
        self.settings, self.read_on = settings, []

    def get(self, model, *_a, **_k):
        self.read_on.append(threading.get_ident())
        return NS(settings=self.settings) if model is Workspace else None


def _chat(settings=None, *, widget=False):
    return NS(db=_Session({"brand_kit": KIT} if settings is None else settings), workspace_id=WS, widget_mode=widget)


def _messages():
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASKED}]


def _turn(chat, then=None):
    """Auto's prepared messages for one turn, the loop's thread, and ``then()`` awaited after them in
    the same task (the turn's answer)."""
    from consumers.chatbot.brand_turn import autos_prompt_carries_the_brand_kit

    sent = _messages()

    async def prepare(self, *args, **kwargs):
        return sent, ["tools"], None

    async def run():
        prepared = await autos_prompt_carries_the_brand_kit(prepare)(chat, "messages", "runtime")
        return prepared, threading.get_ident(), (await then() if then else None)

    (llm_messages, use_tools, orchestrated), loop_thread, after = asyncio.run(run())
    return NS(messages=llm_messages, sent=sent, tools=use_tools, loop=loop_thread, after=after)


def test_autos_system_prompt_ends_with_the_brand_kits_rules_and_the_business_name():
    turn = _turn(_chat())

    system = turn.messages[0]["content"]
    assert turn.messages[0]["role"] == "system" and system.startswith(SYSTEM)
    assert "## The brand's rules" in system and f"- Write as {NAME}." in system
    assert f'Sign it "{SIGNED}"' in system and "Tone: warm, plain, local." in system
    assert 'Never use these words or phrases: "delightful", "exquisite".' in system
    assert turn.messages[1:] == _messages()[1:] and turn.tools == ["tools"]
    assert turn.sent == _messages()                                     # the prepared list is not changed


def test_the_rules_go_in_once():
    from consumers.chatbot.brand_turn import system_prompt_with_rules
    from services.brand_rules import RULES_HEADING

    once = system_prompt_with_rules(_messages(), KIT)
    assert system_prompt_with_rules(once, KIT)[0]["content"].count(RULES_HEADING) == 1


def test_a_workspace_without_a_kit_is_unchanged():
    turn = _turn(_chat({}))

    assert turn.messages == _messages()


def test_a_widget_visitors_turn_is_unchanged_and_reads_no_kit():
    chat = _chat(widget=True)
    turn = _turn(chat)

    assert turn.messages == _messages() and chat.db.read_on == []


def test_the_kit_is_read_off_the_event_loop():
    chat = _chat()
    turn = _turn(chat)

    assert chat.db.read_on and turn.loop not in chat.db.read_on


def test_the_chat_service_runs_each_turn_through_them():
    import consumers.chatbot.service as svc
    from consumers.chatbot import brand_turn as bt

    async def anything(*_a, **_k):
        return None

    assert svc.StreamingChatService._prepare_messages.__code__ is bt.autos_prompt_carries_the_brand_kit(
        anything).__code__
    additions = svc.StreamingChatService._answer_additions.__wrapped__.__wrapped__   # under PRD-256's receipts
    assert additions.__code__ is bt.a_reply_says_its_banned_words(lambda *_a: []).__code__
    assert svc._upload_inline_images.__code__ is bt.a_saved_reply_is_on_brand(anything).__code__


def test_the_saved_reply_names_the_company_and_who_signs():
    import consumers.chatbot.service as svc

    letter = "Dear Quay Coffee House,\n\nYour terms move to 30 days from 1 November.\n\nSincerely,\n[Your Company Name]"
    email = "Thanks Rosa, the 12 kg is booked for Friday.\n\nBest,\n[Your name]"

    async def saved():
        return [await svc._upload_inline_images(letter, workspace_id=WS), await svc._upload_inline_images(email)]

    filled_letter, filled_email = _turn(_chat(), then=saved).after
    assert filled_letter.endswith(f"Sincerely,\n{NAME}") and "[Your Company Name]" not in filled_letter
    assert filled_email.endswith(f"Best,\n{SIGNED}")


def test_a_reply_outside_a_branded_turn_keeps_its_placeholders():
    import consumers.chatbot.service as svc

    letter = "Sincerely,\n[Your Company Name]"

    async def saved():
        return await svc._upload_inline_images(letter)

    assert asyncio.run(saved()) == letter
    assert _turn(_chat({}), then=saved).after == letter


def test_a_banned_word_the_reply_uses_is_said_after_it():
    from consumers.chatbot.service import StreamingChatService
    from services.brand_rules import BANNED_NOTE_LEAD

    async def additions():
        return StreamingChatService._answer_additions(None, NS(content=FLYER_COPY, finish_reason="stop"))

    (note,) = _turn(_chat(), then=additions).after
    assert note.startswith(f"\n\n{BANNED_NOTE_LEAD}") and '"delightful"' in note and "exquisite" not in note


@pytest.mark.parametrize("reply", [NIGHT_10_REPLY, "Your brand kit bans 'delightful' and “exquisite”.",
                                   "Done: the card says it posts Monday 5 October."])
def test_a_reply_that_only_names_a_banned_word_gains_nothing(reply):
    from consumers.chatbot.service import StreamingChatService

    async def additions():
        return StreamingChatService._answer_additions(None, NS(content=reply, finish_reason="stop"))

    assert _turn(_chat(), then=additions).after == []


def test_a_quoted_line_of_copy_is_still_checked():
    from consumers.chatbot.brand_turn import reply_banned_note

    assert '"delightful"' in reply_banned_note('Tagline: "A delightful cup for autumn."', KIT)


def test_autos_standing_rules_say_to_write_only_the_owners_facts():
    from consumers.chatbot.atom_prompt import atom_system_prompt
    from consumers.chatbot.personality import AutomatosPersonality
    from services.brief_facts import AUTO_OWN_FACTS_RULE

    auto = NS(name="Auto", description="", persona=None)
    owners = atom_system_prompt(auto, identity="", memory_block="", facts="## Automatos itself\nLocal edition.")
    visitors = atom_system_prompt(auto, identity="", memory_block="", facts="")

    assert "I use only facts the owner gave me or the workspace holds" in AUTO_OWN_FACTS_RULE
    assert "I never add descriptions, tasting notes, ingredients, colours or claims" in AUTO_OWN_FACTS_RULE
    assert AUTO_OWN_FACTS_RULE in AutomatosPersonality.get_anti_patterns() and AUTO_OWN_FACTS_RULE in owners
    assert AUTO_OWN_FACTS_RULE not in visitors


@pytest.fixture
def london(db_session, seed_workspace):
    """A workspace whose heartbeat runs on UK time."""
    ws = UUID(seed_workspace())
    db_session.execute(text("UPDATE workspaces SET settings = CAST(:s AS json) WHERE id = CAST(:w AS uuid)"),
                       {"s": '{"orchestrator": {"heartbeat": {"timezone": "Europe/London"}}}', "w": str(ws)})
    db_session.expire_all()
    return NS(db=db_session, ws=ws)


def test_autos_short_chat_path_says_today_in_the_workspaces_zone(london, monkeypatch):
    """The full path's DatetimeContextSection already said it in the workspace's zone (F323);
    the ATOM lane said UTC, so from 23:00 UTC on a Sunday in London it still said Sunday."""
    import consumers.chatbot.service as svc
    from modules.context.modes import MODE_CONFIGS, ContextMode
    from services.todays_date import today_line

    monkeypatch.setattr("modules.context.sections.product_facts.product_facts", lambda _db, _ws: "")
    lane = NS(workspace_id=str(london.ws), db=london.db, widget_mode=False, prompt_analyzer=NS(
        convert_to_llm_messages=lambda messages, system_prompt, **_k: [
            {"role": "system", "content": system_prompt}, *messages]))
    llm_messages, _tools, _none = asyncio.run(svc.StreamingChatService._prepare_atom_path(
        lane, [{"role": "user", "content": "Can the club box go out this Saturday?"}],
        NS(agent_id=7, metadata=NS(name="Auto", description="", persona=None)),
        NS(orchestrator=None, get_user_name=lambda: None), force_text_only=True))

    system = llm_messages[0]["content"]
    assert today_line(london.db, london.ws) in system and " in Europe/London." in system
    assert "datetime_context" in MODE_CONFIGS[ContextMode.CHATBOT].sections      # the full path's, unchanged
