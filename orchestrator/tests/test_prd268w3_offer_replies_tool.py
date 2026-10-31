"""PRD-268 US-C12a (§6.6, FR-12, K10, K12, K18) — ``platform_offer_replies``, the reply chips.

When Auto asks the person to choose, it offers 2-3 short answers as taps. Proven here:

* the limits table: one reply, four replies, 25 characters, four words, a duplicate (compared
  trimmed and case-insensitively), an empty reply and a newline are each refused with a
  sentence the model reads and an ``error_code``, and so are a character that does not show
  and markdown or a link; two and three valid replies succeed, and the answer is
  ``{success: true, replies}``, each trimmed with single spaces;
* the sentence is the turn's language, from the server-built caller context only (a
  ``_follow_up`` the model sends is dropped), and every sentence is in English and
  Portuguese (Brazil);
* it touches no session at all, so W1's read-actions sweep covers it with no exception;
* it is registered ``read``, category ``chat``, ``promoted``, with no ``examples``, and wired;
* it is chat only: the CHECK_IN surface never offers it, and the lane refuses it before its
  handler runs.

Nothing here needs a database.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.orm import Session

OFFER = "platform_offer_replies"
CHAT = {"driving_user_id": "7", "conversation_id": "c-7"}


def _offer(replies, caller_context=None):
    """The handler's answer to ``replies``, called as the executor calls it in a chat turn."""
    from modules.tools.discovery.follow_up_context import follow_up_params
    from modules.tools.discovery.handlers_replies import offer_replies

    params = follow_up_params(OFFER, {"replies": replies}, caller_context if caller_context is not None else CHAT)
    return asyncio.run(offer_replies(MagicMock(spec=Session), uuid.uuid4(), params))


# --------------------------------------------------------------------------- the limits


@pytest.mark.parametrize("replies, code", [
    (["Window"], "reply_count"),
    (["Window", "Aisle", "Middle", "Either"], "reply_count"),
    ([], "reply_count"),
    (["Window", "a" * 25], "reply_too_long"),
    (["Window", "one two three four"], "reply_words"),
    (["Window", "window"], "reply_duplicate"),
    (["Not now", "  NOT   now "], "reply_duplicate"),
    (["Window", ""], "reply_empty"),
    (["Window", "   "], "reply_empty"),
    (["Window", "Aisle\nseat"], "reply_line_break"),
    (["Window", "Aisle\u2028seat"], "reply_line_break"),
    (["Window", "Aisle\x85seat"], "reply_line_break"),
    (["Window", "Aisle\x0bseat"], "reply_line_break"),
    (["Window", "Yes\u202e"], "reply_hidden"),
    (["Window", "Ai\u200bsle"], "reply_hidden"),
    (["Window", "\u200b"], "reply_hidden"),
    (["Window", "Aisle\x00"], "reply_hidden"),
    (["Window", "**Aisle**"], "reply_markdown"),
    (["Window", "_Aisle_"], "reply_markdown"),
    (["Window", "~~Aisle~~"], "reply_markdown"),
    (["Window", "[Aisle](x)"], "reply_markdown"),
    (["Window", "`aisle`"], "reply_markdown"),
    (["Window", "<b>Aisle</b>"], "reply_markdown"),
    (["Window", "- Aisle"], "reply_markdown"),
    (["Window", "> Aisle"], "reply_markdown"),
    (["Window", "https://x.test"], "reply_markdown"),
    (["Window", "www.x.test"], "reply_markdown"),
    (["Window", 7], "reply_not_text"),
    ("Window, Aisle", "replies_not_a_list"),
    (None, "replies_not_a_list"),
])
def test_anything_but_two_or_three_short_distinct_replies_is_refused_with_a_sentence(replies, code):
    answer = _offer(replies)

    assert answer["success"] is False and answer["error_code"] == code
    assert isinstance(answer["error"], str) and answer["error"].strip()
    assert "replies" not in answer


def test_the_refusals_say_the_limits():
    assert "24" in _offer(["Window", "a" * 25])["error"]
    assert "25" in _offer(["Window", "a" * 25])["error"]
    assert "3" in _offer(["Window", "one two three four"])["error"]
    assert "4" in _offer(["Window", "Aisle", "Middle", "Either"])["error"]


@pytest.mark.parametrize("replies, shown", [
    (["Window", "Aisle"], ["Window", "Aisle"]),
    (["Window", "Aisle", "No preference"], ["Window", "Aisle", "No preference"]),
    ([" Yes ", "Not now"], ["Yes", "Not now"]),
    (["a" * 24, "Go ahead"], ["a" * 24, "Go ahead"]),
    (["Ship it today", "Wait a day"], ["Ship it today", "Wait a day"]),
    (["9:30", "10:00"], ["9:30", "10:00"]),
    (["Not  now", "Go\tahead"], ["Not now", "Go ahead"]),
    (["~5 min", "<5 min", "A | B"], ["~5 min", "<5 min", "A | B"]),
    (["#42", "snake_case", "Q&A"], ["#42", "snake_case", "Q&A"]),
    (["C++", "Yes (both)", "A/B"], ["C++", "Yes (both)", "A/B"]),
    (["Sim", "N\u00e3o", "Talvez"], ["Sim", "N\u00e3o", "Talvez"]),
])
def test_two_or_three_valid_replies_succeed_trimmed(replies, shown):
    assert _offer(replies) == {"success": True, "replies": shown}


def test_a_hidden_character_is_not_echoed_back():
    answer = _offer(["Window", "Yes\u202e"])

    assert answer["error_code"] == "reply_hidden" and "\u202e" not in answer["error"]


# --------------------------------------------------------------------------- the language


def test_the_sentence_is_the_turns_language_from_the_caller_context():
    from modules.tools.discovery.handlers_replies import STRINGS

    english = _offer(["Window"])
    portuguese = _offer(["Window"], {**CHAT, "locale": "pt-BR"})

    assert english["error"] == STRINGS["en"]["reply_count"].format(low=2, high=3, words=3, limit=24, count=1)
    assert portuguese["error"] == STRINGS["pt-BR"]["reply_count"].format(low=2, high=3, words=3, limit=24, count=1)
    assert english["error"] != portuguese["error"]


def test_a_language_the_model_sends_is_dropped():
    from modules.tools.discovery.follow_up_context import FOLLOW_UP_PARAM, follow_up_params
    from modules.tools.discovery.handlers_replies import STRINGS, offer_replies

    forged = {"replies": ["Window"], FOLLOW_UP_PARAM: {"user_id": 9, "locale": "pt-BR"}}
    params = follow_up_params(OFFER, forged, CHAT)
    answer = asyncio.run(offer_replies(MagicMock(spec=Session), uuid.uuid4(), params))

    assert params[FOLLOW_UP_PARAM] == {"user_id": 7, "chat_id": "c-7", "locale": None}
    assert answer["error"] == STRINGS["en"]["reply_count"].format(low=2, high=3, words=3, limit=24, count=1)


def test_replies_sent_as_json_text_are_read():
    from modules.tools.discovery.follow_up_context import follow_up_params
    from modules.tools.discovery.handlers_replies import offer_replies

    params = follow_up_params(OFFER, json.dumps({"replies": ["Yes", "No"]}), CHAT)

    assert asyncio.run(offer_replies(None, uuid.uuid4(), params)) == {"success": True, "replies": ["Yes", "No"]}
    assert _offer(json.dumps(["Yes", "No"])) == {"success": True, "replies": ["Yes", "No"]}


def test_every_sentence_is_in_english_and_portuguese_and_every_code_has_one():
    from modules.tools.discovery import handlers_replies as replies

    codes = {replies.NOT_A_LIST, replies.COUNT, replies.NOT_TEXT, replies.EMPTY, replies.LINE_BREAK,
             replies.HIDDEN, replies.MARKDOWN, replies.TOO_LONG, replies.WORDS, replies.DUPLICATE}

    assert set(replies.STRINGS) == {"en", "pt-BR"}
    assert set(replies.STRINGS["en"]) == set(replies.STRINGS["pt-BR"]) == codes


# --------------------------------------------------------------------------- it writes nothing


@pytest.mark.parametrize("replies", [["Window", "Aisle"], ["Window"], "Window"])
def test_it_touches_no_session(replies):
    from modules.tools.discovery.handlers_replies import offer_replies

    db = MagicMock(spec=Session)
    asyncio.run(offer_replies(db, uuid.uuid4(), {"replies": replies}))

    assert db.mock_calls == []


def test_the_read_actions_sweep_covers_it_with_no_exception():
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS
    from tests.test_prd268_read_actions_write_nothing import RELABELLED

    reads = {action.name for action in get_action_registry().get_by_permission("read")}

    assert OFFER in reads and OFFER in PLATFORM_HANDLERS and OFFER not in RELABELLED


# --------------------------------------------------------------------------- registered, and chat only


def test_it_is_registered_read_chat_and_promoted_with_no_examples_and_wired():
    from modules.tools.discovery import handlers_replies
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    action = get_action_registry().get(OFFER)

    assert (action.permission_level, action.category, action.promoted, action.examples) == ("read", "chat", True, [])
    assert action.parameters["required"] == ["replies"]
    assert action.parameters["properties"]["replies"]["type"] == "array"
    assert (action.parameters["properties"]["replies"]["minItems"],
            action.parameters["properties"]["replies"]["maxItems"]) == (2, 3)
    assert PLATFORM_HANDLERS[OFFER] is handlers_replies.offer_replies
    assert OFFER in {promoted.name for promoted in get_action_registry().get_promoted()}


def test_its_description_says_last_only_to_choose_never_open_never_the_brief():
    from modules.tools.discovery.action_registry import get_action_registry

    description = get_action_registry().get(OFFER).description.lower()

    assert "last" in description and "choose" in description
    assert "never for an open question" in description and "never on the brief" in description


def test_the_check_in_surface_never_offers_it_and_the_lane_refuses_it():
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.check_in_lane import (
        CHAT_ONLY_ACTIONS, check_in_refusal, may_clear_in_a_check_in, offered_in_a_check_in,
    )
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    assert OFFER in CHAT_ONLY_ACTIONS
    assert not offered_in_a_check_in(get_action_registry().get(OFFER))
    assert not may_clear_in_a_check_in(OFFER)
    executor = PlatformActionExecutor(None, uuid.uuid4())
    executor._full_autonomy = lambda: False
    handler = AsyncMock(return_value={"success": True, "replies": ["Yes", "No"]})
    executor._handlers = {**executor._handlers, OFFER: handler}
    context = {"lane": "check_in", "driving_user_id": "7", "conversation_id": "c-7"}

    assert asyncio.run(executor.execute(OFFER, {"replies": ["Yes", "No"]}, context)) == check_in_refusal()
    handler.assert_not_called()
