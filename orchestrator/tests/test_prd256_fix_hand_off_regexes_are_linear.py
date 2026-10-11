"""The hand-off lane reads owner text in linear time (CodeQL #283, #284 on PR #1048).

The id reply ("267, the operations one") and the receiver ("Jim to handle by Friday") were regexes
whose tails were polynomial on runs of spaces: 20,000 spaces took 2.0 s and 14.7 s (11 Oct). The id is
now an anchored head plus code, the receiver a word loop; they read every reply as the patterns did.
"""
import time

import pytest

from consumers.chatbot.addressed_agents import match_id_reply
from consumers.chatbot.handoffs import receiver_name

SPACES = " " * 20_000
FAST_S = 0.5


@pytest.mark.parametrize("reply, agent, agent_id, rest", [
    ("267", None, "267", None), ("agent 267", "agent ", "267", None), ("267.", None, "267", None),
    ("267!  ", None, "267", None), ("  267  ", None, "267", None), ("267\n", None, "267", None),
    ("267, the operations one", None, "267", "the operations one"),
    ("267 — the shop floor one", None, "267", "the shop floor one"), ("267 (ops)", None, "267", "ops)"),
    ("267,\nops", None, "267", "ops"), ("267, ops\n", None, "267", "ops"),
])
def test_an_id_reply_is_read_as_before(reply, agent, agent_id, rest):
    found = match_id_reply(reply)
    assert found and (found.agent, found.id, found.rest) == (agent, agent_id, rest)


@pytest.mark.parametrize("message", ["267 is my favourite", "267..", "267 x", "267, ops\nmore", "1234567890", "x267"])
def test_a_message_that_only_opens_with_a_number_is_no_id_reply(message):
    assert match_id_reply(message) is None


@pytest.mark.parametrize("words, receiver", [
    ("OPS to do it", "OPS"), ("OPS   to do it", "OPS"), ("the OPS team please", "the OPS team"),
    ("WRITER for the newsletter now", "WRITER"), ("CHRISTMAS BOX", "CHRISTMAS BOX"),
    ("OPS to,", "OPS"), ("OPS to-do", "OPS"), ("Jim Tomorrow", "Jim Tomorrow"), ("Tony", "Tony"),
])
def test_the_receiver_is_read_as_before(words, receiver):
    assert receiver_name(words) == receiver


def test_a_long_run_of_spaces_is_read_in_linear_time():
    started = time.perf_counter()
    assert match_id_reply("1" + SPACES + "x") is None
    assert receiver_name("OPS" + SPACES + "x") == "OPS x"
    assert time.perf_counter() - started < FAST_S
