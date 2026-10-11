"""The hand-off lane's two owner-text patterns run in linear time (CodeQL #283, #284 on PR #1048).

ID_REPLY ("267, the operations one") had \\s*[.!]?\\s* before the end, and RECEIVER_END was retried
by re.sub at every space of a run: a message with a long run of spaces took seconds (20,000 spaces:
2.0 s and 14.7 s on 11 Oct). The patterns keep what they match; these tests pin both.
"""
import time

import pytest

from consumers.chatbot.addressed_agents import ID_REPLY
from consumers.chatbot.handoffs import RECEIVER_END

SPACES = " " * 20_000
FAST_S = 0.5


@pytest.mark.parametrize("reply, agent_id, rest", [
    ("267", "267", None), ("agent 267", "267", None), ("267.", "267", None), ("267!  ", "267", None),
    ("  267  ", "267", None), ("267, the operations one", "267", "the operations one"),
    ("267 — the shop floor one", "267", "the shop floor one"), ("267 (ops)", "267", "ops)"),
])
def test_an_id_reply_is_read_as_before(reply, agent_id, rest):
    match = ID_REPLY.match(reply)
    assert match and match.group("id") == agent_id and match.group("rest") == rest


@pytest.mark.parametrize("message", ["267 is my favourite", "267..", "267 x"])
def test_a_message_that_only_opens_with_a_number_is_no_id_reply(message):
    assert ID_REPLY.match(message) is None


@pytest.mark.parametrize("words, receiver", [
    ("OPS to do it", "OPS"), ("OPS   to do it", "OPS"), ("the OPS team please", "the OPS team"),
    ("WRITER for the newsletter now", "WRITER"), ("CHRISTMAS BOX", "CHRISTMAS BOX"),
])
def test_the_receiver_is_trimmed_as_before(words, receiver):
    assert RECEIVER_END.sub("", words) == receiver


def test_a_long_run_of_spaces_is_read_in_linear_time():
    started = time.perf_counter()
    assert ID_REPLY.match("1" + SPACES + "x") is None
    assert RECEIVER_END.sub("", "OPS" + SPACES + "x") == "OPS" + SPACES + "x"
    assert time.perf_counter() - started < FAST_S
