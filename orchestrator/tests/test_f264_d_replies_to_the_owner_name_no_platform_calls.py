"""F264 (night 8): Auto's replies to the owner never carry the platform's names.

About 60 of night 8's replies named the platform's calls or parameters ("You can find
a list of social posts using `platform_list_social_posts`", "The tool expects
`agent_name` instead of `assigned_agent_name`", "it doesn't accept
`send_back_reason`"), after calls that worked as well as after refusals, and in
the words Auto says before its calls. What Auto streams in a chat turn is now said
in the owner's words, live and in the saved answer; the calls are untouched.
"""
from __future__ import annotations

import asyncio
import pytest

from core.llm.clients.base import LLMResponse
from core.llm.owner_words_stream import OwnerWords, in_owner_words, in_plain_words, offered_names
from core.llm.usage_context import LANE_BOARD_TASK, LANE_CHAT, usage_scope

NAMES = {"platform_list_social_posts", "list_social_posts", "agent_name", "assigned_agent_name",
         "platform_update_task_status", "update_task_status", "review_mode"}


@pytest.mark.parametrize("said, owner, plain", [
    ("You can find a list of social posts using `platform_list_social_posts`.", "Approve #0201 with this note.",
     "You can find a list of social posts using list social posts."),                              # 23:41:31
    ("The tool expects `agent_name` instead of `assigned_agent_name`.", "Give #0207 to the Analyst.",
     "The tool expects agent name instead of assigned agent name."),                               # 00:10:30
    ("It seems it doesn't accept `send_back_reason` as a parameter.", "Send it back with my words.",
     "It seems it doesn't accept send back reason as a parameter."),        # a name Auto made up (01:34:25)
    ("I set review_mode to human, so it waits for you.", "Set review_mode to human on it.",
     "I set review_mode to human, so it waits for you."),                                       # the owner's word
    ("Your file harbourline_brand_voice.md is ready.", "Upload it.", "Your file harbourline_brand_voice.md is ready."),
    ("Here is the query:\n```sql\nSELECT agent_name FROM agents\n```\nIt lists them.", "How do I list them?",
     "Here is the query:\n```sql\nSELECT agent_name FROM agents\n```\nIt lists them."),          # code as written
])
def test_a_reply_says_the_platforms_names_in_plain_words(said, owner, plain):
    words = OwnerWords(NAMES, owner)
    assert in_plain_words(said, words.names, words.owners) == plain


def test_a_name_split_across_deltas_is_still_found():
    words = OwnerWords(NAMES, "Approve #0201.")
    shown = [words.feed("You can use `platform_li"), words.feed("st_social_posts` to see th"), words.feed("em."),
             words.flush()]
    assert "".join(shown) == "You can use list social posts to see them."
    assert "platform" not in shown[0]                                     # held back until the name was whole


def test_the_registry_names_the_calls_night_8_named():
    names = offered_names(None)
    assert {"platform_list_social_posts", "platform_update_task_status", "send_back"} <= names


def _stream(*deltas):
    async def stream(messages, tools, on_delta=None):
        for delta in deltas:
            await on_delta("text", delta)
        return LLMResponse(content="".join(deltas), streamed=True)
    return stream


def _said(lane):
    seen = []

    async def on_delta(kind, text):
        seen.append(text)

    stream = _stream("I made a mistake in the `platform_upd", "ate_task_status` call. It is fixed now.")
    messages = [{"role": "user", "content": "Approve #0231 with my note."}]
    with usage_scope(request_type=lane, execution_id=f"{lane}:1"):
        response = asyncio.run(in_owner_words(stream, messages, None, on_delta))
    return "".join(seen), response.content


def test_autos_streamed_reply_and_its_saved_answer_read_the_same():
    shown, saved = _said(LANE_CHAT)
    assert shown == saved == "I made a mistake in the update task status call. It is fixed now."


def test_an_agent_runs_stream_is_left_as_it_comes():
    shown, saved = _said(LANE_BOARD_TASK)
    assert "`platform_update_task_status`" in shown and "`platform_update_task_status`" in saved


def test_the_chat_streams_through_it():
    import inspect

    from core.llm import manager

    assert "await in_owner_words(stream, messages, tools, on_delta)" in inspect.getsource(
        manager.LLMManager.generate_response)
