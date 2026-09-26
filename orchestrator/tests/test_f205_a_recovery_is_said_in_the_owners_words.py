"""F205 (night 6) — after a failed call, Auto tells the owner what they can do in
their words, never the platform's tool or parameter names.

From 04:46 to 06:04 Auto's replies read "I missed a required parameter for the
create_blog_post tool" and "the document_id needs to be an integer, not the
filename". The owner never sees those names and can do nothing with them.
"""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace as NS

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
TOOLS = [{"type": "function", "function": {"name": "platform_execute", "parameters": {
    "type": "object", "properties": {"action": {"type": "string"}, "params": {"type": "object"}}}}}]
NAMED = "I missed a required parameter for the create_blog_post tool. What title would you like?"
OWNER_WORDS = "I couldn't save the blog post yet: it needs a title. What would you like to call it?"


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


@pytest.mark.parametrize("said, owner, named", [
    (NAMED, "Write a blog post about how we pick the club coffee.", ["create_blog_post"]),
    ("It looks like the document_id needs to be an integer, not the filename.", "Read the orders file.",
     ["document_id"]),
    ("I set review_mode to human, so it waits for you.", "Set review_mode to human on it.", []),   # the owner's word
    ("Your file harbourline_brand_voice.md is ready.", "Upload it.", []),                        # not the platform's
], ids=["tool", "parameter", "the-owners-own", "a-file"])
def test_the_platforms_names_are_found_and_the_owners_are_not(said, owner, named):
    from consumers.chatbot.owner_words import internal_names, internal_vocabulary

    assert internal_names(said, internal_vocabulary(TOOLS), owner) == named


def _call(action):
    return {"id": "call_1", "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": action, "params": {}})}}


def _round(text, calls=None):
    return NS(content=text, tool_calls=calls, usage=None, streamed=bool(text), reasoning=None,
              finish_reason="tool_calls" if calls else "stop")


class _Model:
    def __init__(self, *texts):
        self.texts, self.sent, self.tools = list(texts), [], []

    async def generate_response(self, messages, tools=None, on_delta=None):
        self.sent.append(copy.deepcopy(messages))
        self.tools.append(tools)
        text = self.texts.pop(0)
        if on_delta is not None:
            await on_delta("text", text)
        return _round(text)


class _Router:
    def __init__(self, success):
        self.success = success

    async def execute_and_format(self, tool_name, tool_args, **kwargs):
        if self.success:
            return {"success": True, "llm_context": "Saved.", "raw_result": {"success": True}}
        return {"success": False, "llm_context": "Missing required parameter: title",
                "raw_result": {"success": False, "error": "Missing required parameter: title"}}


def _turn(model, *, success):
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.streaming import get_streaming_handler

    svc = StreamingChatService.__new__(StreamingChatService)
    svc.db, svc.workspace_id, svc.widget_mode = None, WS, False
    svc.streaming_handler, svc.tool_router = get_streaming_handler(), _Router(success)
    svc._release_db_dial, svc._turn_document_ids, svc._turn_chunk_ids = False, set(), set()
    runtime = NS(llm_manager=model, agent_id=322, workspace_id=WS, metadata=NS(name="Auto"))
    messages = [{"role": "system", "content": "You are Auto."},
                {"role": "user", "content": "Write a blog post about how we pick the club coffee."}]
    first = _round("", [_call("platform_create_blog_post")])

    async def run():
        final = None
        async for chunk in svc._stream_tool_loop(first, messages, runtime, {}, TOOLS, streamed_rounds=[first],
                                                 reasoning_log=[]):
            if isinstance(chunk, dict) and chunk.get("_final_response"):
                final = chunk["_final_response"]
        return final
    return asyncio.run(run())


def test_a_reply_after_a_failed_call_is_asked_once_for_the_owners_words():
    model = _Model(NAMED, OWNER_WORDS)
    final = _turn(model, success=False)

    assert final.content == OWNER_WORDS
    assert model.sent[-1][-1]["content"].startswith("Your reply names create_blog_post")
    assert model.tools[-1] is None                                  # a rewording, not another try


def test_a_reply_after_calls_that_worked_is_left_as_it_is():
    model = _Model("Saved the create_blog_post draft for you.")
    final = _turn(model, success=True)
    assert final.content == "Saved the create_blog_post draft for you." and len(model.sent) == 1
