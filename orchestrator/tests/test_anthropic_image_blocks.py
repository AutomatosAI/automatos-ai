"""A photo sent from the app reaches Anthropic as an Anthropic image block (23 Oct 2026).

The Anthropic client sent a message's content list as it came, so an attachment's
OpenAI-style ``image_url`` part (``modules/attachments/resolver.py``) reached the Messages
API, which answered 400: "messages.0.content.1: Input tag 'image_url' found using 'type'
does not match any of the expected tags", and the owner saw "The AI provider couldn't
finish this reply". Images now go as Anthropic image blocks, read the way Bedrock (#1172)
and Gemini (#1180) read them (``core/llm/clients/image_blocks.py``): a data URL as base64,
an https link as a URL source. An image the API can't take stands as a text note, logged,
and the request still goes. A blank text block (a photo sent with no caption) is left out:
the API answers 400 to it. The real SDK runs over an httpx mock transport, as in #1178,
so the tests read the body that leaves the process.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging

import httpx
import pytest

from core.llm.clients.anthropic_client import AnthropicProvider
from core.llm.clients.base import LLMConfig, LLMProvider

JPEG = base64.b64encode(b"\xff\xd8\xff fake jpeg").decode()
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n fake png").decode()
HEIC = base64.b64encode(b"\x00\x00\x00\x18ftypheic fake").decode()
SIGNED = "https://attachments.example/ws/ephemeral-attachments/a1/photo.jpg?X-Amz-Signature=x"
ANSWER = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
          "content": [{"type": "text", "text": "A cat."}], "stop_reason": "end_turn",
          "usage": {"input_tokens": 9, "output_tokens": 2}}
LOGGER = "core.llm.clients.anthropic_client"
NOTE = "[IMAGE UNAVAILABLE]"


@pytest.fixture
def api():
    """A provider whose real SDK client talks to a fake Messages API: (provider, request bodies)."""
    import anthropic

    bodies = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=ANSWER)

    provider = AnthropicProvider.__new__(AnthropicProvider)  # bypass _initialize_client: no key lookup
    provider.config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-5-5", api_key="test-key")
    provider.client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                          http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    return provider, bodies


def _data_url(media_type: str, data: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def _jpeg_block(data: str = JPEG) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def _first_turn(api, *content) -> list:
    """The content of the first turn Anthropic received for one user message of ``content``."""
    provider, bodies = api
    assert provider.generate_response_sync([{"role": "user", "content": list(content)}]).content == "A cat."
    assert len(bodies) == 1
    return bodies[0]["messages"][0]["content"]


def test_the_attachment_photo_goes_as_a_base64_image_after_its_text(api):
    # What the resolver sends for every photo now (a large one made a JPEG first).
    assert _first_turn(api, {"type": "text", "text": "What is this?"}, _data_url("image/jpeg", JPEG)) == [
        {"type": "text", "text": "What is this?"}, _jpeg_block()]


@pytest.mark.parametrize("caption", ["", "  \n"])
def test_a_photo_sent_with_no_caption_goes_without_a_blank_text_block(api, caption):
    # The API answers 400 "text content blocks must be non-empty" to a blank one.
    assert _first_turn(api, {"type": "text", "text": caption}, _data_url("image/jpeg", JPEG)) == [_jpeg_block()]


def test_a_large_attachments_signed_link_goes_as_a_url_source_on_the_chat_path(api):
    # The 23 Oct report: the 1.9 MB photo went as its signed link, text first, the photo at content.1.
    provider, bodies = api
    reply = asyncio.run(provider.generate_response([
        {"role": "system", "content": "You are Auto."},
        {"role": "user", "content": [{"type": "text", "text": "And this one?"},
                                     {"type": "image_url", "image_url": {"url": SIGNED}}]}]))
    assert reply.content == "A cat."
    assert bodies[0]["messages"][0]["content"] == [
        {"type": "text", "text": "And this one?"}, {"type": "image", "source": {"type": "url", "url": SIGNED}}]


@pytest.mark.parametrize("block", [
    _data_url("image/jpg", JPEG),                                                 # the alias
    {"type": "image", "source": {"type": "base64", "media_type": "image/jpg", "data": JPEG}},
    {"type": "image_url", "image_url": {"url": f"data:IMAGE/JPEG;BASE64,{JPEG}"}},  # any case
    _data_url("image/jpeg", f"{JPEG[:8]}\n{JPEG[8:]}"),                           # MIME-wrapped base64
])
def test_an_image_written_another_valid_way_goes_as_clean_jpeg(api, block):
    assert _first_turn(api, block) == [_jpeg_block()]


@pytest.mark.parametrize("block, why", [
    (_data_url("image/heic", HEIC), "image/heic"),                                # an iPhone's own format
    (_data_url("image/svg+xml", PNG), "image/svg+xml"),
    (_data_url("image/png", "!!!"), "base64"),
    ({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "!!!"}}, "base64"),
    ({"type": "image_url", "image_url": {"url": "http://localhost:9000/attachments/photo.jpg"}}, "https"),
    ({"type": "image_url", "image_url": "https://attachments.example/photo.jpg"}, "https"),  # malformed
    ({"type": "image", "source": {"type": "url", "url": "http://localhost:9000/attachments/photo.jpg"}}, "https"),
    ({"type": "image", "source": "not-a-mapping"}, "source"),
])
def test_an_image_anthropic_cant_take_is_a_note_and_the_reply_still_comes(api, caplog, block, why):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        content = _first_turn(api, {"type": "text", "text": "Compare."}, block)
    text, note = content
    assert text == {"type": "text", "text": "Compare."}
    assert note["type"] == "text" and note["text"].startswith(NOTE) and why in note["text"]
    assert "[anthropic]" in caplog.text and "a note stands in for it" in caplog.text


def test_a_message_of_one_unsendable_image_still_has_content(api):
    [note] = _first_turn(api, _data_url("image/heic", HEIC))
    assert note["type"] == "text" and note["text"].startswith(NOTE)


@pytest.mark.parametrize("typed", ['[{"sku": "A1", "qty": 2}]', '["red", "blue"]', "[1, 2, 3]", "[]"])
def test_a_json_array_a_user_typed_goes_as_their_text(api, typed):
    provider, bodies = api
    provider.generate_response_sync([{"role": "user", "content": typed}])
    assert bodies[0]["messages"] == [{"role": "user", "content": typed}]


def test_text_tool_and_reference_blocks_go_as_they_came(api):
    provider, bodies = api
    referenced = [{"type": "text", "text": "Both of these."},
                  {"type": "image", "source": {"type": "url", "url": SIGNED}},
                  {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
                  {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "notes"}}]
    legacy_results = json.dumps([{"type": "tool_result", "tool_use_id": "t0", "content": "done"}])
    provider.generate_response_sync([
        {"role": "user", "content": referenced},
        {"role": "assistant", "content": "Looking.", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "search", "arguments": '{"q": "cats"}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "3 cats"},
        {"role": "user", "content": legacy_results},
        {"role": "user", "content": "Thanks"},
    ])
    assert bodies[0]["messages"] == [
        {"role": "user", "content": referenced},
        {"role": "assistant", "content": [{"type": "text", "text": "Looking."},
                                          {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "cats"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "3 cats"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "done"}]},
        {"role": "user", "content": "Thanks"},
    ]


def test_the_rest_of_the_history_goes_as_it_did_before_the_images_change(api):
    """Every path the converter's rewrite touched, pinned: parallel tool results join one
    user turn in order, arguments that aren't a JSON object go as {}, an assistant with
    nothing to say and an unknown role send nothing, and a non-string content is its text."""
    provider, bodies = api
    provider.generate_response_sync([
        {"role": "user", "content": "Find two things."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "search", "arguments": '{"q": "a"}'}},
            {"id": "t2", "type": "function", "function": {"name": "search", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "found a"},
        {"role": "tool", "tool_call_id": "t2", "content": {"found": "b"}},
        {"role": "assistant", "content": None},
        {"role": "developer", "content": "never sent"},
        {"role": "assistant", "content": "Both found."},
        {"role": "user", "content": 42},
    ])
    assert bodies[0]["messages"] == [
        {"role": "user", "content": "Find two things."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "a"}},
                                          {"type": "tool_use", "id": "t2", "name": "search", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "found a"},
                                     {"type": "tool_result", "tool_use_id": "t2", "content": '{"found": "b"}'}]},
        {"role": "assistant", "content": "Both found."},
        {"role": "user", "content": "42"},
    ]


def test_a_message_of_blank_text_alone_is_not_sent(api):
    provider, bodies = api
    provider.generate_response_sync([
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        {"role": "user", "content": [{"type": "text", "text": " "}]},
        {"role": "user", "content": [{"type": "text", "text": "Still there?"}]},
    ])
    assert bodies[0]["messages"] == [
        {"role": "user", "content": "Hi."},
        {"role": "user", "content": [{"type": "text", "text": "Still there?"}]},
    ]


HISTORY = [
    {"role": "system", "content": "You are Auto."},
    {"role": "user", "content": [{"type": "text", "text": "Image 1:"}, _data_url("image/png", PNG),
                                 {"type": "text", "text": "Image 2:"}, {"type": "image_url", "image_url": {"url": SIGNED}}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Two cats."}, _data_url("image/jpeg", JPEG)]},
    {"role": "assistant", "content": [_data_url("image/jpeg", JPEG)], "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "t1", "content": "none"},
    {"role": "user", "content": [_data_url("image/heic", HEIC), {"type": "text", "text": "And now?"}]},
]


def _without_image_url(value) -> bool:
    return "image_url" not in json.dumps(value, default=str)


@pytest.mark.parametrize("headless", [False, True])
def test_the_full_request_carries_no_image_url_anywhere(api, headless):
    provider, _ = api
    system, turns = provider._convert_messages_to_anthropic_format(HISTORY)
    kwargs = provider._request_kwargs(system, turns, None, None, headless)
    assert _without_image_url(kwargs)
    assert kwargs["messages"][0]["content"] == [
        {"type": "text", "text": "Image 1:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        {"type": "text", "text": "Image 2:"},
        {"type": "image", "source": {"type": "url", "url": SIGNED}}]       # each image in its place


def test_no_image_url_reaches_anthropic_by_either_call(api):
    provider, bodies = api
    provider.generate_response_sync(HISTORY)
    asyncio.run(provider.generate_response(HISTORY))
    assert len(bodies) == 2 and all(_without_image_url(body) for body in bodies)
    assert bodies[0]["messages"] == bodies[1]["messages"]
    assert bodies[0]["messages"][-1]["content"][0]["text"].startswith(NOTE)
