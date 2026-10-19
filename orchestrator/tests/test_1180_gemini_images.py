"""#1180: an image in a message reaches Gemini as inline bytes.

The Gemini provider sent only a message's text blocks: an image was left out, and a
message of images alone was left out entirely. Images now go as Gemini inline-data
parts, read the way Bedrock reads them (``core/llm/clients/image_blocks.py``): an
OpenAI-style data URL, as ``modules/attachments/resolver.py`` builds for an attachment,
or an Anthropic-style base64 block. An image Gemini can't take, or one given only by
URL, is logged and left out. The SDK runs over an httpx mock transport, as in #1169.
"""
from __future__ import annotations

import base64
import json
import logging

import httpx
import pytest

from core.llm.clients import google_client
from core.llm.clients.base import LLMConfig, LLMProvider

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n fake png").decode()
JPEG = base64.b64encode(b"\xff\xd8\xff fake jpeg").decode()
ANSWER = {"candidates": [{"content": {"role": "model", "parts": [{"text": "A cat."}]}, "finishReason": "STOP"}],
          "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2, "totalTokenCount": 11}}


@pytest.fixture
def sent(monkeypatch):
    """The real SDK over a fake Gemini API; yields the request bodies it sent."""
    from google.genai import types

    bodies = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=ANSWER)

    real = google_client.genai.Client

    def client(**kwargs):
        options = kwargs.pop("http_options", None) or types.HttpOptions()
        transport = httpx.Client(transport=httpx.MockTransport(handle))
        return real(**kwargs, http_options=options.model_copy(update={"httpx_client": transport}))

    monkeypatch.setattr(google_client.genai, "Client", client)
    yield bodies


def _ask(*content):
    provider = google_client.GoogleProvider(LLMConfig(provider=LLMProvider.GOOGLE, model="gemini-2.5-flash",
                                                      api_key="key-a"))
    return provider.generate_response_sync([{"role": "user", "content": list(content)}])


def _parts(body):
    """The first turn's parts, with each image as (media type, bytes): the SDK writes
    inline data as URL-safe base64 under ``mime_type``."""
    def readable(part):
        if "inlineData" not in part:
            return part
        inline = part["inlineData"]
        return (inline.get("mime_type") or inline.get("mimeType"), base64.urlsafe_b64decode(inline["data"]))
    return [readable(part) for part in body["contents"][0]["parts"]]


def test_an_attachment_data_url_and_its_text_reach_gemini(sent):
    reply = _ask({"type": "text", "text": "What is this?"},
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}})
    assert reply.content == "A cat."
    assert _parts(sent[0]) == [{"text": "What is this?"}, ("image/png", base64.b64decode(PNG))]


def test_an_anthropic_style_image_and_an_image_alone_are_sent(sent):
    _ask({"type": "image", "source": {"type": "base64", "media_type": "image/jpg", "data": JPEG}})
    assert _parts(sent[0]) == [("image/jpeg", base64.b64decode(JPEG))]


def test_an_image_gemini_cant_take_or_only_has_a_url_for_is_left_out_and_logged(sent, caplog):
    gif = base64.b64encode(b"GIF89a fake").decode()
    with caplog.at_level(logging.WARNING, logger="core.llm.clients.google_client"):
        _ask({"type": "text", "text": "Compare these."},
             {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{gif}"}},
             {"type": "image_url", "image_url": {"url": "https://files.example/signed/cat.png?sig=x"}})
    assert _parts(sent[0]) == [{"text": "Compare these."}]
    assert "'image/gif' has no Gemini form" in caplog.text and "given only by URL" in caplog.text


@pytest.mark.parametrize("url", [
    f"data:IMAGE/PNG;BASE64,{PNG}",                                 # any case
    f"data:image/png;charset=utf-8;base64,{PNG}",                   # a parameter
    f"data:image/png;base64,{PNG[:8]}\n{PNG[8:]}",                  # MIME-wrapped base64
])
def test_a_data_url_written_another_valid_way_still_reaches_gemini(sent, url):
    _ask({"type": "image_url", "image_url": {"url": url}})
    assert _parts(sent[0]) == [("image/png", base64.b64decode(PNG))]


def test_a_malformed_image_block_is_left_out_not_a_failed_request(sent):
    _ask({"type": "text", "text": "Here."}, {"type": "image_url", "image_url": "https://files.example/cat.png"},
         {"type": "image", "source": "not-a-mapping"},
         {"type": "image", "source": {"type": "base64", "media_type": 7, "data": None}},   # fields that aren't text
         {"type": "image_url", "image_url": {"url": 42}},
         {"type": "image_url", "image_url": {"url": "data:image/gif;base64,!!!"}})        # a type Gemini skips, never decoded
    assert _parts(sent[0]) == [{"text": "Here."}]


def test_an_image_that_isnt_base64_fails_before_any_request(sent):
    with pytest.raises(ValueError, match="isn't valid base64"):
        _ask({"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}})
    assert sent == []
