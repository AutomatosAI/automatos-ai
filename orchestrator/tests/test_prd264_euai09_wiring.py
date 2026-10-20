"""PRD-264 EUAI-09 — where the AI marks are written, and what each surface says.

* **Documents.** ``DocumentGenerationService.generate`` with no ``user_id`` is an
  agent's call (the generate_document tool, a Playbook step, a Mission report): the
  file is marked after its last write and its SHA-256 recorded as a ``document``
  provenance row (EUAI-10). A person's own request (``user_id`` set) is not marked. A
  file that cannot be marked is deleted and the call fails: it is never delivered
  unmarked.
* **Images.** ``marked_images``: the model's PNG or JPEG is marked; one with a C2PA
  manifest passes as it came; only a type the mark is written for gets a row. An image
  a chat answer carries is saved marked (``marked_reply_images`` on
  ``service._upload_inline_images``) and recorded once the store has kept it.
* **Chat.** Every assistant message in the chat stream carries ``"ai_generated": true``
  on the frame that opens it (``chat-id``) and the frame that closes it (``finish``).
* **Blog API.** A post an agent wrote (``author_agent_id`` set) says
  ``ai_generated: true``, in the dashboard's API and the public widget's.
* **FR-17.** A provenance write that fails never logs the file's hash.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

WS = UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e64")
DATA = {"columns": ["Coffee", "Kg left"], "rows": [["Kirinyaga", 41], ["Guji Shakiso", 55.5]]}
TRAINED = b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _custom_xml(path: str) -> bytes:
    with zipfile.ZipFile(path) as package:
        return package.read("docProps/custom.xml") if "docProps/custom.xml" in package.namelist() else b""


# --------------------------------------------------------------------------- documents


@pytest.fixture
def generate(tmp_path, monkeypatch):
    """generate() for a spreadsheet, the real service and writer; the provenance rows it records."""
    import modules.documents.brand_logo as bl
    import modules.documents.generation_service as gs
    import services.content_provenance as content_provenance
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    monkeypatch.setattr(gs, "GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setattr(gs, "is_storage_configured", lambda: False)
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    recorded = []
    monkeypatch.setattr(content_provenance, "record_produced_file", lambda *row: recorded.append(row))

    def run(**who):
        db = NS(get=lambda model, key: NS(settings={}), query=lambda *a, **k: None)
        service = gs.DocumentGenerationService(db, WS)
        return asyncio.run(service.generate(title="Green coffee stock", format="xlsx", data=dict(DATA),
                                            workspace_id=WS, **who))

    yield NS(run=run, recorded=recorded, dir=tmp_path / "generated")
    forget_cached_kits()


def test_an_agents_document_is_marked_and_recorded(generate):
    result = generate.run()

    custom = _custom_xml(result.path)
    assert b'name="AIGenerated"' in custom and b"compositeWithTrainedAlgorithmicMedia" in custom
    assert b'name="AIInstanceID"' in custom
    digest = hashlib.sha256(Path(result.path).read_bytes()).hexdigest()
    assert generate.recorded == [(WS, digest, "document", "Automatos AI document generation")]
    assert result.size == Path(result.path).stat().st_size  # the result describes the marked file


def test_a_persons_own_document_is_not_marked(generate):
    result = generate.run(user_id=7)

    assert b"AIGenerated" not in _custom_xml(result.path)
    assert generate.recorded == []


def test_a_document_that_cannot_be_marked_is_never_delivered(generate, monkeypatch):
    import modules.documents.marked_documents as marked_documents

    def refuse(path, fmt, term):
        raise ValueError("the package keeps its custom properties outside docProps/custom.xml")

    monkeypatch.setattr(marked_documents, "mark_document", refuse)
    with pytest.raises(ValueError):
        generate.run()
    assert list(generate.dir.rglob("*.xlsx")) == []
    assert generate.recorded == []


def test_the_service_names_the_two_marking_steps():
    """The decorators sit on generate (who asked) and on _build_result (after the last write)."""
    import modules.documents.generation_service as gs

    source = Path(gs.__file__).read_text()
    assert "@ai_documents_are_marked" in source and "@marked_before_it_is_kept" in source
    assert source.index("@marked_before_it_is_kept") < source.index("def _build_result(")


# --------------------------------------------------------------------------- images


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "#e96235").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def rows(monkeypatch):
    import services.content_provenance as content_provenance

    recorded = []
    monkeypatch.setattr(content_provenance, "record_produced_file", lambda *row: recorded.append(row))
    return recorded


def test_a_model_image_is_marked_then_recorded_by_the_hash_of_what_was_kept(rows):
    from modules.documents.marked_images import marked_model_image, record_model_image

    mime, b64 = marked_model_image("image/png", base64.b64encode(_png()).decode("ascii"))
    kept = base64.b64decode(b64)
    assert mime == "image/png" and TRAINED in kept

    assert record_model_image(WS, mime, b64) is True
    assert rows == [(WS, hashlib.sha256(kept).hexdigest(), "image", "Automatos AI image generation")]


def test_a_model_image_with_a_c2pa_manifest_passes_as_it_came():
    import struct
    import zlib

    from modules.documents.marked_images import marked_model_image

    data = _png()
    body = b"\x00\x00\x00\x10jumbc2pa-manifest"
    chunk = struct.pack(">I", len(body)) + b"caBX" + body + struct.pack(">I", zlib.crc32(b"caBX" + body) & 0xFFFFFFFF)
    signed = base64.b64encode(data[:33] + chunk + data[33:]).decode("ascii")
    assert marked_model_image("image/png", signed) == ("image/png", signed)


def test_a_type_with_no_mark_gets_no_row_and_bad_base64_is_refused(rows):
    from modules.documents.marked_images import marked_model_image, record_model_image

    webp = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPVP8 ").decode("ascii")
    assert marked_model_image("image/webp", webp) == ("image/webp", webp)
    assert record_model_image(WS, "image/webp", webp) is False and rows == []
    with pytest.raises(ValueError):
        marked_model_image("image/png", "not base64!")


def _upload(seen):
    """The upload's stand-in: it keeps each PNG data URL and leaves a link in its place."""
    async def upload(text, workspace_id=None):
        seen.append((text, workspace_id))
        return re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "/api/generated-images/i-1", text)
    return upload


def test_an_image_in_a_chat_answer_is_saved_marked_and_recorded(rows):
    from consumers.chatbot.marked_reply_images import reply_images_are_marked

    seen = []
    b64 = base64.b64encode(_png()).decode("ascii")
    answer = f"Here is the logo:\n\n![logo](data:image/png;base64,{b64})"
    saved = asyncio.run(reply_images_are_marked(_upload(seen))(answer, workspace_id=str(WS)))

    assert saved == "Here is the logo:\n\n![logo](/api/generated-images/i-1)"
    ((sent, workspace),) = seen
    kept = base64.b64decode(re.search(r"base64,([A-Za-z0-9+/=]+)", sent).group(1))
    assert workspace == str(WS) and TRAINED in kept
    assert rows == [(str(WS), hashlib.sha256(kept).hexdigest(), "image", "Automatos AI image generation")]


def test_a_chat_answer_without_an_image_or_whose_image_was_not_kept_records_nothing(rows):
    from consumers.chatbot.marked_reply_images import reply_images_are_marked

    async def keeps_nothing(text, workspace_id=None):
        return text

    plain = asyncio.run(reply_images_are_marked(keeps_nothing)("No picture here.", workspace_id=str(WS)))
    b64 = base64.b64encode(_png()).decode("ascii")
    inline = asyncio.run(reply_images_are_marked(keeps_nothing)(f"![x](data:image/png;base64,{b64})", str(WS)))
    assert plain == "No picture here." and TRAINED in base64.b64decode(inline.split("base64,")[1][:-1])
    assert rows == []


def test_the_chat_service_saves_reply_images_through_the_mark():
    import consumers.chatbot.service as svc
    from consumers.chatbot.marked_reply_images import reply_images_are_marked

    async def anything(*_a, **_k):
        return None

    assert svc._upload_inline_images.__wrapped__.__code__ is reply_images_are_marked(anything).__code__


# --------------------------------------------------------------------------- chat


def _frame(line: str) -> dict:
    assert line.startswith("d:") and line.endswith("\n")
    return json.loads(line[2:])


def test_every_frame_that_opens_or_closes_an_assistant_message_says_ai_generated():
    from consumers.chatbot.streaming import AI_GENERATED_FIELD, StreamingHandler

    handler = StreamingHandler()
    assert AI_GENERATED_FIELD == "ai_generated"
    assert _frame(handler.format_aisdk_chat_id("c-1")) == {"type": "chat-id", "chatId": "c-1", "ai_generated": True}
    assert _frame(handler.format_aisdk_finish()) == {"type": "finish", "finishReason": "stop", "ai_generated": True}
    assert _frame(handler.format_done_with_metadata(stop_reason="end_turn"))["ai_generated"] is True


def test_one_sse_message_read_back_as_the_web_chat_reads_it():
    """A whole assistant message: its opening frame, its text, its closing frame."""
    from consumers.chatbot.streaming import StreamingHandler

    handler = StreamingHandler()

    async def message():
        frames = [handler.format_aisdk_chat_id("c-2")]
        frames += [chunk async for chunk in handler.stream_text_aisdk("Your invoice is ready to send.")]
        return [*frames, handler.format_aisdk_finish()]

    frames = asyncio.run(message())
    text = "".join(json.loads(f[2:]) for f in frames if f.startswith("0:"))
    data = [_frame(f) for f in frames if f.startswith("d:")]
    assert text == "Your invoice is ready to send."
    assert [d["type"] for d in data] == ["chat-id", "finish"]
    assert all(d["ai_generated"] is True for d in data)


# --------------------------------------------------------------------------- blog API


def _post(author_agent_id):
    return NS(id=1, status="published", author_agent_id=author_agent_id,
              to_dict=lambda include_content=False: {"slug": "harbour-blend", "author_agent_id": author_agent_id})


def test_a_post_an_agent_wrote_is_ai_generated_and_a_persons_is_not():
    from core.services.blog_post_view import is_ai_generated, post_view

    agent_post, person_post = _post(12), _post(None)
    assert post_view(agent_post) == {"slug": "harbour-blend", "author_agent_id": 12, "ai_generated": True}
    assert post_view(person_post)["ai_generated"] is False
    assert is_ai_generated(agent_post) and not is_ai_generated(person_post)


def test_the_public_widget_says_which_posts_an_agent_wrote(monkeypatch):
    import api.widgets.blog as widget_blog

    class Posts:
        def __init__(self, db, workspace_id):
            pass

        def list_posts(self, **filters):
            return {"posts": [_post(12), _post(None)], "total": 2, "page": 1, "per_page": 10, "total_pages": 1}

        def get_post_by_slug(self, slug):
            return _post(12)

        def increment_views(self, post_id):
            return None

        async def get_content(self, post):
            return "Roasted on Thursday."

    monkeypatch.setattr(widget_blog, "BlogService", Posts)
    listed = asyncio.run(widget_blog.list_published_posts(page=1, per_page=10, category=None, tag=None,
                                                          workspace_id=WS, db=object()))
    assert [p["ai_generated"] for p in json.loads(listed.body)["posts"]] == [True, False]
    one = asyncio.run(widget_blog.get_published_post("harbour-blend", workspace_id=WS, db=object()))
    assert json.loads(one.body)["ai_generated"] is True


def test_the_dashboard_api_says_it_too(monkeypatch):
    import api.blog as blog

    class Posts:
        def __init__(self, db, workspace_id):
            pass

        def get_post(self, post_id):
            return _post(12)

        async def get_content(self, post):
            return "Roasted on Thursday."

    monkeypatch.setattr(blog, "BlogService", Posts)
    data = asyncio.run(blog.get_post(UUID(int=1), db=object(), ctx=NS(workspace_id=WS)))
    assert data["ai_generated"] is True and data["content"] == "Roasted on Thursday."


# --------------------------------------------------------------------------- FR-17


def test_a_failed_provenance_write_never_logs_the_hash(monkeypatch, caplog):
    import core.database.database as database
    from sqlalchemy.exc import IntegrityError

    import services.content_provenance as content_provenance

    session = NS(rolled_back=False, closed=False)

    def execute(statement):
        raise IntegrityError("INSERT INTO content_provenance ...", {"sha256": SHA}, Exception("fk violation"))

    session.execute = execute
    session.commit = lambda: None
    session.rollback = lambda: setattr(session, "rolled_back", True)
    session.close = lambda: setattr(session, "closed", True)
    monkeypatch.setattr(database, "SessionLocal", lambda: session)

    with caplog.at_level(logging.DEBUG):
        assert content_provenance.record_produced_file(WS, SHA, "document", "Automatos AI document generation") is False
    assert session.rolled_back and session.closed
    assert "IntegrityError" in caplog.text and str(WS) in caplog.text
    assert SHA not in caplog.text
