"""PRD-251B Wave 3, US-B302 and US-B303 — style references and the style profile.

Pinned:

* **References** (``/api/documents/brand-kit/references``): a PNG, JPEG or WebP is stored
  like the logo, listed with its note, stance and image route (never its storage path),
  edited, served and removed. Each refusal answers with its own status: 415 for anything
  that is not one of the three (read from the bytes, whatever its name or declared type),
  413 over the size, 422 over the count, for a long note and for an unknown stance.
  Another workspace's reference is a 404 everywhere; writes need ``workspace:manage``.
* **The profile**, with the model mocked: the read sends the liked references first,
  newest first, then the avoided ones, as images with their notes; the request goes
  through the LLM manager as ``brand_style_read``; an answer that is not a profile is a
  502 and a slow one a 504; any change to the references reads it again in the background.
* **What carries it**: the composer's material, the plan's research tool and the still
  image tool (Template Studio's and Socials' stills) all get the profile as one paragraph.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.llm  # noqa: E402
import modules.documents.brand_logo as bl  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.documents import brand_references as refs  # noqa: E402
from modules.documents import brand_style  # noqa: E402
from modules.socials import compose  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000003b1")
OTHER = uuid.UUID("00000000-0000-0000-0000-0000000003b2")
ROUTE = "/api/documents/brand-kit/references"
PROFILE = {"palette": ["#1F2A44", "#F4B400"], "mood": ["calm", "confident"], "composition": "Wide shots, one subject.",
           "avoid": "Busy collages."}
STYLE_TEXT = (
    "Brand style (from the brand kit's references): palette #1F2A44, #F4B400; mood: calm, confident; "
    "composition: Wide shots, one subject; avoid: Busy collages."
)


def png(size: int = 64) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * size


def jpeg(size: int = 64) -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * size


def webp(size: int = 64) -> bytes:
    return b"RIFF" + (size + 4).to_bytes(4, "little") + b"WEBP" + b"\x00" * size


GIF = b"GIF89a" + b"\x00" * 32
SVG = b"<svg xmlns='http://www.w3.org/2000/svg'/>"


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    return tmp_path


def _ctx(workspace_id):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id="member-1", clerk_user_id="clerk-member-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def api(storage, monkeypatch):
    """The documents router (the brand kit routes ride it) over two workspaces held in memory."""
    import api.document_generation as documents_module

    state = SimpleNamespace(role="owner", caller=WS, launched=[], storage=storage)
    state.workspaces = {ws: SimpleNamespace(id=ws, name=f"ws-{ws.hex[-3:]}", settings={}) for ws in (WS, OTHER)}
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    monkeypatch.setattr(brand_style, "launch_refresh", state.launched.append)
    db = MagicMock()
    db.get.side_effect = lambda model, ident: state.workspaces.get(ident)
    app = FastAPI()
    app.include_router(documents_module.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: _ctx(state.caller)
    app.dependency_overrides[get_db] = lambda: db
    state.client, state.db = TestClient(app), db
    return state


def _upload(api, data=None, *, note="Warm light", stance="like", name="ref.png", declared="image/png"):
    return api.client.post(ROUTE, files={"file": (name, png() if data is None else data, declared)}, data={"note": note, "stance": stance})


# ── the references ─────────────────────────────────────────────────────────


def test_the_type_is_read_from_the_bytes():
    assert [refs.sniff(data) for data in (png(), jpeg(), webp())] == ["png", "jpg", "webp"]
    assert refs.sniff(GIF) is None and refs.sniff(SVG) is None and refs.sniff(b"") is None
    assert refs.sniff(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 8) is None


def test_a_reference_is_stored_listed_served_edited_and_removed(api):
    added = _upload(api)
    assert added.status_code == 201, added.text
    (ref,) = added.json()["references"]
    assert set(ref) == {"id", "note", "stance", "content_type", "bytes", "created_at", "url"}  # never its path
    assert (ref["note"], ref["stance"], ref["content_type"]) == ("Warm light", "like", "image/png")
    assert ref["url"] == f"{ROUTE}/{ref['id']}/image"
    stored = api.storage / str(WS) / "brand" / "references" / f"{ref['id']}.png"
    assert stored.read_bytes() == png()
    assert api.launched == [WS]  # the profile is read again

    image = api.client.get(ref["url"])
    assert (image.status_code, image.content, image.headers["content-type"]) == (200, png(), "image/png")

    edited = api.client.put(f"{ROUTE}/{ref['id']}", json={"note": "Too dark", "stance": "avoid"})
    assert edited.status_code == 200
    assert {k: edited.json()["references"][0][k] for k in ("note", "stance")} == {"note": "Too dark", "stance": "avoid"}
    assert api.client.get(ROUTE).json()["references"][0]["stance"] == "avoid"

    assert api.client.delete(f"{ROUTE}/{ref['id']}").json()["references"] == []
    assert not stored.exists()
    assert api.launched == [WS, WS, WS]


def test_the_declared_type_and_the_name_are_ignored(api):
    assert _upload(api, GIF, name="looks.png", declared="image/png").status_code == 415
    assert _upload(api, SVG, name="logo.svg", declared="image/svg+xml").status_code == 415
    accepted = _upload(api, webp(), name="notes.txt", declared="text/plain")
    assert accepted.status_code == 201 and accepted.json()["references"][0]["content_type"] == "image/webp"
    assert len(api.workspaces[WS].settings["brand_style"]["references"]) == 1


def test_a_reference_over_the_size_is_413_and_nothing_is_kept(api, monkeypatch):
    monkeypatch.setattr(refs, "MAX_REFERENCE_BYTES", 100)
    resp = _upload(api, png(200))
    assert resp.status_code == 413 and "MB" in resp.text
    assert api.launched == [] and not (api.storage / str(WS)).exists()


def test_the_kit_holds_at_most_its_count(api, monkeypatch):
    monkeypatch.setattr(refs, "MAX_REFERENCES", 2)
    assert [_upload(api).status_code for _ in range(2)] == [201, 201]
    third = _upload(api)
    assert third.status_code == 422 and "remove one first" in third.text
    assert api.client.get(ROUTE).json()["limits"] == {"references": 2, "bytes": refs.MAX_REFERENCE_BYTES}


@pytest.mark.parametrize("note, stance, reason", [
    ("x" * (refs.MAX_NOTE_CHARS + 1), "like", "at most"),
    ("Fine", "love", "stance must be one of like, avoid"),
])
def test_a_long_note_or_an_unknown_stance_is_422(api, note, stance, reason):
    resp = _upload(api, note=note, stance=stance)
    assert resp.status_code == 422 and reason in resp.text


def test_an_edit_is_checked_and_an_unknown_reference_is_404(api):
    ref = _upload(api).json()["references"][0]
    assert api.client.put(f"{ROUTE}/{ref['id']}", json={"stance": "love"}).status_code == 422
    assert api.client.put(f"{ROUTE}/{ref['id']}", json={"path": "../x"}).status_code == 422
    assert api.client.put(f"{ROUTE}/{uuid.uuid4().hex}", json={"note": "x"}).status_code == 404
    assert api.client.delete(f"{ROUTE}/{uuid.uuid4().hex}").status_code == 404


def test_another_workspaces_reference_is_404_everywhere(api):
    ref = _upload(api).json()["references"][0]
    api.caller = OTHER
    assert api.client.get(ROUTE).json()["references"] == []
    assert api.client.get(f"{ROUTE}/{ref['id']}/image").status_code == 404
    assert api.client.put(f"{ROUTE}/{ref['id']}", json={"note": "mine now"}).status_code == 404
    assert api.client.delete(f"{ROUTE}/{ref['id']}").status_code == 404
    api.caller = WS
    assert api.client.get(ROUTE).json()["references"][0]["note"] == "Warm light"


def test_writes_need_workspace_manage(api):
    ref = _upload(api).json()["references"][0]
    api.role = "viewer"
    assert _upload(api).status_code == 403
    assert api.client.put(f"{ROUTE}/{ref['id']}", json={"note": "x"}).status_code == 403
    assert api.client.delete(f"{ROUTE}/{ref['id']}").status_code == 403
    assert api.client.put("/api/documents/brand-kit/style", json={"send_liked": False}).status_code == 403
    assert api.client.get(ROUTE).status_code == 200  # reading is for everyone in the workspace


def test_whether_liked_images_go_to_ai_tools_is_a_setting(api):
    assert api.client.get(ROUTE).json()["send_liked"] is True
    off = api.client.put("/api/documents/brand-kit/style", json={"send_liked": False})
    assert off.status_code == 200 and off.json()["send_liked"] is False
    assert refs.style_of(api.workspaces[WS].settings)["send_liked"] is False
    assert api.client.put("/api/documents/brand-kit/style", json={}).status_code == 422


def test_the_kits_own_settings_are_untouched(api):
    api.workspaces[WS].settings = {"brand_kit": {"name": "Acme"}}
    _upload(api)
    assert api.workspaces[WS].settings["brand_kit"] == {"name": "Acme"}


# ── the profile, with the model mocked ─────────────────────────────────────


def test_an_answer_is_read_as_a_profile_checked_and_trimmed():
    fenced = "Here it is:\n```json\n" + json.dumps({
        "palette": ["#1f2a44", "navy", "#F4B400", "#12345"], "mood": ["calm", " ", "x" * 40],
        "composition": "  Wide   shots.  ", "avoid": "Clutter.", "extra": "ignored",
    }) + "\n```"
    assert brand_style.parse_profile(fenced) == {
        "palette": ["#1F2A44", "#F4B400"], "mood": ["calm", "x" * brand_style.MAX_WORD_CHARS],
        "composition": "Wide shots.", "avoid": "Clutter.",
    }
    for answer in (None, "", "no json here", "[1, 2]", json.dumps({"palette": ["red"], "mood": []})):
        assert brand_style.parse_profile(answer) is None


def _ref(stance, note, data, created):
    return {"id": uuid.uuid4().hex, "stance": stance, "note": note, "content_type": "image/png", "data": data, "created_at": created}


def test_the_read_sends_liked_first_newest_first_then_avoided(monkeypatch):
    references = [
        _ref("like", "old like", b"L1", "1"), _ref("avoid", "old avoid", b"A1", "2"),
        _ref("like", "new like", b"L2", "3"), _ref("avoid", "new avoid", b"A2", "4"),
    ]
    monkeypatch.setattr(refs, "load_reference", lambda ref: ref["data"])
    monkeypatch.setattr(brand_style, "_shrunk", lambda data, content_type: (data, content_type))
    images = brand_style.images_for_read(references)
    assert [(stance, note) for stance, note, _data, _type in images] == [
        ("like", "new like"), ("like", "old like"), ("avoid", "new avoid"), ("avoid", "old avoid"),
    ]
    monkeypatch.setattr(brand_style, "MAX_READ_IMAGES", 3)
    assert len(brand_style.images_for_read(references)) == 3
    monkeypatch.setattr(refs, "load_reference", lambda ref: None if ref["note"] == "new like" else ref["data"])
    assert "new like" not in [note for _s, note, _d, _t in brand_style.images_for_read(references)]


def test_a_reference_is_shrunk_for_the_read():
    image_module = pytest.importorskip("PIL.Image")
    out = io.BytesIO()
    image_module.new("RGB", (2000, 1000), (200, 40, 40)).save(out, "PNG")
    data, content_type = brand_style._shrunk(out.getvalue(), "image/png")
    shrunk = image_module.open(io.BytesIO(data))
    assert content_type == "image/jpeg" and max(shrunk.size) == brand_style.READ_EDGE_PX
    assert brand_style._shrunk(png(), "image/png") is None  # unreadable: left out, never a crash


def test_the_messages_carry_each_image_with_its_stance_and_note():
    messages = brand_style.build_messages([("like", "Warm light", b"IMG", "image/jpeg"), ("avoid", "", b"BAD", "image/png")])
    assert messages[0] == {"role": "system", "content": brand_style.SYSTEM}
    content = messages[1]["content"]
    assert content[1] == {"type": "text", "text": "Image 1: LIKE. Note: Warm light"}
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"IMG").decode()
    assert content[3] == {"type": "text", "text": "Image 2: AVOID. Note: none"}


class _Model:
    def __init__(self, answer="", delay=0.0):
        self.answer, self.delay, self.created, self.asked = answer, delay, [], []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return self

    async def generate_response(self, messages):
        self.asked.append(messages)
        await asyncio.sleep(self.delay)
        return SimpleNamespace(content=self.answer)


def test_the_read_goes_through_the_llm_manager_as_brand_style_read(monkeypatch):
    model = _Model(json.dumps(PROFILE))
    monkeypatch.setattr(core.llm, "create_llm_manager", model.create)
    monkeypatch.setattr(config, "BRAND_STYLE_READ_MODEL", "")
    profile = asyncio.run(brand_style.read_profile(WS, [("like", "n", b"IMG", "image/png")]))
    assert profile == {**PROFILE, "palette": ["#1F2A44", "#F4B400"]}
    assert model.created == [{"service_name": "brand_style", "model": None, "workspace_id": WS, "request_type": "brand_style_read"}]
    monkeypatch.setattr(config, "BRAND_STYLE_READ_MODEL", "vision-model")
    asyncio.run(brand_style.read_profile(WS, [("like", "n", b"IMG", "image/png")]))
    assert model.created[-1]["model"] == "vision-model"


def test_an_unreadable_answer_fails_and_a_slow_one_times_out(monkeypatch):
    monkeypatch.setattr(core.llm, "create_llm_manager", _Model("I like these images").create)
    with pytest.raises(brand_style.StyleReadFailed):
        asyncio.run(brand_style.read_profile(WS, [("like", "", b"IMG", "image/png")]))
    monkeypatch.setattr(core.llm, "create_llm_manager", _Model(json.dumps(PROFILE), delay=1).create)
    monkeypatch.setattr(config, "BRAND_STYLE_READ_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(brand_style.read_profile(WS, [("like", "", b"IMG", "image/png")]))


def test_a_refresh_stores_the_profile_stamped_and_no_references_store_none(monkeypatch):
    stored = []
    references = [{"id": "r1", "stance": "like"}]
    monkeypatch.setattr(brand_style, "_load_references", lambda ws: references)
    monkeypatch.setattr(brand_style, "images_for_read", lambda found: [("like", "", b"IMG", "image/png")])

    async def read(ws, images):
        return dict(PROFILE)

    monkeypatch.setattr(brand_style, "read_profile", read)
    monkeypatch.setattr(brand_style, "_store_profile", lambda ws, profile, found: stored.append((ws, profile, found)))
    assert asyncio.run(brand_style.refresh_profile(WS)) == PROFILE
    assert stored == [(WS, PROFILE, references)]
    stamped = brand_style.with_profile({"references": references}, PROFILE, references)["profile"]
    assert stamped["reference_ids"] == ["r1"] and stamped["read_at"]

    monkeypatch.setattr(brand_style, "_load_references", lambda ws: [])
    assert asyncio.run(brand_style.refresh_profile(WS)) is None
    assert stored[-1] == (WS, None, [])


def test_read_again_answers_the_profile_or_why_not(api, monkeypatch):
    async def read(ws):
        api.workspaces[ws].settings = {"brand_style": {"references": [], "profile": PROFILE}}
        return PROFILE

    monkeypatch.setattr(brand_style, "refresh_profile", read)
    done = api.client.post("/api/documents/brand-kit/style/read")
    assert done.status_code == 200 and done.json()["profile"] == PROFILE

    async def unreadable(ws):
        raise brand_style.StyleReadFailed("The model's answer could not be read as a style profile. Try again.")

    monkeypatch.setattr(brand_style, "refresh_profile", unreadable)
    assert api.client.post("/api/documents/brand-kit/style/read").status_code == 502

    async def slow(ws):
        raise asyncio.TimeoutError

    monkeypatch.setattr(brand_style, "refresh_profile", slow)
    assert api.client.post("/api/documents/brand-kit/style/read").status_code == 504


# ── what carries the profile ───────────────────────────────────────────────


def test_the_profile_is_one_paragraph_and_nothing_without_one():
    assert brand_style.style_prompt({"brand_style": {"profile": PROFILE}}) == STYLE_TEXT
    assert brand_style.style_prompt({}) == "" and brand_style.style_prompt(None) == ""


def test_the_composer_is_given_the_profile():
    ctx = compose.ComposeContext(brief="Launch week", format="image", channels=[], templates=[], candidates=[], style=STYLE_TEXT)
    material = json.loads(compose.build_messages(ctx)[1]["content"])
    assert material["brand_style"] == STYLE_TEXT
    bare = compose.ComposeContext(brief="Launch week", format="image", channels=[], templates=[], candidates=[])
    assert "brand_style" not in json.loads(compose.build_messages(bare)[1]["content"])


def test_the_composers_context_reads_the_workspaces_profile():
    from api.socials_compose import brand_style_text

    db = MagicMock()
    db.get.return_value = SimpleNamespace(settings={"brand_style": {"profile": PROFILE}})
    assert brand_style_text(db, WS) == STYLE_TEXT
    db.get.return_value = None
    assert brand_style_text(db, WS) == ""


def test_the_brand_kit_tool_gives_agents_the_profile(monkeypatch):
    from modules.documents import brand_kit
    from modules.tools.discovery import handlers_documents

    monkeypatch.setattr(brand_kit, "brand_kit_suggestions", lambda db, workspace: {})
    db = MagicMock()
    workspace = SimpleNamespace(id=WS, name="Acme", settings={"brand_style": {"profile": PROFILE}})
    db.query.return_value.filter.return_value.first.return_value = workspace
    db.get.return_value = workspace
    result = asyncio.run(handlers_documents.get_brand_kit_tool(db, WS, {}))
    assert result["success"] is True and result["style_profile"] == STYLE_TEXT
