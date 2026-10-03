"""PRD-251B — TESTER's build-6 findings (F237, F238), pinned.

* **F237:** a template's thumbnail is a link a BROWSER opens, so it is minted against the public
  endpoint (``S3_PUBLIC_ENDPOINT_URL``), served inline as an image; the internal client, whose
  host only the compose network resolves, never mints it.
* **F238:** the thumbnail backfill runs on its own thread and event loop, so it brings its own
  media-render HTTP client and closes it, never the server loop's shared one (a pooled connection
  belongs to the loop that opened it). And a connection that fails underneath httpx (uvloop's
  "the handler is closed" RuntimeError) is "unreachable", which the music route answers with 503,
  never a 500.
* **Built-in skills vs an older git import of the skills repo:** the same skill from the same
  source of truth, so a lower version is refreshed from the seed (provenance kept); an equal
  version, or a row from any other source, is left untouched.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import httpx
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.media_render_client as mrc  # noqa: E402
import core.seeds.seed_builtin_skills as seeder  # noqa: E402
import tests.test_prd251w1_builtin_skills as skills_harness  # noqa: E402
from config import config  # noqa: E402
from modules.socials import media_store, template_gallery, template_thumbnails  # noqa: E402
from core.builtin_skills import SKILLS_REPO_SOURCE, read_seed  # noqa: E402
from core.models.core import Skill  # noqa: E402
from modules.socials.media_store import MediaStore  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000f0237")
skills_table = skills_harness.skills_table
db = skills_harness.db
fixture_manifest = skills_harness.fixture_manifest
THUMB = f"social-media/{WS}/thumbnails/tpl-preview-image-270x337.png"


class _Signer:
    def __init__(self, host):
        self.host, self.asked = host, []

    def generate_presigned_url(self, method, Params, ExpiresIn):
        self.asked.append((method, Params, ExpiresIn))
        return f"{self.host}/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"


def test_a_thumbnail_is_a_link_the_browser_can_open(monkeypatch):
    public, internal = _Signer("http://localhost:9000"), _Signer("http://minio:9000")
    monkeypatch.setattr(media_store, "get_public_s3_client", lambda: public)
    monkeypatch.setattr(media_store, "get_s3_client", lambda: internal)
    monkeypatch.setattr(MediaStore, "configured", staticmethod(lambda: True))
    monkeypatch.setattr(config, "SOCIALS_MEDIA_URL_TTL_SECONDS", 600)

    link = template_gallery.thumbnail_link(MediaStore(bucket="documents"), THUMB)
    assert link == f"http://localhost:9000/documents/{THUMB}?ttl=600"
    ((method, params, ttl),) = public.asked
    assert (method, ttl) == ("get_object", 600)
    assert params == {"Bucket": "documents", "Key": THUMB, "ResponseContentDisposition": "inline", "ResponseContentType": "image/png"}
    assert internal.asked == []  # the compose network's host never reaches a browser
    assert template_gallery.thumbnail_link(MediaStore(bucket="documents"), "https://cdn.test/a.png") == "https://cdn.test/a.png"


def test_media_render_still_gets_links_on_its_own_network(monkeypatch):
    internal = _Signer("http://minio:9000")
    monkeypatch.setattr(media_store, "get_s3_client", lambda: internal)
    assert MediaStore(bucket="documents").presigned_get("k/voice-1.mp3", 60).startswith("http://minio:9000/")


def test_the_backfill_brings_its_own_http_client_and_closes_it(monkeypatch):
    seen = []

    async def ensure(workspace_id, ids, *, brand_kit_of, client=None, store=None):
        seen.append(client)
        assert client._http is not None and not client._http.is_closed

    monkeypatch.setattr(template_thumbnails, "ensure_thumbnails", ensure)
    asyncio.run(template_thumbnails._backfill(WS, [uuid.uuid4()], lambda ws: {}))
    (client,) = seen
    assert client._http is not mrc._client and client._http.is_closed


def _failing_client(error: BaseException) -> mrc.MediaRenderClient:
    def answer(request: httpx.Request) -> httpx.Response:
        raise error

    return mrc.MediaRenderClient(http=httpx.AsyncClient(transport=httpx.MockTransport(answer)))


def test_a_connection_that_fails_underneath_httpx_is_unreachable(monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_RENDER_URL", "http://media-render:8090")
    closed = RuntimeError("unable to perform operation on <TCPTransport closed=True>; the handler is closed")
    with pytest.raises(mrc.MediaRenderUnavailable) as raised:
        asyncio.run(_failing_client(closed).music())
    assert raised.value.code == mrc.UNREACHABLE and "connection failed" in str(raised.value)
    with pytest.raises(mrc.MediaRenderUnavailable):
        asyncio.run(_failing_client(httpx.ConnectError("refused")).music())


@pytest.mark.parametrize("version, refreshed", [("0.9.0", True), ("1.2.0", False), ("2.0.0", False), (None, False)])
def test_an_older_git_import_of_the_skills_repo_is_refreshed_from_the_seed(db, fixture_manifest, version, refreshed):
    db.add(skills_harness._git_row(skill_source=SKILLS_REPO_SOURCE, skill_version=version))
    db.commit()
    outcome = seeder.seed_builtin_skills(db, manifest_path=fixture_manifest)
    db.commit()
    db.expire_all()
    row = db.query(Skill).one()
    seed = read_seed(fixture_manifest.parent / "fixture-skill.md")  # version 1.2.0
    assert row.skill_source == SKILLS_REPO_SOURCE  # provenance kept: the git source still owns the row
    if refreshed:
        assert outcome["left_alone"] == [] and outcome["refreshed"] == ["fixture-skill"]
        assert (row.prompt_template, row.content_hash, row.skill_version) == (seed.body, seed.content_hash, "1.2.0")
    else:
        assert outcome["left_alone"] == ["fixture-skill"] and row.prompt_template == "GIT BODY"

