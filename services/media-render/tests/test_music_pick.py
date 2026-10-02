"""PRD-251B: a post's own pick of music.

- ``GET /music`` lists the library by title (id, title, artist, style, length, licence and
  whether it asks for credit), behind the internal token like every path but /health.
- A cue without a start (a post that picked a track other than its template's) starts
  where the library says: the first groove with room for the whole video, else just before
  the first drop with room, else the opening. A start the cue gives is kept.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from helpers import TOKEN
from media_render.bundle import parse_bundle
from media_render.fixture import music_bundle
from media_render.music import DROP_LEAD_SECONDS, Track
from media_render.pipeline import music_report
from media_render.server import TOKEN_HEADER, create_app
from test_server import StubPipeline

ATTRIBUTION = 'Music: "Deep House 003" by Sascha Ende (ende.app), licensed CC BY 4.0.'


def _track(track_id="deep-house-003", title="Deep House 003", cues=None, duration=120.0, style="deep house"):
    return Track(
        id=track_id, path=Path("/library") / f"{track_id}.mp3", duration=duration, title=title, artist="Sascha Ende",
        licence="CC-BY-4.0", attribution=ATTRIBUTION, cues=cues or {}, style=style,
    )


def _start(settings, cues, **music):
    bundle = music_bundle()
    bundle["audio"] = {**bundle["audio"], "music": {"track": "deep-house-003", **music}}
    parsed = parse_bundle(bundle, settings, {"deep-house-003": _track(cues=cues)})
    return music_report(parsed)["start"]


def test_the_library_picks_a_groove_with_room_then_a_drop_then_the_opening(settings):
    assert _start(settings, {"grooves": [{"start": 8.0, "end": 20.0}], "drops": [30.0]}) == 8.0
    late_groove = {"grooves": [{"start": 115.0, "end": 119.0}], "drops": [30.0]}
    assert _start(settings, late_groove) == 30.0 - DROP_LEAD_SECONDS  # the groove leaves no room for the video
    assert _start(settings, {"grooves": [{"start": 115.0, "end": 119.0}]}) == 0.0
    assert _start(settings, {"grooves": [{"start": 8.0, "end": 20.0}]}, start=50.0) == 50.0  # a start given is kept


def test_get_music_lists_the_library_by_title_behind_the_token(settings):
    library = {
        "zz": _track("zz", "Where the Night Begins", style="melodic techno"),
        "aa": _track("aa", "Deep House 003"),
    }

    async def go():
        app = create_app(settings, pipeline=StubPipeline(), library=library)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/music")).status == 401
            response = await client.get("/music", headers={TOKEN_HEADER: TOKEN})
            assert response.status == 200
            tracks = (await response.json())["tracks"]
            assert [t["id"] for t in tracks] == ["aa", "zz"]
            assert tracks[1] == {
                "id": "zz", "title": "Where the Night Begins", "artist": "Sascha Ende", "style": "melodic techno",
                "duration": 120.0, "licence": "CC BY 4.0", "credit_required": True,
            }

    asyncio.run(go())
