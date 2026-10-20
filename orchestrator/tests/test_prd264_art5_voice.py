"""PRD-264 ART5-1 — the spoken script through the generation guard (core/generation_guard.py).

On the voice harness (tests/test_prd251w1_voice.py: SQLite, the voice toolkits faked, a render
driven through the API). Pinned: a refused script sends Kokoro nothing (no bundle reaches
media-render) and is never priced or spoken by a voice toolkit; a clean one renders as before.
The guard's own rules: tests/test_prd264_art5_generation_guard.py.
"""
from __future__ import annotations

import tests.test_prd251w1_voice as harness  # first: it sets the environment the app imports read
from tests.test_prd251w1_voice import (
    FISH_VOICE,
    FakeStore,
    _cache_voice_toolkits,
    _connect,
    _create,
    _media_rows,
    _post,
    _render,
    _script_fish,
)

import core.generation_guard as generation_guard  # noqa: E402

env = harness.env

SCRIPT = ("Lisbon, here we come.", "Three weeks to go.")


def _script_reads(monkeypatch, **answers):
    """The guard's model read answering ``answers`` (every other question no); what it was asked."""
    asked = []

    async def read(request, workspace_id):
        asked.append(request)
        return {"sexual": False, "minor": False, "real_person": False, **answers}

    monkeypatch.setattr(generation_guard, "read_answers", read)
    return asked


def test_art5_a_refused_script_sends_kokoro_nothing(env, monkeypatch):
    asked = _script_reads(monkeypatch, sexual=True, real_person=True)
    post = _create(env)

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is False and renderer.bundles == [], "nothing went to media-render for Kokoro to speak"
    (request,) = asked
    assert (request.path, request.texts) == ("socials.voice", SCRIPT)
    entry = _post(env, post["id"]).review_log[-1]
    assert entry["report"]["code"] == "voice_refused"
    assert entry["comment"] == f"{generation_guard.REFUSED_MESSAGE} Nothing was spoken. Nothing was rendered."


def test_art5_a_refused_script_is_never_priced_or_spoken_by_a_voice_toolkit(env, monkeypatch):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    post = _create(env)
    chosen = env.client.patch(
        f"/api/socials/posts/{post['id']}",
        json={"voice": {"toolkit": "fish_audio", "voice_id": FISH_VOICE, "name": "Energetic narrator"}},
    )
    assert chosen.status_code == 200, chosen.text
    _script_fish(env)
    asked = _script_reads(monkeypatch, sexual=True, minor=True)

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is False and renderer.bundles == []
    assert [request.texts for request in asked] == [SCRIPT + ("The voice it is spoken in: Fish Audio, Energetic narrator",)]
    assert env.composio.calls == [], "no balance read, nothing spoken"
    assert _media_rows(env, post["id"]) == []


def test_art5_a_clean_script_still_renders_with_kokoro(env, monkeypatch):
    asked = _script_reads(monkeypatch)
    post = _create(env)

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is True and len(renderer.bundles) == 1
    assert [request.texts for request in asked] == [SCRIPT]
