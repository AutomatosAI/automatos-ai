"""PRD-264 ART5-1 — Socials footage and stills through the generation guard (core/generation_guard.py).

On the footage harness (tests/test_prd251w1_footage.py: SQLite, the Composio toolkits faked, a
render driven through the API). Pinned: a refused prompt prices, submits and books nothing; the
liked reference is checked with the prompt; a reference that can't be read to check it submits
nothing; a provider's own refusal (Higgsfield's ``nsfw``) ends the shot as a logged refusal and is
never asked of another toolkit. The guard's own rules: tests/test_prd264_art5_generation_guard.py.
"""
from __future__ import annotations

import logging

import pytest

import tests.test_prd251w1_footage as harness  # first: it sets the environment the app imports read
from tests.test_prd251w1_footage import (
    FAL_SUBMIT,
    GUARD,
    HF_VIDEO,
    HF_WAIT,
    PROMPT,
    WS,
    FakeStore,
    _cache_fal,
    _cache_higgsfield,
    _connect,
    _create,
    _last_log,
    _media_rows,
    _ok,
    _render,
    _script_fal,
    _script_higgsfield,
    _set_config,
)

import api.socials_brand as socials_brand  # noqa: E402
import core.generation_guard as generation_guard  # noqa: E402

env = harness.env

REFERENCE = "https://signed/documents/liked-ref.png"
REFERENCE_PNG = b"\x89PNG\r\n\x1a\n" + b"liked reference " * 40


def _guard_reads(monkeypatch, **answers):
    """The guard's model read answering ``answers`` (every other question no); what it was asked."""
    asked = []

    async def read(request, workspace_id):
        asked.append((request, workspace_id))
        return {"sexual": False, "minor": False, "real_person": False, **answers}

    monkeypatch.setattr(generation_guard, "read_answers", read)
    return asked


@pytest.mark.parametrize("answers", [{"sexual": True, "real_person": True}, {"sexual": True, "minor": True}])
def test_art5_a_refused_prompt_prices_submits_and_books_nothing(env, monkeypatch, answers):
    asked = _guard_reads(monkeypatch, **answers)
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is False and renderer.bundles == []
    assert env.composio.slugs() == [], "refused before anything was priced or submitted"
    ((request, workspace_id),) = asked
    # exactly what the provider would get: the prompt as submitted, with the guard words
    assert (request.path, request.texts, request.images, workspace_id) == ("socials.footage", (f"{PROMPT}, {GUARD}",), (), WS)
    entry = _last_log(env, post["id"])
    assert entry["action"] == "render_failed" and entry["report"]["code"] == "footage_refused"
    assert entry["comment"] == f"{generation_guard.REFUSED_MESSAGE} Nothing was submitted. Nothing was rendered."
    assert _media_rows(env, post["id"]) == []


def test_art5_the_liked_reference_is_checked_with_the_prompt(env, monkeypatch):
    asked = _guard_reads(monkeypatch)
    monkeypatch.setattr(socials_brand, "liked_reference_links", lambda settings: (REFERENCE,))
    env.files[REFERENCE] = REFERENCE_PNG
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is True
    ((request, _),) = asked
    assert request.images == ((REFERENCE_PNG, "image/png"),)
    assert FAL_SUBMIT in env.composio.slugs()


def test_art5_the_brand_kits_style_is_checked_as_the_provider_gets_it(env, monkeypatch):
    """The style profile follows the prompt to the provider (prompt_for), so the guard reads it too."""
    asked = _guard_reads(monkeypatch, sexual=True, real_person=True)
    style = "Brand style (from the brand kit's references): mood: candid."
    monkeypatch.setattr(socials_brand, "brand_style_text", lambda db, workspace_id: style)
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)

    assert _render(env, post["id"], FakeStore())[0] is False
    ((request, _),) = asked
    assert request.texts == (f"{PROMPT}, {GUARD} {style}",)
    assert env.composio.slugs() == []


def test_art5_a_reference_that_cannot_be_read_to_check_it_submits_nothing(env, monkeypatch):
    asked = _guard_reads(monkeypatch)
    monkeypatch.setattr(socials_brand, "liked_reference_links", lambda settings: (REFERENCE,))
    env.files[REFERENCE] = b"<html>not an image</html>"
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)

    assert _render(env, post["id"], FakeStore())[0] is False
    assert asked == [] and env.composio.slugs() == []
    entry = _last_log(env, post["id"])
    assert entry["report"]["code"] == "footage_refused"
    assert entry["comment"].startswith(generation_guard.UNCHECKED_MESSAGE)


def test_art5_a_providers_nsfw_refusal_ends_it_and_is_never_asked_of_another_toolkit(env, monkeypatch, caplog):
    _set_config(monkeypatch, SOCIALS_FOOTAGE_TOOLKITS="higgsfield_mcp,fal_ai")
    _cache_higgsfield(env)
    _cache_fal(env)
    _connect(env, "HIGGSFIELD_MCP", "FAL_AI")
    _script_higgsfield(env)
    _script_fal(env)
    env.composio.answers[HF_WAIT] = [_ok({"jobs": [{"id": "job-7", "status": "nsfw"}], "all_terminal": True})]
    post = _create(env)

    with caplog.at_level(logging.WARNING, logger="core.generation_guard"):
        ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is False and renderer.bundles == []
    slugs = env.composio.slugs()
    assert slugs.count(HF_VIDEO) == 1
    assert not [slug for slug in slugs if slug.startswith("FAL_AI")], "never sent to another provider"
    comment = _last_log(env, post["id"])["comment"]
    assert "Higgsfield refused to make this as unsafe content. It was not sent to any other provider." in comment
    logged = [record.getMessage() for record in caplog.records if record.name == "core.generation_guard"]
    assert logged == [f"[GenerationGuard] refused: workspace={WS} path=socials.footage reason=provider_refused"]


def test_art5_with_the_read_off_no_reference_is_fetched_and_the_render_goes_ahead(env, monkeypatch):
    """The model read ships off until after Web Summit: no read, and no reference fetched for one."""
    asked = _guard_reads(monkeypatch, sexual=True, real_person=True)
    monkeypatch.setattr(generation_guard.config, "GENERATION_GUARD_READ_ENABLED", False, raising=False)
    monkeypatch.setattr(socials_brand, "liked_reference_links", lambda settings: (REFERENCE,))
    env.files[REFERENCE] = REFERENCE_PNG
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is True and asked == []
    assert REFERENCE not in [url for url, _, _ in env.fetched], "not fetched for a read that doesn't run"
    assert FAL_SUBMIT in env.composio.slugs()
