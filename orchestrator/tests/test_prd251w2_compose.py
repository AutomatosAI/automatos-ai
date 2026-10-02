"""PRD-251 Wave 2, US-207 (S2.2a) — the composer: a brief becomes a draft proposal.

``POST /api/socials/compose`` on the S0.3b harness (SQLite, the real gate and the
real permission check), with the model faked through ``create_llm_manager`` and
the channel registry and the sources search replaced by their seams. Pinned:

* the proposal carries per-channel copy for each selected channel, a template of
  the workspace, schema-valid variables in the post's shape, and only candidate
  sources; it is not saved;
* the prompt carries the brand voice, the templates, the channels with their
  limits, the candidates and the built-in skills;
* the model's answer is never trusted: another workspace's template is replaced
  with a warning, a non-candidate source is dropped (the claim stays unsourced),
  over-limit copy is trimmed at a word boundary, extra Instagram hashtags go;
* invalid JSON is asked for once more, then 502; a slow model is 504;
* usage is tracked with request_type socials_compose; the route is a plain
  ``def``, gated, in the committed manifest; the channel list carries the limits.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402

import api.socials_channels as channels_api  # noqa: E402
import api.socials_compose as compose_api  # noqa: E402
import core.llm as core_llm  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.socials import compose, copy_limits  # noqa: E402
from tests.test_prd251_api import MANIFEST, WS_A, WS_B  # noqa: E402

api = api_harness.api
ROUTE = "/api/socials/compose"
BRIEF = "Announce the Harvest Club: 1,200 members, opens Friday."
SCHEMA = {
    "headline": {"type": "text", "max_chars": 60},
    "members": {"type": "number", "claim": True},
    "growth": {"type": "text", "claim": True, "default": ""},
}
CANDIDATES = [
    {"kind": "metric", "ref": "members", "title": "members", "value": 1200, "as_of": "2026-09-28T00:00:00+00:00"},
    {"kind": "report", "ref": "11111111-1111-1111-1111-111111111111", "title": "Q3", "as_of": "2026-09-01T00:00:00+00:00"},
]
SKILL_TEXT = "Write like a neighbour, never like an advert."


def _template(api, workspace_id, name, fmt="social_image", schema=SCHEMA):
    template_id = uuid.uuid4()
    api.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks) "
            "VALUES (:id, :ws, :name, :fmt, '{}', :blocks)"
        ),
        {"id": template_id.hex, "ws": workspace_id.hex, "name": name, "fmt": fmt,
         "blocks": json.dumps({"html": "<html></html>", "sizes": ["1080x1350"], "variables_schema": schema})},
    )
    api.session.commit()
    return str(template_id)


class _Model:
    """The workspace's model: answers from a queue, records what it was asked."""

    def __init__(self, answers, delay=0.0):
        self.answers = list(answers)
        self.asked = []
        self.delay = delay

    async def generate_response(self, messages, tools=None):
        self.asked.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        answer = self.answers.pop(0)
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


@pytest.fixture
def composer(api, monkeypatch):
    state = SimpleNamespace(api=api, built=[], model=None, channels=["twitter", "linkedin", "instagram"])

    def build(**kwargs):
        state.built.append(kwargs)
        return state.model

    monkeypatch.setattr(core_llm, "create_llm_manager", build)
    channel = lambda t: SimpleNamespace(toolkit=t, label=t.title())  # noqa: E731
    monkeypatch.setattr(compose_api, "social_channels", lambda db, ws: [channel(t) for t in state.channels])
    monkeypatch.setattr(compose_api, "candidate_sources", lambda db, ws, brief: list(CANDIDATES))
    monkeypatch.setattr(compose_api, "builtin_skills", lambda db: {"social-brand-voice": SKILL_TEXT})
    api.session.execute(
        sa.text("UPDATE workspaces SET settings = :s WHERE id = :id"),
        {"s": json.dumps({"socials": {"enabled": True},
                          "brand_kit": {"voice": {"tone": ["warm", "plain", "local"], "banned_phrases": ["game-changer"]}}}),
         "id": WS_A.hex},
    )
    api.session.commit()
    api.session.expire_all()
    state.template = _template(api, WS_A, "Fact card")
    state.foreign = _template(api, WS_B, "Theirs")
    return state


def _answer(state, **overrides):
    answer = {
        "title": "Harvest Club opens Friday",
        "copy": {"base": "Harvest Club opens Friday.", "per_channel": {
            "twitter": "Harvest Club: Friday.", "linkedin": "We open Harvest Club on Friday.",
            "instagram": "Friday! #harvest",
        }},
        "format": "image",
        "template_id": state.template,
        "variables": {"headline": "Opens Friday", "members": 1200},
        "sources": {"members": {"kind": "metric", "ref": "members"}},
    }
    answer.update(overrides)
    return answer


def _compose(state, answers, body=None, delay=0.0):
    state.model = _Model(answers, delay)
    return state.api.client.post(ROUTE, json=body or {"brief": BRIEF})


# ---------------------------------------------------------------------------
# The proposal
# ---------------------------------------------------------------------------


def test_the_proposal_has_each_channels_copy_a_template_valid_variables_and_candidate_sources(composer):
    resp = _compose(composer, [_answer(composer)])
    assert resp.status_code == 200, resp.text
    proposal = resp.json()
    assert proposal["channels"] == ["twitter", "linkedin", "instagram"]
    assert set(proposal["copy"]["per_channel"]) == {"twitter", "linkedin", "instagram"}
    assert proposal["copy"]["per_channel"]["linkedin"] == "We open Harvest Club on Friday."
    assert proposal["template_id"] == composer.template and proposal["format"] == "image"
    assert proposal["template"] == {
        "id": composer.template, "name": "Fact card", "format": "social_image", "sizes": ["1080x1350"], "variables_schema": SCHEMA,
    }
    assert proposal["variables"] == {
        "headline": {"value": "Opens Friday", "claim": False},
        "members": {"value": 1200, "claim": True},
    }
    assert proposal["sources"] == {"members": {"kind": "metric", "ref": "members", "as_of": CANDIDATES[0]["as_of"]}}
    assert composer.api.session.query(SocialPost).count() == 0  # not saved


def test_the_prompt_carries_the_voice_templates_channels_limits_candidates_and_skills(composer):
    _compose(composer, [_answer(composer)], {"brief": BRIEF, "channels": ["twitter"]})
    (messages,) = composer.model.asked
    system, material = messages[0]["content"], json.loads(messages[1]["content"])
    assert SKILL_TEXT in system
    assert material["brief"] == BRIEF
    assert material["brand_voice"]["tone"] == ["warm", "plain", "local"]
    assert material["brand_voice"]["banned_phrases"] == ["game-changer"]
    assert [t["id"] for t in material["templates"]] == [composer.template]  # never another workspace's
    assert material["templates"][0]["variables_schema"] == SCHEMA
    assert material["channels"] == [{"toolkit": "twitter", "label": "Twitter", "limits": {"text": 280}}]
    assert {c["ref"] for c in material["candidate_sources"]} == {c["ref"] for c in CANDIDATES}


def test_a_channel_that_is_not_connected_is_left_out_with_a_warning(composer):
    resp = _compose(composer, [_answer(composer)], {"brief": BRIEF, "channels": ["linkedin", "tiktok"]})
    proposal = resp.json()
    assert proposal["channels"] == ["linkedin"]
    assert "tiktok is not connected in this workspace; it was left out" in proposal["warnings"]


def test_a_channel_the_model_forgot_starts_from_the_base_copy(composer):
    answer = _answer(composer, copy={"base": "Harvest Club opens Friday.", "per_channel": {"twitter": "Friday."}})
    proposal = _compose(composer, [answer]).json()
    assert proposal["copy"]["per_channel"]["linkedin"] == "Harvest Club opens Friday."
    assert any(w.startswith("linkedin: no text of its own") for w in proposal["warnings"])


# ---------------------------------------------------------------------------
# The model's answer is never trusted
# ---------------------------------------------------------------------------


def test_another_workspaces_template_is_replaced_with_a_warning(composer):
    proposal = _compose(composer, [_answer(composer, template_id=composer.foreign)]).json()
    assert proposal["template_id"] == composer.template
    assert "The model named a template that is not this workspace's; Fact card is used instead" in proposal["warnings"]


def test_with_no_template_of_the_posts_kind_none_is_chosen(composer):
    proposal = _compose(composer, [_answer(composer)], {"brief": BRIEF, "format": "video"}).json()
    assert proposal["template_id"] is None and proposal["format"] == "video"
    assert proposal["variables"] == {} and proposal["sources"] == {}
    assert "No social template of this workspace fits: choose one before rendering" in proposal["warnings"]


def test_a_source_that_is_not_a_candidate_is_dropped_and_the_claim_stays_unsourced(composer):
    invented = {"members": {"kind": "url", "ref": "https://made.up/figures"}}
    proposal = _compose(composer, [_answer(composer, sources=invented)]).json()
    assert proposal["sources"] == {}
    assert proposal["variables"]["members"] == {"value": 1200, "claim": True}
    assert "The source given for members is not one found in this workspace; members is unsourced" in proposal["warnings"]


def test_variables_outside_the_schema_or_of_the_wrong_type_are_dropped(composer):
    answer = _answer(composer, variables={"headline": "x" * 61, "members": "lots", "logo": "mine"})
    proposal = _compose(composer, [answer]).json()
    assert proposal["variables"] == {}
    warnings = " ".join(proposal["warnings"])
    assert "logo" in warnings and "headline is longer than 60 characters" in warnings and "members must be a number" in warnings


def test_over_limit_copy_is_trimmed_at_a_word_boundary_with_a_warning(composer):
    long_tweet = ("Harvest " * 40).strip()  # 319 characters
    tags = " ".join(f"#tag{i}" for i in range(35))
    answer = _answer(composer, copy={"base": "b", "per_channel": {
        "twitter": long_tweet, "linkedin": "fine", "instagram": f"Friday. {tags}",
    }})
    proposal = _compose(composer, [answer]).json()
    tweet = proposal["copy"]["per_channel"]["twitter"]
    assert len(tweet) <= 280 and tweet.endswith("Harvest") and long_tweet.startswith(tweet)
    assert "twitter: the copy was trimmed to 280 characters at a word boundary" in proposal["warnings"]
    insta = proposal["copy"]["per_channel"]["instagram"]
    assert insta.count("#") == 30 and "#tag29" in insta and "#tag30" not in insta
    assert "instagram: 5 hashtags over the limit of 30 were dropped" in proposal["warnings"]


def test_invalid_json_is_asked_once_more(composer):
    resp = _compose(composer, ["Sure! Here is a post.", "```json\n" + json.dumps(_answer(composer)) + "\n```"])
    assert resp.status_code == 200, resp.text
    assert len(composer.model.asked) == 2
    assert composer.model.asked[1][-1]["content"] == compose.RETRY_NOTE


def test_invalid_json_twice_is_502_with_a_clear_message(composer):
    resp = _compose(composer, ["not json", "[1, 2]"])
    assert resp.status_code == 502
    assert "could not be read as a proposal" in resp.json()["detail"]
    assert len(composer.model.asked) == 2


def test_a_model_slower_than_the_timeout_is_504(composer, monkeypatch):
    monkeypatch.setattr(compose_api.config, "SOCIALS_COMPOSE_TIMEOUT_SECONDS", 0.05, raising=False)
    resp = _compose(composer, [_answer(composer)], delay=1.0)
    assert resp.status_code == 504


def test_an_unknown_format_is_422_before_the_model_is_asked(composer):
    resp = _compose(composer, [_answer(composer)], {"brief": BRIEF, "format": "podcast"})
    assert resp.status_code == 422 and composer.model.asked == []


# ---------------------------------------------------------------------------
# Usage, the route and the channel list
# ---------------------------------------------------------------------------


def test_usage_is_tracked_as_socials_compose_for_the_callers_workspace(composer):
    _compose(composer, [_answer(composer)])
    assert composer.built == [{"service_name": "socials", "workspace_id": WS_A, "request_type": "socials_compose"}]


def test_a_viewer_cannot_compose(composer):
    composer.api.role = "viewer"
    assert _compose(composer, [_answer(composer)]).status_code == 403


def test_the_route_is_a_plain_def_in_the_committed_manifest():
    assert not inspect.iscoroutinefunction(compose_api.compose_social_post)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "POST", "path": ROUTE} in manifest["routes"]


def test_the_channel_list_carries_each_channels_copy_limits(monkeypatch):
    kind = SimpleNamespace(to_dict=lambda: {})
    channels = [
        SimpleNamespace(toolkit=t, to_dict=lambda t=t: {"toolkit": t, "post_kinds": [kind.to_dict()]})
        for t in ("twitter", "youtube", "mastodon")
    ]
    monkeypatch.setattr(channels_api, "social_channels", lambda db, ws: channels)
    listed = channels_api.list_social_channels(ctx=SimpleNamespace(workspace_id=WS_A), db=None)
    assert [c["copy_limits"] for c in listed] == [{"text": 280}, {"text": 5000, "title": 100}, None]


# ---------------------------------------------------------------------------
# The limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, limit, fitted",
    [
        ("short", 10, "short"),
        ("one two three", 9, "one two"),
        ("one two three", 7, "one two"),
        ("one two three", 6, "one"),
        ("unbreakable", 5, ""),
        ("line one\nline two", 12, "line one"),
    ],
)
def test_a_trim_never_cuts_a_word(text, limit, fitted):
    assert copy_limits.trim_at_word(text, limit) == fitted


def test_youtube_titles_are_fitted_to_100_characters():
    title, warnings = copy_limits.fit_title(["youtube", "linkedin"], "word " * 30)
    assert len(title) <= 100 and not title.endswith(" ") and warnings
    assert copy_limits.fit_title(["linkedin"], "word " * 30) == ("word " * 30, [])
