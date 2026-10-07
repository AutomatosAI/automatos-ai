"""F379 (night 11, 7 Oct): Auto's Socials post is checked before it is saved, and the answer says if it rendered.

Night 11, through the real ``PlatformActionExecutor`` and the real Socials flows (the US-116
harness: SQLite, media-render's launch recorded):

- ``format: social_image`` failed both of Auto's creations first time (B-I5-3): a template's
  format sent as the post's is the post format it names;
- "there's no instagram-carousel template": the template is "Carousel", found now ignoring case
  and a trailing "template"; a name with no template lists the workspace's social templates;
- carousel 39ee6ae3 was saved with ``slide1_heading``, ``sale_date`` and ``price`` and its render
  answered 422 (B-i3-4): a field the template doesn't have is refused, with its real fields;
- a render refused at once came back ``success: true`` behind the whole post, so the reason was cut
  off and "I've drafted the posts" counted as done (B4): it is a failed call now, saying the post
  is a draft, the empty fields by their labels, and the call that fixes it; and the answer is a
  few lines, never the post's whole content.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w1_social_post_tools as harness  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.tools.discovery import social_post_checks as checks  # noqa: E402
from tests.test_prd251w1_social_post_tools import COMPOSITION, _posts, _row, _template, _tool  # noqa: E402

env = harness.env  # the US-116 fixture: SQLite, the real router and executor, media-render recorded

COMPACT_KEYS = {"id", "title", "status", "format", "template", "template_id", "media"}


def _carousel(env):
    (starter,) = [s for s in social_starters("social_image") if s["slug"] == "carousel"]
    return _template(env, name=starter["name"], blocks=starter["blocks"], fmt="social_image"), starter


def test_a_template_format_sent_as_the_posts_is_the_post_format_it_names(env):
    template_id = _template(env, name="Quote card", blocks=COMPOSITION, fmt="social_image")

    made = _tool(env, "platform_create_social_post", title="Rosa", format="social_image", template="quote card",
                 variables={"headline": "Harbour Blend by name now."}, render=False)

    assert made["success"] is True, made
    row = _row(env, made["post"]["id"])
    assert row.format == "image" and str(row.template_id) == str(template_id)


def test_a_format_no_post_has_is_refused_with_the_post_formats_and_nothing_is_saved(env):
    refused = _tool(env, "platform_create_social_post", title="Rosa", format="quote_card", render=False)

    assert refused["success"] is False
    assert "'quote_card' is not a post format" in refused["error"]
    assert "video, image, carousel, fact_card, infographic, text" in refused["error"]
    assert _posts(env) == 0


def test_a_template_named_in_other_words_is_found_and_a_miss_lists_the_real_names(env):
    _carousel(env)

    found = _tool(env, "platform_create_social_post", title="Harvest", format="carousel",
                  template="Instagram carousel template", render=False)
    missed = _tool(env, "platform_create_social_post", title="Harvest", template="Instagram Image", render=False)

    assert found["success"] is True and found["post"]["template"] == "Carousel", found
    assert missed["success"] is False and "No template 'Instagram Image' in this workspace" in missed["error"]
    assert "This workspace's social templates: image: Carousel." in missed["error"]


def test_a_field_the_template_does_not_have_is_refused_with_its_real_fields(env):
    _carousel(env)

    refused = _tool(env, "platform_create_social_post", title="Harvest", format="carousel", template="Carousel",
                    variables={"headline": "October Harvest Club", "slide1_heading": "Two coffees",
                               "sale_date": "12 October", "price": "£32"}, render=False)

    assert refused["success"] is False
    assert "The 'Carousel' template has no field slide1_heading, sale_date, price. Nothing was saved." in refused["error"]
    assert "point_1_title* (Point 1 heading)" in refused["error"] and "closing_title* (Closing heading)" in refused["error"]
    assert _posts(env) == 0


def test_an_update_refuses_a_field_the_template_does_not_have_and_lets_one_be_cleared(env):
    _carousel(env)
    made = _tool(env, "platform_create_social_post", title="Harvest", format="carousel", template="Carousel",
                 variables={"headline": "October Harvest Club"}, render=False)

    refused = _tool(env, "platform_update_social_post", post_id=made["post"]["id"], variables={"price": "£32"})
    cleared = _tool(env, "platform_update_social_post", post_id=made["post"]["id"],
                    variables={"price": None, "cta": "Order by Friday"})

    assert refused["success"] is False and "has no field price" in refused["error"]
    assert cleared["success"] is True, cleared
    assert _row(env, made["post"]["id"]).variables["cta"] == {"value": "Order by Friday", "claim": False}


def test_a_render_refused_at_once_fails_the_call_and_says_the_draft_its_empty_fields_and_the_fix(env):
    _carousel(env)

    result = _tool(env, "platform_create_social_post", title="Harvest", format="carousel", template="Carousel",
                   variables={"headline": "October Harvest Club"})

    assert result["success"] is False and result["saved_as_draft"] is True, result
    post_id = result["post"]["id"]
    assert result["error"].startswith(f'Saved post {post_id} ("Harvest") as a draft, but it did NOT render: ')
    assert ("these fields are empty: Point 1 heading (point_1_title), Point 2 heading (point_2_title), "
            "Closing heading (closing_title)") in result["error"]
    assert f'platform_update_social_post {{"post_id": "{post_id}", "variables": {{"point_1_title": …' in result["error"]
    assert "Never create this post again" in result["error"]
    assert _row(env, post_id).status == "draft" and env.launched == []


def test_a_post_to_render_without_a_template_is_refused_with_the_templates_and_a_text_post_saves(env):
    """F379 (night 11): three posts were saved with no template and never rendered. A post to
    render needs its template, so nothing is saved and the answer names the ones to choose from;
    a text post is copy alone, and saves without a render."""
    _template(env, name="Quote card", blocks=COMPOSITION, fmt="social_image")
    result = _tool(env, "platform_create_social_post", title="Rosa", format="image", copy={"base": "Hello."})
    assert result["success"] is False and "none was named. Nothing was saved." in result["error"], result
    assert "Quote card" in result["error"] and _posts(env) == 0

    text = _tool(env, "platform_create_social_post", title="Text only", format="text", copy={"base": "Hello."})
    assert text["success"] is True and text["post"]["status"] == "draft" and "render" not in text, text
    assert env.launched == []


def test_the_answer_is_a_few_lines_never_the_posts_whole_content(env):
    template_id = _template(env)

    drafted = _tool(env, "platform_create_social_post", **harness._draft_fields(), template=str(template_id),
                    render=False)
    rendering = _tool(env, "platform_create_social_post", **harness._draft_fields(), template=str(template_id))

    assert set(drafted["post"]) == COMPACT_KEYS and set(rendering["post"]) == COMPACT_KEYS
    assert rendering["success"] is True and rendering["render"] == {"started": True}
    assert "never after approval" in rendering["message"]
    assert drafted["post"]["template"] == "Countdown" and drafted["post"]["media"] == []


def test_the_checks_read_a_render_error_in_the_fields_own_words():
    template = type("T", (), {"name": "Carousel", "blocks": {"variables_schema": {
        "headline": {"type": "text", "label": "Headline"},
        "cta": {"type": "text", "label": "Call to action", "default": "Shop now"},
    }}})()

    assert checks.missing_in_words("fill in headline before rendering", template) == "these fields are empty: Headline (headline)"
    assert checks.missing_in_words("The renderer is busy", template) == "The renderer is busy"
    assert checks.missing_names("fill in headline, cta before rendering") == ["headline", "cta"]
    assert checks.field_list(template) == "headline* (Headline), cta (Call to action)"
    assert checks.post_format("Social Video") == ("video", None)
    assert checks.post_format("fact card") == ("fact_card", None)
    assert checks.post_format(None) == (None, None)
    assert checks.unknown_fields(template, {"headline": {"value": "x"}, "price": None}) is None
