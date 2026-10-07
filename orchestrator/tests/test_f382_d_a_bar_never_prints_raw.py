"""F382 (night 11, 7 Oct), B7: a ``|`` never prints raw on a card.

The IG story Announcement printed "|" separators in its feature text: only the
headline (``data-dress``) splits on "|", and the model wrote it into the features
too. Pins: a field the template prints as plain text has each "|" turned into a
space when the bundle is built (``core/social_line_breaks.py``); display text keeps
its line breaks, and a field the template reads in an attribute or a script (a data
story's rows) keeps its "|"; the composer is told only the fields whose description
says so take it.
"""
from __future__ import annotations

from core.media_render_bundle import build_bundle
from core.social_line_breaks import line_break_fields, unbroken, without_stray_breaks
from core.social_templates import SOCIAL_IMAGE, resolve_variables
from modules.documents.social_starters import social_starters
from modules.socials import compose

PAGE = (
    '<div data-row="{{ row }}"></div><p class="v">{{ plain }}</p>'
    '<h1 class="headline fit" data-dress data-accent="{{ accent }}">{{ head }}</h1>'
    '<blockquote data-dress="quote">{{ quoted }}</blockquote><script>go("{{ scripted }}");</script>'
)


def _announcement():
    return next(s for s in social_starters(SOCIAL_IMAGE) if s["slug"] == "announcement-card")


def test_only_display_text_attributes_and_scripts_read_the_bar():
    assert line_break_fields(PAGE) == {"row", "accent", "head", "quoted", "scripted"}
    values = {name: "one | two|three" for name in ("row", "plain", "head", "quoted", "scripted")}
    cleaned = without_stray_breaks({**values, "count": 3}, PAGE)
    assert cleaned["plain"] == "one two three"
    assert all(cleaned[name] == values[name] for name in ("row", "head", "quoted", "scripted"))
    assert cleaned["count"] == 3 and values["plain"] == "one | two|three"  # the input is untouched
    assert unbroken("|NEW|") == "NEW"


def test_a_script_ends_where_a_browser_ends_it():
    # CodeQL py/bad-tag-filter: "</script >" and "</SCRIPT foo>" end a script too, and what follows is page text.
    page = ('<script>go("{{ one }}");</script ><p>{{ after_one }}</p>'
            '<SCRIPT type="x">go("{{ two }}");</SCRIPT foo><p>{{ after_two }}</p>'
            '<span data-dress>{{ head }}<br>{{ head_2 }}</span><p>{{ plain }}</p>')
    assert line_break_fields(page) == {"one", "two", "head", "head_2"}


def test_the_announcements_features_never_print_a_bar_and_its_headline_keeps_its_lines():
    starter = _announcement()
    html = starter["blocks"]["html"]
    assert "headline" in line_break_fields(html)
    assert not {"feature_1_title", "feature_1_body", "subline"} & line_break_fields(html)
    sample = {**starter["sample_data"], "feature_1_title": "Fresh | weekly", "feature_1_body": "Single origin | roasted|to order"}
    values = resolve_variables(starter["blocks"]["variables_schema"], sample).values
    bundle = build_bundle(workspace_id="ws", reference="r", blocks=starter["blocks"], values=values, brand_kit={},
                          size="1080x1920", fmt=SOCIAL_IMAGE, story_safe=True)
    variables = bundle["variables"]
    assert variables["feature_1_title"] == "Fresh weekly"
    assert variables["feature_1_body"] == "Single origin roasted to order"
    assert "|" in values["headline"] and variables["headline"] == values["headline"]  # its lines are kept


def test_the_composer_is_told_only_the_fields_that_say_so_take_a_bar():
    base = dict(brief="Harvest box", channels=[], templates=[], candidates=[])
    system, _user = compose.build_messages(compose.ComposeContext(format="image", **base))
    assert compose.LINE_BREAK_NOTE in system["content"]
    text_only, _ = compose.build_messages(compose.ComposeContext(format="text", **base))
    assert compose.LINE_BREAK_NOTE not in text_only["content"]
