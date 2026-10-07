"""F382 (night 11, 7 Oct), B21: on the Before/after story the logo and handle no longer sit on the before photo.

The two photos filled the top 66% of the card from its very top, and the brand row
(the logo chip and the handle) sat in the page's flow over them; in a 9:16 story it
landed on the before photo, and the labels (132 design px down) sat in Instagram's
top bar. A tall card now gives the brand row a band of its own (``--band``) above the
photos, which start below it (and below the story's top inset), and each label sits at
the top of its photo. Pins, measured from the template's own numbers at 1080x1920 as a
story renders it.
"""
from __future__ import annotations

import re

from core.media_render_bundle import STORY_SAFE_TOP
from core.social_templates import parse_size
from modules.documents.social_starters import social_starters

STORY = "1080x1920"
TALL = re.compile(r"@media \(max-aspect-ratio: 10/16\) \{(.*?)\n      \}", re.S)


def _html():
    return next(s for s in social_starters() if s["slug"] == "before-after")["blocks"]["html"]


def _number(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    assert match, pattern
    return int(match.group(1))


def test_the_story_size_is_tall_enough_for_the_band():
    width, height = parse_size(STORY)
    assert width / height < 10 / 16  # the band's media query takes the story size, not the 4:5 feed post


def test_the_brand_row_has_a_band_above_the_photos_and_the_labels_clear_the_top_bar():
    html = _html()
    tall = TALL.search(html)
    assert tall, "the tall layout"
    rules = tall.group(1)
    band = _number(r"--band: calc\((\d+) \* var\(--u\)\)", rules)
    assert "top: calc(var(--story-top, 0px) + var(--band));" in rules
    assert "height: calc(66% - var(--story-top, 0px) - var(--band));" in rules
    label_top = _number(r"\.label \{ top: calc\((\d+) \* var\(--u\)\); \}", rules)
    # The brand row: its margin, then a chip of the logo (40) and its padding (10 above, 10 below).
    margin = _number(r"\.top \{[^{}]*margin: calc\((\d+) \* var\(--u\) \+ var\(--story-top, 0px\)\)", html)
    logo = _number(r"\.logo \{[^{}]*width: calc\((\d+) \* var\(--s\)\)", html)
    chip_padding = _number(r"\.brand, \.chip \{[^{}]*padding: calc\((\d+) \* var\(--s\)\)", html)
    row_bottom = margin + logo + 2 * chip_padding
    assert band > row_bottom, (band, row_bottom)  # the photos start below the brand row
    # Each label sits on its photo, below the brand row; in a story the photos start under the top bar's inset,
    # where before the fix a label sat 132 design px down, inside the bar.
    assert band + label_top > row_bottom
    assert label_top < 132 and STORY_SAFE_TOP + band + label_top > STORY_SAFE_TOP + row_bottom
