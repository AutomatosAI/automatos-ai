"""A published blog post's HTML is the reports' markdown reading (Gerard, 7 Oct).

The blog widget's post route rendered markdown with its own renderer
(``core/utils/markdown_renderer.py``: Python-Markdown with nl2br, codehilite and toc),
the last user of that file once F374 moved report pictures onto
``markdown_body.view_html``. The blog now reads a post as the report view does, and
the old renderer is gone. The intended differences:

* a single line break inside a paragraph is a space (it was a ``<br>``);
* ``- [ ]`` / ``- [x]`` are boxes, ☐ and ☑ (they printed as brackets);
* a list written straight under a line of text, or nested two spaces a level, is a list;
* raw HTML in a post is shown as text, never as markup;
* a fenced block is plain ``<pre><code>`` (no codehilite spans: no consumer styles them).

Kept: images (``src``, ``alt``, ``title``), links, tables, headings, and the sanitising
(links and images only to http(s) or mailto). Heading ids never reached a reader: the
old sanitiser stripped them.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import pytest

import api.widgets.blog as blog

WS = UUID("00000000-0000-0000-0000-00000000b109")
POST = (
    "## Harbour Blend is back\n"
    "Roasted on Thursday\n"
    "and packed on Friday.\n"
    "Before you order:\n"
    "- [ ] Check the grind\n"
    "- [x] Pick a bag size\n"
    "  - 250 g\n"
    "\n"
    "![Harbour Blend bag](https://example.com/bag.png \"The new bag\")\n"
    "\n"
    "[Shop](https://example.com/shop) or [this](javascript:void).\n"
    "\n"
    "<script>alert('x')</script>\n"
    "\n"
    "| Size | Price |\n"
    "|---|---|\n"
    "| 250 g | £8 |\n"
)


class _Posts:
    def __init__(self, db, workspace_id):
        self.workspace_id = workspace_id

    def get_post_by_slug(self, slug):
        return SimpleNamespace(id=1, status="published", to_dict=lambda include_content: {"slug": slug})

    def increment_views(self, post_id):
        return None

    async def get_content(self, post):
        return POST


@pytest.fixture
def html(monkeypatch) -> str:
    monkeypatch.setattr(blog, "BlogService", _Posts)
    response = asyncio.run(blog.get_published_post("harbour-blend", workspace_id=WS, db=object()))
    return json.loads(response.body)["content"]


def test_a_line_break_inside_a_paragraph_is_a_space(html):
    assert "Roasted on Thursday\nand packed on Friday." in html
    assert "<br" not in html


def test_jobs_are_boxes_in_a_list_under_their_line(html):
    assert "☐" in html and "☑" in html
    assert "[ ]" not in html and "[x]" not in html
    assert "<ul" in html and html.count("<ul") == 2  # the jobs, and the sizes nested two spaces in


def test_images_links_tables_and_headings_are_kept(html):
    assert 'src="https://example.com/bag.png"' in html
    assert 'alt="Harbour Blend bag"' in html and 'title="The new bag"' in html
    assert '<a href="https://example.com/shop">Shop</a>' in html
    assert "<table>" in html and "<td>£8</td>" in html
    assert "<h2>Harbour Blend is back</h2>" in html


def test_nothing_unsafe_reaches_the_reader(html):
    assert "javascript:" not in html
    assert "<script" not in html
    assert "&lt;script&gt;" in html  # the post's raw HTML is shown as text
