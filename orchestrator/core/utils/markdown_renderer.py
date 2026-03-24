"""
Markdown → HTML renderer with sanitisation.

Used by the public blog API to return safe HTML content.
"""

from __future__ import annotations

import markdown
import bleach

_ALLOWED_TAGS = [
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "a", "img",
    "ul", "ol", "li",
    "code", "pre", "blockquote",
    "table", "thead", "tbody", "tr", "th", "td",
    "strong", "em", "br", "hr",
    "span", "div",
]

_ALLOWED_ATTRS = {
    "a": ["href", "target", "rel"],
    "img": ["src", "alt", "title"],
    "code": ["class"],
    "pre": ["class"],
    "span": ["class"],
    "th": ["align"],
    "td": ["align"],
}

_MD_EXTENSIONS = [
    "tables",
    "fenced_code",
    "codehilite",
    "toc",
    "nl2br",
]


def render_markdown_to_html(content: str) -> str:
    """Convert markdown to sanitised HTML."""
    raw_html = markdown.markdown(content, extensions=_MD_EXTENSIONS)
    return bleach.clean(
        raw_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        strip=True,
    )
