"""The social templates a workspace starts with (PRD-251 S1.2, D4).

The videos: the four reference videos (docs/PRDS/prd251-reference/, PRD-251A)
ported to templates: "UI story promo" (v1), "Cinematic product promo" (v2, four
footage slots), "App promo" (the Academy promo, a phone frame, five footage
slots) and "Data story" (the Markets posh cut, its charts drawn from the post's
data).

The images (US-107): the five automatos-social families ported from its JSX
(title, definition, stats, quote, announcement) and two variants, a fact card
and a carousel, each at the four sizes of automatos-social's schema.json
(1080x1350, 1080x1920, 1200x628, 1600x900) on the brand kit's paper palette.
The infographic (US-113, S1.7) joins them: a chart of a report's top rows (a
bar chart, a line or a number grid), its ``data`` block naming the variables a
report fills (``core/chart_binding.py``).

Each is two seed files under ``templates/social/``: ``<slug>.html``, the
composition, and ``<slug>.json``: its name, description, format, category and
sample_data (the copy a preview renders), and every other block the template
contract takes (css, variables_schema, sizes, audio_plan, slots, stills, data).
The seed files are read once, checked against the template contract
(``core/social_templates.py``), and written to ``document_templates`` rows by
``seed_templates.seed_social_starters``. Rendering reads the row, never these
files (CLAUDE.md §4: no file hacks for DB data). A starter that breaks the
contract never reaches a workspace: loading it raises. So does a seed file
carrying a field that is none of these: nothing in a seed file is dropped unread
(P251W1-RVW-7).

``preview`` in a seed file is for the media-render CI job only (the moments a
video's preview snapshots, and where the brand colour shows at one of them); it
is not stored.

Pure: no database. Standard library plus the contract, so the media-render CI
job loads the very same starters (scripts/ci/social_template_previews.py).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.social_templates import BLOCK_KEYS, SocialTemplateError, validate_social_blocks

STARTERS_DIR = Path(__file__).resolve().parent / "templates" / "social"
SOCIAL_VIDEO_STARTER_SLUGS: Tuple[str, ...] = (
    "ui-story-promo",
    "cinematic-product-promo",
    "app-promo",
    "data-story",
)
SOCIAL_IMAGE_STARTER_SLUGS: Tuple[str, ...] = (
    "title-card",
    "definition-card",
    "stats-card",
    "quote-card",
    "announcement-card",
    "fact-card",
    "carousel",
    "infographic",
)
SOCIAL_STARTER_SLUGS: Tuple[str, ...] = SOCIAL_VIDEO_STARTER_SLUGS + SOCIAL_IMAGE_STARTER_SLUGS
# The composition's own keys in a seed file: every block the contract takes but
# the html, which is the seed's .html file. The row fields describe the row, and
# ``preview`` is the media-render CI job's. A seed file carries nothing else.
BLOCK_FIELDS = tuple(key for key in BLOCK_KEYS if key != "html")
ROW_FIELDS = ("name", "description", "format", "category", "sample_data")
PREVIEW_FIELD = "preview"
SEED_FIELDS = BLOCK_FIELDS + ROW_FIELDS + (PREVIEW_FIELD,)


def _unknown_field_errors(slug: str, meta: Dict[str, Any]) -> List[Dict[str, str]]:
    """Each field of a seed file outside :data:`SEED_FIELDS`, named: none is dropped unread."""
    return [
        {"field": f"{slug}.{key}", "message": f"is not a field of a social starter ({', '.join(SEED_FIELDS)})"}
        for key in meta
        if key not in SEED_FIELDS
    ]


def _load(slug: str) -> Dict[str, Any]:
    meta = json.loads((STARTERS_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    unknown = _unknown_field_errors(slug, meta)
    if unknown:
        raise SocialTemplateError(unknown)
    blocks = {"html": (STARTERS_DIR / f"{slug}.html").read_text(encoding="utf-8")}
    blocks.update({key: meta[key] for key in BLOCK_FIELDS if key in meta})
    try:
        checked = validate_social_blocks(blocks, meta["format"])
    except SocialTemplateError as exc:
        raise SocialTemplateError([{"field": f"{slug}.{e['field']}", "message": e["message"]} for e in exc.errors]) from exc
    starter = {key: meta[key] for key in ROW_FIELDS}
    starter["blocks"] = checked
    starter["slug"] = slug
    starter["preview"] = meta.get(PREVIEW_FIELD) or {}
    return starter


@lru_cache(maxsize=1)
def _starters() -> Tuple[Dict[str, Any], ...]:
    return tuple(_load(slug) for slug in SOCIAL_STARTER_SLUGS)


def social_starters(fmt: Optional[str] = None) -> List[Dict[str, Any]]:
    """The social starters (those of format ``fmt`` when given), each
    ``{name, description, format, category, blocks, sample_data, slug, preview}``.

    ``blocks`` is the checked composition. A copy each call: callers may change it.
    """
    return [starter for starter in json.loads(json.dumps(_starters())) if fmt is None or starter["format"] == fmt]


__all__ = [
    "SOCIAL_IMAGE_STARTER_SLUGS",
    "SOCIAL_STARTER_SLUGS",
    "SOCIAL_VIDEO_STARTER_SLUGS",
    "STARTERS_DIR",
    "social_starters",
]
