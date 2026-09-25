"""The social video templates a workspace starts with (PRD-251 S1.2, D4).

The four reference videos (docs/PRDS/prd251-reference/, PRD-251A) ported to
templates: "UI story promo" (v1), "Cinematic product promo" (v2, four footage
slots), "App promo" (the Academy promo, a phone frame, five footage slots) and
"Data story" (the Markets posh cut, its charts drawn from the post's data).

Each is two seed files under ``templates/social/``: ``<slug>.html``, the
composition, and ``<slug>.json``: its name, description, format, category,
sizes, variables_schema, slots, audio_plan and sample_data (the reference's own
copy, what a preview renders). The seed files are read once, checked against
the template contract (``core/social_templates.py``), and written to
``document_templates`` rows by ``seed_templates.seed_social_starters``.
Rendering reads the row, never these files (CLAUDE.md §4: no file hacks for DB
data). A starter that breaks the contract never reaches a workspace: loading it
raises.

``preview`` in a seed file is for the media-render CI job only (the moments it
snapshots, and where the brand colour shows at one of them); it is not stored.

Pure: no database. Standard library plus the contract, so the media-render CI
job loads the very same starters (scripts/ci/social_template_previews.py).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.social_templates import SocialTemplateError, validate_social_blocks

STARTERS_DIR = Path(__file__).resolve().parent / "templates" / "social"
SOCIAL_STARTER_SLUGS: Tuple[str, ...] = (
    "ui-story-promo",
    "cinematic-product-promo",
    "app-promo",
    "data-story",
)
# The composition's own keys in a seed file; everything else describes the row.
BLOCK_FIELDS = ("variables_schema", "sizes", "audio_plan", "slots")
ROW_FIELDS = ("name", "description", "format", "category", "sample_data")


def _load(slug: str) -> Dict[str, Any]:
    meta = json.loads((STARTERS_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    blocks = {"html": (STARTERS_DIR / f"{slug}.html").read_text(encoding="utf-8")}
    blocks.update({key: meta[key] for key in BLOCK_FIELDS if key in meta})
    try:
        checked = validate_social_blocks(blocks, meta["format"])
    except SocialTemplateError as exc:
        raise SocialTemplateError([{"field": f"{slug}.{e['field']}", "message": e["message"]} for e in exc.errors]) from exc
    starter = {key: meta[key] for key in ROW_FIELDS}
    starter["blocks"] = checked
    starter["slug"] = slug
    starter["preview"] = meta.get("preview") or {}
    return starter


@lru_cache(maxsize=1)
def _starters() -> Tuple[Dict[str, Any], ...]:
    return tuple(_load(slug) for slug in SOCIAL_STARTER_SLUGS)


def social_starters() -> List[Dict[str, Any]]:
    """The social starters, each ``{name, description, format, category, blocks, sample_data, slug, preview}``.

    ``blocks`` is the checked composition. A copy each call: callers may change it.
    """
    return json.loads(json.dumps(_starters()))


__all__ = ["SOCIAL_STARTER_SLUGS", "STARTERS_DIR", "social_starters"]
