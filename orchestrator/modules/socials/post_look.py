"""PRD-251D US-D301 (E10): a post's look, three choices of its own on the template it uses.

``social_posts.look`` is ``{"mode", "accent", "logo"}``, each key optional. A key left out is
the kit's own (the template as authored), and a look with no key left is stored as NULL:

* ``mode``: ``"light"`` or ``"dark"``, for a template built on paper (:func:`built_on_paper`).
  Light is the template as authored. Dark sets the card on the kit's dark stage
  (:func:`dark_tokens`): each paper token takes the stage token that plays its part there, and
  the stage pieces a paper card lays on its page (the dark pill, a dark note) take the paper's,
  so they still stand out; a kit's uploaded logo for dark backgrounds shows where the card shows
  its logo (:func:`dark_logos`). A template on the dark stage or a photograph takes no mode: a mode
  a post keeps after its template changed is left out of its render.
* ``accent``: ``"bold"`` or ``"sparing"``, the kit's ``accent_use`` for this post's render,
  set on a copy of the kit (``core/social_kit_tokens.py`` reads it).
* ``logo``: ``false`` shows no logo: ``brand.logo``, ``brand.logo_mark`` and
  ``brand.logo_on_dark`` are the empty logo every template already hides. The brand name stays.

It is a render setting like the voice and the music: outside the content hash, and what it
changes reaches the hash through the rendered files' digests. A render is refused while a post
holds an approval, so an approved post's file keeps the look it was approved with.

No template file changes. Every starter reads its colours only through ``--brand-*`` tokens and
its logo only through ``{{ brand.* }}`` (``core/social_brand_rule.py``), so a look is applied to
the tokens and variables of one render. Pure, and only the standard library and ``core`` at
import, so the media-render CI driver applies a look exactly as a workspace's render does
(``scripts/ci/social_template_looks.py``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.brand_palette import (
    ACCENT_ON_INK, ACCENT_ON_PAPER, INK, INK_CARD, ON_INK, ON_INK_DIM, ON_INK_MUTED, ON_PAPER, ON_PAPER_DIM,
    ON_PAPER_MUTED, PAPER, PAPER_CARD, PRIMARY_LIGHT, PRIMARY_ON_INK, PRIMARY_ON_PAPER, PRIMARY_ON_PAPER_LARGE,
)
from core.media_render_bundle import NO_LOGO, VAR_BRAND_LOGO, VAR_BRAND_LOGO_MARK, VAR_BRAND_LOGO_ON_DARK
from core.social_kit_tokens import ACCENT_BOLD, ACCENT_USE_FIELD, accent_tokens

LOOK_MODE, LOOK_ACCENT, LOOK_LOGO = "mode", "accent", "logo"
LOOK_KEYS = (LOOK_MODE, LOOK_ACCENT, LOOK_LOGO)
LIGHT, DARK = "light", "dark"
MODES = (LIGHT, DARK)
ACCENT_SPARING = "sparing"
ACCENTS = (ACCENT_BOLD, ACCENT_SPARING)
# What a key means when it is left out: the template as authored, light with its logo. The accent
# has no fixed value of its own: left out, it is the kit's ``accent_use``.
AS_AUTHORED: Dict[str, Any] = {LOOK_MODE: LIGHT, LOOK_LOGO: True}
LOOK_SHAPE = 'look must be null or an object with any of "mode" ("light" or "dark"), "accent" ("bold" or "sparing") and "logo" (true or false)'
# A template built on paper reads the paper's own ground token; a card on it (``--brand-paper-card``)
# does not count, and a template on the dark stage never reads it (its ground is ``--brand-ink``).
PAPER_GROUND = re.compile(r"var\(\s*--brand-paper\s*[,)]")
LOGO_VARIABLES = (VAR_BRAND_LOGO, VAR_BRAND_LOGO_MARK, VAR_BRAND_LOGO_ON_DARK)

# The dark look on a paper template: (token, the token whose value it takes). Each stage tone was
# made to read on the ink, and on the ink's card (``ink-card``) at the paper's own text target, so:
# small accent words take ``primary-light`` (6:1 on the ink, over 5:1 on its card), display words
# ``primary-on-ink`` (4.5:1 on the ink, nearly 4:1 on its card, over the 3:1 large text needs).
# Then back: the stage pieces a paper card lays on its page take the paper's tones.
DARK_SWAP: Tuple[Tuple[str, str], ...] = (
    (PAPER, INK), (PAPER_CARD, INK_CARD), (ON_PAPER, ON_INK), (ON_PAPER_MUTED, ON_INK_MUTED), (ON_PAPER_DIM, ON_INK_DIM),
    (PRIMARY_ON_PAPER, PRIMARY_LIGHT), (PRIMARY_ON_PAPER_LARGE, PRIMARY_ON_INK), (ACCENT_ON_PAPER, ACCENT_ON_INK),
    (INK, PAPER), (INK_CARD, PAPER_CARD), (ON_INK, ON_PAPER), (ON_INK_MUTED, ON_PAPER_MUTED), (ON_INK_DIM, ON_PAPER_DIM),
    (PRIMARY_ON_INK, PRIMARY_ON_PAPER), (PRIMARY_LIGHT, PRIMARY_ON_PAPER), (ACCENT_ON_INK, ACCENT_ON_PAPER),
)


def _allowed(key: str, choice: Any) -> bool:
    """Whether ``choice`` is one of ``key``'s values (a logo is a true boolean, never 0 or 1)."""
    if key == LOOK_LOGO:
        return isinstance(choice, bool)
    allowed = MODES if key == LOOK_MODE else ACCENTS
    return isinstance(choice, str) and choice in allowed


def _chosen(look: Any) -> Dict[str, Any]:
    """The valid choices of a stored look, each that differs from the template as authored; ``{}`` for none."""
    if not isinstance(look, Mapping):
        return {}
    return {
        key: look[key] for key in LOOK_KEYS
        if key in look and _allowed(key, look[key]) and not (key in AS_AUTHORED and AS_AUTHORED[key] == look[key])
    }


def validate_look(value: Any) -> Optional[Dict[str, Any]]:
    """The post's look, checked: ``None`` (or ``{}``) is the kit's own; otherwise any of ``mode``
    (``light``/``dark``), ``accent`` (``bold``/``sparing``) and ``logo`` (``true``/``false``).
    A choice that is the template as authored (light, the logo shown) is left out, and a look
    with nothing left is ``None``. Anything else is :class:`~modules.socials.service.InvalidPost`."""
    from modules.socials.service import InvalidPost  # the service imports this module

    if value is None:
        return None
    if not isinstance(value, Mapping) or not set(value) <= set(LOOK_KEYS):
        raise InvalidPost(LOOK_SHAPE)
    wrong = [key for key in LOOK_KEYS if key in value and not _allowed(key, value[key])]
    if wrong:
        raise InvalidPost(f"look.{wrong[0]} is not one of its choices: {LOOK_SHAPE}")
    return _chosen(value) or None


def built_on_paper(blocks: Any) -> bool:
    """Whether a template lays its page on the kit's paper: its composition reads the paper's own
    ground token, ``var(--brand-paper)``. Only such a template takes light or dark."""
    if not isinstance(blocks, Mapping):
        return False
    text = "\n".join(part for part in (blocks.get("html"), blocks.get("css")) if isinstance(part, str))
    return PAPER_GROUND.search(text) is not None


def looks_of(blocks: Any) -> List[str]:
    """The look choices a template takes, for the gallery: light or dark (``mode``) only when it is
    built on paper; the accent and the logo for every template."""
    return [key for key in LOOK_KEYS if key != LOOK_MODE or built_on_paper(blocks)]


def kit_with_look(kit: Optional[Mapping[str, Any]], look: Any) -> Dict[str, Any]:
    """A copy of the render-ready kit with the look's accent as its ``accent_use``; without an accent
    choice, a copy of the kit as it is. The kit itself is never changed."""
    accent = _chosen(look).get(LOOK_ACCENT)
    copied = dict(kit or {})
    return {**copied, ACCENT_USE_FIELD: accent} if accent else copied


def takes_dark(tokens: Mapping[str, str]) -> bool:
    """Whether a render's tokens hold every stage and paper tone the dark look swaps. A kit without
    them (no hex secondary or text colour: ``stage_palette`` leaves tones out) keeps its light page:
    half a swap would set the template's own fallbacks beside the kit's colours."""
    return {source for _, source in DARK_SWAP} <= set(tokens)


def dark_tokens(tokens: Mapping[str, str], kit: Mapping[str, Any]) -> Dict[str, str]:
    """A paper template's tokens in the dark look (:data:`DARK_SWAP`), with the accent's fill and its
    words (``accent-fill``, ``on-accent-fill``) read again from the swapped tokens under the kit's
    ``accent_use``; a copy of the tokens as they are when the kit cannot take it (:func:`takes_dark`)."""
    if not takes_dark(tokens):
        return dict(tokens)
    swapped = {**tokens, **{token: tokens[source] for token, source in DARK_SWAP}}
    return {**swapped, **accent_tokens(kit, swapped)}


def dark_logos(variables: Mapping[str, Any]) -> Dict[str, Any]:
    """The logo variables on a dark page: a kit with a logo for dark backgrounds shows it where a paper
    card shows its logo and mark, and its mark where the card shows the dark logo on a dark piece
    (which the dark look makes light). A kit without one keeps its variables as they are."""
    mark, dark = variables.get(VAR_BRAND_LOGO_MARK), variables.get(VAR_BRAND_LOGO_ON_DARK)
    if not dark or dark == mark:
        return dict(variables)
    return {**variables, VAR_BRAND_LOGO: dark, VAR_BRAND_LOGO_MARK: dark, VAR_BRAND_LOGO_ON_DARK: mark}


def without_logo(variables: Mapping[str, Any]) -> Dict[str, Any]:
    """The variables with every logo the empty logo (``NO_LOGO``), which each template hides."""
    return {**variables, **{name: NO_LOGO for name in LOGO_VARIABLES}}


def with_look(bundle: Mapping[str, Any], blocks: Any, look: Any, kit: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A copy of one render's ``bundle`` (built from ``kit_with_look(kit, look)``) in the post's look:
    dark on a template built on paper sets its tokens and logos on the stage, and the logo off empties
    every logo variable. A look that leaves everything as authored returns a plain copy."""
    chosen = _chosen(look)
    out = dict(bundle)
    brand = dict(out.get("brand") or {})
    tokens = brand.get("tokens") or {}
    if chosen.get(LOOK_MODE) == DARK and built_on_paper(blocks) and takes_dark(tokens):
        out["brand"] = {**brand, "tokens": dark_tokens(tokens, kit_with_look(kit, chosen))}
        out["variables"] = dark_logos(out.get("variables") or {})
    if chosen.get(LOOK_LOGO) is False:
        out["variables"] = without_logo(out.get("variables") or {})
    return out


__all__ = [
    "DARK", "DARK_SWAP", "LIGHT", "LOOK_ACCENT", "LOOK_KEYS", "LOOK_LOGO", "LOOK_MODE",
    "built_on_paper", "dark_logos", "dark_tokens", "kit_with_look", "looks_of", "takes_dark", "validate_look", "with_look",
    "without_logo",
]
