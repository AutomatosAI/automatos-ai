"""The fonts a PDF can print in without an upload, and what each of the kit's fonts really prints in (F360).

F360 (night 10c, 6 Oct): every PDF was set in DejaVu. The kit named Geist for the
body and Newsreader for the headings with no font files uploaded, and the image
carries DejaVu only (``fonts-dejavu-core``, orchestrator/Dockerfile), so
WeasyPrint fell through the stack to ``sans-serif`` and ``serif``. Nothing on the
documents or the brand board said so, and the Word file (set in Geist by name)
disagreed with the PDF.

Three families now ship with the code, in ``fonts/`` beside this module (copied
into the image by the Dockerfile's ``COPY . .``): Inter (the kit's default body
font), Geist and Newsreader (the Automatos brand's), each as four static faces
(400, 400 italic, 600, 700) in woff2. All three are under the SIL Open Font
License 1.1, with no reserved font name; each family's licence is
``fonts/OFL-<Family>.txt``. They were cut from the variable fonts in google/fonts
(commit 7085eb89a950e85db5b166b7a58d414544b4140c: ``ofl/inter``,
``ofl/geist``, ``ofl/newsreader``) with fontTools: ``varLib.instancer`` at each
weight (Inter at opsz 14, Newsreader at opsz 16), then ``subset`` to Latin,
Latin-1, Latin Extended-A/B and Additional, general punctuation, currency signs
and common symbols, keeping the default layout features plus ``tnum``, ``lnum``
and ``case`` (the tables use tabular figures).

:func:`font_uses` says, for the body and the headings, which family the kit names
first and which one the PDF really prints in: an uploaded face of that family
(the kit's ``font_files``, render-ready), a bundled family, one of the image's
DejaVu families, or what a generic family (``sans-serif``) resolves to. A family
is a *substitute* only when the first named one is none of those, so the mark the
documents and the board print (``page_fonts``, ``brand_board``) is never shown for
a font the PDF really has. :func:`bundled_faces` hands ``page_fonts`` the faces
of each bundled family a stack resolves to (and that the kit did not upload) as
the same ``{family, weight, style, data_uri}`` entries an upload becomes.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

FONTS_DIR = Path(__file__).with_name("fonts")
WOFF2_MIME = "font/woff2"
# Every bundled family has these faces: (weight, style).
BUNDLED_FACE_SET: Tuple[Tuple[int, str], ...] = ((400, "normal"), (400, "italic"), (600, "normal"), (700, "normal"))
BUNDLED_FAMILIES: Tuple[str, ...] = ("Inter", "Geist", "Newsreader")
# The families the orchestrator image has installed (fonts-dejavu-core, orchestrator/Dockerfile).
SYSTEM_FAMILIES: Tuple[str, ...] = ("DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono")
FALLBACK_SANS, FALLBACK_SERIF, FALLBACK_MONO = SYSTEM_FAMILIES[0], SYSTEM_FAMILIES[1], SYSTEM_FAMILIES[2]
# What fontconfig in the image gives a generic family; an unknown family ends as sans-serif.
GENERIC_FAMILIES: Dict[str, str] = {
    "sans-serif": FALLBACK_SANS, "system-ui": FALLBACK_SANS, "ui-sans-serif": FALLBACK_SANS,
    "cursive": FALLBACK_SANS, "fantasy": FALLBACK_SANS, "ui-rounded": FALLBACK_SANS,
    "serif": FALLBACK_SERIF, "ui-serif": FALLBACK_SERIF,
    "monospace": FALLBACK_MONO, "ui-monospace": FALLBACK_MONO,
}
ROLE_BODY, ROLE_HEADINGS = "Body", "Headings"


@dataclass(frozen=True)
class FontUse:
    """One role's font: the family the kit names first and the family the PDF prints it in."""

    role: str
    named: str
    prints_in: str
    bundled: bool

    @property
    def substitute(self) -> bool:
        """True when the PDF prints this role in a family other than the one the kit names."""
        return self.named.casefold() != self.prints_in.casefold()


def families(stack: Any) -> List[str]:
    """The family names in a CSS font stack, quotes and spaces trimmed. Pure."""
    return [name for name in (part.strip().strip("'\"").strip() for part in str(stack or "").split(",")) if name]


def _canonical(names: Iterable[str]) -> Dict[str, str]:
    return {name.casefold(): name for name in names}


_BUNDLED = _canonical(BUNDLED_FAMILIES)
_SYSTEM = _canonical(SYSTEM_FAMILIES)


def resolve_stack(stack: Any, uploaded: Set[str]) -> Tuple[str, str, bool]:
    """``(the first family named, the family the PDF prints in, whether that is a bundled one)``.

    ``uploaded`` holds the casefolded families the kit has a usable uploaded face of. Pure.
    """
    names = families(stack)
    first = names[0] if names else GENERIC_FAMILIES["sans-serif"]
    for index, name in enumerate(names):
        key = name.casefold()
        if key in uploaded:
            return first, name, False
        if key in _BUNDLED:
            return first, _BUNDLED[key], True
        if key in _SYSTEM:
            return first, _SYSTEM[key], False
        if key in GENERIC_FAMILIES:
            # A stack that starts with a generic family names no particular font: never a substitute.
            return (GENERIC_FAMILIES[key] if index == 0 else first), GENERIC_FAMILIES[key], False
    return first, FALLBACK_SANS, False


def font_uses(body_stack: str, heading_stack: Optional[str], uploaded: Set[str]) -> List[FontUse]:
    """The body's font and, when the kit sets one, the headings'. Pure."""
    roles = [(ROLE_BODY, body_stack)] + ([(ROLE_HEADINGS, heading_stack)] if heading_stack else [])
    return [FontUse(role, *resolve_stack(stack, uploaded)) for role, stack in roles]


def substitutes(uses: Iterable[FontUse]) -> List[FontUse]:
    """The uses printed in another family than the kit names, one per named family."""
    found: Dict[str, FontUse] = {}
    for use in uses:
        if use.substitute:
            found.setdefault(use.named.casefold(), use)
    return list(found.values())


def face_file(family: str, weight: int, style: str) -> Path:
    """The bundled woff2 for one face, e.g. ``fonts/Geist-400-italic.woff2``."""
    return FONTS_DIR / f"{family}-{weight}{'-italic' if style == 'italic' else ''}.woff2"


@lru_cache(maxsize=None)
def _data_uri(family: str, weight: int, style: str) -> str:
    data = face_file(family, weight, style).read_bytes()
    return f"data:{WOFF2_MIME};base64,{base64.b64encode(data).decode('ascii')}"


def bundled_faces(uses: Iterable[FontUse]) -> List[Mapping[str, Any]]:
    """Every face of each bundled family the uses print in, as ``{family, weight, style, data_uri}``."""
    wanted = list(dict.fromkeys(use.prints_in for use in uses if use.bundled))
    return [
        {"family": family, "weight": weight, "style": style, "data_uri": _data_uri(family, weight, style)}
        for family in wanted for weight, style in BUNDLED_FACE_SET
    ]


__all__ = [
    "BUNDLED_FACE_SET", "BUNDLED_FAMILIES", "FONTS_DIR", "FontUse", "ROLE_BODY", "ROLE_HEADINGS", "SYSTEM_FAMILIES",
    "bundled_faces", "face_file", "families", "font_uses", "resolve_stack", "substitutes",
]
