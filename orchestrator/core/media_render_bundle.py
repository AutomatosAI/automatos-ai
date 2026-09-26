"""The render bundle media-render takes, built from a social template and the brand kit.

PRD-251 S1.2 (D3, D4, D5). One builder for both callers: a Socials post's render
(``modules/socials/render.py``) and ``generate_document`` with a social format
(``modules/documents/generation_service.py``). It lives in core because those two
feature modules may not import each other (``orchestrator/.importlinter``). Each
caller hands in the workspace brand kit render-ready, every uploaded brand file
already inlined as a data: URI
(``modules/documents/brand_fonts.brand_kit_for_media_render``).

The brand kit becomes:

* ``brand.tokens``: its colours and fonts. media-render declares each as a
  ``--brand-<name>`` custom property, so a template reads ``var(--brand-primary)``
  or ``var(--brand-heading-font)`` and never names a colour or a font (D4). The
  kit's colours come as they are, plus the dark stage a social video reads,
  derived from them with WCAG contrast (``core/brand_palette.py``: ``ink``,
  ``on-ink``, ``primary-on-ink`` and the rest), and the light paper a social
  image reads (``paper``, ``on-paper``, ``primary-on-paper`` and the rest);
* ``files`` and ``brand.fonts``: an uploaded logo at ``assets/brand/logo.<ext>``,
  an uploaded logo mark (D5, the square mark) at ``assets/brand/logo-mark.<ext>``,
  and the kit's font files (D5 ``font_files``) under ``assets/brand/fonts/``, each
  with its ``@font-face``, so ``var(--brand-heading-font)`` can name an uploaded face;
* the variables ``brand.name``, ``brand.tagline``, ``brand.logo`` (the staged
  logo's path, or a transparent pixel when there is no uploaded logo),
  ``brand.logo_mark`` (the staged mark's path; without a mark, whatever
  ``brand.logo`` is), and ``size.width`` / ``size.height`` for the size being
  rendered.

The template becomes the composition, with two things done to it here:

* a slot the caller fills (``slot_media``: slot name → a presigned GET URL on
  our storage) reaches media-render as a media file at the slot's path; a slot
  left empty has its elements taken out of the html, so the template's own
  motion graphics play where the footage would have been. A slot whose file is
  still to come (``keep_slots``: footage a Socials render generates first, S1.8)
  keeps its elements, and ``with_slot_files`` adds its file once it is in our
  storage;
* the audio plan's voice lines are template text: their ``{{ name }}`` are
  filled with the variables, and a line that fills in empty is dropped;
* a ``social_image`` renders as stills (US-107): the bundle asks media-render
  for a PNG snapshot at each of the template's still moments whose ``when``
  variable has a value (``core.social_templates.still_moments``): one for a
  card, one per slide for a carousel.

Kokoro speaks the voice lines from their text inside media-render. When a post
chooses a voice toolkit instead (US-111, ``modules/socials/recipes/voice.py``),
its lines are spoken first and kept in our storage, and ``with_voice_files``
makes each spoken line name its file (a media input at ``assets/voice/``)
instead of its text: nothing else in the bundle changes.

An external ``logo_url`` or ``logo_mark_url`` is never fetched: a render reads
only the files its bundle carries and our own storage (D9), so a logo reaches a
render once it is uploaded. Nothing here generates anything: the renderer assembles (D3).
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from core.brand_palette import paper_palette, stage_palette
from core.social_templates import SOCIAL_IMAGE, SOCIAL_VIDEO, fill_text, parse_size, still_moments, without_slots

logger = logging.getLogger(__name__)

# Token name → the brand kit field it reads.
COLOUR_TOKENS = (
    ("primary", "primary_color"),
    ("secondary", "secondary_color"),
    ("accent", "accent_color"),
    ("text", "text_color"),
)
BODY_FONT_TOKEN = "body-font"
HEADING_FONT_TOKEN = "heading-font"
# media-render's rule for a token value (services/media-render/media_render/bundle.py):
# one CSS value, nothing that ends a declaration, opens a block or fetches.
TOKEN_UNSAFE = re.compile(r"[;{}<>\\\n\r]|/\*|url\(|@import|expression\(", re.IGNORECASE)
MAX_TOKEN_CHARS = 200

BRAND_DIR = "assets/brand/"
FONTS_DIR = "assets/brand/fonts/"
# A voice toolkit's lines (US-111), fetched by media-render from our storage.
VOICE_DIR = "assets/voice/"
LOGO_NAME = "logo"
LOGO_MARK_NAME = "logo-mark"
LOGO_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg"}
FONT_EXTENSIONS = {
    "font/woff2": "woff2",
    "application/font-woff2": "woff2",
    "font/woff": "woff",
    "application/font-woff": "woff",
    "font/ttf": "ttf",
    "font/otf": "otf",
}
FONT_FAMILY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")
FONT_WEIGHTS = frozenset({"normal", "bold"} | {str(weight) for weight in range(100, 1000, 100)})
FONT_STYLES = frozenset({"normal", "italic"})
_DATA_URI = re.compile(r"^data:([a-z0-9.+/-]+);base64,", re.IGNORECASE)
# What {{ brand.logo }} fills in when the kit has no uploaded logo: a 1x1 transparent GIF.
NO_LOGO = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

VAR_BRAND_NAME = "brand.name"
VAR_BRAND_TAGLINE = "brand.tagline"
VAR_BRAND_LOGO = "brand.logo"
VAR_BRAND_LOGO_MARK = "brand.logo_mark"
VAR_SIZE_WIDTH = "size.width"
VAR_SIZE_HEIGHT = "size.height"


def _token(name: str, value: Any) -> Optional[str]:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return None
    if len(text) > MAX_TOKEN_CHARS or TOKEN_UNSAFE.search(text):
        logger.warning("[MediaRender] brand kit %s is not a single CSS value; the template's fallback applies", name)
        return None
    return text


def brand_tokens(kit: Mapping[str, Any]) -> Dict[str, str]:
    """The kit's colours and fonts as the ``--brand-*`` tokens a template reads, its video stage and its paper."""
    body_font = kit.get("font_family")
    raw: Dict[str, Any] = {token: kit.get(field) for token, field in COLOUR_TOKENS}
    raw[BODY_FONT_TOKEN] = body_font
    # D5: the heading font is optional; without one, headings take the body font.
    raw[HEADING_FONT_TOKEN] = kit.get("heading_font") or body_font
    tokens = {name: _token(name, value) for name, value in raw.items()}
    derived = {**stage_palette(kit), **paper_palette(kit)}
    return {**{name: value for name, value in tokens.items() if value is not None}, **derived}


def _data_uri_type(value: Any) -> Optional[str]:
    match = _DATA_URI.match(value) if isinstance(value, str) else None
    return match.group(1).lower() if match else None


def _staged_image(uri: Any, name: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """An uploaded image (a data: URI) as a bundle file named ``name``, and its path; nothing for anything else."""
    ext = LOGO_EXTENSIONS.get(_data_uri_type(uri) or "")
    if ext is None:
        return [], None
    path = f"{BRAND_DIR}{name}.{ext}"
    return [{"path": path, "data_uri": uri}], path


def _logos(kit: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], str, str]:
    """The uploaded logo and logo mark as bundle files, and what ``{{ brand.logo }}`` and ``{{ brand.logo_mark }}`` fill in."""
    logo_files, logo = _staged_image(kit.get("logo_url"), LOGO_NAME)
    mark_files, mark = _staged_image(kit.get("logo_mark_url"), LOGO_MARK_NAME)
    logo = logo or NO_LOGO
    return logo_files + mark_files, logo, mark or logo


def _fonts(kit: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """The kit's font files (D5 ``font_files``, render-ready as data: URIs): bundle files and faces."""
    files: List[Dict[str, str]] = []
    faces: List[Dict[str, str]] = []
    for i, font in enumerate(kit.get("font_files") or []):
        font = font if isinstance(font, Mapping) else {}
        ext = FONT_EXTENSIONS.get(_data_uri_type(font.get("data_uri")) or "")
        family = font.get("family")
        weight, style = str(font.get("weight", 400)), str(font.get("style", "normal"))
        usable = isinstance(family, str) and FONT_FAMILY.match(family) and weight in FONT_WEIGHTS and style in FONT_STYLES
        if ext is None or not usable:
            logger.warning("[MediaRender] brand kit font file %d is not a usable font; skipped", i)
            continue
        path = f"{FONTS_DIR}font-{i}.{ext}"
        files.append({"path": path, "data_uri": font["data_uri"]})
        faces.append({"family": family, "weight": weight, "style": style, "path": path})
    return files, faces


def brand_name(kit: Mapping[str, Any], fallback: str = "") -> str:
    """The name a template shows: the kit's, its company's, else ``fallback`` (the workspace's)."""
    company = kit.get("company") if isinstance(kit.get("company"), Mapping) else {}
    return kit.get("name") or company.get("name") or fallback or ""


def render_size(blocks: Mapping[str, Any], size: Optional[str] = None) -> Tuple[int, int]:
    """``(width, height)`` of ``size``, one the template declares; its first size by default."""
    sizes = list(blocks.get("sizes") or [])
    chosen = size if size is not None else (sizes[0] if sizes else None)
    if chosen not in sizes:
        raise ValueError(f"size {chosen!r} is not one this template declares ({', '.join(map(str, sizes))})")
    return parse_size(chosen)


def _audio(plan: Mapping[str, Any], variables: Mapping[str, Any]) -> Dict[str, Any]:
    """The audio plan with its voice lines filled in; a line that fills in empty is dropped."""
    audio = copy.deepcopy(dict(plan))
    voice = audio.get("voice")
    if isinstance(voice, dict) and isinstance(voice.get("lines"), list):
        lines = []
        for line in voice["lines"]:
            if isinstance(line, dict) and isinstance(line.get("text"), str):
                text = fill_text(line["text"], variables)
                if not text:
                    continue
                line = {**line, "text": text}
            lines.append(line)
        voice["lines"] = lines
        if not lines:
            audio.pop("voice")
    return audio


def _slots(
    blocks: Mapping[str, Any], slot_media: Mapping[str, str], keep_slots: Iterable[str] = ()
) -> Tuple[str, List[Dict[str, str]]]:
    """The html with every empty slot taken out, and the media entries of the
    filled ones. A slot in ``keep_slots`` keeps its elements; its file comes later."""
    slots = blocks.get("slots") or {}
    kept = set(slot_media) | set(keep_slots)
    unknown = sorted(kept - set(slots))
    if unknown:
        raise ValueError(f"this template has no slot {', '.join(unknown)}")
    media = [{"path": slots[name]["path"], "url": url} for name, url in slot_media.items()]
    return without_slots(blocks["html"], slots, keep=kept), media


def build_bundle(
    *,
    workspace_id: Any,
    reference: str,
    blocks: Mapping[str, Any],
    values: Mapping[str, Any],
    brand_kit: Optional[Mapping[str, Any]],
    fallback_name: str = "",
    size: Optional[str] = None,
    slot_media: Optional[Mapping[str, str]] = None,
    keep_slots: Iterable[str] = (),
    fmt: str = SOCIAL_VIDEO,
) -> Dict[str, Any]:
    """The bundle for one render of ``blocks`` (a checked social template of format ``fmt``) at ``size``.

    ``values`` are the template's own variables, already resolved
    (``core.social_templates.resolve_variables``); the brand and size variables
    are added here and always win over a same-named value. ``slot_media`` fills
    slots with footage or stills already in our storage (presigned GET URLs,
    which media-render checks against its allowlist); a slot in ``keep_slots``
    is shown and gets its file later (:func:`with_slot_files`); every other slot
    is empty. A ``social_image`` asks for stills instead of a film.
    """
    kit = brand_kit or {}
    width, height = render_size(blocks, size)
    html, media = _slots(blocks, slot_media or {}, keep_slots)
    logo_files, logo, logo_mark = _logos(kit)
    font_files, faces = _fonts(kit)
    variables = {
        **dict(values),
        VAR_BRAND_NAME: brand_name(kit, fallback_name),
        VAR_BRAND_TAGLINE: kit.get("tagline") or "",
        VAR_BRAND_LOGO: logo,
        VAR_BRAND_LOGO_MARK: logo_mark,
        VAR_SIZE_WIDTH: width,
        VAR_SIZE_HEIGHT: height,
    }
    brand: Dict[str, Any] = {"tokens": brand_tokens(kit)}
    if faces:
        brand["fonts"] = faces
    bundle: Dict[str, Any] = {
        "workspace_id": str(workspace_id),
        "reference": reference,
        "composition": {"html": html, "css": blocks.get("css") or ""},
        "variables": variables,
        "brand": brand,
    }
    if logo_files or font_files:
        bundle["files"] = logo_files + font_files
    if media:
        bundle["media"] = media
    plan = blocks.get("audio_plan")
    audio = _audio(plan, variables) if isinstance(plan, dict) and plan else {}
    if audio:
        bundle["audio"] = audio
    if fmt == SOCIAL_IMAGE:
        bundle["still"] = {"at": still_moments(blocks, values)}
    return bundle


def voice_script(bundle: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``(line id, text)`` of each voice line the bundle speaks from text, in script order."""
    audio = bundle.get("audio") if isinstance(bundle.get("audio"), Mapping) else {}
    voice = audio.get("voice") if isinstance(audio.get("voice"), Mapping) else {}
    return [
        (str(line["id"]), line["text"])
        for line in voice.get("lines") or []
        if isinstance(line, Mapping) and isinstance(line.get("text"), str) and line.get("id") is not None
    ]


def with_voice_files(bundle: Mapping[str, Any], files: Mapping[str, Tuple[str, str]]) -> Dict[str, Any]:
    """A copy of ``bundle`` whose spoken lines are files (US-111, D11).

    ``files`` maps a line id to ``(extension, presigned GET URL on our storage)``.
    Each of those lines keeps its id and start and names
    ``assets/voice/<id>.<extension>`` instead of its text, and that path joins
    the bundle's media, which media-render fetches through its storage
    allowlist. Nothing else changes. Every file must be for a spoken line.
    """
    out = copy.deepcopy(dict(bundle))
    spoken = {line_id for line_id, _ in voice_script(out)}
    unknown = sorted(set(files) - spoken)
    if unknown:
        raise ValueError(f"the bundle speaks no line {', '.join(unknown)}")
    if not files:
        return out
    voice = out["audio"]["voice"]
    media = list(out.get("media") or [])
    lines = []
    for line in voice["lines"]:
        entry = files.get(str(line.get("id"))) if isinstance(line, dict) and "text" in line else None
        if entry is None:
            lines.append(line)
            continue
        extension, url = entry
        path = f"{VOICE_DIR}{line['id']}.{extension}"
        lines.append({"id": line["id"], "at": line["at"], "path": path})
        media.append({"path": path, "url": url})
    voice["lines"] = lines
    out["media"] = media
    return out


def with_slot_files(bundle: Mapping[str, Any], files: Mapping[str, str]) -> Dict[str, Any]:
    """A copy of ``bundle`` whose kept slots carry their files (S1.8, D12).

    ``files`` maps a slot's path (``assets/slots/hook.mp4``) to a presigned GET
    URL on our storage; each joins the bundle's media, which media-render
    fetches through its storage allowlist. A path the composition does not show,
    or one the bundle already carries, is refused. Nothing else changes.
    """
    out = copy.deepcopy(dict(bundle))
    html = str((out.get("composition") or {}).get("html") or "")
    media = list(out.get("media") or [])
    carried = {entry.get("path") for entry in media if isinstance(entry, Mapping)}
    unshown = sorted(path for path in files if path not in html)
    if unshown:
        raise ValueError(f"the composition shows no slot at {', '.join(unshown)}")
    doubled = sorted(path for path in files if path in carried)
    if doubled:
        raise ValueError(f"the bundle already carries {', '.join(doubled)}")
    if not files:
        return out
    out["media"] = media + [{"path": path, "url": url} for path, url in files.items()]
    return out


__all__ = [
    "NO_LOGO",
    "VOICE_DIR",
    "brand_name",
    "brand_tokens",
    "build_bundle",
    "render_size",
    "voice_script",
    "with_slot_files",
    "with_voice_files",
]
