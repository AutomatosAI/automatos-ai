"""PRD-251D US-D201 (E8): the instant preview of an image post, as one self-contained page.

``POST /api/socials/templates/{id}/preview-page`` (``api/socials_instant_preview.py``)
answers with it, and the editor shows it in a frame with ``sandbox="allow-scripts"`` and no
same origin, a moment after the person stops typing. It is labelled a preview: the file a
person approves is always the render.

The page is the template's own document, put together as a render puts it together, from
the same pieces:

* the template is checked (``validate_social_blocks``) and must be an image of one still. A
  carousel's slides are shown one at a time by the renderer's runtime, which a browser does
  not have, so a carousel keeps its rendered preview;
* the fields resolve as a render resolves them (``resolve_variables``): a value that does not
  fit its field is refused with the reason, and a field left empty with no default is empty
  text, so the template's own script takes its block out (Auto writes it when the post is made);
* ``build_bundle`` makes what a render would: the brand and size variables, the brand tokens,
  the story's safe zone at 9:16, every empty slot's element taken out and the logos as files,
  which the logo variables then name as data: URIs;
* before the fill, three things in the markup change: the timeline library's ``<script src>``
  goes (a stub at the top of the page stands in for it), the ``<audio>`` goes (a still has no
  sound), and a photo spot the post holds a picture for shows :data:`PHOTO_MARKER` where its
  file goes. The editor puts the picture there as a data: URI, read through the post's media route;
* the fill is the renderer's (``core/social_composition.py``): every value HTML-escaped, never
  inside a script or an event attribute. The brand styles end with :data:`FONTS_MARKER`, where
  the editor puts the kit's font faces (:func:`font_faces`), fetched once per
  :func:`fonts_version`, so a kit's megabytes of fonts never travel with a keystroke;
* the elements a render refuses (another document, a base URL, a form, a refresh) are refused;
* the page opens with its own content security policy (:data:`POLICY_META`), before any
  script: inline script and style, data: images and fonts, nothing from the network.

Each marker holds a bare ``&``, which an escaped value never holds (it is always ``&amp;``,
``&lt;`` or another entity), so no typed word can be taken for one. Pure: the caller reads
the template and the brand kit.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# ``_fonts`` is the render bundle's own font rule (which faces, which files), kept in one place.
from core.media_render_bundle import VAR_BRAND_LOGO, VAR_BRAND_LOGO_MARK, VAR_BRAND_LOGO_ON_DARK, _fonts, build_bundle
from core.social_composition import (
    GSAP_PATH,
    CompositionError,
    build_document,
    font_face_rule,
    forbidden_elements,
    token_stylesheet,
)
from core.social_templates import IMAGE_SLOT, SOCIAL_IMAGE, SocialTemplateError, resolve_variables, validate_social_blocks
from modules.socials import post_look
from modules.socials.render import composition_of

# The page's own policy: its inline script and styles run, data: images and fonts load, and
# nothing else, so the page fetches nothing from the network. Its head opens with it.
PAGE_POLICY = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:"
POLICY_META = f'<meta http-equiv="Content-Security-Policy" content="{PAGE_POLICY}">'
# Where the editor puts the kit's @font-face rules: a CSS comment at the end of the brand styles.
FONTS_MARKER = "/*&automatos-preview-fonts;*/"
# Where the editor puts a photo spot's picture: the src of the spot's elements.
PHOTO_MARKER = "&automatos-preview-photo-{slot};"
# The stand-in for the timeline library. An image template lays itself out in its own script and
# only then calls gsap.timeline({paused: true}) and .set(...) on it; every call on the stand-in
# answers the stand-in, so those calls, and a workspace template's .to(...) or .fromTo(...), do
# nothing. "then" is not answered, so nothing that awaits it waits forever.
TIMELINE_STUB = """<script>
(function () {
  "use strict";
  var stub;
  var call = function () { return stub; };
  stub = new Proxy(call, {
    get: function (target, key) { return key === "then" || typeof key === "symbol" ? undefined : call; }
  });
  window.gsap = stub;
})();
</script>"""
REFERENCE_PREFIX = "instant-preview:"
LOGO_VARIABLES = frozenset({VAR_BRAND_LOGO, VAR_BRAND_LOGO_MARK, VAR_BRAND_LOGO_ON_DARK})
FONTS_VERSION_CHARS = 16

NOT_AN_IMAGE = "only an image template has an instant preview; a video keeps its rendered preview"
ONE_STILL_ONLY = "a template of several slides has no instant preview: Render preview shows its slides"
CANNOT_PREVIEW = "this template cannot be previewed"

_TIMELINE_SCRIPT = re.compile(
    r"<script\b[^<>]*?(?<![\w-])src\s*=\s*([\"'])" + re.escape(GSAP_PATH) + r"\1[^<>]*>\s*</script\s*>", re.IGNORECASE
)
# An <audio> with nothing inside, as every image template carries the mix; its content stops at the
# next '<', so the scan stays linear however many an unclosed one leaves.
_AUDIO = re.compile(r"<audio\b[^<>]*>[^<]*</audio\s*>", re.IGNORECASE)
_DOCTYPE = re.compile(r"^\s*<!doctype\b[^<>]*>", re.IGNORECASE)


class PreviewRefused(Exception):
    """The template or the values cannot make a preview; the message says why, for the person."""


@dataclass(frozen=True)
class PreviewRequest:
    """What the editor has on screen: the fields' values, one of the template's sizes (its
    first when ``None``), whether the post is a story, and the photo spots it holds a picture for."""

    values: Mapping[str, Any]
    size: Optional[str] = None
    story: bool = False
    photos: Tuple[str, ...] = ()
    # PRD-251D US-D301: the post's look on screen (checked by ``post_look.validate_look``); None is the kit's own.
    look: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class PageParts:
    """What the fill takes, as a render's bundle carries it: the markup with its changes, the
    template css (with the story's insets), the variables (logos as data: URIs) and the brand tokens."""

    html: str
    css: str
    variables: Mapping[str, Any]
    tokens: Mapping[str, str]


def image_blocks(template: Any) -> Dict[str, Any]:
    """The template's checked ``blocks``: an image of one still, or :class:`PreviewRefused`."""
    blocks = composition_of(template)
    if blocks is None or getattr(template, "format", None) != SOCIAL_IMAGE:
        raise PreviewRefused(NOT_AN_IMAGE)
    try:
        checked = validate_social_blocks(blocks, SOCIAL_IMAGE)
    except SocialTemplateError as exc:
        raise PreviewRefused(f"{CANNOT_PREVIEW}: {exc}") from exc
    if len(checked.get("stills") or []) > 1:  # without stills, an image is one still at 0 s
        raise PreviewRefused(ONE_STILL_ONLY)
    return checked


def preview_values(blocks: Mapping[str, Any], typed: Mapping[str, Any]) -> Dict[str, Any]:
    """Each field's value as a render resolves it; a field left empty with no default is empty
    text. A value that does not fit its field is refused, saying why."""
    resolved = resolve_variables(blocks["variables_schema"], typed)
    if resolved.invalid:
        raise PreviewRefused("; ".join(resolved.invalid))
    return {**resolved.values, **{name: "" for name in resolved.missing}}


def checked_photos(blocks: Mapping[str, Any], photos: Sequence[str]) -> Tuple[str, ...]:
    """The photo spots asked for, each once, sorted; each must be one of the template's image slots."""
    slots = blocks.get("slots") or {}
    asked = sorted(set(photos))
    unknown = [name for name in asked if (slots.get(name) or {}).get("kind") != IMAGE_SLOT]
    if unknown:
        raise PreviewRefused(f"this template has no photo spot {', '.join(unknown)}")
    return tuple(asked)


def photo_marker(slot: str) -> str:
    """The marker a photo spot's elements show where the picture goes."""
    return PHOTO_MARKER.format(slot=slot)


def staged_html(blocks: Mapping[str, Any], photos: Sequence[str]) -> str:
    """The template's markup as the page shows it: no timeline library and no audio, and each
    spot in ``photos`` showing its marker in place of its file's path. A template's slot path is
    used only in its own elements' ``src`` (``validate_social_blocks``), so that is all it changes."""
    markup = _AUDIO.sub("", _TIMELINE_SCRIPT.sub("", blocks["html"]))
    slots = blocks.get("slots") or {}
    for slot in photos:
        src = re.compile(r"(?<![\w-])src\s*=\s*([\"'])" + re.escape(slots[slot]["path"]) + r"\1", re.IGNORECASE)
        markup = src.sub(f'src="{photo_marker(slot)}"', markup)
    return markup


def page_parts(
    template: Any, brand_kit: Optional[Mapping[str, Any]], request: PreviewRequest, *, workspace_id: Any, fallback_name: str = "",
) -> PageParts:
    """The pieces of the page for ``request``, as a render's bundle has them (:class:`PageParts`).

    ``brand_kit`` is the workspace's, render-ready; its uploaded fonts are not needed here.
    :class:`PreviewRefused` for a template that is not an image of one still, a value that does
    not fit its field, a photo spot the template lacks or a size it does not declare.
    """
    blocks = image_blocks(template)
    photos = checked_photos(blocks, request.photos)
    values = preview_values(blocks, request.values)
    staged = {**blocks, "html": staged_html(blocks, photos)}
    try:
        bundle = build_bundle(
            workspace_id=workspace_id, reference=f"{REFERENCE_PREFIX}{getattr(template, 'id', '')}", blocks=staged,
            values=values, brand_kit=post_look.kit_with_look(brand_kit, request.look), fallback_name=fallback_name,
            size=request.size, keep_slots=photos, fmt=SOCIAL_IMAGE, story_safe=request.story,
        )
    except ValueError as exc:  # a size the template does not declare
        raise PreviewRefused(str(exc)) from exc
    bundle = post_look.with_look(bundle, blocks, request.look, brand_kit)  # the look, as a render applies it
    files = {entry["path"]: entry["data_uri"] for entry in bundle.get("files") or []}
    variables = {
        name: files.get(value, value) if name in LOGO_VARIABLES else value for name, value in bundle["variables"].items()
    }
    composition = bundle["composition"]
    return PageParts(html=composition["html"], css=composition["css"], variables=variables, tokens=bundle["brand"]["tokens"])


def with_page_head(document: str) -> str:
    """``document`` with the page's policy and the timeline stand-in first: straight after its
    doctype (so the page keeps the standards mode the renderer draws it in), before the template's
    own head and any script of its. A browser puts both in the head it opens there."""
    doctype = _DOCTYPE.match(document)
    at = doctype.end() if doctype else 0
    return document[:at] + POLICY_META + TIMELINE_STUB + document[at:]


def assemble(parts: PageParts) -> str:
    """The page: the renderer's fill of ``parts`` with the font marker after the brand tokens,
    refused when it holds what a render refuses, opened by the policy and the stand-in."""
    try:
        document = build_document(parts.html, parts.css, parts.variables, token_stylesheet(parts.tokens) + FONTS_MARKER)
    except CompositionError as exc:
        raise PreviewRefused(f"{CANNOT_PREVIEW}: {exc}") from exc
    found = forbidden_elements(document)
    if found:
        raise PreviewRefused(f"{CANNOT_PREVIEW}: the composition contains {', '.join(found)}")
    return with_page_head(document)


def preview_page(
    template: Any, brand_kit: Optional[Mapping[str, Any]], request: PreviewRequest, *, workspace_id: Any, fallback_name: str = "",
) -> str:
    """The instant preview of ``template`` for what the editor has on screen, as one page."""
    return assemble(page_parts(template, brand_kit, request, workspace_id=workspace_id, fallback_name=fallback_name))


def font_faces(brand_kit: Mapping[str, Any]) -> str:
    """The kit's font faces as ``@font-face`` rules, each file a data: URI: what the editor puts
    at :data:`FONTS_MARKER`. The faces a render stages (``media_render_bundle._fonts``): the
    uploaded ones, then the bundled ones of the families the kit names and did not upload."""
    files, faces = _fonts(brand_kit)
    uris = {entry["path"]: entry["data_uri"] for entry in files}
    return "".join(
        font_face_rule(face["family"], face["weight"], face["style"], uris[face["path"]], face["path"].rsplit(".", 1)[-1])
        for face in faces
    )


def fonts_version(brand_kit: Mapping[str, Any]) -> str:
    """A short digest of what decides the kit's faces (its uploaded fonts, its body and heading
    families), read from the stored kit: the editor fetches :func:`font_faces` again only when it changes."""
    uploaded = [
        [str(font.get(key)) for key in ("id", "family", "weight", "style")]
        for font in brand_kit.get("font_files") or []
        if isinstance(font, Mapping)
    ]
    basis = json.dumps([uploaded, brand_kit.get("font_family"), brand_kit.get("heading_font")], sort_keys=True, default=str)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:FONTS_VERSION_CHARS]


__all__ = [
    "FONTS_MARKER",
    "PAGE_POLICY",
    "PHOTO_MARKER",
    "POLICY_META",
    "TIMELINE_STUB",
    "PageParts",
    "PreviewRefused",
    "PreviewRequest",
    "assemble",
    "checked_photos",
    "font_faces",
    "fonts_version",
    "image_blocks",
    "page_parts",
    "photo_marker",
    "preview_page",
    "preview_values",
    "staged_html",
    "with_page_head",
]
