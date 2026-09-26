"""Workspace brand kit (PRD-167 S4, extended in place by PRD-251 D5).

The brand kit is stored on ``workspace.settings['brand_kit']`` — no new table, per the
repo rule ("no new table when an existing one fits"). It supplies ``{{brand.*}}``
variables and the palette/fonts/logo applied to rendered PDFs and DOCX, replacing the
hardcoded Automatos branding (the ``#ff6b35`` orange) that used to live in the
renderers.

PRD-251 D5 (S1.3) adds what a social render reads (``core/media_render_bundle.py``):

* ``font_family`` IS the body font (D5's ``body_font``; the PDF renderers read it
  too), and ``heading_font`` is the headings' font, empty for the body font;
* ``font_files``: uploaded woff2 files, stored like the logo
  (``modules/documents/brand_fonts.py``);
* ``logo_mark_url`` / ``logo_mark_path``: a square mark, separate from the wordmark,
  given as a URL or uploaded, like ``logo_url`` / ``logo_path``;
* ``social_handles``: the brand's handle per Composio toolkit, checked against that
  network's own handle rule;
* ``voice``: three to five tone words and the phrases the brand never uses.

PRD-251 US-115 gives agents the kit through ``platform_get_brand_kit`` and
``platform_update_brand_kit``. Those tools and the REST routes
(``api/document_brand_kit.py``) share what is here: :func:`update_brand_kit`
applies a :class:`BrandKitPatch`, :func:`save_brand_kit` is the kit's one writer,
and :func:`brand_kit_suggestions` supplies the prefill candidates.

Defaults are a neutral professional palette — an unconfigured workspace renders cleanly
(and *not* in Automatos orange).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from core.media_render_bundle import FONT_FAMILY, FONT_STYLES, MAX_TOKEN_CHARS, TOKEN_UNSAFE

logger = logging.getLogger(__name__)

BRAND_KIT_SETTINGS_KEY = "brand_kit"

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# Neutral professional defaults (NOT Automatos orange — that was the hardcoded brand
# this PRD removes from the render paths).
DEFAULT_PRIMARY = "#1a1a2e"
DEFAULT_SECONDARY = "#16213e"
DEFAULT_ACCENT = "#0f3460"
DEFAULT_TEXT = "#1a1a2e"
DEFAULT_FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

# PRD-251 D5: the uploaded font files. Six covers a heading face and a body
# family (regular, bold, italic, bold italic) inside media-render's bundle limit.
MAX_FONT_FILES = 6
FONT_WEIGHT_VALUES = tuple(range(100, 1000, 100))
FONT_ID = re.compile(r"^[0-9a-f]{32}$")
FONT_PATH = re.compile(r"^[A-Za-z0-9-]{1,64}/brand/fonts/[0-9a-f]{32}\.woff2$")
MAX_FONT_FILE_NAME_CHARS = 255

# PRD-251 D5: the brand voice.
MIN_TONE_WORDS, MAX_TONE_WORDS = 3, 5
MAX_TONE_WORD_CHARS = 32
MAX_BANNED_PHRASES = 50
MAX_BANNED_PHRASE_CHARS = 120

# PRD-251 D5: the handles, keyed by Composio toolkit slug. A network with a rule
# below is checked against it; any other connected toolkit takes the generic
# rule (the owner: "be flexible, if they connect via composio"). A handle is
# stored without its "@".
MAX_SOCIAL_HANDLES = 20
TOOLKIT_SLUG = re.compile(r"^[a-z][a-z0-9_]{1,49}$")
HANDLE_RULES: Dict[str, Tuple[Pattern[str], str]] = {
    "twitter": (re.compile(r"^[A-Za-z0-9_]{1,15}$"), "1 to 15 letters, digits or underscores"),
    "instagram": (
        re.compile(r"^(?![.])(?!.*[.][.])(?!.*[.]$)[A-Za-z0-9._]{1,30}$"),
        "1 to 30 letters, digits, periods or underscores, with no period first, last or twice in a row",
    ),
    "tiktok": (
        re.compile(r"^(?!.*[.]$)[A-Za-z0-9._]{2,24}$"),
        "2 to 24 letters, digits, periods or underscores, not ending in a period",
    ),
    "youtube": (re.compile(r"^[A-Za-z0-9._-]{3,30}$"), "3 to 30 letters, digits, periods, hyphens or underscores"),
    "linkedin": (
        re.compile(r"^[A-Za-z0-9-]{3,100}$"),
        "the 3 to 100 letters, digits or hyphens after linkedin.com/company/ or linkedin.com/in/",
    ),
}
GENERIC_HANDLE_RULE: Tuple[Pattern[str], str] = (
    re.compile(r"^[A-Za-z0-9._-]{1,100}$"),
    "1 to 100 letters, digits, periods, hyphens or underscores",
)

# Written only by the upload and delete routes: a client patch never points the
# kit at a stored file.
SERVER_MANAGED_FIELDS = frozenset({"logo_path", "logo_mark_path", "font_files"})
# A patch merges into these records key by key; every other field it names is replaced.
MERGED_RECORDS = ("company", "voice")


class CompanyContact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    address: str = ""
    email: str = ""
    phone: str = ""
    website: str = ""


def _one_line(value: str, what: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) > max_chars:
        raise ValueError(f"each {what} is at most {max_chars} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"each {what} is one line of text")
    return text


def _distinct(values: List[str], what: str, max_chars: int) -> List[str]:
    """The values trimmed, blanks dropped, the first of each case-insensitive repeat kept."""
    seen = set()
    kept: List[str] = []
    for value in values:
        text = _one_line(value, what, max_chars)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            kept.append(text)
    return kept


class BrandVoice(BaseModel):
    """How the brand sounds (PRD-251 D5), for the agents that draft its posts."""

    model_config = ConfigDict(extra="forbid")

    tone: List[str] = Field(default_factory=list)
    banned_phrases: List[str] = Field(default_factory=list)

    @field_validator("tone")
    @classmethod
    def _three_to_five_words(cls, words: List[str]) -> List[str]:
        kept = _distinct(words, "tone word", MAX_TONE_WORD_CHARS)
        if any(not any(ch.isalpha() for ch in word) for word in kept):
            raise ValueError("each tone word needs a letter")
        if kept and not MIN_TONE_WORDS <= len(kept) <= MAX_TONE_WORDS:
            raise ValueError(f"give {MIN_TONE_WORDS} to {MAX_TONE_WORDS} tone words, or none (got {len(kept)})")
        return kept

    @field_validator("banned_phrases")
    @classmethod
    def _banned_phrases(cls, phrases: List[str]) -> List[str]:
        kept = _distinct(phrases, "banned phrase", MAX_BANNED_PHRASE_CHARS)
        if len(kept) > MAX_BANNED_PHRASES:
            raise ValueError(f"at most {MAX_BANNED_PHRASES} banned phrases (got {len(kept)})")
        return kept


class BrandFontFile(BaseModel):
    """One uploaded woff2 file (PRD-251 D5): the face it provides and where it is stored.

    Written by the font upload route only (``modules/documents/brand_fonts.py``).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    weight: int = 400
    style: str = "normal"
    path: str
    file_name: str = ""
    bytes: int = 0

    @field_validator("id")
    @classmethod
    def _font_id(cls, v: str) -> str:
        if not FONT_ID.match(v):
            raise ValueError("must be a font file id")
        return v

    @field_validator("family")
    @classmethod
    def _family(cls, v: str) -> str:
        if not FONT_FAMILY.match(v):
            raise ValueError("must be a font family name: letters, digits, spaces, hyphens or underscores, 64 at most")
        return v

    @field_validator("weight")
    @classmethod
    def _weight(cls, v: int) -> int:
        if v not in FONT_WEIGHT_VALUES:
            raise ValueError("must be a font weight from 100 to 900, in hundreds")
        return v

    @field_validator("style")
    @classmethod
    def _style(cls, v: str) -> str:
        if v not in FONT_STYLES:
            raise ValueError(f"must be one of {', '.join(sorted(FONT_STYLES))}")
        return v

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        if not FONT_PATH.match(v):
            raise ValueError("must be a stored brand font path")
        return v

    @field_validator("file_name")
    @classmethod
    def _file_name(cls, v: str) -> str:
        return _one_line(v, "file name", MAX_FONT_FILE_NAME_CHARS)

    @field_validator("bytes")
    @classmethod
    def _bytes(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must not be negative")
        return v


def normalise_handle(toolkit: str, handle: str) -> str:
    """``handle`` without its "@" if it fits ``toolkit``'s rule; ``ValueError`` naming the rule if not."""
    text = handle.strip()
    text = text[1:] if text.startswith("@") else text
    pattern, rule = HANDLE_RULES.get(toolkit, GENERIC_HANDLE_RULE)
    if not pattern.match(text):
        raise ValueError(f"the {toolkit} handle {handle.strip()!r} does not fit: {rule}")
    return text


class BrandKit(BaseModel):
    """Validated brand kit. Stored as a plain dict; this model gates writes and
    supplies defaults on read."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    tagline: str = ""
    logo_url: str = ""
    # PRD-242 S3: storage-relative path of an UPLOADED logo (``<ws>/brand/logo.png``).
    # Server-managed — set by the logo upload route, never by a client PUT. When
    # present the renderers inline it (modules.documents.brand_logo).
    logo_path: str = ""
    primary_color: str = DEFAULT_PRIMARY
    secondary_color: str = DEFAULT_SECONDARY
    accent_color: str = DEFAULT_ACCENT
    text_color: str = DEFAULT_TEXT
    # The body font (PRD-251 D5's body_font): every renderer reads this key.
    font_family: str = DEFAULT_FONT
    company: CompanyContact = Field(default_factory=CompanyContact)
    # PRD-251 D5 (S1.3), what a social render reads.
    # The headings' font; empty means the body font.
    heading_font: str = ""
    # Uploaded woff2 files. Server-managed, like logo_path.
    font_files: List[BrandFontFile] = Field(default_factory=list)
    # A square mark, separate from the wordmark: a URL, or an upload at
    # logo_mark_path (server-managed, like logo_path).
    logo_mark_url: str = ""
    logo_mark_path: str = ""
    social_handles: Dict[str, str] = Field(default_factory=dict)
    voice: BrandVoice = Field(default_factory=BrandVoice)

    @field_validator("primary_color", "secondary_color", "accent_color", "text_color")
    @classmethod
    def _validate_hex(cls, v: str) -> str:
        if v and not _HEX_RE.match(v):
            raise ValueError("must be a hex color such as #1a1a2e or #abc")
        return v

    @field_validator("heading_font")
    @classmethod
    def _one_font_stack(cls, v: str) -> str:
        # media-render declares it as --brand-heading-font: one CSS value.
        text = v.strip()
        if len(text) > MAX_TOKEN_CHARS or TOKEN_UNSAFE.search(text):
            raise ValueError(
                f"must be one font stack such as \"Brand Sans\", sans-serif: {MAX_TOKEN_CHARS} characters "
                "at most, without ; { } < > \\ or url()"
            )
        return text

    @field_validator("font_files")
    @classmethod
    def _font_files(cls, files: List[BrandFontFile]) -> List[BrandFontFile]:
        if len(files) > MAX_FONT_FILES:
            raise ValueError(f"at most {MAX_FONT_FILES} font files")
        if len({f.id for f in files}) != len(files):
            raise ValueError("each font file appears once")
        return files

    @field_validator("social_handles")
    @classmethod
    def _handles(cls, handles: Dict[str, str]) -> Dict[str, str]:
        kept: Dict[str, str] = {}
        for toolkit, handle in handles.items():
            slug = toolkit.strip().lower()
            if not TOOLKIT_SLUG.match(slug):
                raise ValueError(f"{toolkit!r} is not a toolkit name such as linkedin or twitter")
            if handle.strip():  # an empty handle removes that network
                kept[slug] = normalise_handle(slug, handle)
        if len(kept) > MAX_SOCIAL_HANDLES:
            raise ValueError(f"at most {MAX_SOCIAL_HANDLES} social handles")
        return kept


def is_acceptable_logo_url(value: Optional[str]) -> bool:
    """An external logo URL is empty or http(s) — the only schemes the render-time
    fetchers will ever open (PRD-156 S4 keeps the host checks). Applied on the WRITE
    path (:class:`BrandKitPatch`), never on read: a lenient read must not drop a
    whole stored kit over one bad field."""
    if not value:
        return True
    return value.startswith(("http://", "https://"))


class BrandKitPatch(BaseModel):
    """A change to the kit: any of these fields, and one left out (or null) keeps its value.

    The body of ``PUT /api/documents/brand-kit`` and the arguments of
    ``platform_update_brand_kit``. The stored files (:data:`SERVER_MANAGED_FIELDS`)
    are not here: only their upload routes write them. A field this model does not
    know is dropped, as the PUT always did.
    """

    name: Optional[str] = None
    tagline: Optional[str] = None
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    accent_color: Optional[str] = None
    text_color: Optional[str] = None
    font_family: Optional[str] = None
    company: Optional[dict] = None
    # PRD-251 D5 (S1.3).
    heading_font: Optional[str] = None
    logo_mark_url: Optional[str] = None
    social_handles: Optional[Dict[str, str]] = None
    voice: Optional[dict] = None

    @field_validator("logo_url", "logo_mark_url")
    @classmethod
    def _http_only(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        if not is_acceptable_logo_url(v):
            what = "a logo mark" if info.field_name == "logo_mark_url" else "a logo"
            raise ValueError(f"{info.field_name} must be an http(s) URL (or upload {what} instead)")
        return v


PATCH_FIELDS = tuple(BrandKitPatch.model_fields)


def get_brand_kit(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the workspace brand kit as a fully-populated dict (defaults merged in).

    Lenient on read: a stored field that fails validation takes its default, and
    the rest of the kit is kept, so a render never crashes on a bad brand kit and
    one bad field never costs the workspace its colours and logo. Writes go
    through :func:`validate_brand_kit` which is strict.
    """
    raw = (settings or {}).get(BRAND_KIT_SETTINGS_KEY) or {}
    try:
        return BrandKit.model_validate(raw).model_dump()
    except ValidationError as exc:
        bad = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
        logger.warning("[BrandKit] stored brand kit fields %s failed validation; using their defaults", sorted(bad))
        kept = {key: value for key, value in raw.items() if key not in bad} if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001 — read path must not raise
        kept = {}
    try:
        return BrandKit.model_validate(kept).model_dump()
    except Exception:  # noqa: BLE001 — read path must not raise
        logger.warning("[BrandKit] stored brand kit failed validation; using defaults")
        return BrandKit().model_dump()


def validate_brand_kit(patch: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate + merge a brand-kit patch over the existing kit, returning the new dict.

    ``company`` and ``voice`` merge key by key; any other field in the patch
    replaces the stored one (``social_handles`` is the whole map: a network left
    out, or given an empty handle, is removed). Raises ``pydantic.ValidationError``
    (surfaced as 422 by the API) on bad input.
    """
    base = get_brand_kit({BRAND_KIT_SETTINGS_KEY: existing} if existing else None)
    # The stored files (logo, logo mark, fonts) are owned by the upload/delete
    # routes; a client patch cannot point the kit at an arbitrary stored file.
    patch = {k: v for k, v in patch.items() if k not in SERVER_MANAGED_FIELDS}
    merged = {**base, **{k: v for k, v in patch.items() if v is not None}}
    for record in MERGED_RECORDS:
        if isinstance(patch.get(record), dict):
            merged[record] = {**base.get(record, {}), **patch[record]}
    return BrandKit.model_validate(merged).model_dump()


def save_brand_kit(db: Any, workspace: Any, kit: Dict[str, Any]) -> Dict[str, Any]:
    """Store ``kit`` as the workspace's brand kit and commit: the kit's one writer.

    The PUT, the logo, logo mark and font uploads and deletes, and
    ``platform_update_brand_kit`` all save through here.
    """
    # Reassign settings (not in-place mutate) so SQLAlchemy tracks the JSONB change.
    workspace.settings = {**(workspace.settings or {}), BRAND_KIT_SETTINGS_KEY: kit}
    db.commit()
    return kit


def update_brand_kit(db: Any, workspace: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``patch`` to the workspace's stored kit and save the result.

    The patch is read as a :class:`BrandKitPatch` and merged by
    :func:`validate_brand_kit`; either raises ``pydantic.ValidationError`` before
    anything is written (:func:`brand_kit_errors` lists why). The PUT route and
    ``platform_update_brand_kit`` both call this.
    """
    fields = {k: v for k, v in BrandKitPatch.model_validate(patch).model_dump().items() if v is not None}
    existing = (workspace.settings or {}).get(BRAND_KIT_SETTINGS_KEY)
    return save_brand_kit(db, workspace, validate_brand_kit(fields, existing))


def brand_kit_errors(exc: ValidationError) -> List[Dict[str, Any]]:
    """Why a patch was refused: each error's field (``loc``) and rule (``msg``).

    Without the context, which carries the raised exception: the errors must serialise.
    """
    return exc.errors(include_context=False, include_url=False)


def build_brand_suggestions(workspace: Any, business_profile: Any, user: Any) -> Dict[str, Dict[str, str]]:
    """Prefill candidates ``field -> {value, source}``; only fields with a value. Pure."""
    out: Dict[str, Dict[str, str]] = {}

    def put(field: str, value: Any, source: str) -> None:
        if field in out:
            return
        text = str(value).strip() if value is not None else ""
        if text:
            out[field] = {"value": text, "source": source}

    if business_profile is not None:
        put("name", getattr(business_profile, "company_name", None), "business_profile")
        put("company_name", getattr(business_profile, "company_name", None), "business_profile")
        domain = getattr(business_profile, "domain", None)
        if domain:
            website = domain if str(domain).startswith(("http://", "https://")) else f"https://{domain}"
            put("website", website, "business_profile")
        brands = getattr(business_profile, "brands", None) or []
        for brand in brands if isinstance(brands, list) else []:
            if isinstance(brand, dict) and brand.get("logo_url"):
                put("logo_url", brand["logo_url"], "business_profile")
                break
        voice = getattr(business_profile, "voice_notes", None)
        if voice:
            put("tagline", str(voice).strip().splitlines()[0][:120], "business_profile")
    if workspace is not None:
        put("name", getattr(workspace, "name", None), "workspace")
        put("company_name", getattr(workspace, "name", None), "workspace")
    if user is not None:
        put("email", getattr(user, "email", None), "user")
    return out


def brand_kit_suggestions(db: Any, workspace: Any, user: Any = None) -> Dict[str, Dict[str, str]]:
    """Prefill candidates from what the platform already knows (PRD-242 S3).

    The candidates come from the workspace's latest business profile and its
    name, plus, when a person asks, their own ``user`` record. The Brand Kit
    form reads them, and so does ``platform_get_brand_kit``, which has no user.
    """
    from core.models.business_profiles import BusinessProfile

    profile = (
        db.query(BusinessProfile)
        .filter(BusinessProfile.workspace_id == workspace.id)
        .order_by(BusinessProfile.created_at.desc())
        .first()
    )
    return build_brand_suggestions(workspace, profile, user)


__all__ = [
    "BRAND_KIT_SETTINGS_KEY",
    "BrandFontFile",
    "BrandKit",
    "BrandKitPatch",
    "BrandVoice",
    "CompanyContact",
    "HANDLE_RULES",
    "MAX_FONT_FILES",
    "PATCH_FIELDS",
    "SERVER_MANAGED_FIELDS",
    "brand_kit_errors",
    "brand_kit_suggestions",
    "build_brand_suggestions",
    "get_brand_kit",
    "is_acceptable_logo_url",
    "normalise_handle",
    "save_brand_kit",
    "update_brand_kit",
    "validate_brand_kit",
    "DEFAULT_FONT",
]
