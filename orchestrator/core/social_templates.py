"""Social templates as data (PRD-251 S1.2, D4).

A ``document_templates`` row whose format is ``social_image`` or ``social_video``
carries a composition in ``blocks``, which media-render renders:

    {
      "html": "<!doctype html>…",           a full Hyperframes composition
      "css": "…",                            optional, injected after the brand styles
      "variables_schema": {                  every {{ name }} the html and css use
        "headline": {"type": "text", "label": "Headline", "max_chars": 60},
        "stat": {"type": "number", "claim": true},
        "subtitle": {"type": "text", "default": ""}
      },
      "sizes": ["1080x1920", "1080x1350"],   WIDTHxHEIGHT, the first is the default
      "audio_plan": {"voice": …, "music": {"track": "deep-house-003", "start": 32.0}, "sfx": […]},  social_video only
      "slots": {                             optional: footage and stills a post may supply
        "hook": {"kind": "video", "label": "Hook footage", "path": "assets/slots/hook.mp4"},
        "app_loop": {"kind": "video", "path": "assets/slots/app_loop.mp4", "generate": false}
      },
      "stills": [{"at": 0.5}, {"at": 1.5, "when": "point_3"}]   social_image only: one PNG each
      "data": {"rows": 5, "label": "row_{n}_label", "value": "row_{n}_value", "source": "source_label"},
      "made_for": "software"                 optional: the one kind of brief the template is for
    }

* **Variables.** A variable is ``text``, a ``number`` or a ``boolean``, and it
  fills in text, never markup: media-render escapes every value (D4, every word
  on screen is template text). A variable with a ``default`` is optional; one
  without must be supplied. ``claim: true`` marks a fact that needs a source (D7).
  The render bundle also sets ``brand.*`` and ``size.*`` for every template
  (``core/media_render_bundle.py``), so the html may use them undeclared. The
  voice lines of the audio plan are template text too: ``{{ name }}`` in a
  line's ``text`` is filled with the variable's value (:func:`fill_text`).
* **Slots.** A slot is a clip or a still the post may supply: generated footage
  from the workspace's own toolkits (D12), copied into our storage. Every
  element that shows it is a ``<video …></video>`` or an ``<img …>`` carrying
  ``data-slot="<name>"`` and ``src`` the slot's path. A filled slot reaches
  media-render as a media file at that path; an empty one has its elements
  removed (:func:`without_slots`), and the template's own motion graphics play in
  its place. A generation toolkit fills a slot only when the post asks for it
  (S1.8); ``"generate": false`` marks a slot only the workspace's own file may
  fill, such as an app's real screen recording (never generated UI, D12):
  :func:`slot_generatable`. ``"required": true`` (F378) marks a slot a post renders
  only once it is filled, a photo the card is nothing without: :func:`slot_required`.
* **Stills.** An image renders as PNG snapshots of its composition, one per
  moment in ``stills`` (US-107): one for a card, one per slide for a carousel.
  A still with ``when`` is taken only when that variable has a value, so a
  carousel's optional slides drop out; the first still is always taken.
  Without ``stills`` an image is one still at 0 s (:func:`still_moments`).
* **Data.** A chart bound to a report (S1.7) names the variables that hold its
  rows, its source chip and its kind (``core/chart_binding.py``): the report's
  top rows fill them, and a render checks they still match the report.
* **Made for (F377, night 11).** A template that shows one kind of business's own
  material says so: ``"made_for": "software"`` marks the videos built on a software
  product's screens (a chat, a task board, a phone app), which the composer and Auto
  pick only for a software brief. Without it, a template suits any business.
* **The brand comes from the brand kit (D4).** Colours, fonts and the logo reach
  a composition as ``--brand-*`` CSS variables, ``{{ brand.logo }}`` and (D5, the
  square mark) ``{{ brand.logo_mark }}``. ``core/social_brand_rule.py`` finds a
  colour (hex, ``rgb()``/``hsl()`` or named), a named font family or a logo baked
  into the template outside a ``var()`` fallback, and a template carrying one is
  refused on save.

Pure: no database, no IO, and nothing outside the standard library, so the
media-render CI job builds the seeded templates' bundles with this very code.
Used by the template service (validation on save), the documents generator and
a Socials post's render (variables, sizes and slots).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.chart_binding import data_errors
from core.social_brand_rule import brand_literals
from core.social_text_values import text_value_problem

# The two formats a social template has. core/models/core.py reads them from
# here for the document_templates format CHECK (the prd251_wave1 migration).
SOCIAL_IMAGE, SOCIAL_VIDEO = "social_image", "social_video"
SOCIAL_TEMPLATE_FORMATS = (SOCIAL_IMAGE, SOCIAL_VIDEO)

BLOCK_KEYS = (
    "html", "css", "variables_schema", "sizes", "audio_plan", "slots", "stills", "data", "durations", "cuts", "made_for",
)
# F377: the kinds of brief a template may be made for; one without ``made_for`` suits any business.
MADE_FOR_SOFTWARE = "software"
MADE_FOR_KINDS = (MADE_FOR_SOFTWARE,)
# PRD-251B (B5, US-B104): the lengths a video template offers, in whole seconds, each a
# complete timeline the composition selects from the root's data-duration. An image
# template declares none; a video without the list offers its root duration alone.
MAX_DECLARED_LENGTHS = 8
MAX_DECLARED_LENGTH_SECONDS = 600
# A shorter length is a cut (``core/social_cuts.py``): ``cuts`` maps it to the stretches of
# the authored timeline it keeps, in order, adding up to the length. A declared length more
# than UNCUT_SHORTFALL_SECONDS shorter than the timeline needs one.
MAX_CUT_STRETCHES = 12
CUT_SUM_TOLERANCE_SECONDS = 0.01
UNCUT_SHORTFALL_SECONDS = 1.0
REQUIRED_BLOCK_KEYS = ("html", "variables_schema", "sizes")
AUDIO_PLAN_KEYS = ("voice", "music", "sfx")
# The music cue (S1.6): a track of media-render's music library by id, and the
# second of the track the video starts at, with its fades. media-render checks
# that the id is in its library and that the window fits the track.
MUSIC_CUE_KEYS = ("track", "start", "fade_in", "fade_out")
MUSIC_TRACK_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MUSIC_TRACK_ID_MAX_CHARS = 64

TEXT, NUMBER, BOOLEAN = "text", "number", "boolean"
VARIABLE_TYPES = (TEXT, NUMBER, BOOLEAN)
COMMON_SPEC_KEYS = ("type", "label", "description", "default", "claim")
SPEC_KEYS_BY_TYPE = {
    TEXT: COMMON_SPEC_KEYS + ("max_chars",),
    NUMBER: COMMON_SPEC_KEYS + ("min", "max"),
    BOOLEAN: COMMON_SPEC_KEYS,
}
# A template's own variable names are flat; the bundle's are dotted (brand.*, size.*).
VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
BUNDLE_VARIABLE_PREFIXES = ("brand.", "size.")

# Bounds. The media-render service enforces its own on every bundle; these keep
# a stored template inside them. A reference video carries about 120 pieces of
# copy (US-106), and media-render takes 200 variables a bundle, six of them the
# brand and size variables the bundle adds.
MAX_COMPOSITION_CHARS = 1_000_000
MAX_VARIABLES = 160
MAX_TEXT_CHARS = 2000
MAX_LABEL_CHARS = 200
MAX_SIZES = 8
MIN_DIMENSION, MAX_DIMENSION = 16, 4096
MAX_REPORTED_ERRORS = 50

# Slots: footage and stills a post may supply, each at its own fixed path.
VIDEO_SLOT, IMAGE_SLOT = "video", "image"
SLOT_EXTENSIONS = {VIDEO_SLOT: ("mp4", "webm", "mov"), IMAGE_SLOT: ("png", "jpg", "jpeg", "webp")}
SLOT_TAGS = {VIDEO_SLOT: "video", IMAGE_SLOT: "img"}
SLOT_SPEC_KEYS = ("kind", "label", "description", "path", "generate", "required")
SLOT_DIR = "assets/slots/"
REQUIRED_SLOT_WHY = "true: a post renders only once it is filled"
MAX_SLOTS = 12

# Stills: the moments an image render snapshots, one PNG each, ten at most (a
# carousel's cover, points and close); the first still carries no ``when``.
STILL_SPEC_KEYS = ("at", "when")
MAX_STILLS = 10
DEFAULT_STILL_AT = 0.0

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")
_LEFTOVER = re.compile(r"\{\{[^{}]*\}\}")
# F377: a figure as the composer writes it: digits, thousands grouped by commas, a decimal part.
NUMBER_TEXT = re.compile(r"^\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*$")
_SIZE = re.compile(r"^(\d{2,4})x(\d{2,4})$")
_HEAD_CLOSE = re.compile(r"</head\s*>", re.IGNORECASE)
_ROOT = re.compile(r"""data-composition-id\s*=\s*["']main["']""", re.IGNORECASE)
_ROOT_TAG = re.compile(r"""<[^<>]*\bdata-composition-id\s*=\s*["']main["'][^<>]*>""", re.IGNORECASE)
_DURATION = re.compile(r"""(?<![\w-])data-duration\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_SLOT_ATTRIBUTE = re.compile(r"""(?<![\w-])data-slot\s*=\s*(["'])(.*?)\1""", re.IGNORECASE)
_SRC_ATTRIBUTE = re.compile(r"""(?<![\w-])src\s*=\s*(["'])(.*?)\1""", re.IGNORECASE)


class SocialTemplateError(ValueError):
    """A social template that breaks the contract; ``errors`` names each problem."""

    def __init__(self, errors: Sequence[Mapping[str, str]]) -> None:
        self.errors = [dict(e) for e in errors][:MAX_REPORTED_ERRORS]
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in self.errors))


class InvalidVariableValues(SocialTemplateError):
    """Values supplied for a render that do not fit their variables (the input, not the template)."""


def is_social_format(fmt: Optional[str]) -> bool:
    return fmt in SOCIAL_TEMPLATE_FORMATS


def _error(field: str, message: str) -> Dict[str, str]:
    return {"field": field, "message": message}


# ── sizes ───────────────────────────────────────────────────────────────────
def parse_size(value: Any) -> Tuple[int, int]:
    """``"1080x1920"`` → ``(1080, 1920)``; ValueError when it is not a size a render can take."""
    match = _SIZE.match(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{value!r} is not WIDTHxHEIGHT, e.g. 1080x1920")
    width, height = int(match.group(1)), int(match.group(2))
    if not (MIN_DIMENSION <= width <= MAX_DIMENSION and MIN_DIMENSION <= height <= MAX_DIMENSION):
        raise ValueError(f"{value!r}: each side must be {MIN_DIMENSION}-{MAX_DIMENSION} px")
    return width, height


def _is_seconds(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stretch_errors(where: str, stretches: Any, root: float, length: int) -> List[Dict[str, str]]:
    """One cut's stretches: ``[start, end]`` pairs in order, inside the authored timeline."""
    if not isinstance(stretches, list) or not 0 < len(stretches) <= MAX_CUT_STRETCHES:
        return [_error(where, f"must list 1 to {MAX_CUT_STRETCHES} stretches of the timeline, e.g. [[0, 4], [12.5, 23.5]]")]
    errors: List[Dict[str, str]] = []
    previous_end = 0.0
    for i, pair in enumerate(stretches):
        if not (isinstance(pair, list) and len(pair) == 2 and all(_is_seconds(v) for v in pair)):
            errors.append(_error(f"{where}[{i}]", "must be [start, end] in seconds"))
            continue
        start, end = float(pair[0]), float(pair[1])
        if start < previous_end or end <= start or end > root:
            errors.append(_error(f"{where}[{i}]", f"must start where the stretch before it ended or later, and end after its start, by {root:g} s"))
        previous_end = max(previous_end, end)
    if not errors and abs(sum(end - start for start, end in stretches) - length) > CUT_SUM_TOLERANCE_SECONDS:
        errors.append(_error(where, "its stretches must add up to the length"))
    return errors


def _cut_errors(blocks: Mapping[str, Any], fmt: str) -> List[Dict[str, str]]:
    """``cuts`` (PRD-251B US-B104): a shorter declared length keeps stretches of the authored timeline."""
    cuts = blocks.get("cuts")
    if cuts is not None and fmt != SOCIAL_VIDEO:
        return [_error("cuts", "an image template has no cuts")]
    if cuts is not None and not isinstance(cuts, dict):
        return [_error("cuts", 'must map a declared length to the stretches it keeps, e.g. {"15": [[0, 4], [12.5, 23.5]]}')]
    root = root_duration(blocks.get("html") or "")
    if fmt != SOCIAL_VIDEO or root is None:
        return []
    cuts, lengths = cuts or {}, [int(d) for d in blocks.get("durations") or []]
    errors = [_error(f"cuts.{key}", "is not a length in durations") for key in cuts if key not in {str(d) for d in lengths}]
    for length in lengths:
        if str(length) in cuts:
            errors += _stretch_errors(f"cuts.{length}", cuts[str(length)], root, length)
        elif length < root - UNCUT_SHORTFALL_SECONDS:
            errors.append(_error("cuts", f"the {length} s length is shorter than the {root:g} s timeline: declare the stretches it keeps"))
    return errors


def _made_for_errors(made_for: Any) -> List[Dict[str, str]]:
    """``made_for`` (F377): one of :data:`MADE_FOR_KINDS`, or left out for a template any business can use."""
    if made_for in MADE_FOR_KINDS:
        return []
    return [_error("made_for", f"must be one of {list(MADE_FOR_KINDS)}, or left out for any business")]


def made_for(blocks: Any) -> Optional[str]:
    """The kind of brief a template is made for (F377), or ``None`` when any business can use it."""
    value = blocks.get("made_for") if isinstance(blocks, Mapping) else None
    return value if value in MADE_FOR_KINDS else None


def _duration_errors(durations: Any, fmt: str) -> List[Dict[str, str]]:
    """``durations``: for a video, a strictly ascending list of whole seconds (PRD-251B B5)."""
    if fmt != SOCIAL_VIDEO:
        return [_error("durations", "an image template declares no durations")]
    if not isinstance(durations, list) or not durations:
        return [_error("durations", "must be a non-empty list of whole seconds, e.g. [15, 30, 40]")]
    if len(durations) > MAX_DECLARED_LENGTHS:
        return [_error("durations", f"at most {MAX_DECLARED_LENGTHS} lengths")]
    errors = []
    for i, value in enumerate(durations):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > MAX_DECLARED_LENGTH_SECONDS:
            errors.append(_error(f"durations[{i}]", f"must be a whole number of seconds from 1 to {MAX_DECLARED_LENGTH_SECONDS}"))
    if not errors and any(b <= a for a, b in zip(durations, durations[1:])):
        errors.append(_error("durations", "must be strictly ascending, each length once"))
    return errors


def _size_errors(sizes: Any) -> List[Dict[str, str]]:
    if not isinstance(sizes, list) or not sizes:
        return [_error("sizes", "must be a non-empty list of WIDTHxHEIGHT sizes, e.g. [\"1080x1920\"]")]
    if len(sizes) > MAX_SIZES:
        return [_error("sizes", f"at most {MAX_SIZES} sizes")]
    errors = []
    for i, size in enumerate(sizes):
        try:
            parse_size(size)
        except ValueError as exc:
            errors.append(_error(f"sizes[{i}]", str(exc)))
    if len(set(map(str, sizes))) != len(sizes):
        errors.append(_error("sizes", "lists a size twice"))
    return errors


# ── variables ───────────────────────────────────────────────────────────────
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _value_problem(spec: Mapping[str, Any], value: Any) -> Optional[str]:
    """Why ``value`` does not fit the variable, or ``None``."""
    kind = spec.get("type")
    if kind == TEXT:
        if not isinstance(value, str):
            return "must be text"
        limit = spec.get("max_chars", MAX_TEXT_CHARS)
        if len(value) > limit:  # F378: the length given, so a re-ask can say by how much
            return f"is longer than {limit} characters ({len(value)} given)"
        return text_value_problem(value)  # F378: never a placeholder or a bare "true"
    if kind == NUMBER:
        if not _is_number(value):
            return "must be a number"
        if spec.get("min") is not None and value < spec["min"]:
            return f"must be at least {spec['min']:g}"
        if spec.get("max") is not None and value > spec["max"]:
            return f"must be at most {spec['max']:g}"
        return None
    return None if isinstance(value, bool) else "must be true or false"


def _spec_errors(name: str, spec: Any) -> List[Dict[str, str]]:
    where = f"variables_schema.{name}"
    if not VARIABLE_NAME.match(name):
        return [_error(where, "a variable name is letters, digits and _ (64 at most), not starting with a digit")]
    if not isinstance(spec, dict):
        return [_error(where, 'must be an object such as {"type": "text"}')]
    kind = spec.get("type")
    if kind not in VARIABLE_TYPES:
        return [_error(f"{where}.type", f"must be one of {list(VARIABLE_TYPES)}")]
    errors = [
        _error(f"{where}.{key}", f"is not a {kind} variable setting ({', '.join(SPEC_KEYS_BY_TYPE[kind])})")
        for key in spec
        if key not in SPEC_KEYS_BY_TYPE[kind]
    ]
    for key in ("label", "description"):
        if key in spec and (not isinstance(spec[key], str) or len(spec[key]) > MAX_LABEL_CHARS):
            errors.append(_error(f"{where}.{key}", f"must be text of at most {MAX_LABEL_CHARS} characters"))
    if "claim" in spec and not isinstance(spec["claim"], bool):
        errors.append(_error(f"{where}.claim", "must be true or false"))
    if "max_chars" in spec:
        limit = spec["max_chars"]
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 < limit <= MAX_TEXT_CHARS:
            errors.append(_error(f"{where}.max_chars", f"must be a whole number from 1 to {MAX_TEXT_CHARS}"))
    for key in ("min", "max"):
        if key in spec and not _is_number(spec[key]):
            errors.append(_error(f"{where}.{key}", "must be a number"))
    if not errors and _is_number(spec.get("min")) and _is_number(spec.get("max")) and spec["min"] > spec["max"]:
        errors.append(_error(where, "min is greater than max"))
    if not errors and "default" in spec:
        problem = _value_problem(spec, spec["default"])
        if problem:
            errors.append(_error(f"{where}.default", problem))
    return errors


def _schema_errors(schema: Any) -> List[Dict[str, str]]:
    if not isinstance(schema, dict):
        return [_error("variables_schema", 'must be an object of variables, e.g. {"headline": {"type": "text"}}')]
    if len(schema) > MAX_VARIABLES:
        return [_error("variables_schema", f"at most {MAX_VARIABLES} variables")]
    return [error for name, spec in schema.items() for error in _spec_errors(str(name), spec)]


def is_bundle_variable(name: str) -> bool:
    return name.startswith(BUNDLE_VARIABLE_PREFIXES)


def placeholders(text: str) -> List[str]:
    """The ``{{ name }}`` names ``text`` uses, in order, each once."""
    return list(dict.fromkeys(PLACEHOLDER.findall(text or "")))


def _placeholder_errors(field: str, text: str, schema: Mapping[str, Any]) -> List[Dict[str, str]]:
    errors = [
        _error(field, f"uses {{{{ {name} }}}}, which variables_schema does not declare")
        for name in placeholders(text)
        if not is_bundle_variable(name) and name not in schema
    ]
    leftover = _LEFTOVER.search(PLACEHOLDER.sub("", text or ""))
    if leftover:
        errors.append(_error(field, f"has a placeholder that is not a variable name: {leftover.group(0)[:60]}"))
    return errors


@dataclass(frozen=True)
class ResolvedVariables:
    """What a render fills in: the values, and what stops it."""

    values: Dict[str, Any]
    missing: List[str]
    invalid: List[str]


def _as_number(value: Any) -> Any:
    """A figure written as text ("412", "1,240", "12.5") as that number; anything else as it is."""
    if not isinstance(value, str) or not NUMBER_TEXT.match(value):
        return value
    number = float(value.replace(",", ""))
    return int(number) if number.is_integer() and "." not in value else number


def resolve_variables(schema: Mapping[str, Any], supplied: Mapping[str, Any]) -> ResolvedVariables:
    """Each declared variable's value: the one supplied, else its default.

    ``missing`` names the variables with neither; ``invalid`` says which values
    do not fit their variable. Undeclared names in ``supplied`` are ignored. A
    number supplied for a text variable is taken as its text (an agent's 42),
    and (F377) a figure written as text for a number variable as that number
    (the composer's "1,240").
    """
    values: Dict[str, Any] = {}
    missing: List[str] = []
    invalid: List[str] = []
    for name, spec in schema.items():
        value = supplied.get(name) if isinstance(supplied, Mapping) else None
        if spec.get("type") == TEXT and _is_number(value):
            value = str(value)
        if spec.get("type") == NUMBER:
            value = _as_number(value)
        if value is None:
            value = spec.get("default")
        if value is None:
            missing.append(name)
            continue
        problem = _value_problem(spec, value)
        if problem:
            invalid.append(f"{name} {problem}")
            continue
        values[name] = value
    return ResolvedVariables(values=values, missing=missing, invalid=invalid)


def claim_names(schema: Mapping[str, Any]) -> List[str]:
    """The variables marked ``claim: true`` (D7: each needs a source)."""
    return [name for name, spec in schema.items() if isinstance(spec, dict) and spec.get("claim") is True]


def _plain(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return "" if value is None else str(value)


def fill_text(text: str, variables: Mapping[str, Any]) -> str:
    """``text`` with each ``{{ name }}`` replaced by its variable as plain text, spaces collapsed.

    For a voice line: it is spoken, never shown as markup, so nothing is escaped.
    A name with no value fills in as nothing.
    """
    filled = PLACEHOLDER.sub(lambda match: _plain(variables.get(match.group(1))), text or "")
    return " ".join(filled.split())


# ── slots ───────────────────────────────────────────────────────────────────
def _slot_element(kind: str, name: str) -> "re.Pattern[str]":
    """The one form an element showing slot ``name`` takes: a whole ``<video …></video>`` or an ``<img …>``."""
    attribute = r"""(?<![\w-])data-slot\s*=\s*(["'])""" + re.escape(name) + r"\1"
    if kind == VIDEO_SLOT:
        return re.compile(r"<video\b[^<>]*?" + attribute + r"[^<>]*>\s*</video\s*>", re.IGNORECASE)
    return re.compile(r"<img\b[^<>]*?" + attribute + r"[^<>]*>", re.IGNORECASE)


def slot_names_in(html: str) -> List[str]:
    """The slot names the html's ``data-slot`` attributes use, each once."""
    return list(dict.fromkeys(match.group(2) for match in _SLOT_ATTRIBUTE.finditer(html or "")))


def without_slots(html: str, slots: Mapping[str, Any], keep: Iterable[str] = ()) -> str:
    """``html`` without the elements of every slot not in ``keep``: an empty slot is not shown.

    ``slots`` is a checked template's; each of its elements is a whole
    ``<video …></video>`` or ``<img …>`` (``validate_social_blocks``), so the
    removal takes exactly those elements and nothing else.
    """
    kept = set(keep)
    for name, spec in slots.items():
        if name not in kept:
            html = _slot_element(spec["kind"], name).sub("", html)
    return html


def slot_required(spec: Mapping[str, Any]) -> bool:
    """F378: whether a post renders only once the slot is filled (``"required": true``): a
    photo the card is nothing without, such as Just the photo's or a before and after."""
    return spec.get("required") is True


def empty_required_slots(blocks: Mapping[str, Any], filled: Iterable[str]) -> List[str]:
    """F378: the label of each slot marked required (:func:`slot_required`) that ``filled``
    (the slots a render shows) leaves empty, in the template's order."""
    slots = blocks.get("slots") if isinstance(blocks.get("slots"), Mapping) else {}
    shown = set(filled)
    return [
        str(spec.get("label") or name)
        for name, spec in slots.items()
        if isinstance(spec, Mapping) and slot_required(spec) and name not in shown
    ]


def slot_generatable(spec: Mapping[str, Any]) -> bool:
    """Whether a generation toolkit may fill the slot (S1.8): every slot but one marked ``"generate": false``."""
    return spec.get("generate") is not False


def _slot_spec_errors(name: Any, spec: Any) -> List[Dict[str, str]]:
    where = f"slots.{name}"
    if not isinstance(name, str) or not VARIABLE_NAME.match(name):
        return [_error(where, "a slot name is letters, digits and _ (64 at most), not starting with a digit")]
    if not isinstance(spec, dict):
        return [_error(where, 'must be an object such as {"kind": "video", "path": "assets/slots/hook.mp4"}')]
    errors = [
        _error(f"{where}.{key}", f"is not a slot setting ({', '.join(SLOT_SPEC_KEYS)})")
        for key in spec
        if key not in SLOT_SPEC_KEYS
    ]
    for key in ("label", "description"):
        if key in spec and (not isinstance(spec[key], str) or len(spec[key]) > MAX_LABEL_CHARS):
            errors.append(_error(f"{where}.{key}", f"must be text of at most {MAX_LABEL_CHARS} characters"))
    for key, why in (("generate", "false: only the workspace's own file fills it"), ("required", REQUIRED_SLOT_WHY)):
        if key in spec and not isinstance(spec[key], bool):
            errors.append(_error(f"{where}.{key}", f"must be true or false ({why})"))
    kind = spec.get("kind")
    if kind not in SLOT_EXTENSIONS:
        return errors + [_error(f"{where}.kind", f"must be one of {list(SLOT_EXTENSIONS)}")]
    allowed = [f"{SLOT_DIR}{name}.{ext}" for ext in SLOT_EXTENSIONS[kind]]
    if spec.get("path") not in allowed:
        errors.append(_error(f"{where}.path", f"must be one of {', '.join(allowed)}"))
    return errors


def _slot_markup_errors(name: str, spec: Mapping[str, Any], html: str, css: str) -> List[Dict[str, str]]:
    """Every element showing the slot is one the removal takes whole, and nothing else names its file."""
    where, kind, path = f"slots.{name}", spec["kind"], spec["path"]
    tag = SLOT_TAGS[kind]
    marked = sum(1 for match in _SLOT_ATTRIBUTE.finditer(html) if match.group(2) == name)
    elements = [match.group(0) for match in _slot_element(kind, name).finditer(html)]
    if not marked:
        return [_error(where, f'no element shows this slot: mark its <{tag}> with data-slot="{name}"')]
    whole = f"<{tag} …></{tag}>" if kind == VIDEO_SLOT else f"<{tag} …>"
    errors = []
    if len(elements) != marked:
        errors.append(_error(where, f'every element marked data-slot="{name}" must be a whole {whole} with nothing inside'))
    for element in elements:
        src = _SRC_ATTRIBUTE.search(element)
        if src is None or src.group(2) != path:
            errors.append(_error(where, f'an element marked data-slot="{name}" must show src="{path}"'))
    if html.count(path) != len(elements) or path in (css or ""):
        errors.append(_error(where, f"{path} is used outside its data-slot elements; an empty slot removes only those"))
    return errors


def _slot_errors(slots: Any, html: str, css: str) -> List[Dict[str, str]]:
    if slots is None:
        return [_error("html", f'data-slot="{name}" names no slot in slots') for name in slot_names_in(html)]
    if not isinstance(slots, dict):
        return [_error("slots", 'must be an object of slots, e.g. {"hook": {"kind": "video", "path": "assets/slots/hook.mp4"}}')]
    if len(slots) > MAX_SLOTS:
        return [_error("slots", f"at most {MAX_SLOTS} slots")]
    errors = [error for name, spec in slots.items() for error in _slot_spec_errors(name, spec)]
    if errors:
        return errors
    for name, spec in slots.items():
        errors += _slot_markup_errors(name, spec, html, css)
    errors += [_error("html", f'data-slot="{name}" names no slot in slots') for name in slot_names_in(html) if name not in slots]
    return errors


# ── stills ──────────────────────────────────────────────────────────────────
def with_root_attribute(html: str, name: str, value: str) -> str:
    """``html`` with the root's ``name`` attribute set to ``value``, added when the root has
    none. Without a root tag the html is returned as it is (the validator reports that)."""
    root = _ROOT_TAG.search(html or "")
    if not root:
        return html
    tag = root.group(0)
    found = re.search(rf"""(?<![\w-]){re.escape(name)}\s*=\s*["']([^"']*)["']""", tag, re.IGNORECASE)
    if found:
        new_tag = tag[: found.start(1)] + value + tag[found.end(1):]
    else:
        new_tag = tag[:-1].rstrip() + f' {name}="{value}">'
    return html[: root.start()] + new_tag + html[root.end():]


def with_root_duration(html: str, seconds: float) -> str:
    """``html`` with the root's ``data-duration`` set to ``seconds`` (PRD-251B B5): a post
    that chose a length renders that timeline (its cut, ``core/social_cuts.py``)."""
    return with_root_attribute(html, "data-duration", f"{seconds:g}")


def root_duration(html: str) -> Optional[float]:
    """The root's ``data-duration`` in seconds, or ``None`` when it is not a plain number."""
    root = _ROOT_TAG.search(html or "")
    duration = _DURATION.search(root.group(0)) if root else None
    try:
        value = float(duration.group(1)) if duration else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def _still_errors(stills: Any, fmt: str, schema: Mapping[str, Any], html: str) -> List[Dict[str, str]]:
    if stills is None:
        return []
    if fmt != SOCIAL_IMAGE:
        return [_error("stills", "a video renders the whole film: stills are for an image, leave them out")]
    if not isinstance(stills, list) or not stills:
        return [_error("stills", 'must be a non-empty list of moments, e.g. [{"at": 0.5}]')]
    if len(stills) > MAX_STILLS:
        return [_error("stills", f"at most {MAX_STILLS} stills")]
    errors: List[Dict[str, str]] = []
    duration = root_duration(html)
    previous: Optional[float] = None
    for i, still in enumerate(stills):
        where = f"stills[{i}]"
        if not isinstance(still, dict):
            errors.append(_error(where, 'must be an object such as {"at": 0.5}'))
            continue
        errors += [
            _error(f"{where}.{key}", f"is not a still setting ({', '.join(STILL_SPEC_KEYS)})")
            for key in still
            if key not in STILL_SPEC_KEYS
        ]
        at = still.get("at")
        if not _is_number(at) or at < 0:
            errors.append(_error(f"{where}.at", "must be a number of seconds, 0 or more"))
        else:
            if previous is not None and at <= previous:
                errors.append(_error(f"{where}.at", "stills must be in time order, each later than the one before"))
            if duration is not None and at >= duration:
                errors.append(_error(f"{where}.at", f"is past the composition's end ({duration:g} s)"))
            previous = at
        if "when" in still:
            if i == 0:
                errors.append(_error(f"{where}.when", "the first still is always taken: leave its when out"))
            elif not isinstance(still["when"], str) or still["when"] not in schema:
                errors.append(_error(f"{where}.when", "must name a variable of variables_schema"))
    return errors


def _has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return value
    return value is not None and not (_is_number(value) and value == 0)


def still_moments(blocks: Mapping[str, Any], values: Mapping[str, Any]) -> List[float]:
    """The moments an image render snapshots, in order: every still whose ``when`` has a value.

    ``values`` are the resolved variables. A template without stills is one
    still at 0 s.
    """
    stills = blocks.get("stills") or [{"at": DEFAULT_STILL_AT}]
    return [
        float(still["at"])
        for still in stills
        if "when" not in still or _has_value(values.get(still["when"]))
    ]


# ── the whole template ──────────────────────────────────────────────────────
def _text_errors(field: str, value: Any, *, required: bool) -> List[Dict[str, str]]:
    if value is None and not required:
        return []
    if not isinstance(value, str) or (required and not value.strip()):
        return [_error(field, "must be text" + (" and not empty" if required else ""))]
    if len(value) > MAX_COMPOSITION_CHARS:
        return [_error(field, f"is longer than {MAX_COMPOSITION_CHARS} characters")]
    return []


def _document_errors(html: str) -> List[Dict[str, str]]:
    errors = []
    if len(_HEAD_CLOSE.findall(html)) != 1:
        errors.append(_error("html", "must be a full document with exactly one </head>"))
    if len(_ROOT.findall(html)) != 1:
        errors.append(_error("html", 'needs exactly one root element with data-composition-id="main"'))
    return errors


def _audio_errors(plan: Any, fmt: str) -> List[Dict[str, str]]:
    if plan is None or plan == {}:
        return []
    if fmt == SOCIAL_IMAGE:
        return [_error("audio_plan", "an image has no audio: leave audio_plan out")]
    if not isinstance(plan, dict):
        return [_error("audio_plan", f"must be an object with {', '.join(AUDIO_PLAN_KEYS)}")]
    errors = [
        _error(f"audio_plan.{key}", f"is not part of an audio plan ({', '.join(AUDIO_PLAN_KEYS)})")
        for key in plan
        if key not in AUDIO_PLAN_KEYS
    ]
    voice = plan.get("voice")
    if voice is not None and not (isinstance(voice, dict) and isinstance(voice.get("lines"), list)):
        errors.append(_error("audio_plan.voice", 'must be an object with a list of lines, e.g. {"lines": [{"id": "l01", "at": 0.3, "text": "…"}]}'))
    if plan.get("music") is not None:
        errors += _music_errors(plan["music"])
    return errors


def _music_errors(music: Any) -> List[Dict[str, str]]:
    """A music cue names a library track and where in it the video starts (S1.6)."""
    if not isinstance(music, dict):
        return [_error("audio_plan.music", 'must name a music library track, e.g. {"track": "deep-house-003", "start": 32.0}')]
    errors = [
        _error(f"audio_plan.music.{key}", f"is not part of a music cue ({', '.join(MUSIC_CUE_KEYS)})")
        for key in music
        if key not in MUSIC_CUE_KEYS
    ]
    track = music.get("track")
    if not isinstance(track, str) or len(track) > MUSIC_TRACK_ID_MAX_CHARS or not MUSIC_TRACK_ID.match(track):
        errors.append(_error("audio_plan.music.track", "must be a music library track id, e.g. deep-house-003"))
    for key in ("start", "fade_in", "fade_out"):
        value = music.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            errors.append(_error(f"audio_plan.music.{key}", "must be a number of seconds, 0 or more"))
    return errors


def voice_lines(plan: Any) -> List[Mapping[str, Any]]:
    """The voice lines of an audio plan ([] when it has none)."""
    voice = plan.get("voice") if isinstance(plan, dict) else None
    lines = voice.get("lines") if isinstance(voice, dict) else None
    return [line for line in lines if isinstance(line, dict)] if isinstance(lines, list) else []


def _voice_placeholder_errors(plan: Any, schema: Mapping[str, Any]) -> List[Dict[str, str]]:
    """A voice line's ``{{ name }}`` must be a declared variable, like the html's."""
    errors: List[Dict[str, str]] = []
    for i, line in enumerate(voice_lines(plan)):
        text = line.get("text")
        if isinstance(text, str):
            errors += _placeholder_errors(f"audio_plan.voice.lines[{i}].text", text, schema)
    return errors


def validate_social_blocks(blocks: Any, fmt: str) -> Dict[str, Any]:
    """Check a social template's ``blocks`` for ``fmt``; a normalised copy, or :class:`SocialTemplateError`.

    Every problem is reported at once, the brand rule (D4) included.
    """
    if not isinstance(blocks, dict):
        raise SocialTemplateError([_error("blocks", f"a {fmt} template needs blocks {{{', '.join(BLOCK_KEYS)}}}")])
    errors = [_error(key, "is required") for key in REQUIRED_BLOCK_KEYS if key not in blocks]
    errors += [_error(key, f"is not part of a social template ({', '.join(BLOCK_KEYS)})") for key in blocks if key not in BLOCK_KEYS]
    html, css = blocks.get("html"), blocks.get("css")
    errors += _text_errors("html", html, required=True) if "html" in blocks else []
    errors += _text_errors("css", css, required=False)
    if "variables_schema" in blocks:
        errors += _schema_errors(blocks["variables_schema"])
    if "sizes" in blocks:
        errors += _size_errors(blocks["sizes"])
    if "durations" in blocks:
        errors += _duration_errors(blocks["durations"], fmt)
    errors += _made_for_errors(blocks["made_for"]) if "made_for" in blocks else []
    errors += _audio_errors(blocks.get("audio_plan"), fmt)
    if not errors:
        schema = blocks["variables_schema"]
        errors += _document_errors(html)
        errors += _placeholder_errors("html", html, schema) + _placeholder_errors("css", css or "", schema)
        errors += _voice_placeholder_errors(blocks.get("audio_plan"), schema)
        errors += _slot_errors(blocks.get("slots"), html, css or "")
        errors += _still_errors(blocks.get("stills"), fmt, schema, html)
        errors += data_errors(blocks["data"], schema) if "data" in blocks else []
        errors += _cut_errors(blocks, fmt)
        errors += [_error(field, message) for field, message in brand_literals(html, css or "")]
    if errors:
        raise SocialTemplateError(errors)
    return {key: copy.deepcopy(blocks[key]) for key in BLOCK_KEYS if key in blocks}


__all__ = [
    "BLOCK_KEYS",
    "IMAGE_SLOT",
    "InvalidVariableValues",
    "MADE_FOR_KINDS",
    "MADE_FOR_SOFTWARE",
    "ResolvedVariables",
    "SOCIAL_IMAGE",
    "SOCIAL_TEMPLATE_FORMATS",
    "SOCIAL_VIDEO",
    "SocialTemplateError",
    "VIDEO_SLOT",
    "claim_names",
    "empty_required_slots",
    "fill_text",
    "is_bundle_variable",
    "is_social_format",
    "made_for",
    "parse_size",
    "placeholders",
    "resolve_variables",
    "root_duration",
    "slot_generatable",
    "slot_names_in",
    "slot_required",
    "still_moments",
    "validate_social_blocks",
    "voice_lines",
    "with_root_attribute",
    "with_root_duration",
    "without_slots",
]
