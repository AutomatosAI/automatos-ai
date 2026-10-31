"""PRD-261 M-03 (host 1.1.0, plan §2.2.11, PB-35) — WCAG AA contrast for theme.json's ``mobile`` block.

A phone draws a theme that has a ``mobile`` block from its tokens alone, so the review measures the
pairs the phone puts text and marks on (``contract.MOBILE_CONTRAST_PAIRS``): eight text pairs held to
``MOBILE_AA_TEXT`` (4.5:1) and two UI pairs held to ``MOBILE_AA_UI`` (3:1), by the WCAG 2.x ratio. It
measures exactly what ``GET /api/themes/mobile/{id}`` (``mobile.py``) can serve (W-M1's seam rule).

* :func:`contrast_ratio` is ``frontend/lib/look-kit/contrast.ts``'s ``contrastRatio``, step for step,
  over ``mobile_tokens.relative_luminance`` (its ``relativeLuminance``, which the tone rule shares) of
  the ``#RRGGBB`` that ``mobile_tokens.parse_colour`` gives (``hsl.ts`` ``tripletToHex`` for a triplet).
  The shared corpus's ``contrast`` section holds both languages to the same ratios
  (``tests/test_prd261_mobile_contrast.py`` and ``frontend/lib/look-kit/__tests__/contrast-parity.test.ts``).
* :func:`measure_mobile` measures every pair of each subject the phone can be served: each scheme of
  ``mobile.tokens`` (``light``, then ``dark`` when given) or, under ``fromLooks``, the prepaint table's
  ``defaults`` (what every member with no readable stored look gets: kind ``defaults``, id
  ``boot/prepaint.json#defaults``, which no look id can be, so a preset named ``defaults`` keeps its
  own) and then each preset in ``looks/presets/`` (kind ``preset``), each painted as the web paints it
  (``mobile_tokens.paint_look``: Studio tokens as ``H S% L%`` triplets only, the one form the web
  paints them in). A subject that gives no colour for a ``MOBILE_REQUIRED_TOKENS`` token
  (``mobile_tokens.missing_tokens``) fails, named with the tokens it lacks: a phone can't draw it.
  Otherwise a pair whose two tokens are not both set is ``not_measured``; a measured pair is ``pass``
  at or over its minimum (the unrounded ratio, as WCAG reads it) and ``fail`` under it, and each
  failure is a sentence naming the scheme, defaults or preset, both tokens and the ratio. The outcome (``status``) is ``fail`` when any subject
  fails, ``pass`` when at least one was measured and none failed, else ``not_measured``. A manifest
  without the block gives None, and the review keeps its unmeasured contrast check;
  ``review_stage.py`` turns the outcome into a level.

The measured entry, as the review report stores it::

    {"source": "tokens" | "fromLooks", "status": "pass" | "fail" | "not_measured",
     "reason": "<one or two sentences>", "problems": ["<a failing subject or pair, named>", ...],
     "schemes" (tokens) | "presets" (fromLooks; the table's defaults first): [
       {"id": "light" | "boot/prepaint.json#defaults" | "<preset id>", "kind": "scheme" | "defaults" | "preset",
        "status": ..., "missing": ["--background", ...],
        "pairs": [{"foreground": "--foreground", "background": "--background", "kind": "text" | "ui",
                   "minimum": 4.5, "status": ..., "ratio": 17.84 | None}, ...]}]}

Standard library only; a table or preset of the wrong shape paints nothing (and so fails), never raises.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from services.themes.contract import MOBILE_AA_TEXT, MOBILE_AA_UI, MOBILE_CONTRAST_PAIRS, parse_json
from services.themes.css_value import PREPAINT_PATH
from services.themes.look_schema import preset_ids
from services.themes.mobile_manifest import MOBILE_FIELD, MOBILE_SCHEMES
from services.themes.mobile_tokens import missing_tokens, paint_look, parse_colour, relative_luminance

# contrast.ts contrastRatio (WCAG 2.x): its flare.
FLARE = 0.05
# The report's ratios, as the corpus writes them.
RATIO_DECIMALS = 2

KIND_TEXT, KIND_UI = "text", "ui"
MINIMUMS: Dict[str, float] = {KIND_TEXT: MOBILE_AA_TEXT, KIND_UI: MOBILE_AA_UI}
KIND_WORDS: Dict[str, str] = {KIND_TEXT: "text", KIND_UI: "UI marks"}
# The block's two sources of colour (mobile_manifest.MOBILE_FIELDS) and the report key each is listed under.
SOURCE_TOKENS, SOURCE_LOOKS = "tokens", "fromLooks"
SUBJECT_KEYS: Dict[str, str] = {SOURCE_TOKENS: "schemes", SOURCE_LOOKS: "presets"}
OUTCOME_PASS, OUTCOME_FAIL, OUTCOME_NOT_MEASURED = "pass", "fail", "not_measured"

SCHEME_LABEL = "mobile.tokens.{scheme}"
PRESET_LABEL = 'Preset "{preset}"'
# The prepaint table's default look, measured as its own subject under fromLooks. Its id is its label, which
# no look id can be (look_schema.LOOK_ID_PATTERN has no '/' or '#'), so it never collides with a preset's.
DEFAULTS_LABEL = f"{PREPAINT_PATH}#defaults"
DEFAULTS_ID = DEFAULTS_LABEL
# Each subject's "kind" (a pair's "kind" is KIND_TEXT or KIND_UI), so a page listing presets can leave the
# table's defaults out (frontend/components/settings/themes/scan-report.tsx).
SUBJECT_SCHEME, SUBJECT_DEFAULTS, SUBJECT_PRESET = "scheme", "defaults", "preset"
FAILED_PAIR = "{subject}: {foreground} on {background} is {ratio:.2f}:1, under the AA minimum of {minimum:g}:1 for {kind}."
UNPAINTED_SUBJECT = "{subject}: gives no {missing}, so a phone can't draw it."
AA_MINIMUMS = f"WCAG AA ({MOBILE_AA_TEXT:g}:1 for text, {MOBILE_AA_UI:g}:1 for UI marks)"
REASONS: Dict[str, str] = {
    OUTCOME_FAIL: (
        "{failed} measured colour pair(s) of the mobile block are under " + AA_MINIMUMS
        + ", so a phone can't be relied on to read this theme."
    ),
    OUTCOME_PASS: "Every measured colour pair of the mobile block meets " + AA_MINIMUMS + ".",
    OUTCOME_NOT_MEASURED: "Not measured: no colour pair of the mobile block has both of its tokens set.",
}
UNPAINTED_REASON = (
    "{unpainted} of the mobile block's colour sets give no --background or --foreground, so a phone can't draw them."
)
UNMEASURED_NOTE = " {unmeasured} pair(s) are not measured: their two tokens are not both set."


class Subject(NamedTuple):
    """One colour set the phone can be served: a scheme, the table's defaults or a preset."""

    subject_id: str
    kind: str
    label: str
    tokens: Any


def contrast_ratio(foreground: str, background: str) -> float:
    """contrast.ts ``contrastRatio``: from 1 (the same luminance) to 21 (black and white)."""
    first, second = relative_luminance(foreground), relative_luminance(background)
    return (max(first, second) + FLARE) / (min(first, second) + FLARE)


def measure_pair(tokens: Mapping[str, Any], pair: Tuple[str, str, str]) -> Dict[str, Any]:
    """One pair of a scheme or painted look: its ratio and ``pass`` or ``fail``, or ``not_measured``."""
    foreground, background, kind = pair
    entry = {"foreground": foreground, "background": background, "kind": kind, "minimum": MINIMUMS[kind]}
    fore, back = parse_colour(tokens.get(foreground)), parse_colour(tokens.get(background))
    if fore is None or back is None:
        return {**entry, "status": OUTCOME_NOT_MEASURED, "ratio": None}
    ratio = contrast_ratio(fore, back)
    status = OUTCOME_PASS if ratio >= MINIMUMS[kind] else OUTCOME_FAIL
    return {**entry, "status": status, "ratio": round(ratio, RATIO_DECIMALS)}


def outcome_of(statuses: Iterable[str]) -> str:
    """``fail`` when any failed, ``pass`` when any passed and none failed, else ``not_measured``."""
    seen = set(statuses)
    if OUTCOME_FAIL in seen:
        return OUTCOME_FAIL
    return OUTCOME_PASS if OUTCOME_PASS in seen else OUTCOME_NOT_MEASURED


def document_of(files: Mapping[str, bytes], path: str) -> Any:
    """A package file's JSON document, or None when it is absent or not JSON."""
    raw = files.get(path)
    if raw is None:
        return None
    document, error = parse_json(raw, path)
    return None if error is not None else document


def _scheme_tokens(block: Mapping[str, Any]) -> List[Subject]:
    """Each scheme of ``mobile.tokens``, light first, with its label and its tokens."""
    tokens = block.get(SOURCE_TOKENS)
    schemes = tokens if isinstance(tokens, dict) else {}
    return [
        Subject(scheme, SUBJECT_SCHEME, SCHEME_LABEL.format(scheme=scheme), schemes[scheme])
        for scheme in MOBILE_SCHEMES if scheme in schemes
    ]


def _look_tokens(files: Mapping[str, bytes]) -> List[Subject]:
    """The table's defaults (no stored look), then each preset by id, each painted as the web paints it."""
    table = document_of(files, PREPAINT_PATH)
    presets = [
        Subject(preset, SUBJECT_PRESET, PRESET_LABEL.format(preset=preset), paint_look(table, document_of(files, path)))
        for preset, path in preset_ids(files).items()
    ]
    return [Subject(DEFAULTS_ID, SUBJECT_DEFAULTS, DEFAULTS_LABEL, paint_look(table, None)), *presets]


def _failure(label: str, pair: Mapping[str, Any]) -> str:
    return FAILED_PAIR.format(
        subject=label, foreground=pair["foreground"], background=pair["background"], ratio=pair["ratio"],
        minimum=pair["minimum"], kind=KIND_WORDS[pair["kind"]],
    )


def _measure_subject(subject: Subject) -> Tuple[Dict[str, Any], List[str]]:
    """One subject's entry and its problems: it fails on a missing required token, else on its pairs."""
    given = subject.tokens if isinstance(subject.tokens, dict) else {}
    missing = missing_tokens(given)
    pairs = [measure_pair(given, pair) for pair in MOBILE_CONTRAST_PAIRS]
    problems = [UNPAINTED_SUBJECT.format(subject=subject.label, missing=" or ".join(missing))] if missing else []
    problems += [_failure(subject.label, pair) for pair in pairs if pair["status"] == OUTCOME_FAIL]
    status = OUTCOME_FAIL if missing else outcome_of(pair["status"] for pair in pairs)
    entry = {"id": subject.subject_id, "kind": subject.kind, "status": status, "missing": missing, "pairs": pairs}
    return entry, problems


def _reason(outcome: str, subjects: List[Mapping[str, Any]]) -> str:
    """The check's sentence(s): what failed or passed, and how many pairs were not measured."""
    pairs = [pair for subject in subjects for pair in subject["pairs"]]
    failed = sum(1 for pair in pairs if pair["status"] == OUTCOME_FAIL)
    unpainted = sum(1 for subject in subjects if subject["missing"])
    unmeasured = sum(1 for pair in pairs if pair["status"] == OUTCOME_NOT_MEASURED)
    sentences = [REASONS[outcome].format(failed=failed)] if outcome != OUTCOME_FAIL or failed else []
    reason = " ".join(sentences + ([UNPAINTED_REASON.format(unpainted=unpainted)] if unpainted else []))
    if unmeasured and outcome != OUTCOME_NOT_MEASURED:
        return reason + UNMEASURED_NOTE.format(unmeasured=unmeasured)
    return reason


def measure_mobile(manifest: Mapping[str, Any], files: Mapping[str, bytes]) -> Optional[Dict[str, Any]]:
    """The mobile block measured per scheme, or per painting under ``fromLooks`` (module docstring); None without one.

    ``files`` is the checked package's ``{path: bytes}``: the presets and the prepaint table under
    ``fromLooks``.
    """
    block = manifest.get(MOBILE_FIELD)
    if not isinstance(block, dict):
        return None
    source = SOURCE_LOOKS if block.get(SOURCE_LOOKS) is True else SOURCE_TOKENS
    subjects = _look_tokens(files) if source == SOURCE_LOOKS else _scheme_tokens(block)
    measured: List[Dict[str, Any]] = []
    problems: List[str] = []
    for subject in subjects:
        entry, found = _measure_subject(subject)
        measured, problems = [*measured, entry], [*problems, *found]
    outcome = outcome_of(subject["status"] for subject in measured)
    return {
        "source": source, "status": outcome, "reason": _reason(outcome, measured), "problems": problems,
        SUBJECT_KEYS[source]: measured,
    }
