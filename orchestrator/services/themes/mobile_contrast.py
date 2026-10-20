"""PRD-261 M-03 (host 1.1.0, plan §2.2.11, PB-35) — WCAG AA contrast for theme.json's ``mobile`` block.

A phone draws a theme that has a ``mobile`` block from its tokens alone, so the review measures the
pairs the phone puts text and marks on (``contract.MOBILE_CONTRAST_PAIRS``): eight text pairs held to
``MOBILE_AA_TEXT`` (4.5:1) and two UI pairs held to ``MOBILE_AA_UI`` (3:1), by the WCAG 2.x ratio.

* :func:`relative_luminance` and :func:`contrast_ratio` are ``frontend/lib/look-kit/contrast.ts``'s
  ``relativeLuminance`` and ``contrastRatio``, step for step, over the ``#RRGGBB`` that
  ``mobile_tokens.parse_colour`` gives (``hsl.ts`` ``tripletToHex`` for a triplet). The shared corpus's
  ``contrast`` section holds both languages to the same ratios (``tests/test_prd261_mobile_contrast.py``
  and ``frontend/lib/look-kit/__tests__/contrast-parity.test.ts``).
* :func:`measure_mobile` measures every pair of each scheme of ``mobile.tokens`` (``light``, then
  ``dark`` when given) or, under ``fromLooks``, of each preset in ``looks/presets/`` painted through
  ``boot/prepaint.json`` (``mobile_tokens.project_look``). A pair whose two tokens are not both set is
  ``not_measured``; a measured pair is ``pass`` at or over its minimum (the unrounded ratio, as WCAG
  reads it) and ``fail`` under it, and each failure is a sentence naming the scheme or preset, both
  tokens and the ratio. The outcome (``status``) is ``fail`` when any pair fails, ``pass`` when at least
  one pair was measured and none failed, else ``not_measured``. A manifest without the block gives None,
  and the review keeps its unmeasured contrast check; ``review_stage.py`` turns the outcome into a level.

The measured entry, as the review report stores it::

    {"source": "tokens" | "fromLooks", "status": "pass" | "fail" | "not_measured",
     "reason": "<one sentence>", "problems": ["<a failing pair, named>", ...],
     "schemes" (tokens) | "presets" (fromLooks): [
       {"id": "light" | "<preset id>", "status": ...,
        "pairs": [{"foreground": "--foreground", "background": "--background", "kind": "text" | "ui",
                   "minimum": 4.5, "status": ..., "ratio": 17.84 | None}, ...]}]}

Standard library only; a table or preset of the wrong shape measures as unset tokens, never raises.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from services.themes.contract import MOBILE_AA_TEXT, MOBILE_AA_UI, MOBILE_CONTRAST_PAIRS, parse_json
from services.themes.css_value import PREPAINT_PATH
from services.themes.look_schema import preset_ids
from services.themes.mobile_manifest import MOBILE_FIELD, MOBILE_SCHEMES
from services.themes.mobile_tokens import HEX_PATTERN, parse_colour, project_look

# contrast.ts relativeLuminance and contrastRatio (WCAG 2.x), with its constants.
CHANNEL_MAX = 255
HEX_BASE = 16
HEX_CHANNEL_STARTS: Tuple[int, ...] = (1, 3, 5)
LINEAR_KNEE = 0.04045
LINEAR_SLOPE = 12.92
GAMMA_OFFSET = 0.055
GAMMA_SCALE = 1.055
GAMMA = 2.4
RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT = 0.2126, 0.7152, 0.0722
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
FAILED_PAIR = "{subject}: {foreground} on {background} is {ratio:.2f}:1, under the AA minimum of {minimum:g}:1 for {kind}."
AA_MINIMUMS = f"WCAG AA ({MOBILE_AA_TEXT:g}:1 for text, {MOBILE_AA_UI:g}:1 for UI marks)"
REASONS: Dict[str, str] = {
    OUTCOME_FAIL: (
        "{failed} measured colour pair(s) of the mobile block are under " + AA_MINIMUMS
        + ", so a phone can't be relied on to read this theme."
    ),
    OUTCOME_PASS: "Every measured colour pair of the mobile block meets " + AA_MINIMUMS + ".",
    OUTCOME_NOT_MEASURED: "Not measured: no colour pair of the mobile block has both of its tokens set.",
}
UNMEASURED_NOTE = " {unmeasured} pair(s) are not measured: their two tokens are not both set."


def _linear(channel: int) -> float:
    srgb = channel / CHANNEL_MAX
    return srgb / LINEAR_SLOPE if srgb <= LINEAR_KNEE else ((srgb + GAMMA_OFFSET) / GAMMA_SCALE) ** GAMMA


def relative_luminance(hex_colour: str) -> float:
    """contrast.ts ``relativeLuminance`` of a ``#rrggbb`` (either case); ValueError for anything else."""
    if not isinstance(hex_colour, str) or HEX_PATTERN.fullmatch(hex_colour) is None:
        raise ValueError(f"{hex_colour!r} isn't a #rrggbb colour")
    red, green, blue = (int(hex_colour[start:start + 2], HEX_BASE) for start in HEX_CHANNEL_STARTS)
    return RED_WEIGHT * _linear(red) + GREEN_WEIGHT * _linear(green) + BLUE_WEIGHT * _linear(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    """contrast.ts ``contrastRatio``: from 1 (the same luminance) to 21 (black and white)."""
    first, second = relative_luminance(foreground), relative_luminance(background)
    return (max(first, second) + FLARE) / (min(first, second) + FLARE)


def measure_pair(tokens: Mapping[str, Any], pair: Tuple[str, str, str]) -> Dict[str, Any]:
    """One pair of a scheme or projected look: its ratio and ``pass`` or ``fail``, or ``not_measured``."""
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


def _document(files: Mapping[str, bytes], path: str) -> Any:
    """A package file's JSON document, or None when it is absent or not JSON."""
    raw = files.get(path)
    if raw is None:
        return None
    document, error = parse_json(raw, path)
    return None if error is not None else document


def _scheme_tokens(block: Mapping[str, Any]) -> List[Tuple[str, str, Any]]:
    """``(scheme, its label, its tokens)`` for each scheme of ``mobile.tokens``, light first."""
    tokens = block.get(SOURCE_TOKENS)
    schemes = tokens if isinstance(tokens, dict) else {}
    return [(scheme, SCHEME_LABEL.format(scheme=scheme), schemes[scheme]) for scheme in MOBILE_SCHEMES if scheme in schemes]


def _preset_tokens(files: Mapping[str, bytes]) -> List[Tuple[str, str, Any]]:
    """``(preset id, its label, its look projected through the prepaint table)`` for each preset, by id."""
    table = _document(files, PREPAINT_PATH)
    return [
        (preset, PRESET_LABEL.format(preset=preset), project_look(table, _document(files, path)))
        for preset, path in preset_ids(files).items()
    ]


def _failure(label: str, pair: Mapping[str, Any]) -> str:
    return FAILED_PAIR.format(
        subject=label, foreground=pair["foreground"], background=pair["background"], ratio=pair["ratio"],
        minimum=pair["minimum"], kind=KIND_WORDS[pair["kind"]],
    )


def _reason(outcome: str, pairs: List[Mapping[str, Any]]) -> str:
    """The check's sentence: what failed or passed, and how many pairs were not measured."""
    failed = sum(1 for pair in pairs if pair["status"] == OUTCOME_FAIL)
    unmeasured = sum(1 for pair in pairs if pair["status"] == OUTCOME_NOT_MEASURED)
    reason = REASONS[outcome].format(failed=failed)
    if unmeasured and outcome != OUTCOME_NOT_MEASURED:
        return reason + UNMEASURED_NOTE.format(unmeasured=unmeasured)
    return reason


def measure_mobile(manifest: Mapping[str, Any], files: Mapping[str, bytes]) -> Optional[Dict[str, Any]]:
    """The mobile block's pairs measured per scheme or per preset (module docstring), or None without one.

    ``files`` is the checked package's ``{path: bytes}``: the presets and the prepaint table under
    ``fromLooks``.
    """
    block = manifest.get(MOBILE_FIELD)
    if not isinstance(block, dict):
        return None
    source = SOURCE_LOOKS if block.get(SOURCE_LOOKS) is True else SOURCE_TOKENS
    subjects = _preset_tokens(files) if source == SOURCE_LOOKS else _scheme_tokens(block)
    measured = []
    problems: List[str] = []
    for subject_id, label, tokens in subjects:
        pairs = [measure_pair(tokens if isinstance(tokens, dict) else {}, pair) for pair in MOBILE_CONTRAST_PAIRS]
        measured.append({"id": subject_id, "status": outcome_of(pair["status"] for pair in pairs), "pairs": pairs})
        problems += [_failure(label, pair) for pair in pairs if pair["status"] == OUTCOME_FAIL]
    outcome = outcome_of(subject["status"] for subject in measured)
    every_pair = [pair for subject in measured for pair in subject["pairs"]]
    return {
        "source": source, "status": outcome, "reason": _reason(outcome, every_pair), "problems": problems,
        SUBJECT_KEYS[source]: measured,
    }
