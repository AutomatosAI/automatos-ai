"""PRD-271 C04 (D-10) — WCAG AA for the pairs of the layout a ``mobile`` block names, as the review measures them.

A phone draws a theme on its layout from the slots alone, so the review measures the contract's pairs
(``layout_contracts``: a text or mark slot on a ground slot, or on a colour the layout fixes) over the slots
exactly as ``GET /api/themes/mobile/{id}`` resolves them (``layout_paint.resolve_slots``, W-M1's seam rule):

* a block whose slots are all values (or defaults) is measured once (kind ``slots``);
* a block that takes slots from the look is measured for the prepaint table's defaults (what every member
  with no readable stored look gets: kind ``defaults``, id ``boot/prepaint.json#defaults``) and then each
  preset in ``looks/presets/`` (kind ``preset``), each resolved as the answer resolves it: a preset that
  can't fill a slot is measured as the defaults the phone would get instead (``fromDefaults``, and said in the
  reason, so the creator knows members on that preset see the defaults on the phone).

A text pair is held to ``MOBILE_AA_TEXT`` (4.5:1), a ``ui`` pair to ``MOBILE_AA_UI`` (3:1), by
``mobile_contrast.measure_pair``. A subject whose slots can't be filled fails, naming them (the reader
refuses such defaults first, ``MOBILE_CANNOT_PAINT``). :func:`with_layout` adds the measurement to the
skin's (``mobile_contrast.measure_mobile``, still measured: an app without the layout draws the skin)::

    {..., "layout": {"contract": "candy@1", "status": "pass" | "fail" | "not_measured",
                     "subjects": [{"id", "kind": "slots" | "defaults" | "preset", "status", "unresolved": [...],
                                   "fromDefaults": bool,
                                   "pairs": [{"foreground", "background", "kind", "minimum", "status", "ratio"}]}]}}

The check's ``status`` is then the outcome over both, its ``problems`` both lists and its ``reason`` both
sentences; ``review_stage.MOBILE_CONTRAST_FAIL_LEVEL`` turns a failure into the error that blocks a listing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from services.themes.css_value import PREPAINT_PATH
from services.themes.layout_contracts import LayoutContract
from services.themes.layout_paint import Resolved, block_contract, look_slots, resolve_slots
from services.themes.look_schema import preset_ids
from services.themes.mobile_contrast import (
    DEFAULTS_ID, DEFAULTS_LABEL, FAILED_PAIR, KIND_WORDS, OUTCOME_FAIL, OUTCOME_NOT_MEASURED, OUTCOME_PASS,
    PRESET_LABEL, SUBJECT_DEFAULTS, SUBJECT_PRESET, document_of, measure_pair, outcome_of,
)
from services.themes.mobile_manifest import MOBILE_FIELD

SUBJECT_SLOTS = "slots"
SLOTS_ID = "mobile.slots"
SLOTS_LABEL = "mobile.slots ({layout})"
LOOK_LABEL = "{subject}, mobile.slots ({layout})"
UNFILLED = "{subject}: can't fill {slots}, so a phone draws the skin instead of {layout}."
LAYOUT_REASONS: Dict[str, str] = {
    OUTCOME_FAIL: "{failed} measured pair(s) of {layout}'s slots are under WCAG AA, so a phone can't be relied on to read them.",
    OUTCOME_PASS: "Every measured pair of {layout}'s slots meets WCAG AA.",
    OUTCOME_NOT_MEASURED: "No pair of {layout}'s slots was measured.",
}
UNFILLED_REASON = " {unfilled} of {layout}'s slot sets can't be filled."
FELL_BACK_REASON = (
    " {fell_back} preset(s) can't fill every slot of {layout}, so a phone draws them in the defaults (measured as such)."
)


class LayoutSubject(NamedTuple):
    """One set of slots the phone can be served: the block's values, the table's defaults or a preset."""

    subject_id: str
    kind: str
    label: str
    resolved: Resolved


def _subjects(contract: LayoutContract, block: Mapping[str, Any], files: Mapping[str, bytes]) -> List[LayoutSubject]:
    """The block once, or the table's defaults and then each preset by id, each resolved as the answer resolves it."""
    layout = contract.layout
    if not look_slots(block):
        slots = resolve_slots(contract, block, None, None)
        return [LayoutSubject(SLOTS_ID, SUBJECT_SLOTS, SLOTS_LABEL.format(layout=layout), slots)]
    table = document_of(files, PREPAINT_PATH)
    defaults = LayoutSubject(
        DEFAULTS_ID, SUBJECT_DEFAULTS, LOOK_LABEL.format(layout=layout, subject=DEFAULTS_LABEL),
        resolve_slots(contract, block, table, None),
    )
    presets = [
        LayoutSubject(
            preset, SUBJECT_PRESET, LOOK_LABEL.format(layout=layout, subject=PRESET_LABEL.format(preset=preset)),
            resolve_slots(contract, block, table, document_of(files, path)),
        )
        for preset, path in preset_ids(files).items()
    ]
    return [defaults, *presets]


def _failure(label: str, pair: Mapping[str, Any]) -> str:
    return FAILED_PAIR.format(
        subject=label, foreground=pair["foreground"], background=pair["background"], ratio=pair["ratio"],
        minimum=pair["minimum"], kind=KIND_WORDS[pair["kind"]],
    )


def _measure(contract: LayoutContract, subject: LayoutSubject) -> Tuple[Dict[str, Any], List[str]]:
    """One subject's entry and its problems: it fails on a slot it can't fill, else on its pairs."""
    colours = {**contract.fixed, **subject.resolved.slots}
    pairs = [measure_pair(colours, pair) for pair in contract.pairs]
    unresolved = list(subject.resolved.unresolved)
    unfilled = UNFILLED.format(subject=subject.label, slots=", ".join(unresolved), layout=contract.layout)
    problems = [unfilled] if unresolved else []
    problems += [_failure(subject.label, pair) for pair in pairs if pair["status"] == OUTCOME_FAIL]
    status = OUTCOME_FAIL if unresolved else outcome_of(pair["status"] for pair in pairs)
    entry = {
        "id": subject.subject_id, "kind": subject.kind, "status": status, "unresolved": unresolved,
        "fromDefaults": subject.resolved.from_defaults, "pairs": pairs,
    }
    return entry, problems


def measure_layout(block: Mapping[str, Any], files: Mapping[str, bytes]) -> Optional[Tuple[Dict[str, Any], List[str], str]]:
    """The layout's entry, its problems and its sentence(s); None when the block names no layout the host knows."""
    contract = block_contract(block)
    if contract is None:
        return None
    measured, problems = [], []
    for subject in _subjects(contract, block, files):
        entry, found = _measure(contract, subject)
        measured, problems = [*measured, entry], [*problems, *found]
    status = outcome_of(entry["status"] for entry in measured)
    failed = sum(1 for entry in measured for pair in entry["pairs"] if pair["status"] == OUTCOME_FAIL)
    unfilled = sum(1 for entry in measured if entry["unresolved"])
    reason = LAYOUT_REASONS[status].format(failed=failed, layout=contract.layout) if status != OUTCOME_FAIL or failed else ""
    reason += UNFILLED_REASON.format(unfilled=unfilled, layout=contract.layout) if unfilled else ""
    fell_back = sum(1 for entry in measured if entry["fromDefaults"])
    reason += FELL_BACK_REASON.format(fell_back=fell_back, layout=contract.layout) if fell_back else ""
    return {"contract": contract.layout, "status": status, "subjects": measured}, problems, reason.strip()


def with_layout(measured: Dict[str, Any], manifest: Mapping[str, Any], files: Mapping[str, bytes]) -> Dict[str, Any]:
    """The skin's measurement (``measure_mobile``) with the layout's added (module docstring); as given without one."""
    block = manifest.get(MOBILE_FIELD)
    layout = measure_layout(block, files) if isinstance(block, dict) else None
    if layout is None:
        return measured
    entry, problems, reason = layout
    return {
        **measured, "status": outcome_of([measured["status"], entry["status"]]),
        "reason": f'{measured["reason"]} {reason}'.strip(), "problems": [*measured["problems"], *problems], "layout": entry,
    }


__all__ = ["measure_layout", "with_layout"]
