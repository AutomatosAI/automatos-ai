"""PRD-261 §5 — the scan report a private upload stores (``theme_packages.scan_report``).

What the workspace admin sees for an upload. The server-side checks have already run by the
time this is built: ``read_theme_package`` refuses a package that fails any of them, so each
check that ran is recorded as passed, with what it measured:

* ``sizes``: the package, JS plus CSS, the entry module and the largest mascot clip, against the
  caps in force (``contract.CAP_SETTINGS``, Q25 / D1-a);
* ``cssScope``: ``dist/theme.css`` held to the theme's ``htmlClass`` (``css_scope.py``), or
  ``absent`` when the package ships no style sheet;
* ``looks``: every preset checked against ``looks/schema.json`` (``look_schema.py``).

Two checks are recorded as ``not_run``, each with a plain reason, never as passed:

* ``contrast`` (AA per preset): a look is knob values, and neither ``looks/schema.json`` nor
  ``boot/prepaint.json`` declares which painted colours are text on which ground, so the server
  has no pairs to measure. Each preset is listed so the report shows what is unmeasured;
* ``automated``: the review stage, as this base report records it.

Since W-C1 the routes store ``services/themes/review_stage.py``'s report instead (C-02a): it
starts from these checks, drops ``automated`` and adds the ``uses``, ``clonedFrom`` and LLM
checks and a real verdict. Rows stored before it keep this report and the ``not_run`` verdict.
A private upload is never blocked by the verdict (PRD §5: the admin sees the report; the theme
shows as "Private · not reviewed").
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from services.themes.contract import CAP_SETTINGS
from services.themes.css_scope import STYLE_PATH
from services.themes.look_schema import preset_ids
from services.themes.package import CODE_KINDS, ENTRY_PATH, ThemeFile, cap_bytes, file_kind

SCAN_REPORT_SCHEMA = "automatos-theme-scan/v1"
STATUS_PASS = "pass"
STATUS_ABSENT = "absent"
STATUS_NOT_RUN = "not_run"
VERDICT_NOT_RUN = "not_run"
CLIP_KIND = "clip"

CONTRAST_NOT_RUN = (
    "Not measured: the package declares no text and ground colour pairs for its looks, "
    "so the server has nothing to check AA contrast against."
)
AUTOMATED_NOT_RUN = "The automated review stage is not part of this release yet, so it did not run."


def _bytes_of(files: Mapping[str, ThemeFile], kinds: frozenset) -> List[int]:
    return [len(item.data) for path, item in files.items() if file_kind(path).name in kinds]


def size_check(files: Mapping[str, ThemeFile], package_bytes: int) -> Dict[str, Any]:
    """What the size caps measured, and the caps in force, in bytes."""
    entry = files.get(ENTRY_PATH)
    clips = _bytes_of(files, frozenset({CLIP_KIND}))
    return {
        "status": STATUS_PASS,
        "packageBytes": package_bytes,
        "codeBytes": sum(_bytes_of(files, CODE_KINDS)),
        "entryBytes": len(entry.data) if entry is not None else 0,
        "largestClipBytes": max(clips, default=0),
        "limits": {name: cap_bytes(name) for name in CAP_SETTINGS},
    }


def _preset_list(files: Mapping[str, ThemeFile]) -> List[str]:
    return sorted(preset_ids({path: item.data for path, item in files.items()}))


def build_scan_report(files: Mapping[str, ThemeFile], package_bytes: int) -> Dict[str, Any]:
    """The report stored with a private upload: every server-side check and the stages not yet run."""
    presets = _preset_list(files)
    return {
        "schema": SCAN_REPORT_SCHEMA,
        "verdict": VERDICT_NOT_RUN,
        "checks": {
            "sizes": size_check(files, package_bytes),
            "cssScope": {"status": STATUS_PASS if STYLE_PATH in files else STATUS_ABSENT, "path": STYLE_PATH},
            "looks": {"status": STATUS_PASS, "presets": presets},
            "contrast": {
                "status": STATUS_NOT_RUN,
                "reason": CONTRAST_NOT_RUN,
                "presets": [{"id": preset, "status": STATUS_NOT_RUN} for preset in presets],
            },
            "automated": {"status": STATUS_NOT_RUN, "reason": AUTOMATED_NOT_RUN},
        },
    }
