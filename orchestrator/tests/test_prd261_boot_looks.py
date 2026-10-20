"""PRD-261 A-19c — a theme's presets and its tone rule ride its boot (plan §2.2.3, host 1.1.0).

``services/themes/serve.py`` answers the boot of a theme that runs with ``looks``,
``{defaultLook, presets}``: ``looks/presets/*.look.json``, each re-hashed and checked against the
theme's checked ``looks/schema.json``, the default look first and the rest in file-name order, at
most 12 of at most 4 KB each; a preset that fails is dropped and logged. ``boot/prepaint.json``
may carry a tone rule, re-checked with the rest of the table on read (the rule itself is held
to the shared corpus in ``test_prd261_theme_package.py``).

The data is ``e2e/browser/fixtures/hello-a09b/theme/`` (a small theme with looks, not Play), and
``e2e/browser/fixtures/hello-a09b/boot.json`` is the answer, which the frontend's
``boot-server.test.ts`` feeds through the browser's boot gate. The hello package's two presets
ride too.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.models.themes import ThemePackage  # noqa: E402
from services.themes import serve  # noqa: E402
from services.themes.contract import BOOT_PRESET_MAX_BYTES, BOOT_PRESETS_MAX  # noqa: E402
from services.themes.css_value import PREPAINT_PATH, prepaint_errors  # noqa: E402
from services.themes.digest import sha256_hex  # noqa: E402
from services.themes.look_schema import look_file_errors  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

# The browser lane's fixture folder, named literally (scripts/ci/backend_test_plan.py selects this file by it).
A09B_DIR = _ORCH.parent / "e2e" / "browser" / "fixtures" / "hello-a09b"
THEME_DIR = A09B_DIR / "theme"
EXPECTED = json.loads((A09B_DIR / "boot.json").read_text(encoding="utf-8"))
PREFIX = "themes/automatos-e2e/hello-a09b/1.0.0/"
PRESET = "looks/presets/{}.look.json"
SERVE_LOGGER = "services.themes.serve"

Files = Dict[str, bytes]
Reader = Callable[[str, str], Optional[bytes]]


def _a09b_files() -> Files:
    """``{package path: bytes}`` of hello-a09b's theme data."""
    return {path.relative_to(THEME_DIR).as_posix(): path.read_bytes() for path in sorted(THEME_DIR.rglob("*")) if path.is_file()}


def _manifest(**fields: Any) -> Dict[str, Any]:
    return {**EXPECTED["themeJson"], "htmlClass": "hello", **fields}


def _package(files: Files, manifest: Mapping[str, Any], stored: Optional[Files] = None) -> Tuple[ThemePackage, Reader]:
    """A package row listing ``files`` and a reader serving ``stored`` (the same bytes unless given)."""
    rows = {path: {"type": "application/json", "size": len(data), "sha256": sha256_hex(data)} for path, data in files.items()}
    row = ThemePackage(theme_id=manifest["id"], manifest=dict(manifest), file_manifest=rows, storage_prefix=PREFIX)
    served = files if stored is None else stored
    return row, lambda prefix, path: served.get(path) if prefix == PREFIX else None


def _preset(look_id: str, **fields: Any) -> Dict[str, Any]:
    return {**json.loads((THEME_DIR / PRESET.format("paper")).read_text(encoding="utf-8")), "id": look_id, **fields}


def _ids(looks: Optional[Mapping[str, Any]]) -> list:
    return [preset["id"] for preset in (looks or {}).get("presets", [])]


def test_the_a09b_data_is_a_valid_look_set_and_prepaint_table():
    files = _a09b_files()
    schema, errors = look_file_errors(files)
    assert schema is not None and errors == []
    assert prepaint_errors(json.loads(files[PREPAINT_PATH])) == []
    assert json.loads(files[PREPAINT_PATH]) == EXPECTED["prepaint"]


def test_the_boot_carries_the_presets_default_first_and_the_tone_rule():
    row, reader = _package(_a09b_files(), _manifest())
    prepaint = serve.theme_prepaint(row, reader)
    looks = serve.theme_looks(row, reader)
    assert prepaint == EXPECTED["prepaint"]
    assert prepaint["tone"] == {"knob": "ground", "values": {"paper": "light", "ink": "dark", "dusk": "dark"}}
    assert looks["lookGuard"] == EXPECTED["lookGuard"]
    assert looks["looks"] == EXPECTED["looks"]
    assert _ids(looks["looks"]) == ["paper", "dusk", "ink"]
    boot = serve.theme_boot(row, prepaint, looks)
    assert (boot["looks"], boot["prepaint"]["tone"]) == (EXPECTED["looks"], EXPECTED["prepaint"]["tone"])


def test_without_a_default_look_in_theme_json_the_first_preset_by_file_name_leads():
    manifest = {key: value for key, value in _manifest().items() if key != "look"}
    row, reader = _package(_a09b_files(), manifest)
    looks = serve.theme_looks(row, reader)["looks"]
    assert (looks["defaultLook"], _ids(looks)) == ("dusk", ["dusk", "ink", "paper"])


def test_the_hello_package_boot_carries_its_two_presets():
    package = read_theme_package(build.build_package())
    files = {path: item.data for path, item in package.files.items()}
    row, reader = _package(files, package.manifest)
    looks = serve.theme_looks(row, reader)["looks"]
    assert looks == {
        "defaultLook": "default",
        "presets": [json.loads(files[PRESET.format(look_id)]) for look_id in ("default", "night")],
    }


def _failing(kind: str) -> Tuple[Files, Files]:
    """hello-a09b's files with ``dusk`` failing as ``kind``: (listed files, stored files)."""
    files = _a09b_files()
    path = PRESET.format("dusk")
    if kind == "tampered":
        return files, {**files, path: files[path] + b"\n"}
    changed = {
        "schema": json.dumps(_preset("dusk", glow=99)),
        "id": json.dumps(_preset("dawn")),
        "json": "{ not json",
        "size": json.dumps(_preset("dusk")) + " " * BOOT_PRESET_MAX_BYTES,
    }[kind].encode("utf-8")
    return {**files, path: changed}, {**files, path: changed}


@pytest.mark.parametrize("kind", ["tampered", "schema", "id", "json", "size"])
def test_a_preset_that_fails_on_read_is_dropped_and_logged(kind, caplog):
    listed, stored = _failing(kind)
    row, reader = _package(listed, _manifest(), stored)
    with caplog.at_level(logging.ERROR, logger=SERVE_LOGGER):
        looks = serve.theme_looks(row, reader)["looks"]
    assert _ids(looks) == ["paper", "ink"]
    assert any(PRESET.format("dusk") in record.getMessage() for record in caplog.records)


def test_a_preset_over_the_cap_is_dropped_even_when_its_listing_claims_less(caplog):
    files = _a09b_files()
    big = (json.dumps(_preset("dusk")) + " " * BOOT_PRESET_MAX_BYTES).encode("utf-8")
    row, reader = _package({**files, PRESET.format("dusk"): big}, _manifest())
    row.file_manifest = {**row.file_manifest, PRESET.format("dusk"): {**row.file_manifest[PRESET.format("dusk")], "size": 10}}
    with caplog.at_level(logging.ERROR, logger=SERVE_LOGGER):
        assert _ids(serve.theme_looks(row, reader)["looks"]) == ["paper", "ink"]
    assert any(f"over {BOOT_PRESET_MAX_BYTES} bytes" in record.getMessage() for record in caplog.records)


def test_when_the_default_preset_is_dropped_the_next_one_leads(caplog):
    files = _a09b_files()
    path = PRESET.format("paper")
    row, reader = _package(files, _manifest(), {**files, path: files[path] + b" "})
    with caplog.at_level(logging.ERROR, logger=SERVE_LOGGER):
        looks = serve.theme_looks(row, reader)["looks"]
    assert (looks["defaultLook"], _ids(looks)) == ("dusk", ["dusk", "ink"])


def test_more_presets_than_the_cap_ride_as_the_first_twelve_and_are_logged(caplog):
    files = {path: data for path, data in _a09b_files().items() if not path.startswith("looks/presets/")}
    ids = [f"p{number:02d}" for number in range(BOOT_PRESETS_MAX + 2)]
    files.update({PRESET.format(look_id): json.dumps(_preset(look_id)).encode("utf-8") for look_id in ids})
    row, reader = _package(files, _manifest(look="p05"))
    with caplog.at_level(logging.ERROR, logger=SERVE_LOGGER):
        looks = serve.theme_looks(row, reader)["looks"]
    assert BOOT_PRESETS_MAX == 12
    assert _ids(looks) == ["p05", *[look_id for look_id in ids if look_id != "p05"][: BOOT_PRESETS_MAX - 1]]
    assert any(f"more than {BOOT_PRESETS_MAX} presets" in record.getMessage() for record in caplog.records)


def test_no_preset_surviving_gives_no_looks_but_keeps_the_guard(caplog):
    files = _a09b_files()
    stored = {path: (data + b" " if path.startswith("looks/presets/") else data) for path, data in files.items()}
    row, reader = _package(files, _manifest(), stored)
    with caplog.at_level(logging.ERROR, logger=SERVE_LOGGER):
        looks = serve.theme_looks(row, reader)
    assert looks["looks"] is None
    assert looks["lookGuard"] == EXPECTED["lookGuard"]


def test_no_readable_schema_gives_no_looks_and_the_base_boot_carries_none():
    files = {path: data for path, data in _a09b_files().items() if path != "looks/schema.json"}
    row, reader = _package(files, _manifest())
    assert serve.theme_looks(row, reader) == {"lookStorage": None, "lookGuard": None, "looks": None}
    assert serve.base_boot(_manifest())["looks"] is None


def test_a_stored_table_whose_tone_rule_fails_on_read_is_refused_and_logged(caplog):
    table = {**EXPECTED["prepaint"], "tone": {"knob": "ground", "values": {"paper": "light"}}}
    files = {**_a09b_files(), PREPAINT_PATH: json.dumps(table).encode("utf-8")}
    row, reader = _package(files, _manifest())
    with caplog.at_level(logging.ERROR, logger=SERVE_LOGGER), pytest.raises(serve.ThemeFileMissing):
        serve.theme_prepaint(row, reader)
    assert any(PREPAINT_PATH in record.getMessage() for record in caplog.records)
