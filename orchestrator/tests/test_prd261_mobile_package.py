"""PRD-261 M-02 — the reader's half of theme.json's ``mobile`` block (``services/themes/package.py``).

Plan §2.2.11, package rules (host 1.1.0):

* ``assets/fonts/mobile/<NAME>.ttf`` (``font/ttf``) and ``.otf`` (``font/otf``) are package files, capped
  at ``FONT_MAX_BYTES``, each one sfnt face: ``00 01 00 00``, ``true`` or ``OTTO`` (a ``ttcf``
  collection refused), 1 to 64 tables, the table directory and every table inside the file; anything
  else is ``BAD_FILE_TYPE`` with the reason. A .woff2 in that folder and a .ttf outside it match no kind;
* ``fromLooks`` paints from ``boot/prepaint.json``: no table, or a table whose ``defaults`` look, painted
  as the web paints it (``mobile_tokens.paint_look``), gives no ``--background`` or ``--foreground``, is
  ``MOBILE_CANNOT_PAINT``; defaults that don't fit the table paint nothing (boot-sanitize.ts drops the
  whole table, :231), so they can't paint either;
* ``clips`` needs every state's ``assets/mascot/<state>.mp4`` and ``.png`` (PB-36), ``stills`` its
  ``.png``, ``none`` nothing; a missing file is ``MISSING_FILE`` naming it.

The font bytes are headers built here (no real font is in the repo). The corpus's mobile ``packages``
cases are run by ``test_prd261_theme_package.py`` (and by the kit's port, M-06).
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from core.models.themes import ThemePackage  # noqa: E402
from services.themes import contract, serve  # noqa: E402
from services.themes.css_value import PREPAINT_PATH  # noqa: E402
from services.themes.package import (  # noqa: E402
    FONT_MAX_BYTES, MOBILE_MASCOT_FILES, PACKAGE_ERROR_CODES, ThemePackageError, file_kind, read_theme_package,
)
from tests.fixtures.themes import build  # noqa: E402

CASES = json.loads((Path(build.__file__).with_name("cases.json")).read_text(encoding="utf-8"))
MOBILE_ENGINES = {"automatos": f"^{contract.MOBILE_MIN_HOST}", "sdk": "^1.0.0"}
TOKENS = {"light": {"--background": "0 0% 100%", "--foreground": "#14181c"}}
SANS, SERIF = "assets/fonts/mobile/sans.ttf", "assets/fonts/mobile/serif.otf"
TRUETYPE, APPLE_TRUETYPE, OPENTYPE = b"\x00\x01\x00\x00", b"true", b"OTTO"
HEADER = struct.Struct(">4sHHHH")  # version, numTables, searchRange, entrySelector, rangeShift
RECORD = struct.Struct(">4sIII")  # tag, checksum, offset, length
TABLE_BYTES = 4
WEBM_CLIP, WAVE = "assets/mascot/idle.webm", "wave"
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 12
PREFIX = "themes/automatos-e2e/hello/1.0.0/"


def _face(version: bytes = TRUETYPE, count: int = 1, spans: Optional[Sequence[Tuple[int, int]]] = None,
          body: int = TABLE_BYTES) -> bytes:
    """An sfnt face: its header, ``count`` table records and one table after the directory.

    By default every record spans that one table; ``spans`` gives each record's ``(offset, length)``.
    """
    directory_end = HEADER.size + RECORD.size * count
    spans = [(directory_end, body)] * count if spans is None else spans
    records = b"".join(RECORD.pack(b"t%03d" % index, 0, offset, length) for index, (offset, length) in enumerate(spans))
    return HEADER.pack(version, count, RECORD.size, 0, 0) + records + b"\x00" * body


def _with_count(data: bytes, count: int) -> bytes:
    """``data`` with its header's numTables replaced (the records left as they are)."""
    return data[:4] + count.to_bytes(2, "big") + data[6:]


def _manifest(mobile: Mapping[str, Any], states: Optional[Sequence[str]] = None) -> bytes:
    """The hello theme.json with ``mobile``, an engines range that admits it and, given states, a mascot."""
    mascot = {} if states is None else {"mascot": {"states": list(states)}}
    return build.changed_manifest({"engines": MOBILE_ENGINES, "mobile": dict(mobile), **mascot})


def _package(mobile: Mapping[str, Any], files: Optional[Mapping[str, Optional[bytes]]] = None,
             variant: str = "hello", states: Optional[Sequence[str]] = None) -> bytes:
    return build.build_package(variant, replace={build.MANIFEST: _manifest(mobile, states), **(files or {})})


def _errors(data: bytes) -> List[Dict[str, str]]:
    try:
        read_theme_package(data)
    except ThemePackageError as exc:
        return exc.errors
    return []


def _only_error(data: bytes, code: str, path: str) -> str:
    """The one refusal's message, once it is ``code`` at ``path`` and a plain sentence."""
    errors = _errors(data)
    assert [(error["code"], error["path"]) for error in errors] == [(code, path)], errors
    message = errors[0]["message"]
    assert message.strip() and "\n" not in message and "Traceback" not in message
    return message


def _table(knobs: Mapping[str, Any], defaults: Mapping[str, Any], pairs: Sequence[Any] = ()) -> bytes:
    return json.dumps({"knobs": knobs, "pairs": list(pairs), "defaults": dict(defaults)}).encode("utf-8")


def _looks_package(table: Optional[bytes]) -> bytes:
    return _package({"fromLooks": True}, {PREPAINT_PATH: table})


# ---------------------------------------------------------------------------
# Mobile fonts: one sfnt face each
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", [TRUETYPE, APPLE_TRUETYPE, OPENTYPE], ids=["truetype", "true", "otto"])
@pytest.mark.parametrize("path, served", [(SANS, "font/ttf"), (SERIF, "font/otf")])
def test_a_truetype_or_opentype_face_is_read_as_its_font_type(version, path, served):
    package = read_theme_package(_package({"tokens": TOKENS, "fonts": {"sans": path}}, {path: _face(version)}))
    assert package.files[path].content_type == served
    assert package.file_manifest[path]["type"] == served


def test_a_face_whose_tables_each_lie_inside_the_file_passes():
    spans = [(44, 4), (40, 8)]  # the directory ends at byte 44, the file at 48: both tables end on its last byte
    assert _errors(_package({"tokens": TOKENS, "fonts": {"sans": SANS}}, {SANS: _face(count=2, spans=spans)})) == []


@pytest.mark.parametrize("data, reason", [
    (_face(b"ttcf"), "font collection (ttcf)"),
    (_face(b"wOF2"), "not a TrueType or OpenType font"),
    (b"", "the file is empty"),
    (TRUETYPE + b"\x00\x01", "header is cut short"),
    (HEADER.pack(TRUETYPE, 0, 0, 0, 0) + b"\x00" * RECORD.size, "lists 0 tables"),
    (_face(count=65), "lists 65 tables"),
    (_with_count(_face(count=1), 2), "table directory runs past the end"),
    (_face(spans=[(28, TABLE_BYTES + 1)]), "table 1 of 1 runs past the end"),
    (_face(count=2, spans=[(44, 4), (44, 5)]), "table 2 of 2 runs past the end"),
], ids=["ttcf", "woff2", "empty", "short-header", "no-tables", "65-tables", "directory-past-end",
        "record-past-end", "second-record-past-end"])
def test_bytes_that_are_not_one_sfnt_face_are_a_bad_file_type_with_the_reason(data, reason):
    message = _only_error(_package({"tokens": TOKENS, "fonts": {"sans": SANS}}, {SANS: data}), "BAD_FILE_TYPE", SANS)
    assert message.startswith(f"{SANS}: ") and reason in message


def test_a_face_at_the_font_cap_passes_and_one_byte_more_is_too_large():
    directory_end = HEADER.size + RECORD.size
    at_cap = FONT_MAX_BYTES - directory_end
    assert _errors(_package({"tokens": TOKENS}, {SANS: _face(body=at_cap)})) == []
    _only_error(_package({"tokens": TOKENS}, {SANS: _face(body=at_cap + 1)}), "FILE_TOO_LARGE", SANS)


@pytest.mark.parametrize("path", [
    "assets/fonts/mobile/x.woff2", "assets/fonts/x.ttf", "assets/fonts/x.otf", "assets/fonts/mobile/x.TTF",
    "assets/fonts/mobile/inner/x.ttf",
])
def test_a_mobile_font_outside_its_folder_or_of_another_format_matches_no_kind(path):
    assert file_kind(path) is None
    _only_error(_package({"tokens": TOKENS}, {path: _face()}), "NOT_ALLOWED", path)


def test_a_stored_mobile_font_is_served_as_the_type_it_is_listed_with():
    package = read_theme_package(_package(
        {"tokens": TOKENS, "fonts": {"sans": SANS, "serif": SERIF}}, {SANS: _face(), SERIF: _face(OPENTYPE)},
    ))
    row = ThemePackage(theme_id=package.manifest["id"], manifest=package.manifest, file_manifest=package.file_manifest,
                       storage_prefix=PREFIX)
    stored = {path: item.data for path, item in package.files.items()}

    def reader(prefix: str, path: str) -> Optional[bytes]:
        return stored.get(path) if prefix == PREFIX else None

    for path, served in ((SANS, "font/ttf"), (SERIF, "font/otf")):
        data, content_type, _ = serve.verified_file(row, path, reader)
        assert (data, content_type) == (stored[path], served)


# ---------------------------------------------------------------------------
# fromLooks: the defaults look paints the ground and its text
# ---------------------------------------------------------------------------


GROUND = {"paper": {"--background": "40 33.33% 94.12%", "--foreground": "#1a1814"},
          "ink": {"--background": "210 20% 7.84%", "--foreground": "#f0f4f8"}}


def test_from_looks_with_no_prepaint_table_cannot_paint():
    message = _only_error(_looks_package(None), "MOBILE_CANNOT_PAINT", PREPAINT_PATH)
    assert PREPAINT_PATH in message and '"mobile.tokens"' in message


def test_from_looks_whose_defaults_paint_the_ground_and_its_text_passes():
    assert _errors(_looks_package(_table({"ground": GROUND}, {"ground": "ink"}))) == []


def test_a_pair_table_counts_toward_the_painted_tokens():
    knobs = {"ground": {"paper": {"--background": "#f4efe6"}}, "accent": {"blue": {"--primary": "#2f6fed"}}}
    pairs = [{"a": "ground", "b": "accent", "table": {"paper": {"blue": {"--foreground": "#1a1814"}}}}]
    assert _errors(_looks_package(_table(knobs, {"ground": "paper", "accent": "blue"}, pairs))) == []


@pytest.mark.parametrize("drop, named", [
    (("--background",), "paints no --background,"),
    (("--foreground",), "paints no --foreground,"),
    (("--background", "--foreground"), "paints no --background or --foreground,"),
])
def test_from_looks_whose_defaults_paint_no_ground_or_no_text_cannot_paint(drop, named):
    paper = {name: value for name, value in GROUND["paper"].items() if name not in drop}
    table = _table({"ground": {"paper": {**paper, "--hello-bg": "#f4efe6"}, "ink": GROUND["ink"]}}, {"ground": "paper"})
    assert named in _only_error(_looks_package(table), "MOBILE_CANNOT_PAINT", f"{PREPAINT_PATH}#defaults")


def test_only_the_defaults_look_counts_not_another_choice_that_paints():
    table = _table({"ground": {"paper": {"--foreground": "#1a1814"}, "ink": GROUND["ink"]}}, {"ground": "paper"})
    _only_error(_looks_package(table), "MOBILE_CANNOT_PAINT", f"{PREPAINT_PATH}#defaults")


def test_a_ground_value_that_is_not_a_mobile_colour_paints_nothing():
    paper = {"--background": "hsl(40 33% 94%)", "--foreground": "#1a1814"}  # CSS, but not "H S% L%" or #rrggbb
    message = _only_error(_looks_package(_table({"ground": {"paper": paper}}, {"ground": "paper"})),
                          "MOBILE_CANNOT_PAINT", f"{PREPAINT_PATH}#defaults")
    assert "paints no --background," in message


ACCENT = {"blue": {"--primary": "#2f6fed"}}


@pytest.mark.parametrize("defaults, pairs, numeric", [
    ({"ground": "paper", "accent": "violet"}, (), ()),  # a value the accent table lacks
    ({"ground": "paper", "accent": "blue"}, ({"a": "ground", "b": "accent", "table": {"ink": {"blue": {}}}},), ()),
    ({"ground": "paper", "accent": "blue", "glow": 40}, (), (
        {"var": "--hello-glow", "knob": "glow", "min": 0, "max": 30, "integer": True, "scale": 100, "capTable": None,
         "template": "rgba-alpha", "color": {"knob": "accent", "rgb": {"blue": "47,111,237"}}},
    )),
])
def test_defaults_that_do_not_fit_the_table_paint_nothing_and_cannot_paint(defaults, pairs, numeric):
    # paper alone paints the ground and its text, but the web paints nothing from a table its defaults don't fit
    table = json.loads(_table({"ground": GROUND, "accent": ACCENT}, defaults, pairs))
    message = _only_error(_looks_package(json.dumps({**table, "numeric": list(numeric)}).encode("utf-8")),
                          "MOBILE_CANNOT_PAINT", f"{PREPAINT_PATH}#defaults")
    assert "the web paints nothing from the table" in message and '"mobile.tokens"' in message


@pytest.mark.parametrize("table, code", [(b"{", "BAD_JSON"), (b"[]", "BAD_PREPAINT")])
def test_a_prepaint_table_that_cant_be_read_is_refused_once_for_what_it_is(table, code):
    assert [error["code"] for error in _errors(_looks_package(table))] == [code]


def test_a_tokens_block_needs_no_prepaint_table():
    assert _errors(_package({"tokens": TOKENS})) == []


# ---------------------------------------------------------------------------
# The mascot: clips need clip and still, stills the still, none nothing (PB-36)
# ---------------------------------------------------------------------------


def _mascot_package(mode: Optional[str], files: Optional[Mapping[str, Optional[bytes]]] = None,
                    variant: str = "hello-clip", states: Sequence[str] = (build.CLIP_STATE,)) -> bytes:
    mobile = {"tokens": TOKENS, **({} if mode is None else {"mascot": mode})}
    return _package(mobile, files, variant, states)


def test_every_mascot_mode_has_its_files():
    assert set(MOBILE_MASCOT_FILES) == set(contract.MOBILE_MASCOT_MODES)
    files = (MOBILE_MASCOT_FILES["clips"], MOBILE_MASCOT_FILES["stills"], MOBILE_MASCOT_FILES["none"])
    assert files == (("mp4", "png"), ("png",), ())


def test_clips_pass_with_the_hello_clip_variants_clip_and_still():
    package = read_theme_package(_mascot_package("clips"))
    assert package.manifest["mobile"]["mascot"] == "clips"
    assert {build.CLIP, build.STILL} <= set(package.files)


@pytest.mark.parametrize("dropped", [build.STILL, build.CLIP])
def test_clips_without_the_clip_or_the_still_is_missing_file_naming_it(dropped):
    message = _only_error(_mascot_package("clips", {dropped: None}), "MISSING_FILE", dropped)
    assert dropped in message


def test_a_webm_clip_does_not_stand_in_for_the_mp4():
    _only_error(_mascot_package("clips", {build.CLIP: None, WEBM_CLIP: WEBM}), "MISSING_FILE", build.CLIP)


def test_stills_need_only_the_still():
    assert _errors(_mascot_package("stills", {build.CLIP: None})) == []
    _only_error(_mascot_package("stills", {build.STILL: None}), "MISSING_FILE", build.STILL)


@pytest.mark.parametrize("mode", ["none", None], ids=["none", "absent"])
def test_none_or_no_mode_needs_no_mascot_file(mode):
    assert _errors(_mascot_package(mode, variant="hello")) == []


def test_every_state_is_checked_in_order():
    errors = _errors(_mascot_package("clips", states=(build.CLIP_STATE, WAVE)))
    assert [(error["code"], error["path"]) for error in errors] == [
        ("MISSING_FILE", f"assets/mascot/{WAVE}.mp4"), ("MISSING_FILE", f"assets/mascot/{WAVE}.png"),
    ]


# ---------------------------------------------------------------------------
# The codes and the corpus
# ---------------------------------------------------------------------------


def test_mobile_cannot_paint_is_a_package_code():
    assert "MOBILE_CANNOT_PAINT" in PACKAGE_ERROR_CODES


def test_the_corpus_carries_a_mobile_package_case_for_each_verdict():
    mobile = [case for case in CASES["packages"]["cases"] if case["name"].startswith("mobile")]
    verdicts = {case["code"] for case in mobile}
    assert {None, "BAD_FILE_TYPE", "NOT_ALLOWED", "MOBILE_CANNOT_PAINT", "MISSING_FILE", "BAD_MOBILE"} <= verdicts
