"""PRD-261 A-06a — the .autotheme reader and its checks, held to the shared corpus.

``tests/fixtures/themes/cases.json`` is the corpus both validators run (A-06b runs the
TypeScript half): a case's ``code`` is null for a valid input, otherwise the check returns at
least one error and every error carries that code. The zip-level refusals a JSON corpus can't
carry (a symlink, zip64, an over-cap clip, encrypted or duplicate members) are written here, and
so is the check that the frontend's shared-modules.json carries the contract's versions (PA-2).
"""
from __future__ import annotations

import io
import json
import re
import stat
import struct
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config  # noqa: E402
from services.themes import contract  # noqa: E402
from services.themes.css_scope import CSS_ERROR_CODES, MAX_NESTING, css_scope_errors  # noqa: E402
from services.themes.css_value import (  # noqa: E402
    PREPAINT_ERROR_CODES, css_value_problem, prepaint_drop_errors, prepaint_errors,
)
from services.themes.digest import package_digest, sha256_hex  # noqa: E402
from services.themes.first_paint import FIRST_PAINT_ERROR_CODES, first_paint_table_errors  # noqa: E402
from services.themes.look_schema import LOOK_ERROR_CODES, validate_look, validate_look_schema  # noqa: E402
from services.themes.manifest import MANIFEST_ERROR_CODES, validate_manifest  # noqa: E402
from services.themes.package import PACKAGE_ERROR_CODES, ThemePackageError, read_theme_package  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

CASES = json.loads((Path(build.__file__).with_name("cases.json")).read_text(encoding="utf-8"))
KNOWN_CODES = frozenset(PACKAGE_ERROR_CODES + MANIFEST_ERROR_CODES + LOOK_ERROR_CODES + CSS_ERROR_CODES + PREPAINT_ERROR_CODES)
KB = 1024
MB = 1024 * KB
CENTRAL_FIXED = 46  # a central directory entry's fixed bytes, before the name
FLAG_BITS_AT = 8  # its general purpose bit flag (bit 0: encrypted)
EOCD = b"PK\x05\x06"
# PA-2: the browser's copy of the host and SDK versions. This path, named here literally, is what
# makes the backend planner (scripts/ci/backend_test_plan.py) select this file on a frontend change.
SHARED_MODULES = ORCH_ROOT.parent / "frontend/lib/theme-host/shared-modules.json"
# A-20 (QB-11): the <html> classes the host keeps, and the one it no longer reserves for any id.
HOST_HTML_CLASSES = ("studio", "classic", "dark", "light", "system")
FREED_HTML_CLASS = "play"
RESERVED_CODE = "RESERVED_HTML_CLASS"
FREED_CLASS_IDS_MIN = 3


def _patched(base: Mapping[str, Any], case: Mapping[str, Any]) -> Any:
    """The case's input: ``document`` as given, or ``base`` with ``set`` applied and ``unset`` removed."""
    if "document" in case:
        return case["document"]
    merged = {**base, **case.get("set", {})}
    return {key: value for key, value in merged.items() if key not in case.get("unset", [])}


def _assert_plain(errors: List[Dict[str, str]]) -> None:
    for error in errors:
        assert set(error) == {"code", "path", "message"}
        assert error["code"] in KNOWN_CODES
        assert isinstance(error["message"], str) and error["message"].strip()
        assert "\n" not in error["message"] and "Traceback" not in error["message"]


def _assert_verdict(errors: List[Dict[str, str]], code: Any) -> None:
    _assert_plain(errors)
    codes = [error["code"] for error in errors]
    if code is None:
        assert codes == [], errors
    else:
        assert codes and set(codes) == {code}, errors


def _ids(section: Mapping[str, Any]) -> List[str]:
    return [case["name"] for case in section["cases"]]


def _package_errors(data: bytes) -> List[Dict[str, str]]:
    try:
        read_theme_package(data)
    except ThemePackageError as exc:
        return exc.errors
    return []


def _codes(data: bytes) -> List[str]:
    errors = _package_errors(data)
    _assert_plain(errors)
    return [error["code"] for error in errors]


# ---------------------------------------------------------------------------
# The shared corpus
# ---------------------------------------------------------------------------


def test_the_corpus_bases_are_the_hello_fixture():
    hello = build.HELLO_DIR
    assert CASES["manifests"]["base"] == build.hello_manifest()
    assert CASES["looks"]["schema"] == json.loads((hello / "looks/schema.json").read_text(encoding="utf-8"))
    assert CASES["looks"]["base"] == json.loads((hello / "looks/presets/default.look.json").read_text(encoding="utf-8"))


def test_every_corpus_code_is_a_known_code():
    sections = ("manifests", "lookSchemas", "looks", "prepaint", "prepaintDrops", "firstPaint", "cssScope", "packages")
    codes = {case["code"] for name in sections for case in CASES[name]["cases"]} - {None}
    assert codes <= KNOWN_CODES


def _html_class_cases(section: str, field: str, html_class: str) -> List[Mapping[str, Any]]:
    return [case for case in CASES[section]["cases"] if case.get(field, {}).get("htmlClass") == html_class]


def test_the_corpus_gives_play_to_any_theme_and_refuses_each_host_class():
    # A-20: htmlClass 'play' is valid for any id, in theme.json and in a package; each class the host keeps is refused.
    freed = _html_class_cases("manifests", "set", FREED_HTML_CLASS)
    ids = {case["set"].get("id", CASES["manifests"]["base"]["id"]) for case in freed}
    assert len(ids) >= FREED_CLASS_IDS_MIN and [case["code"] for case in freed] == [None] * len(freed)
    assert [case["code"] for case in _html_class_cases("packages", "manifestSet", FREED_HTML_CLASS)] == [None]
    for html_class in HOST_HTML_CLASSES:
        assert {case["code"] for case in _html_class_cases("manifests", "set", html_class)} == {RESERVED_CODE}, html_class


@pytest.mark.parametrize("case", CASES["manifests"]["cases"], ids=_ids(CASES["manifests"]))
def test_every_manifest_case_gives_its_code(case):
    result = validate_manifest(_patched(CASES["manifests"]["base"], case))
    _assert_verdict(result.errors, case["code"])
    assert (result.manifest is None) == (case["code"] is not None)
    if "messages" in case:  # PRD-271: the layout's sentences, the same in both languages
        assert [error["message"] for error in result.errors] == case["messages"]


@pytest.mark.parametrize("case", CASES["lookSchemas"]["cases"], ids=_ids(CASES["lookSchemas"]))
def test_every_look_schema_case_gives_its_code(case):
    schema, errors = validate_look_schema(case["document"])
    _assert_verdict(errors, case["code"])
    assert (schema is None) == (case["code"] is not None)


@pytest.mark.parametrize("case", CASES["looks"]["cases"], ids=_ids(CASES["looks"]))
def test_every_look_case_gives_its_code(case):
    schema, schema_errors = validate_look_schema(CASES["looks"]["schema"])
    assert schema_errors == []
    _assert_verdict(validate_look(schema, _patched(CASES["looks"]["base"], case)), case["code"])


@pytest.mark.parametrize("value", CASES["cssValues"]["valid"])
def test_every_valid_css_value_passes(value):
    assert css_value_problem(value) is None


@pytest.mark.parametrize("value", CASES["cssValues"]["invalid"])
def test_every_invalid_css_value_is_refused_with_a_reason(value):
    problem = css_value_problem(value)
    assert isinstance(problem, str) and problem and "\n" not in problem


@pytest.mark.parametrize("case", CASES["prepaint"]["cases"], ids=_ids(CASES["prepaint"]))
def test_every_prepaint_case_gives_its_code(case):
    errors = prepaint_errors(case["document"])
    _assert_verdict(errors, case["code"])
    if "messages" in case:  # A-19c: the tone rule's sentences, the same in both languages
        assert [error["message"] for error in errors] == case["messages"]


@pytest.mark.parametrize("case", CASES["prepaintDrops"]["cases"], ids=_ids(CASES["prepaintDrops"]))
def test_every_prepaint_drop_case_gives_its_code_and_sentences(case):
    # The values the browser's first paint drops (boot-sanitize.ts), which the reader refuses (first_paint.py).
    errors = prepaint_drop_errors(case["document"])
    _assert_verdict(errors, case["code"])
    if "messages" in case:
        assert [error["message"] for error in errors] == case["messages"]


@pytest.mark.parametrize("case", CASES["firstPaint"]["cases"], ids=_ids(CASES["firstPaint"]))
def test_every_first_paint_case_gives_its_code_and_sentences(case):
    # What the browser keeps of a table, against the looks section's schema (first_paint.py; first-paint.ts its twin).
    schema, schema_errors = validate_look_schema(CASES["looks"]["schema"])
    assert schema_errors == []
    errors = first_paint_table_errors(schema, case["document"])
    _assert_verdict(errors, case["code"])
    assert {error["code"] for error in errors} <= set(FIRST_PAINT_ERROR_CODES)
    if "messages" in case:
        assert [error["message"] for error in errors] == case["messages"]


def test_the_corpus_has_a_valid_tone_rule_and_each_refusal_kind():
    tone_cases = [case for case in CASES["prepaint"]["cases"] if "tone" in (case["document"] or {})]
    assert any(case["code"] is None for case in tone_cases)
    refused = {message.split(": ", 1)[0] for case in tone_cases for message in case.get("messages", [])}
    assert {"boot/prepaint.json#tone", "boot/prepaint.json#tone.knob"} <= refused
    assert any(path.startswith("boot/prepaint.json#tone.values.") for path in refused)


@pytest.mark.parametrize("case", CASES["cssScope"]["cases"], ids=_ids(CASES["cssScope"]))
def test_every_css_scope_case_gives_its_code(case):
    _assert_verdict(css_scope_errors(case["css"], CASES["cssScope"]["htmlClass"]), case["code"])


# The scope check is one linear pass, whatever the stylesheet holds: a long run inside an unquoted
# url( or very deep nesting is refused without backtracking or copying (18 Oct review). With the old
# forms these inputs took quadratic time, so a regression shows up as a hung test.
LONG_RUN = 200_000
SCOPE_CLASS = CASES["cssScope"]["htmlClass"]


def test_a_long_whitespace_run_in_an_unclosed_plain_url_is_refused_in_one_pass():
    css = f"html.{SCOPE_CLASS}{{a:url(" + " " * LONG_RUN
    _assert_verdict(css_scope_errors(css, SCOPE_CLASS), "CSS_SYNTAX")


def test_brackets_nested_deeper_than_the_cap_are_refused():
    css = f"html.{SCOPE_CLASS}{{a:" + "(" * LONG_RUN + "}"
    _assert_verdict(css_scope_errors(css, SCOPE_CLASS), "CSS_SYNTAX")


def test_blocks_nested_deeper_than_the_cap_are_refused():
    css = f"html.{SCOPE_CLASS}{{" + "&.a{" * MAX_NESTING + "}" * (MAX_NESTING + 1)
    _assert_verdict(css_scope_errors(css, SCOPE_CLASS), "CSS_SYNTAX")


def test_blocks_nested_up_to_the_cap_are_read():
    css = f"html.{SCOPE_CLASS}{{" + "&.a{" * (MAX_NESTING - 1) + "}" * MAX_NESTING
    _assert_verdict(css_scope_errors(css, SCOPE_CLASS), None)


def _corpus_package(case: Mapping[str, Any]) -> bytes:
    replace: Dict[str, Any] = {
        path: None if text is None else text.encode("utf-8") for path, text in case.get("replace", {}).items()
    }
    if "manifestSet" in case or "manifestUnset" in case:
        replace = {build.MANIFEST: build.changed_manifest(case.get("manifestSet"), case.get("manifestUnset", ())), **replace}
    return build.build_package(case.get("variant", "hello"), replace=replace)


@pytest.mark.parametrize("case", CASES["packages"]["cases"], ids=_ids(CASES["packages"]))
def test_every_package_case_gives_its_code(case):
    _assert_verdict(_package_errors(_corpus_package(case)), case["code"])


# ---------------------------------------------------------------------------
# The package read
# ---------------------------------------------------------------------------


def test_the_hello_package_reads_with_its_manifest_files_and_digest():
    package = read_theme_package(build.build_package())
    assert package.manifest["id"] == "automatos-e2e/hello"
    assert package.files["dist/entry.js"].content_type == "text/javascript"
    assert package.files["dist/theme.css"].content_type == "text/css"
    assert set(package.file_manifest) == set(build.hello_members())
    manifest_bytes = build.hello_members()["theme.json"]
    assert package.file_manifest["theme.json"] == {
        "type": "application/json", "size": len(manifest_bytes), "sha256": sha256_hex(manifest_bytes),
    }
    assert re.fullmatch(r"[0-9a-f]{64}", package.digest)


def test_the_digest_is_the_sha256_of_the_sorted_path_and_hash_lines():
    assert package_digest({"b.js": "22", "a.js": "11"}) == sha256_hex(b"a.js\x0011\nb.js\x0022\n")


def test_the_digest_is_stable_across_member_order_and_compression():
    items = sorted(build.hello_members().items())
    forward = read_theme_package(build.build_zip(items))
    backward = read_theme_package(build.build_zip(list(reversed(items)), compression=zipfile.ZIP_STORED))
    assert forward.digest == backward.digest == build.members_digest(dict(items))
    assert forward.file_manifest == backward.file_manifest


@pytest.mark.parametrize("variant", build.VARIANTS)
def test_the_cli_writes_a_package_read_theme_package_accepts(variant, tmp_path, capsys):
    out = tmp_path / f"{variant}.autotheme"
    assert build.main(["--variant", variant, "--out", str(out)]) == 0
    package = read_theme_package(out.read_bytes())
    assert package.digest == build.members_digest(build.hello_members(variant))
    assert package.digest in capsys.readouterr().out


def test_the_clip_variant_carries_a_playable_clip_and_its_still():
    package = read_theme_package(build.build_package("hello-clip"))
    assert package.manifest["mascot"] == {"states": [build.CLIP_STATE]}
    assert package.files[build.CLIP].content_type == "video/mp4"
    assert package.files[build.STILL].content_type == "image/png"


def test_the_throws_variant_swaps_only_the_entry():
    hello, throws = build.hello_members(), build.hello_members("hello-throws")
    assert {path for path in hello if hello[path] != throws[path]} == {build.ENTRY}


# ---------------------------------------------------------------------------
# Zip-level refusals the corpus can't carry
# ---------------------------------------------------------------------------


def test_a_package_over_the_cap_is_too_large(monkeypatch):
    monkeypatch.setattr(config, contract.CAP_SETTINGS["package"].config_name, 1, raising=False)
    error = pytest.raises(ThemePackageError, read_theme_package, b"PK\x03\x04" + b"\x00" * MB).value
    assert [item["code"] for item in error.errors] == ["PACKAGE_TOO_LARGE"]
    assert error.too_large


@pytest.mark.parametrize("data", [b"", b"not a zip", b"PK\x03\x04 but not a zip", b"%PDF-1.7"])
def test_anything_but_a_zip_is_refused(data):
    assert _codes(data) == ["NOT_A_ZIP"]


def test_a_symlink_is_refused():
    link = zipfile.ZipInfo("dist/chunks/link.js")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert _codes(build.build_package(extra=[(link, b"/etc/passwd")])) == ["SYMLINK"]


def test_a_zip64_end_record_is_refused():
    data = bytearray(build.build_package())
    at = bytes(data).rfind(EOCD)
    data[at + 8:at + 10] = b"\xff\xff"  # entries on this disk: zip64's marker
    assert _codes(bytes(data)) == ["ZIP64"]


def test_a_zip64_locator_is_refused():
    data = build.build_package()
    at = data.rfind(EOCD)
    locator = b"PK\x06\x07" + b"\x00" * 16
    assert _codes(data[:at] + locator + data[at:]) == ["ZIP64"]


def test_an_svg_is_refused_wherever_it_sits():
    extra = [("assets/brand/logo.svg", b"<svg/>"), ("assets/previews/home.SVG", b"<svg/>")]
    assert _codes(build.build_package(extra=extra)) == ["SVG_NOT_ALLOWED", "SVG_NOT_ALLOWED"]


def test_a_dot_dot_path_is_refused():
    assert _codes(build.build_package(extra=[("looks/../theme.json", b"{}")])) == ["UNSAFE_PATH"]


def test_a_clip_over_its_cap_is_refused():
    cap = contract.CAP_SETTINGS["clip"].default * contract.CAP_SETTINGS["clip"].unit
    clip = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * cap
    assert _codes(build.build_package("hello-clip", replace={build.CLIP: clip})) == ["CLIP_TOO_LARGE"]


def test_an_entry_over_its_cap_is_refused():
    entry = b"export default 1\n" + b"/" * (300 * KB)
    assert _codes(build.build_package(replace={build.ENTRY: entry})) == ["ENTRY_TOO_LARGE"]


def test_js_and_css_over_the_code_cap_together_are_refused():
    chunk = b"/" * (1600 * KB)  # each under the 3 MB cap, together over it
    extra = [("dist/chunks/a.js", chunk), ("dist/chunks/b.js", chunk)]
    assert _codes(build.build_package(extra=extra)) == ["CODE_TOO_LARGE"]


def test_more_members_than_a_package_holds_are_refused():
    extra = [(f"dist/chunks/c{index}.js", b"") for index in range(500)]
    assert _codes(build.build_package(extra=extra)) == ["TOO_MANY_FILES"]


def test_a_duplicate_member_is_refused():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about the duplicate it is asked to write
        data = build.build_package(extra=[("README.md", b"# again")])
    assert _codes(data) == ["DUPLICATE_FILE"]


def test_a_member_differing_only_in_capitals_is_a_duplicate():
    # Both unpack to one file on a case-insensitive disk (macOS, Windows), so one would hide the other.
    assert _codes(build.build_package(extra=[("readme.md", b"# again")])) == ["DUPLICATE_FILE"]


def test_a_style_sheet_starting_with_a_bom_is_read_without_it():
    css = (build.HELLO_DIR / "dist/theme.css").read_bytes()
    assert _codes(build.build_package(replace={"dist/theme.css": b"\xef\xbb\xbf" + css})) == []


def _with_flag(data: bytes, name: str, flag: int) -> bytes:
    """``data`` with ``flag`` written into ``name``'s central directory entry."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        position = zf.start_dir
    out = bytearray(data)
    while out[position:position + 4] == b"PK\x01\x02":
        name_len, extra_len, comment_len = struct.unpack("<HHH", out[position + 28:position + 34])
        if out[position + CENTRAL_FIXED:position + CENTRAL_FIXED + name_len].decode("utf-8") == name:
            out[position + FLAG_BITS_AT:position + FLAG_BITS_AT + 2] = flag.to_bytes(2, "little")
            return bytes(out)
        position += CENTRAL_FIXED + name_len + extra_len + comment_len
    raise AssertionError(f"{name} is not in the zip")


def test_an_encrypted_member_is_refused():
    assert _codes(_with_flag(build.build_package(), "README.md", 0x1)) == ["ENCRYPTED"]


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_the_contract_pins_the_versions_and_the_reserved_names():
    # PB-3 (D7, QB-3): host 1.1.0 for the boot tone rule and looks presets. Theme voice (C5): SDK 1.2.0 for its
    # voice area; PRD-268 (K12): SDK 1.3.0 for its proactive area. PRD-271 C02: host 1.2.0 for the mobile block's
    # layout and slots, which a theme may use only on a 1.2.0 host (LAYOUT_MIN_HOST).
    assert contract.THEME_HOST_VERSION == "1.2.0"
    assert (contract.MOBILE_MIN_HOST, contract.LAYOUT_MIN_HOST) == ("1.1.0", "1.2.0")
    assert contract.SDK_VERSION == "1.3.0"
    assert contract.RESERVED_PUBLISHERS == ("automatos",)
    assert contract.RESERVED_HTML_CLASSES == HOST_HTML_CLASSES
    assert not hasattr(contract, "HTML_CLASS_OWNERS")  # A-20: no theme's id keeps a reserved class
    assert set(contract.REQUIRED_SLOTS) == {"lookEditor@1", "styleSheet@1"} <= set(contract.KNOWN_SLOTS)


def test_the_contract_pins_the_tone_rule_and_the_boot_preset_caps():
    # A-19c (plan §2.2.3): a tone rule of a knob and its choices' tones; at most 12 presets of 4 KB each.
    assert (contract.TONE_RULE_KEY, contract.TONE_RULE_FIELDS, contract.TONE_RULE_TONES) == ("tone", ("knob", "values"), ("light", "dark"))
    assert (contract.BOOT_PRESETS_MAX, contract.BOOT_PRESET_MAX_BYTES) == (12, 4 * KB)
    assert "BAD_TONE_RULE" in PREPAINT_ERROR_CODES


def test_the_frontend_carries_the_same_host_and_sdk_versions():
    table = json.loads(SHARED_MODULES.read_text(encoding="utf-8"))
    assert table["hostVersion"] == contract.THEME_HOST_VERSION
    assert table["sdkVersion"] == contract.SDK_VERSION


def test_every_page_in_scope_has_sources_and_no_host_route_is_in_scope():
    assert set(contract.PAGE_SOURCES) == set(contract.PAGE_SCOPE) == set(contract.PAGE_AREAS)
    assert set(contract.PAGE_AREAS.values()) == {
        "chat", "deliverables", "missions", "agents", "knowledge", "playbooks", "command-center", "home",
    }
    owned = [
        route for route in contract.PAGE_SCOPE for host in contract.HOST_OWNED_ROUTES
        if route == host or route.startswith(f"{host}/")
    ]
    assert owned == []
    assert len(contract.page_source_files()) == sum(len(paths) for paths in contract.PAGE_SOURCES.values())
