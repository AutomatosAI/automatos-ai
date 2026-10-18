"""PRD-261 A-06a — the .autotheme reader and its checks, held to the shared corpus.

``tests/fixtures/themes/cases.json`` is the corpus both validators run (A-06b runs the
TypeScript half): a case's ``code`` is null for a valid input, otherwise the check returns at
least one error and every error carries that code. The zip-level refusals a JSON corpus can't
carry (a symlink, zip64, an over-cap clip, encrypted or duplicate members) are written here.
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
from services.themes.css_scope import CSS_ERROR_CODES, css_scope_errors  # noqa: E402
from services.themes.css_value import PREPAINT_ERROR_CODES, css_value_problem, prepaint_errors  # noqa: E402
from services.themes.digest import package_digest, sha256_hex  # noqa: E402
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
    sections = ("manifests", "lookSchemas", "looks", "prepaint", "cssScope", "packages")
    codes = {case["code"] for name in sections for case in CASES[name]["cases"]} - {None}
    assert codes <= KNOWN_CODES


@pytest.mark.parametrize("case", CASES["manifests"]["cases"], ids=_ids(CASES["manifests"]))
def test_every_manifest_case_gives_its_code(case):
    result = validate_manifest(_patched(CASES["manifests"]["base"], case))
    _assert_verdict(result.errors, case["code"])
    assert (result.manifest is None) == (case["code"] is not None)


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
    _assert_verdict(prepaint_errors(case["document"]), case["code"])


@pytest.mark.parametrize("case", CASES["cssScope"]["cases"], ids=_ids(CASES["cssScope"]))
def test_every_css_scope_case_gives_its_code(case):
    _assert_verdict(css_scope_errors(case["css"], CASES["cssScope"]["htmlClass"]), case["code"])


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
    assert contract.THEME_HOST_VERSION == "1.0.0"
    assert contract.RESERVED_PUBLISHERS == ("automatos",)
    assert {"studio", "classic", "play", "dark", "light"} <= set(contract.RESERVED_HTML_CLASSES)
    assert contract.HTML_CLASS_OWNERS == {"play": "automatos/play"}
    assert set(contract.REQUIRED_SLOTS) == {"lookEditor@1", "styleSheet@1"} <= set(contract.KNOWN_SLOTS)


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
