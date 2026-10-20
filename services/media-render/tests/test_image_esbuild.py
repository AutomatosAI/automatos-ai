"""GO-2026-4337 and GO-2026-6612: the image's esbuild is built with a patched Go.

Hyperframes requires esbuild ^0.25.12, and npm's 0.25.12 binary was built with Go
1.23.12. The Dockerfile's esbuild stage builds the same version from source on Go's
patched 1.26 line, and the js stage puts it where npm put its own, with
esbuild/bin/esbuild hard-linked to it. This reads the Go release each esbuild
binary in the js tree records in its build info, and runs it through esbuild's
JavaScript API, as every render does. Runs inside the image (the media-render CI job).
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

JS_ROOT = Path("/opt/media-render/js")
JS_TREE = JS_ROOT / "node_modules"
# GO-2026-4337 (crypto/tls) is fixed in Go 1.24.13, 1.25.7 and 1.26; GO-2026-6612
# (net/http) in 1.26.9 (and 1.27.2).
PATCHED_GO = (1, 26, 9)
# Go 1.18 and later: the build info blob starts with this magic, its flags byte
# sits at offset 15, and with the inline-strings flag the Go version follows the
# 32-byte header as a uvarint-length-prefixed string (debug/buildinfo).
BUILDINFO_MAGIC = b"\xff Go buildinf:"
BUILDINFO_FLAGS_AT = 15
BUILDINFO_HEADER_BYTES = 32
INLINE_STRINGS = 0x2
ELF_MAGIC = b"\x7fELF"
NPM_ARCH = {"x86_64": "x64", "aarch64": "arm64"}
TRANSFORM = "console.log(require('esbuild').transformSync('let a = 1').code)"


def _uvarint(data: bytes, at: int) -> Tuple[int, int]:
    """The unsigned varint at ``at`` and the offset just past it."""
    value, shift = 0, 0
    while True:
        byte = data[at]
        at += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, at
        shift += 7


def _go_version(binary: Path) -> Tuple[int, ...]:
    """The Go release ``binary`` was built with, read from its build info."""
    data = binary.read_bytes()
    start = data.find(BUILDINFO_MAGIC)
    assert start >= 0, f"{binary} carries no Go build info"
    assert data[start + BUILDINFO_FLAGS_AT] & INLINE_STRINGS, f"{binary}: build info older than Go 1.18"
    length, at = _uvarint(data, start + BUILDINFO_HEADER_BYTES)
    version = data[at : at + length].decode()
    match = re.fullmatch(r"go(\d+)\.(\d+)(?:\.(\d+))?", version)
    assert match, f"{binary}: unexpected Go version {version!r}"
    return tuple(int(part or 0) for part in match.groups())


def _esbuild_binaries() -> List[Path]:
    """Every esbuild executable in the js tree (not the .bin link)."""
    return sorted(
        path for path in JS_TREE.rglob("esbuild")
        if path.is_file() and not path.is_symlink() and path.read_bytes()[:4] == ELF_MAGIC
    )


def test_the_js_tree_holds_one_esbuild_binary_where_npm_put_it():
    arch = NPM_ARCH[platform.machine()]
    binaries = _esbuild_binaries()
    relative = {str(path.relative_to(JS_TREE)) for path in binaries}
    assert relative == {"esbuild/bin/esbuild", f"@esbuild/linux-{arch}/bin/esbuild"}
    assert len({path.stat().st_ino for path in binaries}) == 1, "the two paths are one hard-linked file"


def test_every_esbuild_binary_is_built_with_a_patched_go():
    versions = {str(path): _go_version(path) for path in _esbuild_binaries()}
    assert versions
    assert {path: version for path, version in versions.items() if version < PATCHED_GO} == {}


def test_esbuild_runs_through_its_javascript_api_at_the_locked_version():
    locked = json.loads((JS_TREE / "esbuild" / "package.json").read_text())["version"]
    for binary in _esbuild_binaries():
        run = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30)
        assert (run.returncode, run.stdout.strip()) == (0, locked), run.stderr
    run = subprocess.run(["node", "-e", TRANSFORM], cwd=JS_ROOT, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "let a = 1;"
