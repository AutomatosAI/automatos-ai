"""PRD-262 #24 (the 19 Oct breakdown): the frontend runtime's node binary is stripped.

The official node:24-alpine binary keeps a symbol table and debug sections that nothing
loads (.strtab 11.40, .symtab 5.60, .debug_* 1.77 MB on arm64). The Dockerfile's
node-strip stage, built on base, installs binutils and runs `strip --strip-all -p` on
base's /usr/local/bin/node: .dynsym stays, so JS stack traces, the V8 snapshot and the
exported symbols are unchanged, and -p keeps the file's dates, so a rerun of the stage
(after a lock change, say) makes the same layer. The build fails if readelf still lists a
symbol table or a debug section. The runtime copies the binary from that stage, so the
Node pin stays on base's FROM line and binutils never ships.

In CI, frontend/scripts/check-standalone-image.sh runs node in the built image and has it
read its own section headers (the image has no readelf): test.yml's prod-image-parity on
both editions, images.yml on the image each build pushed. The section reader is run here
against small ELF files made in the test.

File reads, plus the reader under the host's node when there is one.
"""
from __future__ import annotations

import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "frontend" / "Dockerfile").read_text()
CHECK_SCRIPT = (ROOT / "frontend" / "scripts" / "check-standalone-image.sh").read_text()
NODE = "/usr/local/bin/node"
STRIP = f"strip --strip-all -p {NODE}"
COPY_NODE = f"COPY --from=node-strip {NODE} {NODE}"
_FROM = re.compile(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", re.M | re.I)

ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
ELF64_LITTLE_ENDIAN = b"\x7fELF\x02\x01\x01" + bytes(9)


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in code.replace("\\\n", " ").splitlines() if line.strip()]


def _stage(name: str) -> list[str]:
    """The instructions of stage `name`, without its FROM line."""
    match = re.search(rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}\s*$", DOCKERFILE, re.M | re.I)
    assert match, f"no stage named {name}"
    rest = DOCKERFILE[match.end():]
    following = re.search(r"^FROM\s", rest, re.M)
    return _instructions(rest[: following.start()] if following else rest)


def _stage_images() -> dict[str, str]:
    return {name.lower(): image for image, name in _FROM.findall(DOCKERFILE) if name}


# ── the Dockerfile ──────────────────────────────────────────────────────────


def test_node_strip_is_built_on_base_and_strips_its_node_in_one_run():
    assert _stage_images()["node-strip"] == "base", "base's binary, so the Node pin stays on base's FROM line"
    runs = [line for line in _stage("node-strip") if line.startswith("RUN")]
    assert len(runs) == 1, runs
    run = " ".join(runs[0].split())
    assert run.startswith("RUN apk add --no-cache binutils && ")
    assert f" {STRIP} && " in run
    # The build fails while readelf still lists a symbol table or a debug section, and the binary still runs.
    assert "if readelf -S --wide /usr/local/bin/node | grep -E '] \\.(symtab|strtab|debug_)'; then exit 1; fi" in run
    assert run.endswith("&& node -v")


def test_the_runtime_takes_the_stripped_binary_and_never_binutils():
    production = _stage("production")
    assert COPY_NODE in production
    assert [line for line in production if f"{NODE} " in line] == [COPY_NODE]
    runs = "\n".join(line for line in production if line.startswith("RUN"))
    for tool in ("binutils", "strip", "readelf"):
        assert not re.search(rf"\b{tool}\b", runs), tool
    assert _FROM.findall(DOCKERFILE)[-1][1].lower() == "production", "production stays the last stage"
    assert not _stage_images()["production"].startswith("node-strip"), "node-strip is copied from, never built on"


def test_only_the_node_strip_stage_installs_binutils():
    stages_with_binutils = [
        name for name in _stage_images() if any("binutils" in line for line in _stage(name))
    ]
    assert stages_with_binutils == ["node-strip"]


# ── the in-image check ──────────────────────────────────────────────────────


def _reader() -> str:
    match = re.search(r"^ELF_SECTIONS_JS='(.*?)'$", CHECK_SCRIPT, re.M | re.S)
    assert match, "check-standalone-image.sh defines no ELF_SECTIONS_JS"
    return match.group(1)


def test_the_check_script_runs_node_and_reads_its_sections_in_the_image():
    assert 'docker run --rm "$IMAGE" node -v' in CHECK_SCRIPT
    assert 'docker run --rm "$IMAGE" node -e "$ELF_SECTIONS_JS"' in CHECK_SCRIPT
    assert re.search(r"^check_node$", CHECK_SCRIPT, re.M), "the script runs check_node"
    assert "process.argv[1] || process.execPath" in _reader(), "the image's own node binary by default"


def _elf(section_names: list[str]) -> bytes:
    """A 64-bit little-endian ELF file holding only the named, empty sections."""
    names = ["", *section_names, ".shstrtab"]
    table = b"\0".join(name.encode() for name in names) + b"\0"
    offsets = [table.index(name.encode() + b"\0") if name else 0 for name in names]
    table_at = ELF_HEADER.size
    headers_at = table_at + len(table) + (-len(table)) % 8
    sections = b"".join(
        SECTION_HEADER.pack(offset, 3 if name == ".shstrtab" else 1, 0, 0,
                            table_at if name == ".shstrtab" else 0, len(table) if name == ".shstrtab" else 0,
                            0, 0, 1, 0)
        for name, offset in zip(names, offsets)
    )
    header = ELF_HEADER.pack(ELF64_LITTLE_ENDIAN, 2, 0x3E, 1, 0, 0, headers_at, 0, ELF_HEADER.size, 0, 0,
                             SECTION_HEADER.size, len(names), len(names) - 1)
    return header + table + bytes(headers_at - table_at - len(table)) + sections


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def _read_sections(tmp_path: Path, data: bytes) -> subprocess.CompletedProcess:
    binary = tmp_path / "binary"
    binary.write_bytes(data)
    return subprocess.run(["node", "-e", _reader(), str(binary)], capture_output=True, text=True, timeout=30)


@needs_node
def test_the_reader_lists_the_sections_strip_removes(tmp_path):
    unstripped = [".text", ".dynsym", ".dynstr", ".debug_info", ".debug_line", ".symtab", ".strtab"]
    proc = _read_sections(tmp_path, _elf(unstripped))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == [".debug_info", ".debug_line", ".symtab", ".strtab"]


@needs_node
def test_the_reader_prints_nothing_for_a_stripped_binary(tmp_path):
    proc = _read_sections(tmp_path, _elf([".text", ".dynsym", ".dynstr", ".gnu_debuglink"]))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@needs_node
def test_the_reader_fails_on_a_file_that_is_not_elf(tmp_path):
    proc = _read_sections(tmp_path, b"#!/bin/sh\necho not a binary\n" * 4)
    assert proc.returncode != 0
    assert "not a 64-bit little-endian ELF file" in proc.stderr
