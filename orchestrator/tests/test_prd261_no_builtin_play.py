"""PRD-261 A-19f3 — the backend no longer knows builtin Play.

Play is an installed theme like any other: its package lives in the theme kit, and core keeps
no choice, constant or look schema of its own for it. This guard reads every file under
``orchestrator/`` and fails on any name builtin Play left behind: the stored choice string, the
server's constants and the committed look schema's file name. It skips only itself.

Two cases keep it from passing on nothing: the walk must reach the files A-19f3 changed, and
the same scan run over a sample holding each banned name must find every one.

Pure file reads: no database, no network.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List

ORCH_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
BANNED = ("builtin:play", "BUILTIN_PLAY", "play-look-schema")
# Caches and environments a local run or CI leaves under orchestrator/: never the source.
SKIPPED_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules", ".git",
})
# The files A-19f3 took builtin Play out of: the walk has to read each.
MUST_SCAN = (
    "services/themes/choices.py",
    "services/themes/defaults.py",
    "services/themes/looks.py",
    "api/themes_install.py",
    "api/themes_looks.py",
    "tests/test_prd261_theme_install.py",
    "tests/test_prd261_theme_looks.py",
)
# Deleted with builtin Play (A-19f3): the server no longer holds Play's look schema.
DELETED = ("services/themes/data/play-look-schema.json", "tests/test_prd261_play_schema_parity.py")


def _files(root: Path) -> Iterator[Path]:
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in SKIPPED_DIRS)
        for name in sorted(names):
            path = Path(directory, name)
            if path.resolve() != THIS_FILE and path.is_file():
                yield path


def _hits(root: Path) -> Dict[str, List[str]]:
    """Each banned name, with the files under ``root`` (relative paths) named by it or holding it."""
    found: Dict[str, List[str]] = {needle: [] for needle in BANNED}
    for path in _files(root):
        relative, data = path.relative_to(root).as_posix(), path.read_bytes()
        for needle in BANNED:
            if needle in relative or needle.encode("utf-8") in data:
                found[needle].append(relative)
    return found


def test_no_file_under_orchestrator_names_builtin_play():
    hits = {needle: paths for needle, paths in _hits(ORCH_ROOT).items() if paths}
    assert hits == {}, f"builtin Play is gone from core; remove these names: {hits}"


def test_the_walk_reads_the_files_builtin_play_left():
    scanned = {path.relative_to(ORCH_ROOT).as_posix() for path in _files(ORCH_ROOT)}
    assert [path for path in MUST_SCAN if path not in scanned] == []
    assert THIS_FILE.relative_to(ORCH_ROOT).as_posix() not in scanned


def test_the_scan_finds_each_banned_name(tmp_path):
    for index, needle in enumerate(BANNED):
        (tmp_path / f"sample_{index}.py").write_text(f"VALUE = {needle!r}\n", encoding="utf-8")
    (tmp_path / "clean.py").write_text("VALUE = 'studio'\n", encoding="utf-8")
    cached = tmp_path / "__pycache__"
    cached.mkdir()
    (cached / "stale.pyc").write_bytes(BANNED[0].encode("utf-8"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / f"{BANNED[2]}.json").write_text("{}\n", encoding="utf-8")
    hits = {needle: sorted(paths) for needle, paths in _hits(tmp_path).items()}
    expected = {needle: [f"sample_{index}.py"] for index, needle in enumerate(BANNED)}
    assert hits == {**expected, BANNED[2]: [f"data/{BANNED[2]}.json", "sample_2.py"]}


def test_plays_server_look_schema_and_its_parity_test_are_deleted():
    assert [path for path in DELETED if (ORCH_ROOT / path).exists()] == []
